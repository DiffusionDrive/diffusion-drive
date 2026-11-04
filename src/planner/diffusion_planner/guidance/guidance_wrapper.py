"""Soft, lane-width-aware guidance for diffusion samples.

Route geometry is used only to guide sampling. It never replaces model output.
All trajectories still require independent collision and dynamics validation.
"""
import torch
import torch.nn.functional as F


def _ordered_route_geometry(route, max_join_gap=3.0):
    """Return connected (center, half_width) samples from route [R,L,12]."""
    pieces = []
    for lane in route:
        valid = lane.abs().sum(dim=-1) > 1e-6
        data = lane[valid]
        if len(data) < 2:
            continue
        keep = torch.cat((torch.ones(1, dtype=torch.bool, device=data.device),
                          torch.linalg.vector_norm(data[1:, :2] - data[:-1, :2], dim=-1) > 1e-4))
        data = data[keep]
        if len(data) < 2:
            continue
        if pieces and torch.linalg.vector_norm(pieces[-1][-1, :2] - data[0, :2]).item() > max_join_gap:
            break
        pieces.append(data if not pieces else data[1:])
    if not pieces:
        return None, None
    joined = torch.cat(pieces, dim=0)
    left = torch.linalg.vector_norm(joined[:, 4:6], dim=-1)
    right = torch.linalg.vector_norm(joined[:, 6:8], dim=-1)
    # Conservative symmetric usable half-width, with fallback for missing boundaries.
    half_width = torch.minimum(left, right)
    half_width = torch.where(half_width > 0.2, half_width,
                             torch.full_like(half_width, 1.75))
    return joined[:, :2], half_width


def _ordered_route_points(route):
    """Backward-compatible helper used by diagnostics."""
    points, _ = _ordered_route_geometry(route)
    return points


class GuidanceWrapper:
    def __init__(self, route_weight=0.6, samples=12,
                 boundary_weight=0.25, vehicle_width=2.30, clearance_margin=0.20,
                 heading_weight=0.25):
        self.route_weight = float(route_weight)
        self.samples = int(samples)
        self.boundary_weight = float(boundary_weight)
        self.vehicle_width = float(vehicle_width)
        self.clearance_margin = float(clearance_margin)
        self.heading_weight = float(heading_weight)
        self._guidance_fns = [self.route_energy]

    def route_energy(self, predicted, route):
        points, half_widths = _ordered_route_geometry(route)
        if points is None or len(points) < 2 or len(predicted) == 0:
            return predicted.sum() * 0.0
        edges = points[1:] - points[:-1]
        lengths = torch.linalg.vector_norm(edges, dim=-1)
        good = lengths > 1e-4
        start, edges, lengths = points[:-1][good], edges[good], lengths[good]
        widths0, widths1 = half_widths[:-1][good], half_widths[1:][good]
        if lengths.numel() == 0:
            return predicted.sum() * 0.0
        arc = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        idx = torch.linspace(0, len(predicted)-1, min(self.samples, len(predicted)),
                             device=predicted.device).round().long()
        xy = predicted[idx, :2]
        delta = xy[:, None, :] - start[None, :, :]
        fraction = ((delta * edges[None]).sum(-1) /
                    lengths.square().clamp_min(1e-6)[None]).clamp(0, 1)
        projected = start[None] + fraction[..., None] * edges[None]
        distances2 = (xy[:, None] - projected).square().sum(-1)
        nearest = distances2.argmin(dim=1)
        selected_fraction = fraction.gather(1, nearest[:, None]).squeeze(1)
        offset = torch.sqrt(distances2.gather(1, nearest[:, None]).squeeze(1) + 1e-6)
        progress = (arc[:-1][None] + fraction * lengths[None]).gather(
            1, nearest[:, None]).squeeze(1)
        available_half_width = (widths0[nearest] * (1-selected_fraction)
                                + widths1[nearest] * selected_fraction)
        # A point near the boundary receives a stronger, smooth penalty.
        # This is an approximate center-clearance cost, not footprint validation.
        clearance = (available_half_width - self.vehicle_width/2
                     - self.clearance_margin).clamp_min(0.0)
        edge_violation = F.softplus((offset - clearance).clamp(-10.0, 10.0) * 2.0) / 2.0
        lateral = F.smooth_l1_loss(offset.clamp(max=10.0), torch.zeros_like(offset),
                                   reduction='mean', beta=2.0)
        boundary = F.smooth_l1_loss(edge_violation, torch.zeros_like(edge_violation), reduction="mean", beta=1.0)
        backward = (torch.relu(progress[:-1] - progress[1:]).clamp(max=10.0).mean()
                    if len(progress) > 1 else lateral * 0.0)
        origin = predicted.new_zeros((1, 2))
        t0 = (((origin - start) * edges).sum(-1) /
              lengths.square().clamp_min(1e-6)).clamp(0, 1)
        q0 = start + t0[:, None] * edges
        j0 = torch.linalg.vector_norm(q0 - origin, dim=-1).argmin()
        s0 = (arc[:-1] + t0 * lengths)[j0].detach()
        forward = torch.clamp(progress[-1] - s0, min=0, max=30) / 30.0
        # Directional guidance uses the tangent of the nearest ROUTE segment.
        # It is a differentiable soft cost on the model output, not a path replacement.
        # Evaluate only where predicted displacement is meaningful; otherwise a
        # stopped prediction would receive an arbitrary heading penalty.
        sample_delta = xy[1:] - xy[:-1]
        travel = torch.linalg.vector_norm(sample_delta, dim=-1)
        route_tangent = edges[nearest[:-1]] / lengths[nearest[:-1], None].clamp_min(1e-6)
        model_tangent = sample_delta / travel[:, None].clamp_min(1e-6)
        heading_error = (1.0 - (route_tangent * model_tangent).sum(-1).clamp(-1, 1))
        heading_error = (heading_error * (travel > 0.15).to(travel.dtype)).sum() / (travel > 0.15).to(travel.dtype).sum().clamp_min(1)
        return -self.route_weight * (lateral + self.boundary_weight * boundary
                                     + 0.5 * backward - 0.2 * forward
                                     + self.heading_weight * heading_error)

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        state_norm = kwargs['state_normalizer']
        obs_norm = kwargs['observation_normalizer']
        with torch.enable_grad():
            b, p, _ = x_in.shape
            states = state_norm.inverse(x_in.reshape(b, p, -1, 4))
            route = obs_norm.inverse(kwargs['inputs'])['route_lanes']
            energy = states.sum() * 0.0
            for i in range(b):
                energy = energy + self.route_energy(states[i, 0, 1:], route[i])
            return energy
