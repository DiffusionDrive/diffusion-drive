import torch

from src.planner.diffusion_planner.diffusion_utils.sde import VPSDE_linear
from src.planner.diffusion_planner.guidance.collision import collision_guidance_fn

N = 1
sde = VPSDE_linear()

def goal_guidance_fn(x_in, t_input, cond, **kwargs):
    """Softly guide the final ego position to a reachable route waypoint (meters)."""
    inputs = kwargs["inputs"]  # already denormalized by GuidanceWrapper
    route = inputs["route_lanes"][..., :2]  # [B, R, L, 2]
    ego = inputs["ego_current_state"]
    targets = []
    valid_batches = []
    horizon_seconds = 8.0  # 80 samples at 0.1 s; adjust if model differs
    for b in range(route.shape[0]):
        # Deduplicate repeated/padded route points and ignore all-zero slots.
        points = route[b].reshape(-1, 2)
        valid = points.abs().sum(dim=-1) > 1e-5
        points = points[valid]
        if points.numel() == 0:
            continue
        keep = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        keep[1:] = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1) > 1e-3
        points = points[keep]
        if points.shape[0] == 0:
            continue
        # Begin at the closest route point, rather than assuming index zero is nearby.
        origin = ego[b, :2]
        start = torch.argmin(torch.linalg.vector_norm(points - origin, dim=-1))
        points = torch.cat((origin[None], points[start:]), dim=0)
        if points.shape[0] < 2:
            continue
        distances = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
        cumulative = torch.cat((distances.new_zeros(1), distances.cumsum(0)))
        speed = torch.linalg.vector_norm(ego[b, 4:6]).detach()
        travel = torch.clamp(speed * horizon_seconds, min=3.0, max=50.0)
        idx = int(torch.searchsorted(cumulative.contiguous(), travel).item())
        if idx >= len(points):
            target = points[-1]
        elif idx == 0:
            target = points[0]
        else:
            fraction = (travel - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1]).clamp_min(1e-6)
            target = points[idx - 1] + fraction.clamp(0, 1) * (points[idx] - points[idx - 1])
        targets.append(target.detach())
        valid_batches.append(b)
    if not targets:
        return x_in.sum() * 0.0
    target_xy = torch.stack(targets)
    final_xy = x_in[valid_batches, 0, -1, :2]
    # Conservative starting strength; tune after examining unguided predictions.
    return -2.0 * torch.nn.functional.smooth_l1_loss(final_xy, target_xy, reduction="sum", beta=5.0)

class GuidanceWrapper:
    def __init__(self):
        # Ensure goal_guidance is active
        self._guidance_fns = [collision_guidance_fn, goal_guidance_fn]

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """
        This function is a wrapper for the guidance functions in the model.
        """
        energy = 0

        state_normalizer = kwargs["state_normalizer"]
        observation_normalizer = kwargs["observation_normalizer"]

        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]

        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x_in = x_in + x_fix.reshape(B, P, -1)

        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))

        # We must assign the denormalized dict to a NEW variable so we don't accidentally
        # un-normalize the PyTorch tensors for the rest of the DPM-Solver timesteps!
        denorm_inputs = observation_normalizer.inverse(kwargs["inputs"])
        kwargs["inputs"] = denorm_inputs

        for guidance_fn in self._guidance_fns:
            energy += guidance_fn(x_in, t_input, cond, **kwargs)

        assert not torch.isnan(energy).any()

        return energy