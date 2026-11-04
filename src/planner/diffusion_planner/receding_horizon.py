"""Diffusion receding horizon planner with experimental route guidance.

Route-guided candidates blend model predictions with map centreline geometry.
All executed prefixes still undergo road, collision and dynamics validation.
This is an experimental simulation planner, not safety certified.
"""
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from time import perf_counter
import numpy as np
import torch
from commonroad.scenario.state import KSState
from src.core.types import PlannerInput
from src.planner.diffusion_planner.adapter import (
    extract_map_tensors, extract_neighbor_tensors, extract_navigation_route,
    extract_static_objects, normalize_to_ego_frame,
    select_initial_route_lanelet_ids, route_remaining_ids,
)
from src.planner.diffusion_planner.trajectory_validation import (
    local_to_global_states, check_collision_and_road, check_vehicle_dynamics,
)
from src.planner.diffusion_planner.collision_aware_speed import collision_aware_stop


@dataclass
class ReplanResult:
    states: list
    success: bool
    reason: str
    cycles: int
    source: str = 'diffusion_with_route_guidance'
    diagnostics: list = field(default_factory=list)


def _inputs(scenario, problem, state, route_lanelet_ids):
    view = SimpleNamespace(initial_state=state, goal=problem.goal,
                           planning_problem_id=problem.planning_problem_id)
    lanes, speed, has_speed = extract_map_tensors(scenario, view)
    route, route_speed, route_has = extract_navigation_route(
        scenario, view, route_lanelet_ids=route_lanelet_ids)
    neighbors = extract_neighbor_tensors(scenario, current_time_step=int(state.time_step))
    static = extract_static_objects(scenario)
    yaw = float(state.orientation)
    pose = np.array([*state.position, np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    local_lanes, local_neigh, local_route, local_static = normalize_to_ego_frame(
        pose, lanes, neighbors, route, static)
    v = float(state.velocity)
    a = float(getattr(state, 'acceleration', 0.0))
    ego = np.array([0, 0, 1, 0, v, 0, a, 0, 2.30, 5.18], dtype=np.float32)
    return PlannerInput(ego, local_neigh, local_lanes, speed, has_speed,
                        local_static, local_route, route_speed, route_has), pose


def _evaluate(scenario, states, initial_speed, dt, max_acceleration, max_jerk):
    """Validate the actual proposed prefix, not the full prediction."""
    if not states:
        return False, 'empty', {}
    if any(not np.isfinite(s.position).all() or not np.isfinite(s.velocity)
           for s in states):
        return False, 'nonfinite', {}
    # User-requested goal-reaching experiment: do not reject on acceleration/jerk.
    # Collision and drivable-area checks remain mandatory.
    safety = check_collision_and_road(scenario, states, 5.18, 2.30)
    if not safety['collision_free']:
        return False, 'collision:' + str(safety.get('colliding_obstacle_ids', [])), safety
    if not safety['ego_center_on_road']:
        return False, 'off_road', safety
    return True, 'valid', {'safety': safety}


def _goal_distance(shape, position):
    """Distance-to-goal proxy for candidate ranking, not a geometric path."""
    if shape.contains_point(np.asarray(position, dtype=float)):
        return 0.0
    shapes = getattr(shape, 'shapes', [shape])
    centers = [np.asarray(s.center, dtype=float) for s in shapes
               if hasattr(s, 'center')]
    if not centers:
        return float('inf')
    return min(float(np.linalg.norm(np.asarray(position) - c)) for c in centers)



def _route_progress(local_route, local_xy):
    """Estimate arc progress on ordered, connected route lane polylines.

    This is a ranking signal ONLY: never generates or changes a trajectory.
    Rejects disconnected/degenerate route tensor fragments instead of
    interpreting zero padding or lanelet jumps as a physical route.
    """
    lanes = np.asarray(local_route, dtype=float)
    if lanes.ndim != 3 or lanes.shape[-1] < 2:
        return None
    polylines = []
    for lane in lanes:
        pts = lane[:, :2]
        pts = pts[np.isfinite(pts).all(axis=1)]
        # Zero XY is a legitimate ego-frame coordinate. Remove only padded rows
        # using the full lane feature vector, not the XY position.
        valid_rows = np.isfinite(lane).all(axis=1) & (np.abs(lane).sum(axis=1) > 1e-6)
        pts = lane[valid_rows, :2]
        if len(pts) < 2:
            continue
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-4]
        pts = pts[keep]
        if len(pts) >= 2:
            polylines.append(pts)
    if not polylines:
        return None
    route = polylines[0].copy()
    for lane in polylines[1:]:
        # Route slots must be ordered and connected. Do not fabricate links.
        if np.linalg.norm(route[-1] - lane[0]) > 3.0:
            break
        route = np.vstack((route, lane[1:]))
    edges = np.diff(route, axis=0)
    ds = np.linalg.norm(edges, axis=1)
    valid = ds > 1e-5
    if not np.any(valid):
        return None
    start = route[:-1][valid]
    edges = edges[valid]
    ds = ds[valid]
    arc = np.r_[0., np.cumsum(ds)]
    query = np.asarray(local_xy, float)
    frac = np.clip(np.sum((query-start)*edges, axis=1) / ds**2, 0., 1.)
    proj = start + frac[:, None]*edges
    j = int(np.argmin(np.linalg.norm(proj-query, axis=1)))
    return float(arc[j]+frac[j]*ds[j]), float(np.linalg.norm(proj[j]-query))


def _guided_model_geometry(local, route_local, blend=0.85):
    """Simulation experiment: bend diffusion geometry toward the locked route.

    The route is assembled from connected, ordered lanelet centreline tensors.
    It is not a collision-free or dynamically feasible trajectory by itself.
    """
    lanes = np.asarray(route_local, dtype=float)
    if lanes.ndim != 3 or lanes.shape[-1] < 2:
        return None
    pieces = []
    for lane in lanes:
        valid = np.isfinite(lane).all(axis=1) & (np.abs(lane).sum(axis=1) > 1e-6)
        pts = lane[valid, :2]
        if len(pts) < 2:
            continue
        pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-4]]
        if len(pts) >= 2:
            if pieces and np.linalg.norm(pieces[-1][-1] - pts[0]) > 3.0:
                break
            pieces.append(pts)
    if not pieces:
        return None
    route = pieces[0]
    for pts in pieces[1:]:
        route = np.vstack((route, pts[1:]))
    segment = np.diff(route, axis=0)
    ds = np.linalg.norm(segment, axis=1)
    if not np.any(ds > 1e-5):
        return None
    arc = np.r_[0.0, np.cumsum(ds)]
    keep = np.r_[True, np.diff(arc) > 1e-5]
    route, arc = route[keep], arc[keep]
    if len(route) < 2:
        return None
    # Find the nearest forward point to the ego origin on the locked route.
    edges = np.diff(route, axis=0)
    lengths2 = np.maximum(np.sum(edges * edges, axis=1), 1e-8)
    t = np.clip(-np.sum(route[:-1] * edges, axis=1) / lengths2, 0., 1.)
    projected = route[:-1] + t[:, None] * edges
    j = int(np.argmin(np.linalg.norm(projected, axis=1)))
    start_s = arc[j] + t[j] * (arc[j+1] - arc[j])
    original = np.asarray(local, float)
    if original.ndim != 2 or original.shape[1] < 2 or not np.isfinite(original).all():
        return None
    model_xy = np.vstack((np.zeros((1, 2)), original[:, :2]))
    model_s = np.cumsum(np.r_[0., np.linalg.norm(np.diff(model_xy, axis=0), axis=1)])
    target_s = np.clip(start_s + model_s[1:], 0., arc[-1])
    target = np.column_stack([np.interp(target_s, arc, route[:, k]) for k in range(2)])
    # Ramp in guidance to avoid a discontinuity at the current ego pose.
    ramp = np.clip(model_s[1:] / 6.0, 0., 1.)
    alpha = blend * ramp
    guided = original.copy()
    guided[:, :2] = (1. - alpha[:, None]) * original[:, :2] + alpha[:, None] * target
    if not np.isfinite(guided).all():
        return None
    return guided


def _route_reference_geometry(local, route_local):
    """Simulation-only reference route candidate using diffusion step distances.

    The diffusion prediction supplies progression, while the locked route supplies
    geometry. Not a learned trajectory; all executed points are collision checked.
    """
    lanes = np.asarray(route_local, float)
    pieces = []
    for lane in lanes:
        valid = np.isfinite(lane).all(axis=1) & (np.abs(lane).sum(axis=1) > 1e-6)
        pts = lane[valid, :2]
        if len(pts) < 2:
            continue
        pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-4]]
        if len(pts) < 2:
            continue
        if pieces and np.linalg.norm(pieces[-1][-1] - pts[0]) > 3.0:
            break
        pieces.append(pts)
    if not pieces:
        return None
    route = np.vstack([pieces[0]] + [p[1:] for p in pieces[1:]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-5]
    route, arc = route[keep], arc[keep]
    if len(arc) < 2:
        return None
    # Project ego origin onto route segments.
    edges = np.diff(route, axis=0)
    t = np.clip(-np.sum(route[:-1] * edges, axis=1) /
                np.maximum(np.sum(edges * edges, axis=1), 1e-8), 0., 1.)
    projections = route[:-1] + t[:, None] * edges
    j = int(np.argmin(np.linalg.norm(projections, axis=1)))
    start_s = arc[j] + t[j] * (arc[j+1] - arc[j])
    raw = np.asarray(local, float)
    distances = np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), raw[:, :2])), axis=0), axis=1)
    # Avoid pathological model jumps in the route-reference candidate.
    distances = np.clip(distances, 0.0, 1.2)
    progress = np.clip(start_s + np.cumsum(distances), 0., arc[-1])
    result = raw.copy()
    result[:, :2] = np.column_stack([np.interp(progress, arc, route[:, k]) for k in range(2)])
    return result if np.isfinite(result).all() else None


def _turn_lookahead_score(route_local, model_local, *, horizon_m=30.0):
    """Score raw MODEL geometry against the locked route ahead of the ego.

    This is a ranking/diagnostic signal, never a generated path. Sampling
    raw points by their own traveled arc avoids confusing time with distance.
    Returns (score, info), with a neutral score when the route is unavailable.
    """
    xy = np.asarray(model_local, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2:
        return 0.0, {'status': 'invalid_model'}
    xy = np.vstack([np.zeros((1, 2)), xy[:, :2]])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if not np.isfinite(arc[-1]) or arc[-1] > 250.0:
        return -1e6, {"status": "implausible_model_arc", "model_arc_m": float(arc[-1])}
    if arc[-1] < 1.0:
        return 0.0, {'status': 'short_model'}
    start = _route_progress(route_local, [0., 0.])
    if start is None:
        return 0.0, {'status': 'missing_route'}
    sample_s = np.linspace(min(5., arc[-1]), min(horizon_m, arc[-1]), 7)
    samples = np.column_stack([np.interp(sample_s, arc, xy[:, d]) for d in (0, 1)])
    projections = [_route_progress(route_local, p) for p in samples]
    if any(p is None for p in projections):
        return 0.0, {'status': 'route_projection_unavailable'}
    offsets = np.asarray([p[1] for p in projections])
    progress = np.asarray([p[0] for p in projections])
    # Focus on the future branch: a straight model prediction can look good
    # over the next 0.5 s but diverge 10-25 m farther ahead.
    far_offset = float(np.mean(offsets[-3:]))
    backtrack = float(np.maximum(0., -np.diff(progress)).sum())
    score = -4.0 * far_offset - 1.5 * float(np.mean(offsets)) - 3.0 * backtrack
    return score, {'status': 'ok', 'far_offset_m': round(far_offset, 2),
                   'mean_offset_m': round(float(np.mean(offsets)), 2),
                   'backtrack_m': round(backtrack, 2),
                   'model_arc_m': round(float(arc[-1]), 2)}


def _lane_clearance_penalty(route_local, model_local, vehicle_width=2.30,
                            margin=0.20, sample_count=16):
    """Soft ranking penalty for MODEL points near route-lane boundaries.

    The model geometry is unchanged. Missing/disconnected route data is neutral.
    This is NOT a vehicle-footprint road-containment certificate.
    """
    route = np.asarray(route_local, dtype=float)
    model = np.asarray(model_local, dtype=float)
    if model.ndim != 2 or len(model) == 0 or not np.isfinite(model).all():
        return 0.0
    if np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), model[:, :2])), axis=0), axis=1).sum() > 250.0:
        return 0.0
    parts = []
    for lane in route:
        valid = np.abs(lane).sum(axis=1) > 1e-6
        data = lane[valid]
        if len(data) < 2:
            continue
        keep = np.r_[True, np.linalg.norm(np.diff(data[:, :2], axis=0), axis=1) > 1e-4]
        data = data[keep]
        if len(data) < 2:
            continue
        if parts and np.linalg.norm(parts[-1][-1, :2] - data[0, :2]) > 3.0:
            break
        parts.append(data if not parts else data[1:])
    if not parts:
        return 0.0
    data = np.concatenate(parts, axis=0)
    a, b = data[:-1, :2], data[1:, :2]
    edge = b-a
    lengths2 = (edge*edge).sum(axis=1)
    good = lengths2 > 1e-8
    a, edge, lengths2 = a[good], edge[good], lengths2[good]
    widths = np.minimum(np.linalg.norm(data[:, 4:6], axis=1),
                        np.linalg.norm(data[:, 6:8], axis=1))
    widths = np.where(widths > 0.2, widths, 1.75)
    w0, w1 = widths[:-1][good], widths[1:][good]
    if len(a) == 0:
        return 0.0
    idx = np.linspace(0, len(model)-1, min(sample_count, len(model))).astype(int)
    xy = model[idx, :2]
    delta = xy[:, None, :] - a[None, :, :]
    t = np.clip((delta * edge[None]).sum(axis=-1)/lengths2[None], 0, 1)
    proj = a[None] + t[..., None]*edge[None]
    dist2 = ((xy[:, None] - proj)**2).sum(axis=-1)
    nearest = np.argmin(dist2, axis=1)
    fraction = t[np.arange(len(xy)), nearest]
    offset = np.sqrt(dist2[np.arange(len(xy)), nearest])
    half_width = w0[nearest]*(1-fraction)+w1[nearest]*fraction
    usable = np.maximum(0., half_width-vehicle_width/2-margin)
    violation = np.maximum(0., offset-usable)
    return float(np.mean(np.minimum(violation, 10.0)**2))


def _curvature_speed_envelope(local, max_lateral_acceleration=3.5,
                              max_braking=3.0):
    """Preview turn-speed bounds along diffusion geometry (no route replacement)."""
    xy = np.vstack((np.zeros((1, 2)), np.asarray(local[:, :2], float)))
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(ds)]
    keep = np.r_[True, np.diff(arc) > 1e-5]
    xy, arc = xy[keep], arc[keep]
    if len(arc) < 3:
        return arc, np.full(len(arc), 25.0)
    dx = np.gradient(xy[:, 0], arc)
    dy = np.gradient(xy[:, 1], arc)
    ddx = np.gradient(dx, arc)
    ddy = np.gradient(dy, arc)
    curvature = np.abs(dx*ddy-dy*ddx) / np.maximum((dx*dx+dy*dy)**1.5, 1e-5)
    # Discrete noisy corners can be overconservative: smooth curvature locally.
    if len(curvature) >= 5:
        curvature = np.convolve(np.pad(curvature, (2,2), mode='edge'),
                                np.ones(5)/5, mode='valid')
    cap = np.minimum(25., np.sqrt(max_lateral_acceleration / np.maximum(curvature, 1e-5)))
    # Backward speed envelope: allow slowing BEFORE a bend.
    for i in range(len(cap)-2, -1, -1):
        cap[i] = min(cap[i], np.sqrt(cap[i+1]**2 + 2*max_braking*(arc[i+1]-arc[i])))
    return arc, cap


def _goal_path_distance(local, pose, goal_shape, progress, arc, xy):
    """First goal entry and an interior stopping target measured along path arc.

    Uses sampled world-space centre positions; does not certify full footprint.
    """
    c, sn = float(pose[2]), float(pose[3])
    rot = np.array([[c, -sn], [sn, c]])
    samples = np.linspace(float(progress), float(arc[-1]),
                          max(50, int((arc[-1] - progress) / 0.20) + 1))
    pts = np.column_stack([np.interp(samples, arc, xy[:, j]) for j in (0, 1)])
    world = pts @ rot.T + np.asarray(pose[:2], float)
    inside = np.array([bool(goal_shape.contains_point(p)) for p in world])
    if not np.any(inside):
        return None
    first = int(np.flatnonzero(inside)[0])
    last = first
    while last + 1 < len(inside) and inside[last + 1]:
        last += 1
    # Target 35% into the contiguous goal intersection, not the far boundary.
    target = samples[first] + 0.35 * (samples[last] - samples[first])
    return max(0.0, float(target - progress))


def _route_turn_speed_cap(route_local, current_speed, max_braking=3.0):
    """Speed preview from the connected route centreline, not raw model geometry."""
    lanes = np.asarray(route_local, float)
    pieces = []
    if lanes.ndim != 3:
        return 25.0
    for lane in lanes:
        valid = np.isfinite(lane).all(axis=1) & (np.abs(lane).sum(axis=1) > 1e-6)
        pts = lane[valid, :2]
        if len(pts) < 2:
            continue
        pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-4]]
        if len(pts) < 2:
            continue
        if pieces and np.linalg.norm(pieces[-1][-1] - pts[0]) > 3.0:
            break
        pieces.append(pts)
    if not pieces:
        return 25.0
    route = np.vstack([pieces[0]] + [piece[1:] for piece in pieces[1:]])
    ds = np.linalg.norm(np.diff(route, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(ds)]
    if arc[-1] < 3.0:
        return 25.0
    samples = np.arange(0.0, arc[-1], 0.5)
    if len(samples) < 7:
        return 25.0
    xy = np.column_stack([np.interp(samples, arc, route[:, j]) for j in (0, 1)])
    delta = np.diff(xy, axis=0)
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    curvature = np.abs(np.diff(heading)) / 0.5
    origin = _route_progress(route_local, [0., 0.])
    if origin is None:
        return 25.0
    s0 = origin[0]
    ahead = np.where((samples[1:-1] >= s0) & (samples[1:-1] <= s0 + 30.0))[0]
    if not len(ahead):
        return 25.0
    cap = 25.0
    for i in ahead:
        if curvature[i] < 0.025:
            continue
        turn_v = float(np.sqrt(2.2 / max(curvature[i], 1e-5)))
        distance = max(0.0, samples[i+1] - s0 - 3.0)
        cap = min(cap, np.sqrt(turn_v**2 + 2.0 * max_braking * distance))
    return float(cap)


def _track_model_geometry(local, pose, current, dt, steps,
                          max_acceleration, max_jerk, initial_acceleration=0.0,
                          goal_shape=None, route_speed_cap=None):
    """Jerk-limited longitudinal tracking on diffusion-predicted geometry.

    The model supplies the entire geometric path. This function only chooses
    how far along that path to travel each timestep. It does not use lanelet
    centerlines or replace the model's turns. Returns KSState samples.
    """
    xy = np.vstack((np.zeros((1, 2)), np.asarray(local[:, :2], float)))
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(ds)]
    keep = np.r_[True, np.diff(arc) > 1e-7]
    xy, arc = xy[keep], arc[keep]
    if len(arc) < 2:
        return []

    # Raw model speed is a tracking preference, NOT a commanded speed.
    raw_speed = np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), local[:, :2])), axis=0), axis=1) / dt
    x0, y0, c, s = [float(v) for v in pose]
    rot = np.array([[c, -s], [s, c]], dtype=float)
    curve_arc, curve_cap = _curvature_speed_envelope(
        local, max_lateral_acceleration=3.5,
        max_braking=max_acceleration)
    speed = max(0.0, float(current.velocity))
    accel = float(np.clip(initial_acceleration, -max_acceleration, max_acceleration))
    progress = 0.0
    states = []
    for k in range(min(int(steps), len(local))):
        # Cap desired speed by upcoming curvature, without modifying geometry.
        preview_s = min(progress + max(speed, 0.0)*dt, curve_arc[-1])
        turn_cap = float(np.interp(preview_s, curve_arc, curve_cap))
        desired = float(np.clip(min(raw_speed[k], turn_cap), 0.0, 25.0))
        if route_speed_cap is not None and np.isfinite(route_speed_cap):
            desired = min(desired, float(route_speed_cap))
        # Begin braking before the goal rather than entering it at cruising
        # speed. This changes only speed along the MODEL path, never geometry.
        # Use a conservative distance-to-center proxy; the final goal-stop
        # profile still decides the exact far-edge stopping location.
        if goal_shape is not None:
            centers = [np.asarray(sh.center, float) for sh in
                       getattr(goal_shape, 'shapes', [goal_shape])
                       if hasattr(sh, 'center')]
            if centers:
                here_local = np.array([np.interp(progress, arc, xy[:, j])
                                       for j in range(2)])
                here_world = here_local @ rot.T + np.array([x0, y0])
                dist = min(float(np.linalg.norm(here_world - center))
                           for center in centers)
                if dist < 35.0:
                    # Allow extra distance for jerk-limited brake onset.
                    effective = max(0.0, dist - 3.0)
                    approach_cap = np.sqrt(2.0 * 1.5 * effective)
                    # Avoid forcing a premature standstill outside the box.
                    if not goal_shape.contains_point(here_world):
                        approach_cap = max(1.0, approach_cap)
                    desired = min(desired, float(approach_cap))
        # Direct speed tracking: no acceleration or jerk restrictions.
        # Retain geometry-based turn and goal-approach speed preferences.
        next_speed = max(0.0, desired)
        next_accel = (next_speed - speed) / dt
        next_progress = progress + next_speed * dt
        if next_progress > arc[-1] - 1e-6:
            break
        local_pos = np.array([np.interp(next_progress, arc, xy[:, j]) for j in range(2)])
        tangent_step = min(0.15, max(0.01, arc[-1] - next_progress))
        left = max(0.0, next_progress - tangent_step)
        right = min(arc[-1], next_progress + tangent_step)
        left_xy = np.array([np.interp(left, arc, xy[:, j]) for j in range(2)])
        right_xy = np.array([np.interp(right, arc, xy[:, j]) for j in range(2)])
        direction = right_xy - left_xy
        if np.linalg.norm(direction) < 1e-7:
            yaw = float(current.orientation)
        else:
            world_dir = direction @ rot.T
            yaw = float(np.arctan2(world_dir[1], world_dir[0]))
        world_pos = local_pos @ rot.T + np.array([x0, y0])
        states.append(KSState(position=world_pos, orientation=yaw,
                              velocity=float(next_speed),
                              time_step=int(current.time_step) + k + 1))
        progress, speed, accel = next_progress, next_speed, next_accel
    return states

def _model_arc_length(local):
    points = np.vstack((np.zeros((1, 2)), np.asarray(local[:, :2], float)))
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _braking_on_model_path(local, pose, current, dt, max_acceleration,
                           max_jerk, initial_acceleration=0.0,
                           max_steps=120):
    """Try a jerk-bounded, discrete stop entirely on model-generated geometry.

    A set of braking strengths is searched to avoid the common terminal-step
    quantization failure (the last step must reach v=0 without violating jerk).
    Nothing is extrapolated beyond the model's final point. [] means unverified.
    """
    points = np.vstack((np.zeros((1, 2)), np.asarray(local[:, :2], float)))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-7]
    points, arc = points[keep], arc[keep]
    if len(arc) < 2:
        return []
    x0, y0, c, sn = [float(v) for v in pose]
    rot = np.array([[c, -sn], [sn, c]])
    initial_speed = max(0., float(current.velocity))
    if initial_speed < 1e-8:
        return [KSState(position=np.asarray(current.position, float),
                        orientation=float(current.orientation), velocity=0.,
                        time_step=int(current.time_step) + 1)]
    a0 = float(np.clip(initial_acceleration, -max_acceleration, max_acceleration))

    # A discrete braking ramp can fail at the final timestep if its last
    # acceleration jump is too large. Try multiple bounded braking rates.
    for strength in np.linspace(1.0, 0.12, 89):
        speed, accel, progress = initial_speed, a0, 0.
        states = []
        for k in range(int(max_steps)):
            desired_accel = max(-max_acceleration, accel - strength * max_jerk * dt)
            next_speed = speed + desired_accel * dt
            if next_speed <= 0.0:
                required_accel = -speed / dt
                if (abs(required_accel) > max_acceleration + 1e-7 or
                        abs(required_accel - accel) > max_jerk * dt + 1e-7):
                    break
                next_speed, desired_accel = 0., required_accel
            progress += next_speed * dt
            if progress > arc[-1] - 1e-5:
                break
            local_xy = np.array([np.interp(progress, arc, points[:, j]) for j in range(2)])
            lo, hi = max(0., progress - .1), min(arc[-1], progress + .1)
            direction = np.array([np.interp(hi, arc, points[:, j]) -
                                  np.interp(lo, arc, points[:, j]) for j in range(2)])
            world_dir = direction @ rot.T
            yaw = (float(np.arctan2(world_dir[1], world_dir[0]))
                   if np.linalg.norm(world_dir) > 1e-8 else float(current.orientation))
            world_xy = local_xy @ rot.T + [x0, y0]
            states.append(KSState(position=world_xy, orientation=yaw,
                                  velocity=float(next_speed),
                                  time_step=int(current.time_step) + k + 1))
            speed, accel = next_speed, desired_accel
            if speed == 0.0:
                return states
    return []


def _safe_braking_continuation(scenario, current, prefix, brake_states, dt,
                               max_acceleration, max_jerk, initial_acceleration):
    """Check a candidate's complete safe-stop continuation, including obstacles.

    This checks braking from the CURRENT state on the same model path. A moving
    prefix is eligible only if it matches the first samples of that safe
    braking trajectory; this conservative policy avoids false claims about
    braking feasibility after an arbitrary prefix.
    """
    if not brake_states or len(prefix) > len(brake_states):
        return False
    if any(np.linalg.norm(np.asarray(a.position)-np.asarray(b.position)) > 1e-3
           for a, b in zip(prefix, brake_states)):
        return False
    ok, _, _ = _evaluate(scenario, brake_states, float(current.velocity), dt,
                         max_acceleration, max_jerk)
    if not ok:
        return False
    accelerations = np.diff([float(current.velocity)] +
                            [float(s.velocity) for s in brake_states]) / dt
    if len(accelerations) and abs(accelerations[0] - initial_acceleration) > max_jerk * dt + 1e-5:
        return False
    return brake_states[-1].velocity < 0.1


def _project_progress(points, arc, local_end):
    """Project onto continuous polyline segments, not the nearest vertex."""
    edges = np.diff(points, axis=0)
    lengths2 = np.einsum('ij,ij->i', edges, edges)
    valid = lengths2 > 1e-12
    if not np.any(valid):
        return 0.0
    t = np.zeros(len(edges))
    t[valid] = np.clip(
        np.einsum('ij,ij->i', local_end - points[:-1], edges)[valid]
        / lengths2[valid], 0.0, 1.0)
    projected = points[:-1] + t[:, None] * edges
    index = int(np.argmin(np.linalg.norm(projected - local_end, axis=1)))
    return float(arc[index] + t[index] * (arc[index + 1] - arc[index]))


def _brake_after_prefix(local, pose, prefix, dt, max_acceleration,
                        max_jerk, terminal_acceleration):
    """Emergency stopping continuation after a tracked prefix on model geometry."""
    points = np.vstack((np.zeros((1, 2)), np.asarray(local[:, :2], float)))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-7]
    points, arc = points[keep], arc[keep]
    if len(arc) < 2:
        return []
    x0, y0, c, sn = [float(v) for v in pose]
    rot = np.array([[c, -sn], [sn, c]])
    local_end = (np.asarray(prefix[-1].position)-np.array([x0,y0])) @ rot
    progress = _project_progress(points, arc, local_end)
    speed = float(prefix[-1].velocity)
    accel = float(terminal_acceleration)
    states = []
    for k in range(120):
        next_a = max(-max_acceleration, accel-max_jerk*dt)
        next_v = max(0.0, speed+next_a*dt)
        if next_v == 0.0 and speed > 0.0:
            required = -speed/dt
            if required < -max_acceleration-1e-6 or abs(required-accel)>max_jerk*dt+1e-6:
                return []
            next_a = required
        progress += next_v*dt
        if progress > arc[-1]-1e-5:
            return []
        xy = np.array([np.interp(progress,arc,points[:,j]) for j in range(2)])
        lo,hi=max(0.,progress-.1),min(arc[-1],progress+.1)
        tangent=np.array([np.interp(hi,arc,points[:,j])-np.interp(lo,arc,points[:,j]) for j in range(2)]) @ rot.T
        yaw=float(np.arctan2(tangent[1],tangent[0])) if np.linalg.norm(tangent)>1e-8 else float(prefix[-1].orientation)
        states.append(KSState(position=xy@rot.T+[x0,y0], orientation=yaw,
                              velocity=next_v,time_step=int(prefix[-1].time_step)+k+1))
        speed,accel=next_v,next_a
        if speed < 1e-8:
            return states
    return []


def _route_terminal_error(scenario, ids, goal_shape):
    """Measure whether the selected route actually terminates in the goal."""
    if not ids:
        return float('inf')
    lane = scenario.lanelet_network.find_lanelet_by_id(int(ids[-1]))
    if lane is None:
        return float('inf')
    end = np.asarray(lane.center_vertices[-1], float)
    return _goal_distance(goal_shape, end)


def _choose_goal_route(scenario, problem, goal_shape):
    """Select a CommonRoad route using its actual goal intersection, not first-route order."""
    from commonroad_route_planner.route_planner import RoutePlanner
    candidates = RoutePlanner(scenario, problem).plan_routes().retrieve_all_routes()
    if not candidates:
        raise ValueError('No CommonRoad route to the goal')
    # RoutePlanner versions may return (routes, num_routes), a list of
    # Route objects, or a list of lanelet-ID sequences.
    if (isinstance(candidates, tuple) and len(candidates) == 2
            and isinstance(candidates[1], (int, np.integer))):
        candidates = candidates[0]
    scored = []
    for route in candidates:
        raw_ids = getattr(route, 'list_ids_lanelets', route)
        if callable(raw_ids):
            raw_ids = raw_ids()
        # Unwrap a single nested Route object if returned by an adapter.
        if (isinstance(raw_ids, (list, tuple)) and len(raw_ids) == 1
                and hasattr(raw_ids[0], 'list_ids_lanelets')):
            raw_ids = raw_ids[0].list_ids_lanelets
            if callable(raw_ids):
                raw_ids = raw_ids()
        ids = [int(i) for i in raw_ids]
        if not ids:
            continue
        last = scenario.lanelet_network.find_lanelet_by_id(ids[-1])
        if last is None:
            continue
        vertices = np.asarray(last.center_vertices, float)
        inside = [i for i, pt in enumerate(vertices) if goal_shape.contains_point(pt)]
        # Prefer routes that actually enter the goal, and avoid unnecessary branches.
        goal_error = 0.0 if inside else min(_goal_distance(goal_shape, pt) for pt in vertices)
        length = sum(float(np.linalg.norm(np.diff(
            np.asarray(scenario.lanelet_network.find_lanelet_by_id(j).center_vertices, float),
            axis=0), axis=1).sum()) for j in ids)
        scored.append(((0 if inside else 1, goal_error, length), ids))
    if not scored:
        raise ValueError('No usable CommonRoad route candidates')
    scored.sort(key=lambda item: item[0])
    ids = scored[0][1]
    print(f'[ROUTE LOCK] Goal-aligned route: {ids}; candidates={len(scored)}; ranking={scored[0][0]}')
    return ids


def _candidate_route_metrics(route_local, pose, states):
    """Evaluate executed prefix against ordered route; return metrics, never alter geometry."""
    if not states:
        return None
    c, sn = float(pose[2]), float(pose[3])
    rotation = np.array([[c, -sn], [sn, c]])
    samples = np.asarray([st.position for st in states], float)
    local = (samples - np.asarray(pose[:2], float)) @ rotation
    projections = [_route_progress(route_local, pt) for pt in local]
    origin = _route_progress(route_local, [0., 0.])
    if origin is None or any(x is None for x in projections):
        return None
    progress = np.array([x[0] for x in projections])
    offsets = np.array([x[1] for x in projections])
    return {'delta_m': float(progress[-1] - origin[0]),
            'max_offset_m': float(offsets.max()),
            'mean_offset_m': float(offsets.mean()),
            'end_offset_m': float(offsets[-1]),
            'backtrack_m': float(np.maximum(0, -np.diff(progress)).sum())}


def plan_receding_horizon(planner, scenario, planning_problem, *,
                          replan_steps=5, max_cycles=100, seeds_per_cycle=8,
                          max_acceleration=3.0, max_jerk=5.0,
                          max_planning_time=25.0, seed=42,
                          max_stationary_cycles=30, diagnostic_mode=False,
                          goal_progress_debug=False,
                          centerline_weight=7.0,
                          centerline_edge_threshold_m=0.65,
                          centerline_edge_weight=18.0,
                          diagnostic_goal_speed_tolerance=0.20,
                          recovery_seeds_per_cycle=20):
    """Diffusion-only receding horizon with conservative safe-stop screening.

    Each moving prefix follows diffusion geometry. A moving prefix is executed
    only if the model path admits a collision-free, jerk-limited braking rollout.
    The emergency brake rollout is a fallback, not a substitute route.
    """
    if goal_progress_debug:
        diagnostic_mode = True
        print("[GOAL PROGRESS DEBUG] Simulation-only route-progress experiment; not safety certified.")
    if diagnostic_mode:
        print('[DIAGNOSTIC MODE] Simulation only: future stopping guarantee is NOT required.')
        print('[DIAGNOSTIC MODE] Collision, road and dynamics checks remain enabled.')
    if replan_steps < 1 or seeds_per_cycle < 1:
        raise ValueError('replan_steps and seeds_per_cycle must be positive')
    dt = float(scenario.dt)
    goal_shape = planning_problem.goal.state_list[0].position
    current = planning_problem.initial_state
    current_acceleration = float(getattr(current, 'acceleration', 0.0) or 0.0)
    # Lock the intended route from the ORIGINAL planning problem.
    # Replanning a synthetic problem can silently switch branches.
    locked_route_ids = _choose_goal_route(scenario, planning_problem, goal_shape)
    route_index = 0
    executed, history = [], []
    # A certified stopping continuation is retained as a safety fallback.
    committed_stop = []
    stationary_cycles = 0
    planning_latencies = []
    for cycle in range(max_cycles):
        # Diagnostic completion is intentionally separate from certified
        # stopping. Do not reject an already-in-goal, near-stationary state
        # solely because a new rollout fails its jerk boundary check.
        if (diagnostic_mode and goal_shape.contains_point(current.position)
                and abs(float(current.velocity)) <= diagnostic_goal_speed_tolerance):
            print('[DIAGNOSTIC GOAL] inside region and near stationary; '
                  'terminal dynamics NOT certified')
            return ReplanResult(executed, True,
                                'DIAGNOSTIC goal reached; NOT safety certified',
                                cycle, diagnostics=history)
        cycle_start = perf_counter()
        # Keep deterministic, distinct diffusion seeds across all cycles.
        remaining_ids, route_index = route_remaining_ids(
            scenario, locked_route_ids, current.position, route_index)
        live_input, pose = _inputs(
            scenario, planning_problem, current, remaining_ids)
        route_local = live_input.route_lanes
        failures, proposals, brake_options = Counter(), [], []
        lookahead_debug = []
        scored_debug = []
        uncertified_candidates = 0
        # The previously certified continuation remains usable even when the
        # latest model samples are poor. Revalidate its remaining states at
        # their absolute timestamps before offering it as a fallback.
        if committed_stop:
            remaining = [s for s in committed_stop
                         if int(s.time_step) > int(current.time_step)]
            if remaining:
                valid, why, _ = _evaluate(
                    scenario, remaining, float(current.velocity), dt,
                    max_acceleration, max_jerk)
                boundary_a = (float(remaining[0].velocity)
                              - float(current.velocity)) / dt
                valid = (valid and abs(boundary_a - current_acceleration)
                         <= max_jerk * dt + 1e-5)
                if valid:
                    brake_options.append(remaining)
                    # Preserve the certified goal stop as a candidate without
                    # forcing it to beat every newly sampled safe trajectory.
                    if (goal_shape.contains_point(remaining[-1].position)
                            and float(remaining[-1].velocity) < 0.1):
                        prefix = remaining[:min(replan_steps, len(remaining))]
                        proposals.append((40.0, prefix, 'committed_goal_stop', remaining))
                else:
                    failures['committed_stop:' + why] += 1
            committed_stop = []
        # Keep the fast original sampling path. Only expand the search if
        # none of the initial seeds produces a collision-free prefix.
        trial_budget = max(seeds_per_cycle, recovery_seeds_per_cycle)
        for trial in range(trial_budget):
            if trial >= seeds_per_cycle and proposals:
                break
            if trial == seeds_per_cycle:
                print(f'[RECOVERY] cycle={cycle} no valid base candidates; '
                      f'trying {trial_budget-seeds_per_cycle} additional diffusion seeds')
            torch.manual_seed(seed + cycle * trial_budget + trial)
            with torch.no_grad():
                prediction = planner.plan(live_input)
            local = np.asarray(prediction.ego_states, dtype=float)
            if local.ndim != 2 or local.shape[1] != 4 or not len(local) or not np.isfinite(local).all():
                failures['invalid_prediction'] += 1
                continue
            # Reject numerically unstable predictions before any geometric processing.
            raw_xy = np.vstack((np.zeros((1, 2)), local[:, :2]))
            raw_steps = np.linalg.norm(np.diff(raw_xy, axis=0), axis=1)
            raw_arc = float(raw_steps.sum())
            if not np.isfinite(raw_arc) or raw_arc > 250.0 or float(raw_steps.max()) > 15.0:
                failures['implausible_model_geometry'] += 1
                lookahead_debug.append({'trial': trial, 'status': 'implausible_model_geometry', 'model_arc_m': raw_arc})
                continue
            turn_score, turn_info = _turn_lookahead_score(route_local, local)
            lane_penalty = _lane_clearance_penalty(route_local, local)
            lookahead_debug.append({'trial': trial, 'score': round(turn_score, 2), 'lane_penalty': round(lane_penalty, 3),
                                    **turn_info})
            # An emergency stop must be checked against ALL of its future
            # positions, not just the next replan_steps samples.
            brake = ([] if diagnostic_mode else _braking_on_model_path(
                local, pose, current, dt, max_acceleration, max_jerk,
                initial_acceleration=current_acceleration))
            brake_ok = False
            if brake and not diagnostic_mode:
                brake_ok, why, _ = _evaluate(
                    scenario, brake, float(current.velocity), dt,
                    max_acceleration, max_jerk)
                acc0 = (float(brake[0].velocity) - float(current.velocity)) / dt
                brake_ok = (brake_ok and brake[-1].velocity < 0.1 and
                            abs(acc0-current_acceleration) <= max_jerk*dt+1e-5)
                if not brake_ok:
                    failures['brake:' + why] += 1
            else:
                failures['brake:no_stopping_path'] += 1
                failures['brake:available_arc_m=' + str(round(_model_arc_length(local), 2))] += 1
            if brake_ok and not diagnostic_mode:
                brake_options.append(brake)

            # A validated full stopping profile is also a certified continuation
            # when it uses correct absolute obstacle timestamps.
            # A geometric goal intersection is necessary before expensive
            # stopping-profile optimization. Check local points directly;
            # no velocity or full trajectory conversion is required here.
            world_xy = local[:, :2] @ np.array([[pose[2], pose[3]],
                                                [-pose[3], pose[2]]]) + pose[:2]
            candidate_sets = []
            if not diagnostic_mode and any(goal_shape.contains_point(point) for point in world_xy):
                try:
                    stopped, _ = collision_aware_stop(
                        local, goal_shape, pose, float(current.velocity),
                        scenario, planning_problem, dt=dt,
                        max_acceleration=max_acceleration,
                        max_deceleration=max_acceleration, max_jerk=max_jerk,
                        max_planning_time=max_planning_time,
                        current_time_step=int(current.time_step))
                    stop_states = local_to_global_states(
                        stopped, pose, dt, int(current.time_step))
                    full_ok, full_reason, _ = _evaluate(
                        scenario, stop_states, float(current.velocity), dt,
                        max_acceleration, max_jerk)
                    if (full_ok and goal_shape.contains_point(stop_states[-1].position)
                            and float(stop_states[-1].velocity) < 0.1):
                        candidate_sets.append(('goal_stop', stop_states, True, local))
                    else:
                        failures['goal_stop_full:' + full_reason] += 1
                except (RuntimeError, ValueError) as exc:
                    failures['goal_stop_unavailable'] += 1
            tracked = _track_model_geometry(
                local, pose, current, dt, replan_steps,
                max_acceleration, max_jerk,
                initial_acceleration=current_acceleration,
                goal_shape=goal_shape)
            candidate_sets.append(('tracked', tracked, False, local))
            # Additional route-guided candidate, never a replacement for the
            # pretrained diffusion output. Subject to identical validation.
            # Search several strengths of centreline guidance. Near an obstacle,
            # the strongest guidance can collide while a weaker, model-shaped
            # alternative remains feasible. Every candidate is independently
            # checked for collision, road containment, dynamics and jerk below.
            for blend, label in ((0.35, 'route_guided_soft'),
                                 (0.65, 'route_guided_medium'),
                                 (0.85, 'route_guided'),
                                 (0.97, 'route_guided_strong')):
                guided_local = _guided_model_geometry(local, route_local, blend=blend)
                if guided_local is None:
                    continue
                guided_states = _track_model_geometry(
                    guided_local, pose, current, dt, replan_steps,
                    max_acceleration, max_jerk,
                    initial_acceleration=current_acceleration,
                    goal_shape=goal_shape)
                candidate_sets.append((label, guided_states, False, guided_local))
            if diagnostic_mode:
                route_reference = _route_reference_geometry(local, route_local)
                if route_reference is not None:
                    reference_states = _track_model_geometry(
                        route_reference, pose, current, dt, replan_steps,
                        max_acceleration, max_jerk,
                        initial_acceleration=current_acceleration,
                        goal_shape=goal_shape)
                    candidate_sets.append(('route_reference_diagnostic', reference_states,
                                           False, route_reference))
            for kind, states, full_stop_validated, candidate_geometry in candidate_sets:
                if not states:
                    continue
                # Only trust an independently checked braking continuation.
                # A goal_stop candidate already passed full-horizon validation.
                for n in range(min(replan_steps, len(states)), 0, -1):
                    prefix = states[:n]
                    ok, reason, _ = _evaluate(
                        scenario, prefix, float(current.velocity), dt,
                        max_acceleration, max_jerk)
                    if not ok:
                        failures[kind + ':' + reason] += 1
                        continue
                    continuation = []
                    if not full_stop_validated and not diagnostic_mode:
                        prev_v = (float(prefix[-2].velocity) if len(prefix)>1
                                  else float(current.velocity))
                        terminal_a = (float(prefix[-1].velocity)-prev_v)/dt
                        continuation = _brake_after_prefix(
                            candidate_geometry, pose, prefix, dt, max_acceleration,
                            max_jerk, terminal_a)
                        if not continuation:
                            failures[kind + ':no_braking_distance'] += 1
                            failures[kind + ':model_arc_m=' + str(round(_model_arc_length(local), 2))] += 1
                            continue
                        # Validate full future stopping rollout, not just prefix.
                        ok_stop, why_stop, _ = _evaluate(
                            scenario, continuation, float(prefix[-1].velocity),
                            dt, max_acceleration, max_jerk)
                        first_a = (float(continuation[0].velocity)-float(prefix[-1].velocity))/dt
                        if not ok_stop or abs(first_a-terminal_a)>max_jerk*dt+1e-5:
                            failures[kind + ':unsafe_braking_continuation'] += 1
                            continue
                    # Never execute a moving prefix through the goal and out
                    # the other side. Goal crossing must end in a stop.
                    reached_indices = [i for i, st in enumerate(prefix)
                                       if goal_shape.contains_point(st.position)]
                    if reached_indices:
                        if any(not goal_shape.contains_point(st.position)
                               for st in prefix[reached_indices[0]:]):
                            failures[kind + ':overshoots_goal'] += 1
                            continue
                    before = _goal_distance(goal_shape, current.position)
                    after = _goal_distance(goal_shape, prefix[-1].position)
                    improvement = before-after
                    # Do not require monotonic Euclidean progress on curved roads.
                    score = (0.25 if goal_progress_debug else 0.7)*improvement + 0.1*n
                    metrics = _candidate_route_metrics(route_local, pose, prefix)
                    # Favor diffusion samples whose FUTURE geometry follows
                    # the intended turn, not merely the next short prefix.
                    # Route alignment is a soft tie-breaker, not a replacement for
                    # model trajectory feasibility. Large saturated penalties
                    # previously dominated the entire candidate ranking.
                    if kind != 'route_reference_diagnostic':
                        score += (0.45 if goal_progress_debug else 0.55) * float(np.clip(turn_score, -30.0, 0.0))
                    # Keep lane clearance as a diagnostic until verified on turns.
                    # Hard road and collision checks remain in _evaluate().
                    # Prefer progress on the intended route, especially before
                    # turns where Euclidean goal distance can be misleading.
                    p0 = _route_progress(route_local, [0.0, 0.0])
                    local_end = ((np.asarray(prefix[-1].position, float)
                                  - np.asarray(pose[:2], float))
                                 @ np.array([[pose[2], pose[3]],
                                             [-pose[3], pose[2]]]))
                    p1 = _route_progress(route_local, local_end)
                    if p0 is not None and p1 is not None:
                        route_delta = p1[0] - p0[0]
                        # Route progress matters more than direct goal distance
                        # on turns. Large lateral deviation suggests a wrong branch.
                        score += (0.0 if goal_shape.contains_point(current.position) else
                                  (9.0 if goal_progress_debug else 3.5))*np.clip(route_delta, -5.0, 5.0)
                        score -= centerline_weight * min(p1[1], 8.0)
                        # Reject only clearly divergent branches, not valid
                        # turning paths that momentarily project far from a
                        # coarse or discontinuous route polyline.
                        # Do not hard-reject a short prefix solely because its
                        # nearest route projection is distant. At junctions the
                        # projection can jump between adjacent route segments.
                        # Full collision/road/dynamics checks above remain hard.
                        if p1[1] > 6.0 and route_delta < 1.0:
                            score -= 18.0 + 2.0 * min(p1[1] - 6.0, 12.0)
                            failures[kind + ':branch_ranking_penalty'] += 1
                    # If the raw diffusion prediction misses the upcoming branch,
                    # reject its short prefix rather than rewarding forward travel.
                    if kind != 'route_reference_diagnostic' and turn_info.get('status') == 'ok':
                        far_offset = float(turn_info.get('far_offset_m', 0.0))
                        # A distant mismatch is not proof that the immediately
                        # executed prefix is unsafe: the model may turn after
                        # its current prediction horizon. Penalize it, but
                        # retain collision and dynamics validation as hard gates.
                        score -= 2.0 * min(max(far_offset - 3.0, 0.0), 15.0)
                    if metrics is not None:
                        # Strongly prefer centerline tracking BEFORE a prefix
                        # drifts to the lane edge. This changes ranking only;
                        # it does not move candidate points or bypass validation.
                        mean_d = metrics['mean_offset_m']
                        end_d = metrics['end_offset_m']
                        max_d = metrics['max_offset_m']
                        edge_excess = max(0.0, max_d - centerline_edge_threshold_m)
                        score -= centerline_weight * (0.6 * mean_d + 0.4 * end_d)
                        score -= centerline_edge_weight * edge_excess ** 2
                        score -= 4.0 * metrics['backtrack_m']
                        if metrics['delta_m'] < 0.1 and float(current.velocity) > 0.5:
                            score -= 15.0
                    if goal_progress_debug and metrics is not None:
                        # Avoid choosing stationary prefixes far from the goal.
                        if metrics['delta_m'] < 0.25 and after > 2.0:
                            score -= 35.0
                    # Once inside the goal, prioritize slowing down rather
                    # than rewarding more distance along the route.
                    if goal_shape.contains_point(prefix[-1].position):
                        score += 45.0 - 25.0 * float(prefix[-1].velocity)
                    elif after < 12.0:
                        score -= 1.5 * max(0.0, float(prefix[-1].velocity) -
                                           np.sqrt(2.0 * 1.5 * max(after, 0.0)))
                    if kind == 'route_reference_diagnostic':
                        score += 5.0
                    if kind == 'goal_stop':
                        # Prioritize validated trajectories ending at the goal
                        # rather than maximum immediate forward progress.
                        score += 60.0
                    # Save the complete validated stop corresponding to this
                    # prefix. Do not discard it after executing a few steps.
                    certified = (states if full_stop_validated else
                                 (prefix if diagnostic_mode else prefix + continuation))
                    if diagnostic_mode and not full_stop_validated:
                        uncertified_candidates += 1
                    proposals.append((score, prefix, kind, certified))
                    scored_debug.append({'trial': trial, 'kind': kind, 'score': round(float(score), 3),
                                         'prefix_steps': len(prefix), 'route': metrics,
                                         'goal_improvement_m': round(float(improvement), 3),
                                         'centerline_edge_excess_m': round(max(0.0, metrics['max_offset_m'] - centerline_edge_threshold_m), 3) if metrics else None})
                    break
        record = {'cycle': cycle, 'step': int(current.time_step),
                  'route_lanelets': remaining_ids,
                  'route_turn_speed_cap_mps': _route_turn_speed_cap(route_local, current_speed=float(current.velocity), max_braking=max_acceleration),
                  'rejections': dict(failures), 'proposals': len(proposals),
                  'safe_brakes': len(brake_options),
                  'turn_lookahead': lookahead_debug,
                  'scored_candidates': sorted(scored_debug, key=lambda x: -x['score'])[:12],
                  'candidate_count': len(proposals),
                  'diagnostic_mode': bool(diagnostic_mode),
                  'goal_progress_debug': bool(goal_progress_debug),
                  'uncertified_candidates': uncertified_candidates,
                  'recovery_seeds_per_cycle': recovery_seeds_per_cycle,
                  'samples_evaluated': trial + 1}
        history.append(record)
        cycle_latency = perf_counter() - cycle_start
        planning_latencies.append(cycle_latency)
        record['planning_latency_s'] = round(cycle_latency, 5)
        record['average_planning_latency_s'] = round(float(np.mean(planning_latencies)), 5)
        print(f'[LATENCY] cycle={cycle} planning={cycle_latency:.3f}s '
              f'average={np.mean(planning_latencies):.3f}s')
        print(f'[REPLAN] cycle={cycle} step={current.time_step} '
              f'accepted={len(proposals)} safe_brakes={len(brake_options)} '
              f'rejections={dict(failures)}')
        if scored_debug:
            print('[CANDIDATES] best:', sorted(scored_debug, key=lambda x: -x['score'])[:3])
        else:
            print('[CANDIDATES] none; rejections:', failures.most_common(8))
        if not proposals and brake_options:
            # Only a fully validated brake is eligible. Do not invent motion.
            stop = min(brake_options, key=len)
            segment = stop[:min(replan_steps, len(stop))]
            proposals.append((-100.0, segment, 'emergency_brake', stop))
        if not proposals and float(current.velocity) < 1e-5:
            # A stopped vehicle can wait for dynamic obstacles to clear and
            # sample the diffusion model again at the next absolute timestep.
            # Waiting is accepted only if the stationary footprint is safe.
            if stationary_cycles < max_stationary_cycles:
                wait = KSState(position=np.asarray(current.position, dtype=float),
                               orientation=float(current.orientation),
                               velocity=0.0,
                               time_step=int(current.time_step) + 1)
                ok_wait, why_wait, _ = _evaluate(
                    scenario, [wait], 0.0, dt, max_acceleration, max_jerk)
                if ok_wait:
                    executed.append(wait)
                    current = wait
                    current_acceleration = 0.0
                    stationary_cycles += 1
                    history[-1]['action'] = 'stationary_wait'
                    continue
                failures['wait:' + why_wait] += 1
        if (not proposals and diagnostic_mode
                and goal_shape.contains_point(current.position)
                and abs(float(current.velocity)) <= diagnostic_goal_speed_tolerance):
            return ReplanResult(executed, True,
                                'DIAGNOSTIC goal reached; NOT safety certified',
                                cycle, diagnostics=history)
        if not proposals:
            history[-1]['failure_diagnostics'] = {
                'speed_mps': float(current.velocity),
                'acceleration_mps2': float(current_acceleration),
                'position': np.asarray(current.position, float).tolist(),
                'goal_distance_proxy_m': _goal_distance(goal_shape, current.position),
                'route_lanelets': list(remaining_ids),
                'rejection_counts': dict(failures),
                'candidate_lookahead': lookahead_debug,
                'last_selected_candidate': next((h.get('selected_kind') for h in reversed(history[:-1]) if h.get('selected_kind')), None),
                'recommendation': 'Inspect collision/road/dynamics rejection counts and route lookahead before relaxing constraints',
            }
            print('[NO CANDIDATE] state:', history[-1]['failure_diagnostics'])
            print(f'[LATENCY SUMMARY] mean={np.mean(planning_latencies):.3f}s '
                  f'max={np.max(planning_latencies):.3f}s cycles={len(planning_latencies)}')
            return ReplanResult(executed, False,
                                (f'no locally valid candidate at cycle {cycle}' if diagnostic_mode
                                 else f'no verified safe continuation at cycle {cycle}'),
                                cycle, diagnostics=history)
        best_score, segment, kind, certified_stop = max(proposals, key=lambda x: x[0])
        history[-1]['selected_kind'] = kind
        history[-1]['selected_score'] = float(best_score)
        history[-1]['selected_steps'] = len(segment)
        history[-1]['selected_end_speed_mps'] = float(segment[-1].velocity)
        history[-1]['selected_goal_distance_m'] = _goal_distance(goal_shape, segment[-1].position)
        history[-1]['selected_end_xy'] = np.asarray(segment[-1].position, float).tolist()
        committed_stop = [] if diagnostic_mode else list(certified_stop)
        if any(goal_shape.contains_point(st.position) for st in segment):
            # A goal-stop segment may be executed, but do not let its prefix
            # continue outside the goal after the first stopped state.
            for i, st in enumerate(segment):
                if goal_shape.contains_point(st.position) and float(st.velocity) < 0.1:
                    segment = segment[:i + 1]
                    break
        executed.extend(segment)
        last = segment[-1]
        if (goal_shape.contains_point(last.position) and
                float(last.velocity) < (diagnostic_goal_speed_tolerance if diagnostic_mode else 0.1)):
            print(f'[LATENCY SUMMARY] mean={np.mean(planning_latencies):.3f}s '
                  f'max={np.max(planning_latencies):.3f}s cycles={len(planning_latencies)}')
            return ReplanResult(executed, True,
                                ('DIAGNOSTIC goal reached; NOT safety certified'
                                 if diagnostic_mode else 'goal reached and stopped'),
                                cycle+1, diagnostics=history)
        prev_speed = (float(segment[-2].velocity) if len(segment)>1
                      else float(current.velocity))
        current_acceleration = (float(last.velocity)-prev_speed)/dt
        if float(last.velocity) >= 0.1:
            stationary_cycles = 0
        current = KSState(position=np.asarray(last.position, dtype=float),
                          orientation=float(last.orientation),
                          velocity=float(last.velocity),
                          time_step=int(last.time_step))
        try:
            current.acceleration = float(current_acceleration)
        except (AttributeError, TypeError):
            pass
    if planning_latencies:
        print(f'[LATENCY SUMMARY] mean={np.mean(planning_latencies):.3f}s '
              f'max={np.max(planning_latencies):.3f}s cycles={len(planning_latencies)}')
    return ReplanResult(executed, False, 'maximum replanning cycles reached',
                        max_cycles, diagnostics=history)
