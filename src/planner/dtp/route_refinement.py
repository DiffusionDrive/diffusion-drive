"""Experimental route-guided refinement of DTP XY candidates.

This is not a dynamics controller or a collision-free trajectory optimizer.
All returned paths require downstream footprint and obstacle validation.
"""
import numpy as np
from shapely.geometry import Point


def _headings(path, initial_yaw):
    headings = np.empty(len(path), dtype=float)
    headings[0] = float(initial_yaw)
    for i in range(1, len(path)):
        d = path[i] - path[i - 1]
        headings[i] = np.arctan2(d[1], d[0]) if np.linalg.norm(d) >= 0.15 else headings[i-1]
    return np.unwrap(headings)


def refine_route_candidate(world, current, initial_yaw, route,
                           strength=0.55, max_lateral_shift=1.2,
                           max_heading_step=np.deg2rad(9),
                           max_curvature=0.25, max_acceleration=5.0,
                           dt=0.1, horizon=20):
    """Return corrected (N,2) XY prediction or None if geometric checks fail.

    Correct only the first `horizon` samples, smoothly ramping correction from
    zero at the ego position. Preserve longitudinal progress by projecting
    onto the A* reference line and shifting toward the closest point.
    """
    original = np.asarray(world, dtype=float)
    current = np.asarray(current, dtype=float)
    if original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all():
        return None
    if len(original) < 2 or route.length <= 0:
        return None
    count = min(len(original), int(horizon))
    # Retry progressively weaker corrections. Never return an unchecked path.
    # The first successful correction retains the strongest route guidance.
    for trial_strength in (strength, strength * 0.65, strength * 0.35,
                           strength * 0.15):
        out = original.copy()
        for i in range(count):
            p = original[i]
            station = route.project(Point(float(p[0]), float(p[1])))
            q = route.interpolate(station)
            correction = np.array([q.x, q.y]) - p
            ramp = min(1.0, (i + 1) / 7.0)
            offset = correction * (trial_strength * ramp)
            shift = np.linalg.norm(offset)
            if shift > max_lateral_shift:
                offset *= max_lateral_shift / shift
            out[i] = p + offset

        # Smooth the correction, leaving the original prediction intact.
        delta = out[:count] - original[:count]
        padded = np.vstack([np.zeros((2, 2)), delta,
                            np.repeat(delta[-1:], 2, axis=0)])
        weights = np.array([1, 2, 3, 2, 1], dtype=float) / 9.0
        smooth = np.array([(padded[i:i+5] * weights[:, None]).sum(axis=0)
                           for i in range(count)])
        out[:count] = original[:count] + smooth

        full = np.vstack([current, out[:count]])
        steps_xy = np.diff(full, axis=0)
        lengths = np.linalg.norm(steps_xy, axis=1)
        if np.any(lengths < 1e-4):
            continue
        headings = _headings(full, initial_yaw)
        if np.max(np.abs(np.diff(headings))) > max_heading_step:
            continue
        curvature = np.abs(np.diff(headings[1:])) / np.maximum(lengths[1:], 1e-4)
        if len(curvature) and np.max(curvature) > max_curvature:
            continue
        velocities = lengths / dt
        if len(velocities) > 1 and np.max(np.abs(np.diff(velocities)) / dt) > max_acceleration:
            continue
        return out
    return None
