"""Route-based reference fallback for predictions that miss the goal.

This constructs geometry only; it does NOT certify a safe trajectory.
"""
import numpy as np


def build_route_reference(scenario, planning_problem, ego_pose, spacing=0.15):
    from commonroad_route_planner.route_planner import RoutePlanner

    planner = RoutePlanner(scenario, planning_problem)
    routes = planner.plan_routes()
    route = routes.retrieve_first_route()
    ids = list(route.list_ids_lanelets)
    if not ids:
        raise RuntimeError("Route planner returned no lanelets")

    start = np.asarray(ego_pose[:2], dtype=float)
    chunks = []
    for lanelet_id in ids:
        lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        pts = np.asarray(lanelet.center_vertices, dtype=float)
        if len(pts) < 2:
            continue
        if not chunks:
            # Lanelet center vertices follow traffic direction.
            # Start from the closest point on the first lanelet polyline.
            d = np.linalg.norm(pts - start, axis=1)
            index = int(np.argmin(d))
            pts = pts[index:]
        if not len(pts):
            continue
        if chunks and np.linalg.norm(chunks[-1][-1] - pts[0]) > 4.0:
            raise RuntimeError("Route centerlines have a large discontinuity")
        chunks.append(pts)

    if not chunks:
        raise RuntimeError("No usable route centerlines")

    world = np.vstack([start[None, :]] + chunks)
    steps = np.linalg.norm(np.diff(world, axis=0), axis=1)
    keep = np.r_[True, steps > 1e-5]
    world = world[keep]
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(world, axis=0), axis=1))]
    if distances[-1] < 1.0:
        raise RuntimeError("Route geometry is too short")
    samples = np.arange(0.0, distances[-1], spacing)
    samples = np.r_[samples, distances[-1]]
    dense = np.column_stack([
        np.interp(samples, distances, world[:, 0]),
        np.interp(samples, distances, world[:, 1]),
    ])
    goal_shape = planning_problem.goal.state_list[0].position
    inside = np.array([goal_shape.contains_point(p) for p in dense])
    if not inside.any():
        raise RuntimeError("Route centerline does not intersect goal region")

    # Keep path through the far end of the first goal passage.
    entry = int(np.flatnonzero(inside)[0])
    end = entry
    while end + 1 < len(inside) and inside[end + 1]:
        end += 1
    dense = dense[:min(len(dense), end + max(2, int(2.0 / spacing))) + 1]

    x0, y0, c, s = ego_pose
    rot = np.array([[c, -s], [s, c]], dtype=float)
    local = (dense - [x0, y0]) @ rot
    tangents = np.gradient(local, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(norms, 1e-8)
    return np.column_stack([local[1:], tangents[1:]]).astype(np.float32), ids
