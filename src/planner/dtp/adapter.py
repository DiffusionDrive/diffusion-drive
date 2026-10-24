"""CommonRoad to Waymo-trained DTP feature schema (approximate history)."""
import numpy as np
import torch


def local_xy(points, origin, yaw):
    p = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, -s], [s, c]])


def global_xy(points, origin, yaw):
    p = np.asarray(points, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, s], [-s, c]]) + np.asarray(origin)


def _resample(points, n=10):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros((n, 2))
    if len(points) < 2:
        return np.repeat(points[:1], n, axis=0)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if d[-1] < 1e-8:
        return np.repeat(points[:1], n, axis=0)
    return np.column_stack([np.interp(np.linspace(0, d[-1], n), d, points[:, i]) for i in range(2)])


def goal_center(
        scenario,
        planning_problem,
        end_margin=1.0,
        route=None
):
    """
    Select the midpoint of the far end of the goal region,
    measured along the ego vehicle's initial heading.

    Move the target inward by end_margin metres.
    """
    import numpy as np
    from shapely.ops import unary_union
    from shapely.geometry import Point

    if route is not None:
        # Obtain the direction of the final A* route segment.
        coords = np.asarray(route.coords, dtype=float)

        if len(coords) < 2:
            raise ValueError("A* route has insufficient points.")

        # Use the last non-zero-length segment.
        forward = None

        for i in range(len(coords) - 1, 0, -1):
            direction = coords[i] - coords[i - 1]
            length = np.linalg.norm(direction)

            if length > 1e-6:
                forward = direction / length
                break

        if forward is None:
            raise ValueError("A* route has no valid direction.")

    else:
        # Fallback to the ego's initial heading.
        heading = float(
            planning_problem.initial_state.orientation
        )

        forward = np.array([
            np.cos(heading),
            np.sin(heading)
        ], dtype=float)

    goal_shapes = []

    for goal_state in planning_problem.goal.state_list:
        goal_position = getattr(
            goal_state, "position", None
        )

        if goal_position is None:
            continue

        if hasattr(goal_position, "shapely_object"):
            goal_shapes.append(
                goal_position.shapely_object
            )

        elif hasattr(goal_position, "shapes"):
            for shape in goal_position.shapes:
                if hasattr(shape, "shapely_object"):
                    goal_shapes.append(
                        shape.shapely_object
                    )

    # Fallback to goal lanelet geometry if necessary.
    if not goal_shapes:
        goal = planning_problem.goal

        ids = [
            ll
            for values in (
                    getattr(
                        goal,
                        "lanelets_of_goal_position",
                        None
                    ) or {}
            ).values()
            for ll in values
        ]

        if ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(
                int(ids[0])
            )

            goal_shapes.append(
                lanelet.polygon.shapely_object
            )

    if not goal_shapes:
        raise ValueError(
            "Cannot resolve goal region geometry."
        )

    goal_polygon = unary_union(goal_shapes)

    polygons = (
        list(goal_polygon.geoms)
        if goal_polygon.geom_type == "MultiPolygon"
        else [goal_polygon]
    )

    polygon = max(
        polygons,
        key=lambda p: p.area
    )

    coords = np.asarray(
        polygon.exterior.coords[:-1],
        dtype=float
    )

    best_score = -np.inf
    best_midpoint = None

    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]

        midpoint = (p1 + p2) / 2.0
        edge = p2 - p1

        edge_length = np.linalg.norm(edge)

        if edge_length < 1e-6:
            continue

        edge_direction = edge / edge_length

        # Prefer an edge perpendicular to travel direction.
        perpendicular_score = 1.0 - abs(
            np.dot(edge_direction, forward)
        )

        # Prefer the edge farther along the road.
        progress = np.dot(midpoint, forward)

        score = progress + 5.0 * perpendicular_score

        if score > best_score:
            best_score = score
            best_midpoint = midpoint

    if best_midpoint is None:
        raise ValueError(
            "Could not identify goal end edge."
        )

    # Move inward from the far edge.
    target = best_midpoint - forward * end_margin

    if not polygon.covers(Point(target)):
        raise ValueError(
            "Target is outside the goal region. "
            "Adjust end_margin or inspect goal geometry."
        )

    return np.asarray(target, dtype=float)



def _recorded_states(obstacle):
    """Known states only; an expired prediction is not evidence of parking."""
    states = [obstacle.initial_state]
    trajectory = getattr(getattr(obstacle, "prediction", None), "trajectory", None)
    states.extend(getattr(trajectory, "state_list", []) or [])
    return sorted({int(s.time_step): s for s in states}.values(),
                  key=lambda s: int(s.time_step))


def classify_parked(obstacle, dt, speed_threshold=0.5):
    """Match training's all-valid-steps slow criterion using available CR states.

    Position differences are checked as well because CR state.velocity can be
    absent or inconsistent. Insufficient data is classified as unknown (False).
    """
    states = _recorded_states(obstacle)
    if len(states) < 2:
        return False
    for before, after in zip(states[:-1], states[1:]):
        delta_t = (int(after.time_step) - int(before.time_step)) * dt
        if delta_t <= 0:
            return False
        measured = float(np.linalg.norm(
            np.asarray(after.position, dtype=float) -
            np.asarray(before.position, dtype=float))) / delta_t
        if not np.isfinite(measured) or measured >= speed_threshold:
            return False
    for state in states:
        v = getattr(state, "velocity", None)
        if v is not None and (not np.isfinite(float(v)) or abs(float(v)) >= speed_threshold):
            return False
    return True


def _history_headings(history, current_yaw):
    """Approximate relative heading from successive executed XY positions."""
    history = np.asarray(history, dtype=float)
    angles = np.zeros(len(history), dtype=float)
    previous = float(current_yaw)
    for i in range(1, len(history)):
        d = history[i] - history[i - 1]
        if np.linalg.norm(d) > 1e-5:
            previous = float(np.arctan2(d[1], d[0]))
        angles[i] = (previous - current_yaw + np.pi) % (2 * np.pi) - np.pi
    if len(angles) > 1:
        angles[0] = angles[1]
    return angles

def adapt(scenario, planning_problem, device='cpu', ego_history=None, goal_override=None):
    ego = planning_problem.initial_state
    origin = np.asarray(ego.position, dtype=float)
    yaw = float(ego.orientation)
    goal = (
        np.asarray(goal_override, dtype=float)
        if goal_override is not None
        else goal_center(scenario, planning_problem)
    )
    state = {
        'ego_history': np.zeros((1, 11, 3), np.float32),
        'ego_history_mask': np.zeros((1, 11), bool),
        'agents': np.zeros((1, 16, 11), np.float32),
        'agents_mask': np.zeros((1, 16), bool),
        'map': np.zeros((1, 64, 10, 2), np.float32),
        'map_mask': np.zeros((1, 64), bool),
        'goal': np.zeros((1, 5, 2), np.float32),
        'goal_mask': np.zeros((1, 5), bool),
        'traffic_lights': np.zeros((1, 2), np.float32),
        'traffic_lights_mask': np.ones((1, 1), bool),
    }
    # CommonRoad provides an initial state, not 1 second of measured history.
    # Extrapolation is explicitly an approximation, with mask marking only real observation.
    speed = float(getattr(ego, 'velocity', 0.0))
    state['ego_history'][0, :, 0] = np.arange(-10, 1) * 0.1 * speed
    state['ego_history_mask'][0, -1] = True
    if ego_history is not None:
        hist = np.asarray(ego_history, dtype=float)
        if hist.ndim != 2 or hist.shape[1] != 2:
            raise ValueError('ego_history must have shape (N,2)')
        hist = hist[-11:]
        n = len(hist)
        state['ego_history'][0, -n:, :2] = local_xy(hist, origin, yaw)
        state['ego_history'][0, -n:, 2] = _history_headings(hist, yaw)
        state['ego_history_mask'][0, -n:] = True
    state['goal'][0, 0] = local_xy(goal[None], origin, yaw)[0]
    state['goal_mask'][0, 0] = True

    lanes = sorted(scenario.lanelet_network.lanelets,
                   key=lambda ll: np.min(np.linalg.norm(np.asarray(ll.center_vertices) - origin, axis=1)))
    # Training extractor uses centerlines only, not left/right boundaries.
    for i, ll in enumerate(lanes[:64]):
        state['map'][0, i] = local_xy(_resample(ll.center_vertices), origin, yaw)
        state['map_mask'][0, i] = True

    agents = []
    for obs in scenario.obstacles:
        try:
            os = obs.state_at_time_step(int(ego.time_step))
            if os is None:
                continue
            xy = local_xy(np.asarray(os.position)[None], origin, yaw)[0]
            if not (-15 < xy[0] < 80 and abs(xy[1]) < 30):
                continue
            heading = float(getattr(os, 'orientation', 0.0)) - yaw
            v = float(getattr(os, 'velocity', 0.0))
            length = float(getattr(obs.obstacle_shape, 'length', 0.0))
            width = float(getattr(obs.obstacle_shape, 'width', 0.0))
            name = str(getattr(obs, 'obstacle_type', '')).lower()
            vehicle = float(any(k in name for k in ('car', 'truck', 'bus', 'vehicle', 'motorcycle')))
            pedestrian = float(any(k in name for k in ('pedestrian', 'bicycle', 'cyclist')))
            feature = [xy[0], xy[1], v * np.cos(heading), v * np.sin(heading),
                       np.cos(heading), np.sin(heading), length, width,
                       vehicle, pedestrian, float(classify_parked(obs, float(scenario.dt)))]
            agents.append((np.linalg.norm(xy), feature))
        except (AttributeError, ValueError, TypeError):
            continue
    for i, (_, feature) in enumerate(sorted(agents, key=lambda item: item[0])[:16]):
        state['agents'][0, i] = feature
        state['agents_mask'][0, i] = True
    return {k: torch.as_tensor(v, device=device) for k, v in state.items()}, origin, yaw, goal
