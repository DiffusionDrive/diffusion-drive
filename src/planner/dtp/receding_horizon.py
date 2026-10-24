"""Simplified offline DTP MPC. Sampled geometric checks; not safety certification."""
from copy import copy
from dataclasses import dataclass
import heapq
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from .adapter import adapt, global_xy, goal_center
from .route_refinement import refine_route_candidate

EGO_LENGTH = 4.5
EGO_WIDTH = 1.8


def rectangle(xy, yaw, length, width):
    corners = np.array([[length/2, width/2], [length/2, -width/2],
                        [-length/2, -width/2], [-length/2, width/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon(corners @ np.array([[c, -s], [s, c]]).T + xy)


def shape_polygon(shape):
    if hasattr(shape, 'shapes'):
        return unary_union([shape_polygon(s) for s in shape.shapes])
    if hasattr(shape, 'vertices'):
        return Polygon(np.asarray(shape.vertices, dtype=float))
    if hasattr(shape, 'length') and hasattr(shape, 'width'):
        return rectangle(np.asarray(shape.center), float(getattr(shape, 'orientation', 0)),
                         float(shape.length), float(shape.width))
    if hasattr(shape, 'radius'):
        return Point(*shape.center).buffer(float(shape.radius))
    raise TypeError(f'Unknown obstacle shape: {type(shape)}')

def get_obstacle_occupancy(obstacle, time_step):
    """
    Return obstacle geometry at a requested time.

    If the obstacle's prediction has ended, assume it
    remains stationary at its last known position.

    Returns:
        geometry: Shapely geometry or None
        status: 'available', 'stationary_assumed',
                or 'unknown'
    """
    time_step = int(time_step)

    if time_step < int(obstacle.initial_state.time_step):
        return None, "not_started"

    # First use the actual CommonRoad occupancy.
    occupancy = obstacle.occupancy_at_time(time_step)

    if occupancy is not None:
        return shape_polygon(occupancy.shape), "available"

    # Find the last recorded prediction.
    prediction = getattr(obstacle, "prediction", None)
    trajectory = getattr(prediction, "trajectory", None)
    states = list(getattr(trajectory, "state_list", []) or [])

    if not states:
        return None, "unknown"

    last_state = states[-1]
    last_step = int(last_state.time_step)

    # Do not fill gaps inside the recorded prediction.
    if time_step <= last_step:
        return None, "unknown"

    last_occupancy = obstacle.occupancy_at_time(last_step)

    if last_occupancy is None:
        return None, "unknown"

    # Experimental assumption: obstacle stays at
    # its last known position and orientation.
    return (
        shape_polygon(last_occupancy.shape),
        "stationary_assumed"
    )

def road_polygon(scenario):
    polys = []
    for lane in scenario.lanelet_network.lanelets:
        p = Polygon(np.vstack([lane.left_vertices, lane.right_vertices[::-1]]))
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            polys.append(p)
    if not polys:
        raise ValueError('No drivable lanelets')
    return unary_union(polys).buffer(0.05)


def astar_route(scenario, start, goal):
    network = scenario.lanelet_network
    starts = network.find_lanelet_by_position([np.asarray(start)])[0]
    goals = set(network.find_lanelet_by_position([np.asarray(goal)])[0])
    if not starts or not goals:
        raise ValueError('Start or goal reference is outside all lanelets')
    lanes = {l.lanelet_id: l for l in network.lanelets}
    queue = []
    best = {}
    parent = {}
    for sid in starts:
        best[sid] = 0.0
        parent[sid] = None
        heapq.heappush(queue, (0.0, sid))
    while queue:
        cost, lid = heapq.heappop(queue)
        if cost > best[lid] + 1e-8:
            continue
        if lid in goals:
            ids = []
            while lid is not None:
                ids.append(lid)
                lid = parent[lid]
            ids.reverse()
            vertices = []
            for rid in ids:
                pts = np.asarray(lanes[rid].center_vertices, dtype=float)
                if len(vertices) and np.linalg.norm(vertices[-1] - pts[0]) < 1e-3:
                    pts = pts[1:]
                vertices.extend(pts)
            return ids, LineString(vertices)
        for nxt in lanes[lid].successor:
            if nxt not in lanes:
                continue
            pts = np.asarray(lanes[nxt].center_vertices, dtype=float)
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            new_cost = cost + length
            if new_cost < best.get(nxt, float('inf')):
                best[nxt] = new_cost
                parent[nxt] = lid
                heapq.heappush(queue, (new_cost, nxt))
    raise ValueError('No connected successor-only lanelet route to goal')


def obstacle_may_reach(obstacle, ego_footprint, requested_step, dt,
                       max_speed=20.0, max_acceleration=4.0, margin=2.0):
    """Conservative radial bound for missing future prediction; not a proof."""
    prediction = getattr(obstacle, 'prediction', None)
    states = list(getattr(getattr(prediction, 'trajectory', None), 'state_list', []) or [])
    if not states:
        return True
    last = states[-1]
    if requested_step <= int(last.time_step):
        return True
    occ = obstacle.occupancy_at_time(int(last.time_step))
    if occ is None:
        return True
    elapsed = (requested_step - int(last.time_step)) * dt
    radius = max_speed * elapsed + 0.5 * max_acceleration * elapsed**2 + margin
    return shape_polygon(occ.shape).buffer(radius).intersects(ego_footprint)


def check_segment(path, scenario, road, step, yaw,
                  strict_missing=True):
    """
    Check ego footprints against road and obstacles.

    Recorded occupancies are used directly.
    Expired obstacle predictions are held stationary.

    This is waypoint-sampled validation, not a
    continuous-time collision guarantee.
    """
    unknown_ids = set()
    stationary_ids = set()

    road_bad = 0
    collisions = 0
    prev_yaw = float(yaw)

    for j, xy in enumerate(path):

        if j > 0:
            delta = path[j] - path[j - 1]

            # Tiny displacements give numerically unstable travel headings.
            # Retain the last meaningful heading rather than rotating the car.
            if np.linalg.norm(delta) >= 0.15:
                prev_yaw = float(np.arctan2(delta[1], delta[0]))

        ego = rectangle(
            xy,
            prev_yaw,
            EGO_LENGTH,
            EGO_WIDTH
        )

        """if not road.covers(ego):
            road_bad += 1"""

        time_step = step + j

        for obstacle in scenario.obstacles:

            try:
                geometry, status = get_obstacle_occupancy(
                    obstacle,
                    time_step
                )

                if status == "not_started":
                    continue

                if status == "unknown":
                    unknown_ids.add(
                        int(obstacle.obstacle_id)
                    )
                    continue

                if status == "stationary_assumed":
                    stationary_ids.add(
                        int(obstacle.obstacle_id)
                    )

                if geometry is not None:
                    if ego.intersects(geometry):
                        collisions += 1

            except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    IndexError
            ):
                unknown_ids.add(
                    int(obstacle.obstacle_id)
                )

    valid = (
            # road_bad == 0 and
            collisions == 0 and
            len(unknown_ids) == 0
    )

    return {
        "valid": valid,
        "road": road_bad,
        "collisions": collisions,
        "unknown_ids": sorted(unknown_ids),
        "stationary_assumed_ids": sorted(stationary_ids),
        "assumption_based": bool(stationary_ids),
    }


def route_metrics(path, route):
    """
    Measure forward progress and deviation from the
    intended A* lane centerline.
    """
    path = np.asarray(path, dtype=float)

    points = [
        Point(float(x), float(y))
        for x, y in path
    ]

    projections = np.array([
        route.project(p) for p in points
    ])

    lateral_errors = np.array([
        route.distance(p) for p in points
    ])

    return {
        "forward_progress": float(
            np.max(projections) - projections[0]
        ),
        "mean_lateral_error": float(
            np.mean(lateral_errors[1:])
        ),
        "max_lateral_error": float(
            np.max(lateral_errors[1:])
        ),
    }

def score_candidate(path, current, goal, route, check):
    """
    Score a DTP candidate.

    Higher scores reward goal progress and lane keeping.
    Unsafe candidates are still rejected separately.
    """
    distance_before = float(np.linalg.norm(current - goal))
    distance_after = float(np.linalg.norm(path[-1] - goal))

    goal_progress = distance_before - distance_after

    metrics = route_metrics(path, route)

    forward_progress = metrics["forward_progress"]
    mean_lateral = metrics["mean_lateral_error"]
    max_lateral = metrics["max_lateral_error"]

    collision_penalty = check["collisions"]
    road_penalty = check["road"]

    score = (
            20.0 * goal_progress
            + 20.0 * forward_progress
            - 2.0 * mean_lateral
            - 4.0 * max_lateral
            - 100.0 * collision_penalty
            - 100.0 * road_penalty
    )

    return float(score)

def expired_obstacle_cannot_reach(
        obstacle,
        ego_path,
        scenario,
        start_step,
        ego_length=4.5,
        ego_width=1.8,
        max_obstacle_speed=20.0,
        safety_margin=3.0,
):
    """
    Conservative reachability exclusion for expired predictions.

    Returns True only if the obstacle's reachable region
    cannot intersect the ego segment.

    max_obstacle_speed is an assumed upper bound, not a
    guaranteed physical limit.
    """
    prediction = getattr(obstacle, "prediction", None)
    trajectory = getattr(prediction, "trajectory", None)

    if trajectory is None or not trajectory.state_list:
        return False

    last_state = trajectory.state_list[-1]
    last_step = int(last_state.time_step)

    last_occupancy = obstacle.occupancy_at_time(last_step)

    if last_occupancy is None:
        return False

    # Use your existing shape_polygon() function.
    last_geometry = shape_polygon(last_occupancy.shape)

    ego_path = np.asarray(ego_path, dtype=float)

    if len(ego_path) < 2:
        return False

    # Conservative centerline sweep with a circular
    # bound enclosing the ego vehicle footprint.
    ego_radius = 0.5 * np.hypot(
        ego_length,
        ego_width
    )

    swept_ego = LineString(ego_path).buffer(
        ego_radius + safety_margin
    )

    final_step = start_step + len(ego_path) - 1

    elapsed = max(
        0.0,
        (final_step - last_step) * scenario.dt
    )

    reachable_radius = max_obstacle_speed * elapsed

    reachable_region = last_geometry.buffer(
        reachable_radius
    )

    return not swept_ego.intersects(reachable_region)

@dataclass
class Rollout:
    trajectory: np.ndarray
    times: np.ndarray
    status: str
    cycles: int
    diagnostics: list
    goal_distance: float


def plan_receding_horizon(scenario, problem, predictor, execute_steps=5,
                          samples=12, ddim_steps=50, max_steps=150,
                          reference_tolerance=1.0, enable_route_refinement=True,
                          strict_missing=False):
    """Choose one valid 5-step segment, execute, and repeat toward purple reference."""
    if not 1 <= execute_steps <= 80:
        raise ValueError('execute_steps must be 1..80')
    if scenario.dt <= 0:
        raise ValueError('Scenario dt must be positive')

    road = road_polygon(scenario)

    initial = problem.initial_state
    current = np.asarray(initial.position, dtype=float).copy()

    # 1. Preliminary target using the goal region's centroid.
    goal_position = next(
        state.position
        for state in problem.goal.state_list
        if getattr(state, "position", None) is not None
    )

    if isinstance(goal_position, np.ndarray):
        preliminary_goal = np.asarray(
            goal_position,
            dtype=float
        )

    elif hasattr(goal_position, "shapely_object"):
        geometry = goal_position.shapely_object
        preliminary_goal = np.array([
            geometry.centroid.x,
            geometry.centroid.y
        ], dtype=float)

    elif hasattr(goal_position, "shapes"):
        from shapely.ops import unary_union

        geometries = [
            shape.shapely_object
            for shape in goal_position.shapes
            if hasattr(shape, "shapely_object")
        ]

        if not geometries:
            raise ValueError(
                "ShapeGroup contains no usable goal geometry."
            )

        geometry = unary_union(geometries)

        preliminary_goal = np.array([
            geometry.centroid.x,
            geometry.centroid.y
        ], dtype=float)

    elif hasattr(goal_position, "center"):
        preliminary_goal = np.asarray(
            goal_position.center,
            dtype=float
        )

    else:
        raise TypeError(
            f"Unsupported goal position type: "
            f"{type(goal_position)}"
        )

    # 2. Find the A* route to the preliminary goal.
    route_ids, route = astar_route(
        scenario,
        current,
        preliminary_goal
    )

    # 3. Recalculate the final target using the
    #    A* route's final direction.
    goal = goal_center(
        scenario,
        problem,
        end_margin=1.0,
        route=route
    )

    goal = np.asarray(goal, dtype=float)

    print("A* lanelet route:", route_ids)
    print("Final goal target:", goal)
    yaw = float(initial.orientation)
    speed = float(getattr(initial, 'velocity', 0.0))
    step = int(initial.time_step)
    end_step = step + max_steps
    points, times, diagnostics = [current.copy()], [step], []
    cycles, status = 0, 'max_steps'
    while step < end_step:
        distance = float(np.linalg.norm(current - goal))
        if distance <= reference_tolerance:
            status = 'goal_reached'
            break
        pp = copy(problem)
        state = copy(initial)
        state.position = current.copy()
        state.orientation = yaw
        state.velocity = speed
        state.time_step = step
        pp.initial_state = state
        features, origin, heading, model_goal = adapt(
            scenario,
            pp,
            ego_history=np.asarray(points[-11:]),
            goal_override=goal
        )
        predictions = predictor.predict(features, samples=samples, steps=ddim_steps)
        n = min(execute_steps, end_step - step)
        choices = []
        rejected = {'road': 0, 'collision': 0, 'unknown': 0, 'route': 0}
        unknown_rejected_ids = set()
        refinement_stats = {'accepted': 0, 'returned_none': 0, 'failed_validation': 0, 'valid': 0}
        for k, local in enumerate(predictions):
            raw = np.asarray(global_xy(local, origin, heading), dtype=float)
            if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < n or not np.isfinite(raw).all():
                continue
            variants = [('raw', raw)]
            if enable_route_refinement:
                refined = refine_route_candidate(raw, current, yaw, route,
                                                 horizon=max(20, n), dt=scenario.dt)
                if refined is not None:
                    refinement_stats['accepted'] += 1
                    variants.append(('refined', refined))
                else:
                    refinement_stats['returned_none'] += 1
            for variant, world in variants:
                path = np.vstack([current, world[:n]])
                metrics = route_metrics(path, route)

                if metrics["forward_progress"] <= 0.00:
                    rejected["route"] += 1
                    continue

                distance_before = np.linalg.norm(current - goal)
                distance_after = np.linalg.norm(path[-1] - goal)

                goal_progress = distance_before - distance_after

                # Reject candidates that do not get closer to the goal
                if goal_progress <= 0.0:
                    rejected["route"] += 1
                    continue

                result = check_segment(
                    path,
                    scenario,
                    road,
                    step,
                    yaw,
                    strict_missing=strict_missing,
                )

                unknown_rejected_ids.update(
                    result["unknown_ids"]
                )

                if not result['valid']:
                    if variant == 'refined':
                        refinement_stats['failed_validation'] += 1
                    rejected['road'] += int(result['road'] > 0)
                    rejected['collision'] += int(result['collisions'] > 0)
                    rejected['unknown'] += int(bool(result['unknown_ids']))
                    continue

                if variant == 'refined':
                    refinement_stats['valid'] += 1

                score = score_candidate(
                    path=path,
                    current=current,
                    goal=goal,
                    route=route,
                    check=result,
                )

                choices.append((
                    score,
                    k,
                    variant,
                    path,
                    result
                ))
        if not choices:
            status = "no_safe_candidate"

            print(f"\n[DTP] No safe candidate at step {step}")
            print("[DTP] Rejection counts:", rejected)

            diagnostics.append({
                "step": step,
                "status": status,
                "rejected": rejected,
                "unknown_obstacle_ids": sorted(
                    unknown_rejected_ids
                ),
            })

            break

        best_score, index, variant, chosen, result = max(
            choices,
            key=lambda x: x[0]
        )

        # Find the first predicted waypoint that reaches the goal.
        goal_distances = np.linalg.norm(
            chosen[1:] - goal,
            axis=1
        )

        reached_indices = np.where(
            goal_distances <= reference_tolerance
        )[0]

        if len(reached_indices) > 0:
            # chosen[0] is the current ego position.
            committed = int(reached_indices[0]) + 1
            goal_reached_in_segment = True
        else:
            # Execute the entire selected segment.
            committed = len(chosen) - 1
            goal_reached_in_segment = False

        # Extract only the waypoints to execute.
        executed = chosen[1:committed + 1]

        # Update the ego state using executed waypoints.
        for j, xy in enumerate(executed, start=1):
            previous = points[-1]
            delta = xy - previous

            if np.linalg.norm(delta) >= 0.15:
                yaw = float(np.arctan2(delta[1], delta[0]))

            speed = float(
                np.linalg.norm(delta) / scenario.dt
            )

            points.append(xy.copy())
            times.append(step + j)

        # Update current position and simulation time.
        current = points[-1].copy()
        step += committed

        current = points[-1].copy()
        step += committed
        cycles += 1
        diagnostics.append({
            "cycle": cycles,
            "step": step,
            "candidate": index,
            "variant": variant,
            "executed_steps": committed,
            "score": float(best_score),
            "refinement_stats": refinement_stats,
            "distance_to_goal": float(
                np.linalg.norm(current - goal)
            ),
            "route_metrics": route_metrics(
                chosen,
                route
            ),
            "check": result,
        })

        if goal_reached_in_segment:
            status = "goal_reached"
            break

    return Rollout(np.asarray(points), np.asarray(times), status, cycles,
                   diagnostics, float(np.linalg.norm(current - goal)))
