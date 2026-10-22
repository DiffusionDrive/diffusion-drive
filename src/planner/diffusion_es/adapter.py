import numpy as np
import torch
from typing import Tuple
from commonroad.scenario.traffic_light import TrafficLightState
from commonroad.scenario.lanelet import LaneletType

def resample_polyline(polyline, num_points=20):
    points = np.asarray(polyline, dtype=np.float32)

    if len(points) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    if len(points) == 1:
        return np.repeat(points, num_points, axis=0)

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))

    keep = np.concatenate(([True], np.diff(cumulative) > 1e-6))
    cumulative = cumulative[keep]
    points = points[keep]

    if len(points) == 1:
        return np.repeat(points, num_points, axis=0)

    targets = np.linspace(0, cumulative[-1], num_points)

    return np.stack([
        np.interp(targets, cumulative, points[:, 0]),
        np.interp(targets, cumulative, points[:, 1])
    ], axis=-1).astype(np.float32)

def extract_vector_map(scenario, planning_problem):
    points_per_element = 20
    ego_pos = planning_problem.initial_state.position
    ego_yaw = planning_problem.initial_state.orientation
    ego_dir = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])

    # 1. Identify start lanelet best aligned with Ego heading
    ego_lanelets = scenario.lanelet_network.find_lanelet_by_position([ego_pos])[0]
    best_start_id = None
    min_angle_diff = float('inf')

    for l_id in ego_lanelets:
        let = scenario.lanelet_network.find_lanelet_by_id(l_id)
        if let is not None and len(let.center_vertices) >= 2:
            vec = let.center_vertices[1] - let.center_vertices[0]
            lane_yaw = np.arctan2(vec[1], vec[0])
            angle_diff = np.abs(np.arctan2(np.sin(lane_yaw - ego_yaw), np.cos(lane_yaw - ego_yaw)))
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_start_id = l_id

    if best_start_id is None and ego_lanelets:
        best_start_id = ego_lanelets[0]

    # 2. Extract goal shape and all intersecting lanelets
    goal_state = planning_problem.goal.state_list[0]
    goal_shape = goal_state.position

    raw_goal_lanelets = set()
    if hasattr(goal_shape, 'shapes'):
        for shape in goal_shape.shapes:
            raw_goal_lanelets.update(scenario.lanelet_network.find_lanelet_by_shape(shape))
    else:
        raw_goal_lanelets.update(scenario.lanelet_network.find_lanelet_by_shape(goal_shape))

    print("\n========== GOAL DIAGNOSTICS ==========")
    print("Ego position:", ego_pos)
    print("Ego heading:", ego_yaw)
    print("Goal shape type:", type(goal_shape).__name__)
    print("Goal shape center:", getattr(goal_shape, "center", None))
    print("Raw goal lanelets:", sorted(raw_goal_lanelets))

    for i, goal_state in enumerate(planning_problem.goal.state_list):
        print(f"\nGoal state {i}:")
        print("Position:", getattr(goal_state, "position", None))
        print("Orientation:", getattr(goal_state, "orientation", None))
        print("Velocity:", getattr(goal_state, "velocity", None))
        print("Time step:", getattr(goal_state, "time_step", None))

    print("======================================\n")

    # 3. Select single target lanelet aligned with Ego heading & furthest forward
    best_goal_id = None
    max_forward_dist = -float('inf')

    for g_id in raw_goal_lanelets:
        g_let = scenario.lanelet_network.find_lanelet_by_id(g_id)
        if g_let is not None and len(g_let.center_vertices) >= 2:
            vec = g_let.center_vertices[-1] - g_let.center_vertices[0]
            g_yaw = np.arctan2(vec[1], vec[0])
            angle_diff = np.abs(np.arctan2(np.sin(g_yaw - ego_yaw), np.cos(g_yaw - ego_yaw)))

            # Keep only lanelets within 45 degrees of Ego heading
            if angle_diff < np.pi / 4:
                lane_center = np.mean(g_let.center_vertices, axis=0)
                forward_dist = np.dot(lane_center - ego_pos, ego_dir)
                if forward_dist > max_forward_dist:
                    max_forward_dist = forward_dist
                    best_goal_id = g_id

    if best_goal_id is None and raw_goal_lanelets:
        best_goal_id = list(raw_goal_lanelets)[0]

    # 3 and 4. Find a reachable goal lanelet using directed BFS
    from collections import deque

    route_path = []
    best_goal_id = None

    if best_start_id is not None and raw_goal_lanelets:
        queue = deque([[best_start_id]])
        visited = {best_start_id}

        while queue:
            path = queue.popleft()
            current_id = path[-1]

            if current_id in raw_goal_lanelets:
                route_path = path
                best_goal_id = current_id
                break

            lanelet = scenario.lanelet_network.find_lanelet_by_id(
                current_id
            )

            if lanelet is None:
                continue

            for successor_id in lanelet.successor:
                if successor_id not in visited:
                    visited.add(successor_id)
                    queue.append(path + [successor_id])

    if not route_path:
        raise RuntimeError(
            "No successor-connected route found from ego "
            f"lanelet {best_start_id} to goal lanelets "
            f"{sorted(raw_goal_lanelets)}"
        )

    print("[ROUTE] Start lanelet:", best_start_id)
    print("[ROUTE] Goal lanelet:", best_goal_id)
    print("[ROUTE] Lanelet sequence:", route_path)

    print("\n========== ROUTE DIAGNOSTICS ==========")
    print("Start lanelet:", best_start_id)
    print("Goal lanelet:", best_goal_id)
    print("Route path:", route_path)

    for lanelet_id in route_path:
        lanelet = scenario.lanelet_network.find_lanelet_by_id(
            lanelet_id
        )

        if lanelet is not None:
            print(
                f"Lanelet {lanelet_id}: "
                f"successors={lanelet.successor}, "
                f"start={lanelet.center_vertices[0]}, "
                f"end={lanelet.center_vertices[-1]}"
            )

    print("=======================================\n")

    coords_dict = {
        'LANE': np.zeros((1, 30, points_per_element, 2), dtype=np.float32),
        'LEFT_BOUNDARY': np.zeros((1, 30, points_per_element, 2), dtype=np.float32),
        'RIGHT_BOUNDARY': np.zeros((1, 30, points_per_element, 2), dtype=np.float32),
        'STOP_LINE': np.zeros((1, 20, points_per_element, 2), dtype=np.float32),
        'CROSSWALK': np.zeros((1, 20, points_per_element, 2), dtype=np.float32),
        'ROUTE_LANES': np.zeros((1, 30, points_per_element, 2), dtype=np.float32)
    }

    avails_dict = {k: np.zeros((1, v.shape[1], points_per_element), dtype=np.bool_) for k, v in coords_dict.items()}
    tl_dict = {k: np.zeros((1, v.shape[1], points_per_element, 4), dtype=np.float32) for k, v in coords_dict.items()}
    for key in tl_dict:
        tl_dict[key][..., 3] = 1.0

    counters = {k: 0 for k in coords_dict.keys()}

    def get_tl_state(let):
        tl_state_array = [0.0, 0.0, 0.0, 1.0]
        if let.traffic_lights:
            tl = scenario.lanelet_network.find_traffic_light_by_id(list(let.traffic_lights)[0])
            if tl is not None:
                state = tl.get_state_at_time_step(0)
                if state == TrafficLightState.GREEN: tl_state_array = [1.0, 0.0, 0.0, 0.0]
                elif state in [TrafficLightState.YELLOW, TrafficLightState.RED_YELLOW]: tl_state_array = [0.0, 1.0, 0.0, 0.0]
                elif state == TrafficLightState.RED: tl_state_array = [0.0, 0.0, 1.0, 0.0]
        return tl_state_array

    # 6. Populate ROUTE_LANES sequentially
    for l_id in route_path:
        if counters['ROUTE_LANES'] >= 30: break
        let = scenario.lanelet_network.find_lanelet_by_id(l_id)
        if let is not None:
            coords_dict['ROUTE_LANES'][0, counters['ROUTE_LANES']] = resample_polyline(let.center_vertices, points_per_element)
            avails_dict['ROUTE_LANES'][0, counters['ROUTE_LANES']] = True
            tl_dict['ROUTE_LANES'][0, counters['ROUTE_LANES']] = get_tl_state(let)
            counters['ROUTE_LANES'] += 1

    # 7. Populate remaining map elements sorted by distance to Ego
    lanelets = list(scenario.lanelet_network.lanelets)
    lanelets.sort(key=lambda l: np.min(np.linalg.norm(l.center_vertices - ego_pos, axis=1)))

    for let in lanelets:
        min_dist = np.min(np.linalg.norm(let.center_vertices - ego_pos, axis=1))
        if min_dist > 35.0: continue

        tl_state_array = get_tl_state(let)

        if counters['LANE'] < 30:
            coords_dict['LANE'][0, counters['LANE']] = resample_polyline(let.center_vertices, points_per_element)
            avails_dict['LANE'][0, counters['LANE']] = True
            tl_dict['LANE'][0, counters['LANE']] = tl_state_array
            counters['LANE'] += 1

        if counters['LEFT_BOUNDARY'] < 30 and let.left_vertices is not None:
            coords_dict['LEFT_BOUNDARY'][0, counters['LEFT_BOUNDARY']] = resample_polyline(let.left_vertices, points_per_element)
            avails_dict['LEFT_BOUNDARY'][0, counters['LEFT_BOUNDARY']] = True
            tl_dict['LEFT_BOUNDARY'][0, counters['LEFT_BOUNDARY']] = tl_state_array
            counters['LEFT_BOUNDARY'] += 1

        if counters['RIGHT_BOUNDARY'] < 30 and let.right_vertices is not None:
            coords_dict['RIGHT_BOUNDARY'][0, counters['RIGHT_BOUNDARY']] = resample_polyline(let.right_vertices, points_per_element)
            avails_dict['RIGHT_BOUNDARY'][0, counters['RIGHT_BOUNDARY']] = True
            tl_dict['RIGHT_BOUNDARY'][0, counters['RIGHT_BOUNDARY']] = tl_state_array
            counters['RIGHT_BOUNDARY'] += 1

        if counters['STOP_LINE'] < 20 and let.stop_line is not None:
            sl_pts = np.array([let.stop_line.start, let.stop_line.end])
            coords_dict['STOP_LINE'][0, counters['STOP_LINE']] = resample_polyline(sl_pts, points_per_element)
            avails_dict['STOP_LINE'][0, counters['STOP_LINE']] = True
            counters['STOP_LINE'] += 1

        if counters['CROSSWALK'] < 20 and LaneletType.CROSSWALK in let.lanelet_type:
            coords_dict['CROSSWALK'][0, counters['CROSSWALK']] = resample_polyline(let.center_vertices, points_per_element)
            avails_dict['CROSSWALK'][0, counters['CROSSWALK']] = True
            counters['CROSSWALK'] += 1

    return {k: torch.tensor(v) for k, v in coords_dict.items()}, {k: torch.tensor(v) for k, v in avails_dict.items()}, {k: torch.tensor(v) for k, v in tl_dict.items()}

def extract_kinematics(state) -> np.ndarray:
    x, y = state.position
    yaw = state.orientation
    v = getattr(state, 'velocity', 0.0)
    a = getattr(state, 'acceleration', 0.0)
    return np.array([
        x, y, yaw,
        v * np.cos(yaw), v * np.sin(yaw),
        a * np.cos(yaw), a * np.sin(yaw)
    ], dtype=np.float32)

def extract_ego_history(planning_problem) -> torch.Tensor:
    history = np.zeros((1, 5, 7), dtype=np.float32)
    state = planning_problem.initial_state
    x, y = state.position
    yaw = state.orientation
    v = float(getattr(state, "velocity", 0.0))
    a = getattr(state, 'acceleration', 0.0)
    dt = 0.1

    history_times = np.linspace(-1.5, 0.0, 5)
    for i, time_offset in enumerate(history_times):
        time_offset = (i - 4) * dt
        history[0, i, 0] = x + (v * np.cos(yaw) * time_offset)
        history[0, i, 1] = y + (v * np.sin(yaw) * time_offset)
        history[0, i, 2] = yaw
        history[0, i, 3] = v * np.cos(yaw)
        history[0, i, 4] = v * np.sin(yaw)
        history[0, i, 5] = a * np.cos(yaw)
        history[0, i, 6] = a * np.sin(yaw)

    return torch.tensor(history)

def extract_agent_history(scenario) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.zeros((1, 5, 30, 7), dtype=np.float32)
    mask = np.zeros((1, 5, 30), dtype=np.bool_)

    for i, obs in enumerate(scenario.dynamic_obstacles[:30]):
        state_array = extract_kinematics(obs.initial_state)
        data[0, :, i, :] = state_array
        mask[0, :, i] = True

    return torch.tensor(data), torch.tensor(mask)