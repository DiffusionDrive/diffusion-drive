import torch
import numpy as np
from commonroad.scenario.obstacle import ObstacleType


def resample_polyline(polyline: np.ndarray, num_points: int = 20) -> np.ndarray:
    """Resamples a 2D polyline to have exactly num_points evenly spaced."""
    if len(polyline) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(polyline) == 1:
        return np.tile(polyline, (num_points, 1)).astype(np.float32)

    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))

    if cum_dists[-1] == 0:
        return np.tile(polyline[0], (num_points, 1)).astype(np.float32)

    target_dists = np.linspace(0, cum_dists[-1], num_points)
    resampled_x = np.interp(target_dists, cum_dists, polyline[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, polyline[:, 1])

    return np.column_stack((resampled_x, resampled_y)).astype(np.float32)


def extract_map_tensors(scenario, planning_problem, max_lanes=70, points_per_lane=20):
    """Extracts map polylines sorted by proximity to the ego vehicle position."""
    ego_pos = np.array(planning_problem.initial_state.position, dtype=np.float32)

    lane_polylines = np.zeros((max_lanes, points_per_lane, 12), dtype=np.float32)
    lanes_speed_limit = np.zeros((max_lanes, 1), dtype=np.float32)
    lanes_has_speed_limit = np.zeros((max_lanes, 1), dtype=bool)

    all_lanelets = list(scenario.lanelet_network.lanelets)
    all_lanelets.sort(key=lambda let: np.min(np.linalg.norm(let.center_vertices - ego_pos, axis=1)))

    num_lanes = min(len(all_lanelets), max_lanes)

    for i in range(num_lanes):
        let = all_lanelets[i]
        center = resample_polyline(let.center_vertices, points_per_lane)
        left = resample_polyline(let.left_vertices, points_per_lane)
        right = resample_polyline(let.right_vertices, points_per_lane)

        for j in range(points_per_lane):
            cx, cy = center[j]
            dx = center[j + 1, 0] - cx if j < points_per_lane - 1 else cx - center[j - 1, 0]
            dy = center[j + 1, 1] - cy if j < points_per_lane - 1 else cy - center[j - 1, 1]

            # Normalize tangent vector to unit length [cos(theta), sin(theta)]
            norm = np.hypot(dx, dy)
            if norm > 1e-5:
                dx /= norm
                dy /= norm

            lx, ly = left[j]
            rx, ry = right[j]

            lane_polylines[i, j, 0:2] = [cx, cy]
            lane_polylines[i, j, 2:4] = [dx, dy]
            lane_polylines[i, j, 4:6] = [lx - cx, ly - cy]
            lane_polylines[i, j, 6:8] = [rx - cx, ry - cy]

        lanes_speed_limit[i, 0] = 15.0
        lanes_has_speed_limit[i, 0] = True

    return lane_polylines, lanes_speed_limit, lanes_has_speed_limit


def extract_neighbor_tensors(scenario, current_time_step=0, max_agents=32, time_len=21):
    neighbors_history = np.zeros((max_agents, time_len, 11), dtype=np.float32)
    obstacles = scenario.dynamic_obstacles
    num_agents = min(len(obstacles), max_agents)

    for i in range(num_agents):
        obs = obstacles[i]
        length = obs.obstacle_shape.length
        width = obs.obstacle_shape.width

        type_one_hot = [1.0, 0.0, 0.0]
        if obs.obstacle_type == ObstacleType.PEDESTRIAN:
            type_one_hot = [0.0, 1.0, 0.0]
        elif obs.obstacle_type in [ObstacleType.BICYCLE, ObstacleType.MOTORCYCLE]:
            type_one_hot = [0.0, 0.0, 1.0]

        start_step = current_time_step - time_len + 1
        for t_idx, t in enumerate(range(start_step, current_time_step + 1)):
            state = obs.state_at_time(t)
            if state is not None:
                x, y = state.position
                yaw = getattr(state, "orientation", 0.0)
                vel = getattr(state, "velocity", 0.0)

                neighbors_history[i, t_idx, 0:2] = [x, y]
                neighbors_history[i, t_idx, 2:4] = [np.cos(yaw), np.sin(yaw)]
                neighbors_history[i, t_idx, 4:6] = [vel * np.cos(yaw), vel * np.sin(yaw)]
                neighbors_history[i, t_idx, 6:8] = [width, length]
                neighbors_history[i, t_idx, 8:11] = type_one_hot

    return neighbors_history


def extract_ego_state(planning_problem, current_time_step=0):
    """Extracts a 10-dimensional global ego state for the network."""
    state = planning_problem.initial_state
    x, y = state.position
    yaw = getattr(state, "orientation", 0.0)
    vel = getattr(state, "velocity", 0.0)
    accel = getattr(state, "acceleration", 0.0)

    # [x, y, cos, sin, vx, vy, ax, ay, width, length]
    return np.array([
        x, y, np.cos(yaw), np.sin(yaw),
        vel * np.cos(yaw), vel * np.sin(yaw),
        accel * np.cos(yaw), accel * np.sin(yaw),
        2.30, 5.18  # nuPlan defaults
    ], dtype=np.float32)


def extract_static_objects(scenario, max_static=5):
    static_tensor = np.zeros((max_static, 10), dtype=np.float32)
    obstacles = scenario.static_obstacles
    num_static = min(len(obstacles), max_static)

    for i in range(num_static):
        obs = obstacles[i]
        x, y = obs.initial_state.position
        yaw = getattr(obs.initial_state, "orientation", 0.0)

        type_one_hot = [1.0, 0.0, 0.0, 0.0]
        if obs.obstacle_type == ObstacleType.CAR:
            type_one_hot = [0.0, 1.0, 0.0, 0.0]
        elif obs.obstacle_type == ObstacleType.PEDESTRIAN:
            type_one_hot = [0.0, 0.0, 1.0, 0.0]
        elif obs.obstacle_type in [ObstacleType.BICYCLE, ObstacleType.MOTORCYCLE]:
            type_one_hot = [0.0, 0.0, 0.0, 1.0]

        static_tensor[i, 0:2] = [x, y]
        static_tensor[i, 2:4] = [np.cos(yaw), np.sin(yaw)]
        static_tensor[i, 4:6] = [obs.obstacle_shape.width, obs.obstacle_shape.length]
        static_tensor[i, 6:10] = type_one_hot

    return static_tensor


def extract_navigation_route(scenario, planning_problem, max_route_lanes=25, points_per_lane=20):
    """Extracts sequential route lanelets leading to the goal region."""
    from commonroad_route_planner.route_planner import RoutePlanner

    route_lanes = np.zeros((max_route_lanes, points_per_lane, 12), dtype=np.float32)
    route_speed_limit = np.zeros((max_route_lanes, 1), dtype=np.float32)
    route_has_speed_limit = np.zeros((max_route_lanes, 1), dtype=bool)

    try:
        route_planner = RoutePlanner(scenario, planning_problem)
        candidate_routes = route_planner.plan_routes()
        if len(candidate_routes.retrieve_all_routes()) > 0:
            route = candidate_routes.retrieve_first_route()
            route_lanelet_ids = route.list_ids_lanelets
            num_lanes = len(route_lanelet_ids)

            for i in range(min(num_lanes, max_route_lanes)):
                l_id = route_lanelet_ids[i]
                let = scenario.lanelet_network.find_lanelet_by_id(l_id)
                if let is None:
                    continue

                center = resample_polyline(let.center_vertices, points_per_lane)
                left = resample_polyline(let.left_vertices, points_per_lane)
                right = resample_polyline(let.right_vertices, points_per_lane)

                for j in range(points_per_lane):
                    cx, cy = center[j]
                    dx = center[j + 1, 0] - cx if j < points_per_lane - 1 else cx - center[j - 1, 0]
                    dy = center[j + 1, 1] - cy if j < points_per_lane - 1 else cy - center[j - 1, 1]

                    # Normalize tangent vector to unit length [cos(theta), sin(theta)]
                    norm = np.hypot(dx, dy)
                    if norm > 1e-5:
                        dx /= norm
                        dy /= norm

                    lx, ly = left[j]
                    rx, ry = right[j]

                    route_lanes[i, j, 0:2] = [cx, cy]
                    route_lanes[i, j, 2:4] = [dx, dy]
                    route_lanes[i, j, 4:6] = [lx - cx, ly - cy]
                    route_lanes[i, j, 6:8] = [rx - cx, ry - cy]

                route_speed_limit[i, 0] = 15.0
                route_has_speed_limit[i, 0] = True

            # Fill remaining route slots with the target goal lanelet
            if 0 < num_lanes < max_route_lanes:
                last_idx = min(num_lanes - 1, max_route_lanes - 1)
                for i in range(num_lanes, max_route_lanes):
                    route_lanes[i] = route_lanes[last_idx].copy()
                    route_speed_limit[i] = route_speed_limit[last_idx].copy()
                    route_has_speed_limit[i] = route_has_speed_limit[last_idx].copy()
    except Exception as e:
        print(f"Route extraction notice: {e}")

    return route_lanes, route_speed_limit, route_has_speed_limit

def extract_ego_pose(planning_problem):
    """Extracts [x, y, cos(yaw), sin(yaw)] to use ONLY for coordinate rotation."""
    state = planning_problem.initial_state
    x, y = state.position
    yaw = getattr(state, 'orientation', 0.0)
    return np.array([x, y, np.cos(yaw), np.sin(yaw)], dtype=np.float32)

def extract_ego_dynamics(planning_problem):
    """Extracts the ego vehicle's [vx, vy, ax, ay] for the neural network."""
    state = planning_problem.initial_state

    # CommonRoad states typically store scalar velocity and acceleration
    vel = getattr(state, 'velocity', 0.0)
    accel = getattr(state, 'acceleration', 0.0)

    # In the ego-centric local frame, forward is X, lateral is Y.
    return np.array([vel, 0.0, accel, 0.0], dtype=np.float32)

def normalize_to_ego_frame(ego_world_pose, polylines, neighbors, nav_route, static_objs):
    """Vectorized transformation of all global map/agent coordinates to the ego-centric frame."""
    import numpy as np

    ego_x, ego_y, cos_yaw, sin_yaw = ego_world_pose
    yaw = np.arctan2(sin_yaw, cos_yaw)
    cos_inv, sin_inv = np.cos(-yaw), np.sin(-yaw)

    rot_matrix_inv = np.array([
        [cos_inv, -sin_inv],
        [sin_inv,  cos_inv]
    ], dtype=np.float32)

    ego_pos = np.array([ego_x, ego_y], dtype=np.float32)

    def transform(tensor, has_vec=False, has_lane_bounds=False, has_vel=False):
        t = tensor.copy()

        # 1. Translate & Rotate Positions (x, y)
        mask = (t[..., 0] != 0) | (t[..., 1] != 0)
        if mask.any():
            t[mask, 0:2] = (t[mask, 0:2] - ego_pos) @ rot_matrix_inv.T

        # 2. Rotate Direction Vectors (dx, dy)
        if has_vec:
            mask_vec = (t[..., 2] != 0) | (t[..., 3] != 0)
            if mask_vec.any():
                t[mask_vec, 2:4] = t[mask_vec, 2:4] @ rot_matrix_inv.T

        # 3. Rotate Lane Boundaries
        if has_lane_bounds:
            mask_l = (t[..., 4] != 0) | (t[..., 5] != 0)
            if mask_l.any():
                t[mask_l, 4:6] = t[mask_l, 4:6] @ rot_matrix_inv.T
            mask_r = (t[..., 6] != 0) | (t[..., 7] != 0)
            if mask_r.any():
                t[mask_r, 6:8] = t[mask_r, 6:8] @ rot_matrix_inv.T

        # 4. Rotate Velocities
        if has_vel:
            mask_v = (t[..., 4] != 0) | (t[..., 5] != 0)
            if mask_v.any():
                t[mask_v, 4:6] = t[mask_v, 4:6] @ rot_matrix_inv.T

        return t

    local_poly = transform(polylines, has_vec=True, has_lane_bounds=True)
    local_neigh = transform(neighbors, has_vec=True, has_vel=True)
    local_route = transform(nav_route, has_vec=True, has_lane_bounds=True)  # Updated: has_lane_bounds=True
    local_stat = transform(static_objs, has_vec=True)

    return local_poly, local_neigh, local_route, local_stat

def map_input_to_tensors(inputs, device):
    """Converts PlannerInput numpy arrays to batched PyTorch tensors (B=1)."""
    import torch
    import dataclasses

    tensor_dict = {}
    for field in dataclasses.fields(inputs):
        val = getattr(inputs, field.name)

        if val is not None:
            # Convert boolean arrays to boolean tensors, everything else to float32
            if val.dtype == bool:
                tensor_dict[field.name] = torch.from_numpy(val).bool().unsqueeze(0).to(device)
            else:
                tensor_dict[field.name] = torch.from_numpy(val).float().unsqueeze(0).to(device)

    return tensor_dict