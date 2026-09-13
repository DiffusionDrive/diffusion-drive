import torch
import numpy as np
from src.core.types import PlannerInput
from commonroad.scenario.obstacle import ObstacleType
from commonroad_route_planner.route_planner import RoutePlanner


def resample_polyline(polyline: np.ndarray, num_points: int = 20) -> np.ndarray:
    """Resamples a 2D polyline to have exactly num_points evenly spaced."""
    if len(polyline) == 0:
        return np.zeros((num_points, 2))
    if len(polyline) == 1:
        return np.tile(polyline, (num_points, 1))

    # Calculate cumulative distances along the line
    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))

    # Create target evenly spaced distances
    target_dists = np.linspace(0, cum_dists[-1], num_points)

    # Interpolate x and y separately
    resampled_x = np.interp(target_dists, cum_dists, polyline[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, polyline[:, 1])

    return np.column_stack((resampled_x, resampled_y))


def extract_map_tensors(scenario, max_lanes=70, points_per_lane=20):
    """Converts CommonRoad lanelet network into PyTorch-ready map tensors."""
    lane_polylines = np.zeros((max_lanes, points_per_lane, 12), dtype=np.float32)
    lanes_speed_limit = np.zeros((max_lanes, 1), dtype=np.float32)
    lanes_has_speed_limit = np.zeros((max_lanes, 1), dtype=bool)

    lanelets = list(scenario.lanelet_network.lanelets)
    num_lanes = min(len(lanelets), max_lanes)

    for i in range(num_lanes):
        let = lanelets[i]

        # 1. Resample center, left, and right bounds to exactly 20 points
        center = resample_polyline(let.center_vertices, points_per_lane)
        left = resample_polyline(let.left_vertices, points_per_lane)
        right = resample_polyline(let.right_vertices, points_per_lane)

        # 2. Compute the 12 features for all points in this lanelet
        for j in range(points_per_lane):
            cx, cy = center[j]

            # Direction vector (difference to next point)
            if j < points_per_lane - 1:
                dx = center[j + 1, 0] - cx
                dy = center[j + 1, 1] - cy
            else:
                dx = cx - center[j - 1, 0]
                dy = cy - center[j - 1, 1]

            lx, ly = left[j]
            rx, ry = right[j]

            lane_polylines[i, j, 0] = cx
            lane_polylines[i, j, 1] = cy
            lane_polylines[i, j, 2] = dx
            lane_polylines[i, j, 3] = dy
            lane_polylines[i, j, 4] = lx - cx
            lane_polylines[i, j, 5] = ly - cy
            lane_polylines[i, j, 6] = rx - cx
            lane_polylines[i, j, 7] = ry - cy
            # Indices 8-11 remain 0.0 (No traffic light feature extracted for MVP)

        # 3. Speed limit (Setting a mock 15.0 m/s default for urban environment)
        lanes_speed_limit[i, 0] = 15.0
        lanes_has_speed_limit[i, 0] = True

    return lane_polylines, lanes_speed_limit, lanes_has_speed_limit


def extract_neighbor_tensors(scenario, current_time_step=0, max_agents=32, time_len=21):
    """Converts CommonRoad dynamic obstacles into PyTorch-ready history tensors."""
    neighbors_history = np.zeros((max_agents, time_len, 11), dtype=np.float32)

    obstacles = scenario.dynamic_obstacles
    num_agents = min(len(obstacles), max_agents)

    for i in range(num_agents):
        obs = obstacles[i]

        # 1. Extract physical dimensions
        length = obs.obstacle_shape.length
        width = obs.obstacle_shape.width

        # 2. Extract agent type as One-Hot Encoding
        type_one_hot = [1.0, 0.0, 0.0]  # Default to Car
        if obs.obstacle_type == ObstacleType.PEDESTRIAN:
            type_one_hot = [0.0, 1.0, 0.0]
        elif obs.obstacle_type in [ObstacleType.BICYCLE, ObstacleType.MOTORCYCLE]:
            type_one_hot = [0.0, 0.0, 1.0]

        # 3. Extract the last 21 timesteps ending at current_time_step
        start_step = current_time_step - time_len + 1

        for t_idx, t in enumerate(range(start_step, current_time_step + 1)):
            state = obs.state_at_time(t)

            if state is not None:
                x, y = state.position
                yaw = getattr(state, "orientation", 0.0)
                vel = getattr(state, "velocity", 0.0)

                neighbors_history[i, t_idx, 0] = x
                neighbors_history[i, t_idx, 1] = y
                neighbors_history[i, t_idx, 2] = np.cos(yaw)
                neighbors_history[i, t_idx, 3] = np.sin(yaw)
                neighbors_history[i, t_idx, 4] = vel * np.cos(yaw)  # vx
                neighbors_history[i, t_idx, 5] = vel * np.sin(yaw)  # vy
                neighbors_history[i, t_idx, 6] = width
                neighbors_history[i, t_idx, 7] = length
                neighbors_history[i, t_idx, 8:11] = type_one_hot
            else:
                # If the car didn't exist at timestep `t`, the array remains 0.0
                pass

    return neighbors_history


def extract_ego_state(planning_problem, current_time_step=0):
    """Extracts the ego vehicle's current [x, y, cos(yaw), sin(yaw)] state."""
    # For time step 0, we use the initial state from the planning problem
    # (Student A's simulation loop will eventually pass the live ego state here)
    state = planning_problem.initial_state

    x, y = state.position
    yaw = getattr(state, "orientation", 0.0)

    return np.array([x, y, np.cos(yaw), np.sin(yaw)], dtype=np.float32)


def extract_static_objects(scenario, max_static=5):
    """Extracts static obstacles into a (5, 10) PyTorch-ready tensor.
    Features: [x, y, cos, sin, width, length, type_one_hot(4)]
    """
    static_tensor = np.zeros((max_static, 10), dtype=np.float32)
    obstacles = scenario.static_obstacles
    num_static = min(len(obstacles), max_static)

    for i in range(num_static):
        obs = obstacles[i]
        x, y = obs.initial_state.position
        yaw = getattr(obs.initial_state, "orientation", 0.0)

        # Static type one-hot: [Unknown/Other, Car, Pedestrian, Bicycle/Motorcycle]
        type_one_hot = [1.0, 0.0, 0.0, 0.0]
        if obs.obstacle_type == ObstacleType.CAR:
            type_one_hot = [0.0, 1.0, 0.0, 0.0]
        elif obs.obstacle_type == ObstacleType.PEDESTRIAN:
            type_one_hot = [0.0, 0.0, 1.0, 0.0]
        elif obs.obstacle_type in [ObstacleType.BICYCLE, ObstacleType.MOTORCYCLE]:
            type_one_hot = [0.0, 0.0, 0.0, 1.0]

        static_tensor[i, 0] = x
        static_tensor[i, 1] = y
        static_tensor[i, 2] = np.cos(yaw)
        static_tensor[i, 3] = np.sin(yaw)
        static_tensor[i, 4] = obs.obstacle_shape.width
        static_tensor[i, 5] = obs.obstacle_shape.length
        static_tensor[i, 6:10] = type_one_hot

    return static_tensor


def extract_navigation_route(
    scenario, planning_problem, max_route_lanes=25, points_per_lane=20
):
    """Calculates the shortest path to the goal using the official CommonRoad A* RoutePlanner."""
    route_tensor = np.zeros((max_route_lanes, points_per_lane, 4), dtype=np.float32)

    # 1. Use CommonRoad's built-in A* search to find the route
    route_planner = RoutePlanner(scenario, planning_problem)
    candidate_routes = route_planner.plan_routes()

    if len(candidate_routes.retrieve_all_routes()) == 0:
        print("Warning: No route found to the goal! Returning empty route tensor.")
        return route_tensor

    # Get the best route
    route = candidate_routes.retrieve_first_route()
    route_lanelet_ids = route.list_ids_lanelets

    num_lanes = min(len(route_lanelet_ids), max_route_lanes)

    # 2. Extract and format the lanelets in the route sequence
    for i in range(num_lanes):
        l_id = route_lanelet_ids[i]
        let = scenario.lanelet_network.find_lanelet_by_id(l_id)

        center = resample_polyline(let.center_vertices, points_per_lane)

        for j in range(points_per_lane):
            cx, cy = center[j]

            # Direction vector (dx, dy)
            if j < points_per_lane - 1:
                dx = center[j + 1, 0] - cx
                dy = center[j + 1, 1] - cy
            else:
                dx = cx - center[j - 1, 0]
                dy = cy - center[j - 1, 1]

            route_tensor[i, j, 0] = cx
            route_tensor[i, j, 1] = cy
            route_tensor[i, j, 2] = dx
            route_tensor[i, j, 3] = dy

    return route_tensor


def map_input_to_tensors(inputs: PlannerInput, device: torch.device) -> dict:
    return {
        "ego_current_state": torch.tensor(
            inputs.ego_current_state, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "neighbor_agents_past": torch.tensor(
            inputs.neighbors_history, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "lanes": torch.tensor(
            inputs.lane_polylines, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "lanes_speed_limit": torch.tensor(
            inputs.lanes_speed_limit, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "lanes_has_speed_limit": torch.tensor(
            inputs.lanes_has_speed_limit, dtype=torch.bool, device=device
        ).unsqueeze(0),
        "static_objects": torch.tensor(
            inputs.static_objects, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "route_lanes": torch.tensor(
            inputs.navigation_route, dtype=torch.float32, device=device
        ).unsqueeze(0),
    }
