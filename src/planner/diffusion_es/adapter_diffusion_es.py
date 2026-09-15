import numpy as np
from typing import Tuple


def resample_polyline(polyline: np.ndarray, num_points: int = 20) -> np.ndarray:
    """Reused from Zheng's model: Resamples a 2D polyline to evenly spaced points."""
    if len(polyline) == 0:
        return np.zeros((num_points, 2))
    if len(polyline) == 1:
        return np.tile(polyline, (num_points, 1))

    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    target_dists = np.linspace(0, cum_dists[-1], num_points)

    resampled_x = np.interp(target_dists, cum_dists, polyline[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, polyline[:, 1])

    return np.column_stack((resampled_x, resampled_y))


def extract_pgp_map_tensor(
    scenario, max_nodes=200, poses_per_node=20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts map into Diffusion-ES PGP format.
    Output shape: (max_nodes, poses_per_node, 7)
    Features: [x, y, yaw, boundary_flag, stop_line, crosswalk, carpark]
    """
    # Initialize the [N, P, C] tensor
    lane_node_feats = np.zeros((max_nodes, poses_per_node, 7), dtype=np.float32)
    # Mask to tell the network which nodes actually exist vs zero-padding
    lane_node_masks = np.ones((max_nodes, poses_per_node), dtype=np.float32)

    lanelets = list(scenario.lanelet_network.lanelets)
    num_lanes = min(len(lanelets), max_nodes)

    for i in range(num_lanes):
        let = lanelets[i]
        center = resample_polyline(let.center_vertices, poses_per_node)

        # Valid nodes are set to 0 in the mask (PGP logic: 0 = valid, 1 = missing)
        lane_node_masks[i, :] = 0.0

        for j in range(poses_per_node):
            cx, cy = center[j]

            # Calculate yaw (heading)
            if j < poses_per_node - 1:
                dx = center[j + 1, 0] - cx
                dy = center[j + 1, 1] - cy
            else:
                dx = cx - center[j - 1, 0]
                dy = cy - center[j - 1, 1]
            yaw = np.arctan2(dy, dx)

            # CommonRoad lanelets are boundaries by default
            boundary_flag = 1.0

            # Stop lines, crosswalks, carparks (default to 0.0 for basic MVP)
            stop_line = 1.0 if let.stop_line else 0.0
            crosswalk = 0.0
            carpark = 0.0

            lane_node_feats[i, j, 0] = cx
            lane_node_feats[i, j, 1] = cy
            lane_node_feats[i, j, 2] = yaw
            lane_node_feats[i, j, 3] = boundary_flag
            lane_node_feats[i, j, 4] = stop_line
            lane_node_feats[i, j, 5] = crosswalk
            lane_node_feats[i, j, 6] = carpark

    return lane_node_feats, lane_node_masks


def extract_pgp_agent_tensors(
    scenario, planning_problem, max_vehicles=32, max_pedestrians=8, history_steps=21
):
    """
    Extracts agent history into Diffusion-ES PGP format.
    Output Shapes:
      ego_feats: (1, history_steps, 7)
      vehicle_feats: (max_vehicles, history_steps, 7)
      pedestrian_feats: (max_pedestrians, history_steps, 7)
    """
    # Initialize the [Batch, Time, Features] tensors
    ego_feats = np.zeros((1, history_steps, 7), dtype=np.float32)
    vehicle_feats = np.zeros((max_vehicles, history_steps, 7), dtype=np.float32)
    pedestrian_feats = np.zeros((max_pedestrians, history_steps, 7), dtype=np.float32)

    # Masks (PGP Logic: 0 = valid data, 1 = missing/padded data)
    vehicle_masks = np.ones((max_vehicles, history_steps), dtype=np.float32)
    pedestrian_masks = np.ones((max_pedestrians, history_steps), dtype=np.float32)

    # Extract current Ego state (at t = -1, the most recent timestep)
    # We will expand this to extract full history in future iterations
    initial_state = planning_problem.initial_state
    ego_feats[0, -1, 0] = initial_state.position[0]
    ego_feats[0, -1, 1] = initial_state.position[1]
    ego_feats[0, -1, 2] = initial_state.orientation
    ego_feats[0, -1, 3] = initial_state.velocity * np.cos(initial_state.orientation)
    ego_feats[0, -1, 4] = initial_state.velocity * np.sin(initial_state.orientation)
    ego_feats[0, -1, 5] = 2.0  # Default width
    ego_feats[0, -1, 6] = 4.5  # Default length

    # Extract dynamic obstacles (Vehicles)
    obstacles = scenario.dynamic_obstacles
    num_vehicles = min(len(obstacles), max_vehicles)

    for i in range(num_vehicles):
        obs = obstacles[i]
        # Flag this vehicle as valid at the current timestep
        vehicle_masks[i, -1] = 0.0

        vehicle_feats[i, -1, 0] = obs.initial_state.position[0]
        vehicle_feats[i, -1, 1] = obs.initial_state.position[1]
        vehicle_feats[i, -1, 2] = obs.initial_state.orientation
        vehicle_feats[i, -1, 3] = obs.initial_state.velocity * np.cos(
            obs.initial_state.orientation
        )
        vehicle_feats[i, -1, 4] = obs.initial_state.velocity * np.sin(
            obs.initial_state.orientation
        )
        vehicle_feats[i, -1, 5] = (
            obs.obstacle_shape.width if hasattr(obs.obstacle_shape, "width") else 2.0
        )
        vehicle_feats[i, -1, 6] = (
            obs.obstacle_shape.length if hasattr(obs.obstacle_shape, "length") else 4.5
        )

    return ego_feats, vehicle_feats, pedestrian_feats, vehicle_masks, pedestrian_masks
