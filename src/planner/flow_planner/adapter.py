import os
import torch
import numpy as np
from shapely.geometry import Point
from hydra.utils import instantiate
from commonroad.scenario.obstacle import ObstacleType

from src.planner.flow_planner.data.normalization.obs_normalize import ObservationNormalizer
from src.planner.flow_planner.data.commonroad_data_sample import CommonRoadDataSample

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

class FlowPlannerCommonRoadAdapter:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)

        if "PROJECT_ROOT" not in os.environ:
            os.environ["PROJECT_ROOT"] = "."
        if "SAVE_DIR" not in os.environ:
            os.environ["SAVE_DIR"] = "."
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"

        from omegaconf import OmegaConf, open_dict

        # 1. Load config using OmegaConf
        self.cfg = OmegaConf.load(config_path)
        OmegaConf.set_struct(self.cfg, False)

        # ---------------------------------------------------------
        # 1st: Inject the training dataset variables and clean up
        # ---------------------------------------------------------
        with open_dict(self.cfg):
            # Inject required normalizer data
            if "data" not in self.cfg:
                self.cfg.data = {
                    "dataset": {
                        "train": {
                            "future_downsampling_method": "uniform",
                            "predicted_neighbor_num": 32
                        }
                    }
                }

            # Delete the TensorBoard logging section used during training
            if "recorder" in self.cfg:
                del self.cfg["recorder"]

        # ---------------------------------------------------------
        # 2nd: Recursively fix Hydra paths
        # ---------------------------------------------------------
        def update_targets(cfg_node):
            if OmegaConf.is_dict(cfg_node):
                if "_target_" in cfg_node and isinstance(cfg_node["_target_"], str):
                    if cfg_node["_target_"].startswith("flow_planner."):
                        cfg_node["_target_"] = "src.planner." + cfg_node["_target_"]
                for key in cfg_node.keys():
                    update_targets(cfg_node[key])
            elif OmegaConf.is_list(cfg_node):
                for item in cfg_node:
                    update_targets(item)

        # Apply the path fixes AFTER data is injected
        update_targets(self.cfg)

        # 2. Instantiate the entire Flow Planner architecture dynamically
        model_cfg = self.cfg.model if "model" in self.cfg else self.cfg
        self.model = instantiate(model_cfg).to(self.device)
        self.model.device = self.device

        # 3. Load pre-trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Determine where the weights are stored in the checkpoint dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # 4. Clean PyTorch Lightning / DDP prefixes
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if k.startswith("model."):
                k = k[6:]  # Strip leading "model." if Lightning added it
            clean_state_dict[k] = v

        # 5. Inject weights
        self.model.load_state_dict(clean_state_dict)
        self.model.eval()

        # Normalization / Core helper
        self.normalizer = ObservationNormalizer(config=self.cfg.normalization_stats)

        self.cfg_scale = self.cfg.get("cfg_scale", 1.80)
        self.ode_steps = self.cfg.get("ode_steps", 4)
        print(f"[FlowPlanner] Model successfully loaded on {self.device} (CFG={self.cfg_scale}, ODE_steps={self.ode_steps})")

    # =========================================================
    # DATA EXTRACTION METHODS
    # =========================================================
    def _extract_map_tensors(self, scenario, max_lanes=70, points_per_lane=20):
        lane_polylines = np.zeros((max_lanes, points_per_lane, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((max_lanes, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((max_lanes, 1), dtype=bool)

        lanelets = list(scenario.lanelet_network.lanelets)

        # Select lanelets closest to the ego vehicle
        ego_position = np.asarray(
            self._map_ego_position, dtype=np.float32
        )

        lanelets.sort(
            key=lambda lanelet: np.min(
                np.linalg.norm(
                    np.asarray(lanelet.center_vertices) - ego_position,
                    axis=1
                )
            )
        )

        num_lanes = min(len(lanelets), max_lanes)

        for i in range(num_lanes):
            let = lanelets[i]
            center = resample_polyline(let.center_vertices, points_per_lane)
            left = resample_polyline(let.left_vertices, points_per_lane)
            right = resample_polyline(let.right_vertices, points_per_lane)

            for j in range(points_per_lane):
                cx, cy = center[j]
                dx = center[j + 1, 0] - cx if j < points_per_lane - 1 else cx - center[j - 1, 0]
                dy = center[j + 1, 1] - cy if j < points_per_lane - 1 else cy - center[j - 1, 1]

                lx, ly = left[j]
                rx, ry = right[j]

                lane_polylines[i, j, 0:2] = [cx, cy]
                lane_polylines[i, j, 2:4] = [dx, dy]
                lane_polylines[i, j, 4:6] = [lx - cx, ly - cy]
                lane_polylines[i, j, 6:8] = [rx - cx, ry - cy]
                # EXPLICITLY mark traffic light as 'unknown' [0, 0, 0, 1] rather than zeros
                lane_polylines[i, j, 8:12] = [0.0, 0.0, 0.0, 1.0]

            lanes_speed_limit[i, 0] = 15.0
            lanes_has_speed_limit[i, 0] = True

        return lane_polylines, lanes_speed_limit, lanes_has_speed_limit

    def _extract_neighbor_tensors(self, scenario, current_time_step=0, max_agents=32, time_len=21):
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

    def _extract_ego_state(self, planning_problem, current_time_step=0):
        state = planning_problem.initial_state
        x, y = state.position
        yaw = getattr(state, "orientation", 0.0)
        vel = getattr(state, "velocity", 15.0)

        # Dimensions based on CommonRoad standard (adjust if your scenario defines custom dimensions)
        width = 2.0
        length = 4.5

        # Build the exact 16-feature array expected by the model
        ego_state_array = np.zeros(16, dtype=np.float32)
        ego_state_array[0] = x
        ego_state_array[1] = y
        ego_state_array[2] = np.cos(yaw)
        ego_state_array[3] = np.sin(yaw)
        ego_state_array[4] = vel * np.cos(yaw)  # vx
        ego_state_array[5] = vel * np.sin(yaw)  # vy
        # indices 6-9 are ax, ay, steering, yaw_rate (leaving as 0.0 for step 0)
        ego_state_array[10] = width
        ego_state_array[11] = length
        ego_state_array[12] = 1.0  # ego_type indicator

        return ego_state_array

    def _extract_static_objects(self, scenario, max_static=5):
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

    def _extract_navigation_route(self, scenario, planning_problem, max_route_lanes=25, points_per_lane=20):
        from commonroad_route_planner.route_planner import RoutePlanner

        route_tensor = np.zeros((max_route_lanes, points_per_lane, 12), dtype=np.float32)
        route_planner = RoutePlanner(scenario, planning_problem)
        candidate_routes = route_planner.plan_routes()

        if len(candidate_routes.retrieve_all_routes()) == 0:
            return route_tensor

        route = candidate_routes.retrieve_first_route()
        route_lanelet_ids = route.list_ids_lanelets
        num_lanes = min(len(route_lanelet_ids), max_route_lanes)

        for i in range(num_lanes):
            l_id = route_lanelet_ids[i]
            let = scenario.lanelet_network.find_lanelet_by_id(l_id)
            center = resample_polyline(let.center_vertices, points_per_lane)
            left = resample_polyline(let.left_vertices, points_per_lane)
            right = resample_polyline(let.right_vertices, points_per_lane)

            for j in range(points_per_lane):
                cx, cy = center[j]
                dx = center[j + 1, 0] - cx if j < points_per_lane - 1 else cx - center[j - 1, 0]
                dy = center[j + 1, 1] - cy if j < points_per_lane - 1 else cy - center[j - 1, 1]

                lx, ly = left[j]
                rx, ry = right[j]

                route_tensor[i, j, 0:2] = [cx, cy]
                route_tensor[i, j, 2:4] = [dx, dy]
                route_tensor[i, j, 4:6] = [lx - cx, ly - cy]
                route_tensor[i, j, 6:8] = [rx - cx, ry - cy]
                # EXPLICITLY mark traffic light as 'unknown' [0, 0, 0, 1]
                route_tensor[i, j, 8:12] = [0.0, 0.0, 0.0, 1.0]

        return route_tensor

    # =========================================================
    # CORE INFERENCE LOGIC
    # =========================================================
    def _extract_inputs(self, scenario, planning_problem, current_step=0):
        self._map_ego_position = np.asarray(
            planning_problem.initial_state.position,
            dtype=np.float32
        )

        # Now calls its own internal private methods instead of importing them!
        lane_polylines, lanes_speed_limit, lanes_has_speed_limit = self._extract_map_tensors(scenario)
        neighbors_history = self._extract_neighbor_tensors(scenario, current_time_step=current_step)
        ego_state = self._extract_ego_state(planning_problem, current_time_step=current_step)
        static_objects = self._extract_static_objects(scenario)
        nav_route = self._extract_navigation_route(scenario, planning_problem)

        # Convert to torch and send to device with batch dim (B=1)
        inputs = {
            "ego_current_state": torch.from_numpy(ego_state).float().unsqueeze(0).to(self.device),
            "neighbor_agents_past": torch.from_numpy(neighbors_history).float().unsqueeze(0).to(self.device),
            "lanes": torch.from_numpy(lane_polylines).float().unsqueeze(0).to(self.device),
            "lanes_speed_limit": torch.from_numpy(lanes_speed_limit).float().unsqueeze(0).to(self.device),
            "lanes_has_speed_limit": torch.from_numpy(lanes_has_speed_limit).bool().unsqueeze(0).to(self.device),
            "static_objects": torch.from_numpy(static_objects).float().unsqueeze(0).to(self.device),
            "route_lanes": torch.from_numpy(nav_route).float().unsqueeze(0).to(self.device),
        }
        return inputs, ego_state

    @torch.no_grad()
    def plan(self, scenario, planning_problem, current_step=0) -> np.ndarray:
        """
        Executes ODE sampling to generate the planned trajectory in world coordinates.
        Output: (N_steps, 2) or (N_steps, 4) numpy trajectory.
        """
        initial_state = planning_problem.initial_state

        print("\n========== FLOW-PLANNER DIAGNOSTICS ==========")
        print(f"Ego initial position: {initial_state.position}")
        print(f"Ego initial orientation: {initial_state.orientation:.4f} rad")
        print(f"Ego initial velocity: {initial_state.velocity:.4f} m/s")
        print(f"Scenario dt: {scenario.dt:.4f} s")
        print(f"Prediction horizon: {80 * scenario.dt:.2f} s")
        print("==============================================\n")

        inputs, ego_world_state = self._extract_inputs(scenario, planning_problem, current_step)

        # Transform inputs into ego-centric coordinate frame
        # If the model expects ego-centric frame (X=0, Y=0, heading=0):
        ego_x, ego_y, cos_yaw, sin_yaw = ego_world_state[0], ego_world_state[1], ego_world_state[2], ego_world_state[3]
        yaw = np.arctan2(sin_yaw, cos_yaw)

        # --- SHIFT MAP TO EGO-CENTRIC FRAME ---
        # 1. Create inverse rotation matrix (-yaw)
        cos_inv, sin_inv = np.cos(-yaw), np.sin(-yaw)
        rot_matrix_inv = torch.tensor([
            [cos_inv, -sin_inv],
            [sin_inv,  cos_inv]
        ], dtype=torch.float32, device=self.device)

        ego_pos = torch.tensor([ego_x, ego_y], dtype=torch.float32, device=self.device)

        # 2. Iterate through all spatial tensors to translate and rotate them
        spatial_keys = [
            'ego_agent_past', 'ego_past', 'neighbor_agents_past', 'neighbor_past',
            'lanes', 'route_lanes', 'routes', 'static_objects', 'map_objects'
        ]

        for key in spatial_keys:
            if key in inputs and isinstance(inputs[key], torch.Tensor) and inputs[key].numel() > 0:
                tensor = inputs[key].clone().to(self.device)

                # 1. Identify valid entries before transforming
                valid_mask = torch.any(tensor != 0, dim=-1)

                xy_translated = tensor[..., :2] - ego_pos
                xy_rotated = torch.matmul(
                    xy_translated,
                    rot_matrix_inv.T
                )

                # Preserve zero padding
                tensor[..., :2] = torch.where(
                    valid_mask.unsqueeze(-1),
                    xy_rotated,
                    torch.zeros_like(xy_rotated)
                )

                # 2. Rotate Direction/Heading vectors (Always indices 2:4 for all elements)
                if tensor.shape[-1] >= 4:
                    tensor[..., 2:4] = torch.matmul(tensor[..., 2:4], rot_matrix_inv.T)

                # 3. Rotate specific additional vectors based on category
                if key in ['lanes', 'route_lanes', 'routes']:
                    # Indices 4:6 and 6:8 are left & right boundary offset vectors (lx-cx, ly-cy)
                    if tensor.shape[-1] >= 8:
                        tensor[..., 4:6] = torch.matmul(tensor[..., 4:6], rot_matrix_inv.T)
                        tensor[..., 6:8] = torch.matmul(tensor[..., 6:8], rot_matrix_inv.T)

                elif key in ['neighbor_agents_past', 'neighbor_past']:
                    # Indices 4:6 are velocity vectors (vx, vy)
                    if tensor.shape[-1] >= 6:
                        tensor[..., 4:6] = torch.matmul(tensor[..., 4:6], rot_matrix_inv.T)

                # Note: We do NOT rotate indices 4:6 for static_objects because those are Width & Length!

                inputs[key] = tensor

        # 3. Hardcode ego's current state to the origin
        vel = np.hypot(ego_world_state[4], ego_world_state[5])
        origin_state = torch.tensor([0.0, 0.0, 1.0, 0.0, vel, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.5, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        for key in ['ego_current_state', 'ego_current']:
            if key in inputs:
                inputs[key] = origin_state.unsqueeze(0)
        # -------------------------------------------

        data_obj = CommonRoadDataSample(
            batched=(inputs.get('ego_current_state', inputs.get('ego_current')).dim() > 1),
            ego_past=inputs.get('ego_agent_past', inputs.get('ego_past')),
            ego_current=inputs.get('ego_current_state', inputs.get('ego_current')),
            neighbor_past=inputs.get('neighbor_agents_past', inputs.get('neighbor_past')),
            lanes=inputs.get('lanes'),
            lanes_speedlimit=inputs.get('lanes_speed_limit', inputs.get('lanes_speedlimit')),
            lanes_has_speedlimit=inputs.get('lanes_has_speed_limit', inputs.get('lanes_has_speedlimit')),
            routes=inputs.get('route_lanes', inputs.get('routes')),
            routes_speedlimit=inputs.get('route_lanes_speed_limit', inputs.get('routes_speedlimit')),
            routes_has_speedlimit=inputs.get('route_lanes_has_speed_limit', inputs.get('routes_has_speedlimit')),
            map_objects=inputs.get('static_objects', inputs.get('map_objects'))
        )

        # Execute the exact Flow-Planner ODE generation logic
        planned_traj_ego = self.model.forward_inference(
            data=data_obj,
            use_cfg=True,
            cfg_weight=self.cfg_scale
        )

        if isinstance(planned_traj_ego, torch.Tensor):
            planned_traj_ego = planned_traj_ego.detach().cpu().numpy()

        print("[DEBUG] Raw trajectory shape:", planned_traj_ego.shape)

        # Expected: (batch, candidates, time, features)
        if planned_traj_ego.ndim == 4:
            planned_traj_ego = planned_traj_ego[0, 0]

        # Expected: (batch, time, features)
        elif planned_traj_ego.ndim == 3:
            planned_traj_ego = planned_traj_ego[0]

        elif planned_traj_ego.ndim != 2:
            raise ValueError(
                f"Unexpected trajectory shape: {planned_traj_ego.shape}"
            )

        assert planned_traj_ego.shape[-1] >= 2, (
            "Trajectory must contain x and y coordinates."
        )

        # Transform planned trajectory from ego-centric back to CommonRoad world coordinates:
        # [x_world, y_world]^T = R(yaw) * [x_ego, y_ego]^T + [ego_x, ego_y]^T
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        # Grab all time steps, but only the first 2 features (x, y)
        xy_ego = planned_traj_ego[:, :2]
        xy_world = (rot_matrix @ xy_ego.T).T + np.array([ego_x, ego_y])

        step_distances = np.linalg.norm(
            np.diff(xy_world, axis=0),
            axis=1
        )

        total_distance = np.sum(step_distances)

        print("\n========== TRAJECTORY MOTION ==========")
        print(f"Total path length: {total_distance:.2f} m")
        print(f"Average step distance: {np.mean(step_distances):.4f} m")
        print(f"Maximum step distance: {np.max(step_distances):.4f} m")
        print(f"Minimum step distance: {np.min(step_distances):.4f} m")
        print("=======================================\n")

        # ==========================================
        # PREDICTED SPEED DIAGNOSTICS
        # ==========================================

        # Use the model's prediction timestep.
        # 0.1 is appropriate only if Flow-Planner
        # predicts one state every 0.1 seconds.
        model_dt = 0.1

        step_distances = np.linalg.norm(
            np.diff(xy_world, axis=0),
            axis=1
        )

        predicted_speeds = step_distances / model_dt

        print("\n========== SPEED DIAGNOSTICS ==========")
        print(f"Initial CommonRoad speed: {planning_problem.initial_state.velocity:.2f} m/s")
        print(f"First predicted speed: {predicted_speeds[0]:.2f} m/s")
        print(f"Mean predicted speed: {np.mean(predicted_speeds):.2f} m/s")
        print(f"Last predicted speed: {predicted_speeds[-1]:.2f} m/s")

        print("\nSpeed at selected prediction steps:")
        for i in [0, 9, 19, 39, 59, 78]:
            if i < len(predicted_speeds):
                print(f"Step {i + 1:02d}: {predicted_speeds[i]:.2f} m/s")

        print("=======================================\n")

        from shapely.geometry import Point
        from shapely.ops import unary_union

        goal_poly = None
        if hasattr(planning_problem.goal, "state_list") and len(planning_problem.goal.state_list) > 0:
            goal_state = planning_problem.goal.state_list[0]
            if hasattr(goal_state, "position"):
                pos = goal_state.position
                if hasattr(pos, "shapely_object"):
                    goal_poly = pos.shapely_object
                elif hasattr(pos, "shapes"):  # Handles ShapeGroup properly
                    goal_poly = unary_union([s.shapely_object for s in pos.shapes if hasattr(s, "shapely_object")])

        if goal_poly is not None:

            start_point = Point(
                float(xy_world[0, 0]),
                float(xy_world[0, 1])
            )

            end_point = Point(
                float(xy_world[-1, 0]),
                float(xy_world[-1, 1])
            )

            start_distance = start_point.distance(goal_poly)
            end_distance = end_point.distance(goal_poly)

            print("\n========== GOAL PROGRESS ==========")
            print(f"Start distance to goal: {start_distance:.2f} m")
            print(f"End distance to goal: {end_distance:.2f} m")
            print(f"Distance reduction: {start_distance - end_distance:.2f} m")
            print("===================================\n")

        if goal_poly is not None:
            goal_reached = False

            for i, pt in enumerate(xy_world):
                point = Point(float(pt[0]), float(pt[1]))

                if goal_poly.covers(point):
                    # Stop immediately upon entering goal region
                    xy_world = xy_world[:i + 1]
                    goal_reached = True

                    print(
                        f"[FlowPlanner] Goal position reached "
                        f"at trajectory step {i}"
                    )
                    break

            if not goal_reached:
                final_point = Point(
                    float(xy_world[-1, 0]),
                    float(xy_world[-1, 1])
                )

                distance_to_goal = final_point.distance(goal_poly)

                print(
                    "[FlowPlanner] WARNING: Predicted trajectory "
                    "did not enter the goal region."
                )
                print(
                    f"[FlowPlanner] Final distance to goal: "
                    f"{distance_to_goal:.2f} m"
                )

        return xy_world