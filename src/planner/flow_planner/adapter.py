import os
import torch
import numpy as np
from shapely.geometry import Polygon, Point
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

                if j < points_per_lane - 1:
                    direction = center[j + 1] - center[j]
                else:
                    direction = center[j] - center[j - 1]

                norm = np.linalg.norm(direction)

                if norm > 1e-6:
                    direction = direction / norm
                else:
                    direction = np.array([1.0, 0.0], dtype=np.float32)

                dx, dy = direction

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

    def _extract_ego_state(self, planning_problem, current_time_step=0, ego_state=None):
        state = (
            ego_state
            if ego_state is not None
            else planning_problem.initial_state
        )
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

    def _build_debug_ego_history(
            self,
            planning_problem,
            dt=0.1,
            history_len=21,
            ego_state=None
    ):
        """
        Diagnostic constant-velocity ego history.

        Output:
            (history_len, 14)

        Layout follows Flow-Planner ego_agent_past:
            x, y, cos_h, sin_h,
            vx, vy, ax, ay,
            width, length, ego_type,
            reserved, reserved, reserved
        """

        state = (
            ego_state
            if ego_state is not None
            else planning_problem.initial_state
        )

        position = np.asarray(
            state.position,
            dtype=np.float32
        )

        heading = float(state.orientation)
        speed = float(state.velocity)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        history = np.zeros(
            (history_len, 14),
            dtype=np.float32
        )

        for i in range(history_len):
            time_offset = (
                                  i - history_len + 1
                          ) * dt

            history[i, 0] = position[0] + vx * time_offset
            history[i, 1] = position[1] + vy * time_offset

            history[i, 2] = np.cos(heading)
            history[i, 3] = np.sin(heading)

            history[i, 4] = vx
            history[i, 5] = vy

            history[i, 6] = 0.0
            history[i, 7] = 0.0

            history[i, 8] = 2.0
            history[i, 9] = 4.5

            history[i, 10] = 1.0

        return history

    def _estimate_obstacle_history(
            self,
            state,
            history_len,
            dt
    ):
        """
        Reconstruct an approximate history using
        constant-velocity backward extrapolation.

        This is for debugging missing history,
        not a replacement for observed states.
        """

        position = np.asarray(
            state.position,
            dtype=np.float32
        )

        orientation = float(state.orientation)
        speed = float(state.velocity)

        velocity = np.array([
            speed * np.cos(orientation),
            speed * np.sin(orientation)
        ], dtype=np.float32)

        positions = np.zeros(
            (history_len, 2),
            dtype=np.float32
        )

        velocities = np.zeros(
            (history_len, 2),
            dtype=np.float32
        )

        for i in range(history_len):

            time_offset = (
                                  i - history_len + 1
                          ) * dt

            positions[i] = (
                    position + velocity * time_offset
            )

            velocities[i] = velocity

        return positions, velocities

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

    def _make_ego_polygon(
            self,
            position,
            heading,
            length=4.5,
            width=2.0
    ):
        """
        Construct the ego vehicle footprint in world coordinates.

        Assumes position represents the vehicle's geometric center.
        """

        x, y = position

        half_length = length / 2.0
        half_width = width / 2.0

        corners = np.array([
            [ half_length,  half_width],
            [ half_length, -half_width],
            [-half_length, -half_width],
            [-half_length,  half_width]
        ])

        c = np.cos(heading)
        s = np.sin(heading)

        rotation = np.array([
            [c, -s],
            [s,  c]
        ])

        world_corners = corners @ rotation.T
        world_corners += np.array([x, y])

        return Polygon(world_corners)

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

                # ==========================================
                # NORMALIZE ROUTE DIRECTION
                # ==========================================

                if j < points_per_lane - 1:
                    direction = center[j + 1] - center[j]
                else:
                    direction = center[j] - center[j - 1]

                norm = np.linalg.norm(direction)

                if norm > 1e-6:
                    direction = direction / norm
                else:
                    direction = np.array(
                        [1.0, 0.0],
                        dtype=np.float32
                    )

                dx, dy = direction

                # ==========================================
                # LEFT AND RIGHT BOUNDARIES
                # ==========================================

                lx, ly = left[j]
                rx, ry = right[j]

                route_tensor[i, j, 0:2] = [cx, cy]
                route_tensor[i, j, 2:4] = [dx, dy]
                route_tensor[i, j, 4:6] = [lx - cx, ly - cy]
                route_tensor[i, j, 6:8] = [rx - cx, ry - cy]

                route_tensor[i, j, 8:12] = [
                    0.0, 0.0, 0.0, 1.0
                ]

                route_tensor[i, j, 0:2] = [cx, cy]
                route_tensor[i, j, 2:4] = [dx, dy]
                route_tensor[i, j, 4:6] = [lx - cx, ly - cy]
                route_tensor[i, j, 6:8] = [rx - cx, ry - cy]
                # EXPLICITLY mark traffic light as 'unknown' [0, 0, 0, 1]
                route_tensor[i, j, 8:12] = [0.0, 0.0, 0.0, 1.0]

        return route_tensor

    def _check_trajectory_collisions(
            self,
            scenario,
            trajectory,
            start_step,
            initial_position,
            ego_length=4.5,
            ego_width=2.0
    ):
        """
        Check predicted ego footprints against CommonRoad obstacles.

        trajectory:
            (N, 2) world-coordinate future positions

        start_step:
            Current CommonRoad timestep

        Returns:
            (is_collision_free, collision_step, obstacle_id)
        """

        trajectory = np.asarray(trajectory, dtype=float)
        initial_position = np.asarray(initial_position, dtype=float)

        for i, position in enumerate(trajectory):

            # Prediction index 0 corresponds to start_step + 1
            future_step = start_step + i + 1

            previous_position = (
                initial_position
                if i == 0
                else trajectory[i - 1]
            )

            direction = position - previous_position

            if np.linalg.norm(direction) > 1e-6:
                heading = np.arctan2(
                    direction[1],
                    direction[0]
                )
            else:
                heading = 0.0

            ego_polygon = self._make_ego_polygon(
                position,
                heading,
                length=ego_length,
                width=ego_width
            )

            for obstacle in (
                    list(scenario.dynamic_obstacles)
                    + list(scenario.static_obstacles)
            ):

                obstacle_occupancy = obstacle.occupancy_at_time(
                    future_step
                )

                if obstacle_occupancy is None:
                    continue

                obstacle_shape = obstacle_occupancy.shape

                if not hasattr(obstacle_shape, "shapely_object"):
                    continue

                obstacle_polygon = obstacle_shape.shapely_object

                if ego_polygon.intersects(obstacle_polygon):
                    return (
                        False,
                        future_step,
                        obstacle.obstacle_id
                    )

        return True, None, None

    def _check_road_boundaries(
            self,
            scenario,
            trajectory,
            initial_position,
            ego_length=4.5,
            ego_width=2.0
    ):
        """
        Check whether the ego vehicle remains inside
        the union of CommonRoad lanelet polygons.
        """

        from shapely.ops import unary_union

        lane_polygons = []

        for lanelet in scenario.lanelet_network.lanelets:

            polygon = lanelet.polygon.shapely_object

            if polygon is not None:
                lane_polygons.append(polygon)

        if not lane_polygons:
            raise ValueError(
                "No lanelet polygons available for road validation."
            )

        road_polygon = unary_union(lane_polygons)

        trajectory = np.asarray(trajectory, dtype=float)
        initial_position = np.asarray(initial_position, dtype=float)

        for i, position in enumerate(trajectory):

            previous_position = (
                initial_position
                if i == 0
                else trajectory[i - 1]
            )

            direction = position - previous_position

            heading = (
                np.arctan2(direction[1], direction[0])
                if np.linalg.norm(direction) > 1e-6
                else 0.0
            )

            ego_polygon = self._make_ego_polygon(
                position,
                heading,
                length=ego_length,
                width=ego_width
            )

            if not road_polygon.covers(ego_polygon):
                return False, i

        return True, None

    def _is_destination_reached(
            self,
            planning_problem,
            position,
            target_position=None,
            tolerance=0.5
    ):
        """
        Check whether the ego reference point has reached
        the selected target inside the goal region.

        No arrival-speed requirement.
        """

        if target_position is None:
            return False

        position = np.asarray(position, dtype=float)
        target_position = np.asarray(
            target_position,
            dtype=float
        )

        distance = float(
            np.linalg.norm(position - target_position)
        )

        return distance <= tolerance

    def _get_goal_target(
            self,
            scenario,
            planning_problem,
            end_margin=1.0
    ):
        """
        Select the midpoint of the far end of the goal box,
        measured along the ego vehicle's initial heading.

        Move the target slightly inward from the boundary.
        """

        import numpy as np
        from shapely.ops import unary_union
        from shapely.geometry import Point

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
                goal_shapes.append(goal_position.shapely_object)

            elif hasattr(goal_position, "shapes"):
                for shape in goal_position.shapes:
                    if hasattr(shape, "shapely_object"):
                        goal_shapes.append(shape.shapely_object)

        if not goal_shapes:
            raise ValueError("No geometric goal region found.")

        goal_polygon = unary_union(goal_shapes)

        # Select the relevant polygon.
        polygons = (
            list(goal_polygon.geoms)
            if goal_polygon.geom_type == "MultiPolygon"
            else [goal_polygon]
        )

        polygon = max(polygons, key=lambda p: p.area)

        # Find the farthest boundary edge.
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
            raise ValueError("Could not identify goal end edge.")

        # Move slightly backward into the goal region.
        target = best_midpoint - forward * end_margin

        if not polygon.covers(Point(target)):
            raise ValueError(
                "Target is outside the goal region. "
                "Adjust end_margin or inspect goal geometry."
            )

        print("\n========== GOAL TARGET DEBUG ==========")
        print("Far edge midpoint:", best_midpoint)
        print("Selected target:", target)
        print("=======================================\n")

        return target

    # =========================================================
    # CORE INFERENCE LOGIC
    # =========================================================
    def _extract_inputs(self, scenario, planning_problem, current_step=0, ego_state=None):
        self._map_ego_position = np.asarray(
            planning_problem.initial_state.position,
            dtype=np.float32
        )

        # Now calls its own internal private methods instead of importing them!
        lane_polylines, lanes_speed_limit, lanes_has_speed_limit = self._extract_map_tensors(scenario)
        neighbors_history = self._extract_neighbor_tensors(scenario, current_time_step=current_step)
        ego_state = self._extract_ego_state(planning_problem, current_time_step=current_step, ego_state=ego_state)
        ego_history = self._build_debug_ego_history(planning_problem, dt=scenario.dt, history_len=21)
        static_objects = self._extract_static_objects(scenario)
        nav_route = self._extract_navigation_route(scenario, planning_problem)

        # Convert to torch and send to device with batch dim (B=1)
        inputs = {
            "ego_current_state": torch.from_numpy(ego_state).float().unsqueeze(0).to(self.device),
            "ego_agent_past": torch.from_numpy(ego_history).float().unsqueeze(0).to(self.device),
            "neighbor_agents_past": torch.from_numpy(neighbors_history).float().unsqueeze(0).to(self.device),
            "lanes": torch.from_numpy(lane_polylines).float().unsqueeze(0).to(self.device),
            "lanes_speed_limit": torch.from_numpy(lanes_speed_limit).float().unsqueeze(0).to(self.device),
            "lanes_has_speed_limit": torch.from_numpy(lanes_has_speed_limit).bool().unsqueeze(0).to(self.device),
            "static_objects": torch.from_numpy(static_objects).float().unsqueeze(0).to(self.device),
            "route_lanes": torch.from_numpy(nav_route).float().unsqueeze(0).to(self.device),
        }

        # ==========================================
        # DEBUG: MAP DIRECTION MAGNITUDES
        # ==========================================

        lanes = inputs["lanes"].detach().cpu().numpy()

        valid = np.any(lanes != 0, axis=-1)

        direction_norms = np.linalg.norm(
            lanes[..., 2:4],
            axis=-1
        )

        valid_norms = direction_norms[valid]

        print("\n========== MAP DIRECTION DEBUG ==========")

        if len(valid_norms) > 0:
            print("Min:", valid_norms.min())
            print("Max:", valid_norms.max())
            print("Mean:", valid_norms.mean())

        print("========================================\n")

        return inputs, ego_state

    @torch.no_grad()
    def plan(
            self,
            scenario,
            planning_problem,
            current_step=0,
            ego_state=None,
            debug_disable_agents=False,
            debug_seed=None,
            debug_keep_first_n_agents=None,
            debug_agent_indices=None,
            debug_disable_ego_history=False,
            truncate_at_goal=True
    ) -> np.ndarray:
        """
        Executes ODE sampling to generate the planned trajectory in world coordinates.
        Output: (N_steps, 2) or (N_steps, 4) numpy trajectory.
        """
        initial_state = (
            ego_state
            if ego_state is not None
            else planning_problem.initial_state
        )

        print("\n========== FLOW-PLANNER DIAGNOSTICS ==========")
        print(f"Ego initial position: {initial_state.position}")
        print(f"Ego initial orientation: {initial_state.orientation:.4f} rad")
        print(f"Ego initial velocity: {initial_state.velocity:.4f} m/s")
        print(f"Scenario dt: {scenario.dt:.4f} s")
        print(f"Prediction horizon: {80 * scenario.dt:.2f} s")
        print("==============================================\n")

        # ==========================================
        # DEBUG: REPRODUCIBLE INFERENCE
        # ==========================================

        if debug_seed is not None:
            import random

            random.seed(debug_seed)
            np.random.seed(debug_seed)
            torch.manual_seed(debug_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(debug_seed)

            print(f"[DEBUG] Random seed: {debug_seed}")

        inputs, ego_world_state = self._extract_inputs(
            scenario,
            planning_problem,
            current_step=current_step,
            ego_state=ego_state
        )

        if debug_disable_ego_history:
            inputs["ego_agent_past"] = torch.zeros_like(
                inputs["ego_agent_past"]
            )
            print("[DEBUG] Ego history disabled")

        print("\n========== INPUT DICTIONARY DEBUG ==========")

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"{key}: shape={tuple(value.shape)}, "
                    f"nonzero={torch.count_nonzero(value).item()}"
                )
            else:
                print(f"{key}: type={type(value).__name__}")

        print("============================================\n")

        # ==========================================
        # DEBUG: EGO HISTORY
        # ==========================================

        ego_history = inputs.get(
            "ego_agent_past",
            inputs.get("ego_past")
        )

        print("\n========== EGO HISTORY DEBUG ==========")

        if ego_history is None:
            print("Ego history is missing.")

        else:
            print("Shape:", tuple(ego_history.shape))

            history_np = ego_history.detach().cpu().numpy()

            print("History values:")
            print(history_np)

            valid_steps = np.any(
                history_np != 0,
                axis=-1
            )

            print("Valid history steps:", valid_steps.sum())

        print("======================================\n")

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

                elif key in [
                    'neighbor_agents_past',
                    'neighbor_past',
                    'ego_agent_past',
                    'ego_past'
                ]:
                    if tensor.shape[-1] >= 6:
                        tensor[..., 4:6] = torch.matmul(
                            tensor[..., 4:6],
                            rot_matrix_inv.T
                        )

                inputs[key] = tensor

        # 3. Hardcode ego's current state to the origin
        vel = np.hypot(ego_world_state[4], ego_world_state[5])
        origin_state = torch.tensor([0.0, 0.0, 1.0, 0.0, vel, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.5, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        for key in ['ego_current_state', 'ego_current']:
            if key in inputs:
                inputs[key] = origin_state.unsqueeze(0)

        print("\n========== FINAL EGO INPUT ==========")

        ego_final = inputs["ego_current_state"]

        print("Shape:", tuple(ego_final.shape))
        print("Values:", ego_final.detach().cpu().numpy())

        print("====================================\n")

        print("\n========== MODEL INPUT KEYS ==========")

        print("Input keys:", list(inputs.keys()))

        print(
            "Ego history shape:",
            tuple(inputs["ego_agent_past"].shape)
        )

        print(
            "Ego history nonzero:",
            torch.count_nonzero(
                inputs["ego_agent_past"]
            ).item()
        )

        print("======================================\n")

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

        assert inputs["ego_agent_past"].shape == (1, 21, 14), (
            f"Unexpected ego history shape: "
            f"{inputs['ego_agent_past'].shape}"
        )

        assert inputs["ego_current_state"].shape == (1, 16), (
            f"Unexpected current ego shape: "
            f"{inputs['ego_current_state'].shape}"
        )

        assert inputs["neighbor_agents_past"].shape[-1] == 11, (
            "Neighbor agent features must have dimension 11.r"
        )

        print("[DEBUG] Flow-Planner input shapes verified")

        print("\n========== DATA SAMPLE DEBUG ==========")
        print("ego_current:", tuple(data_obj.ego_current.shape))
        print("ego_past:", tuple(data_obj.ego_past.shape))
        print("ego_future:", tuple(data_obj.ego_future.shape))
        print("=======================================\n")

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

        print("\n========== RAW MODEL OUTPUT ==========")

        print("Output shape:", planned_traj_ego.shape)

        # planned_traj_ego is already (80, 4)
        raw = planned_traj_ego

        print("\nFirst 5 predicted states:")
        print(raw[:5, :])

        print("\nLast 5 predicted states:")
        print(raw[-5:, :])

        print("\nFeature statistics:")
        for i in range(raw.shape[1]):
            print(
                f"Feature {i}: "
                f"min={raw[:, i].min():.4f}, "
                f"max={raw[:, i].max():.4f}"
            )

        print("====================================\n")

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
        print(f"Current ego speed: {initial_state.velocity:.2f} m/s")
        print(f"First predicted speed: {predicted_speeds[0]:.2f} m/s")
        print(f"Mean predicted speed: {np.mean(predicted_speeds):.2f} m/s")
        print(f"Last predicted speed: {predicted_speeds[-1]:.2f} m/s")

        print("\nSpeed at selected prediction steps:")
        for i in [0, 9, 19, 39, 59, 78]:
            if i < len(predicted_speeds):
                print(f"Step {i + 1:02d}: {predicted_speeds[i]:.2f} m/s")

        print("=======================================\n")

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
                    if truncate_at_goal:
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

            # ==========================================
            # DEBUG: GOAL DISTANCE OVER TIME
            # ==========================================

            if goal_poly is not None:

                goal_distances = np.array([
                    Point(float(p[0]), float(p[1])).distance(goal_poly)
                    for p in xy_world
                ])

                print("\n========== GOAL DISTANCE PROFILE ==========")

                for idx in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
                    if idx < len(goal_distances):
                        print(
                            f"Step {idx:02d}: "
                            f"{goal_distances[idx]:.3f} m"
                        )

                closest_idx = int(np.argmin(goal_distances))

                print(f"Closest step: {closest_idx}")
                print(
                    f"Minimum goal distance: "
                    f"{goal_distances[closest_idx]:.3f} m"
                )

                print("===========================================\n")

            # ==========================================
            # DEBUG: ROUTE FEATURES
            # ==========================================

            route_tensor = inputs["route_lanes"]

            print("\n========== ROUTE DEBUG ==========")
            print("Route tensor shape:", tuple(route_tensor.shape))

            route_np = route_tensor.detach().cpu().numpy()

            valid_route_segments = np.any(
                route_np != 0,
                axis=(-1, -2)
            )

            print(
                "Valid route segments:",
                int(valid_route_segments.sum())
            )

            print("First route segment:")
            print(route_np[0, 0])

            print("=================================\n")

        return xy_world


    def plan_receding_horizon(
            self,
            scenario,
            planning_problem,
            replan_steps=10,
            max_cycles=20,
            seed=42
    ):
        """
        Repeatedly generate and validate short trajectory segments.

        Returns:
            Executed trajectory in world coordinates, shape (N, 2).
        """

        from commonroad.scenario.state import InitialState

        if replan_steps < 1:
            raise ValueError("replan_steps must be >= 1")

        current_state = planning_problem.initial_state
        current_step = int(current_state.time_step)

        # ==========================================
        # SELECT TARGET NEAR CENTRE OF GOAL
        # ==========================================

        target_position = self._get_goal_target(
            scenario,
            planning_problem
        )

        executed = [
            np.asarray(current_state.position, dtype=float)
        ]

        arrival_tolerance = 0.5
        for cycle in range(max_cycles):

            if self._is_destination_reached(
                    planning_problem,
                    current_state.position,
                    target_position=target_position,
                    tolerance=arrival_tolerance
            ):
                print(
                    f"[REPLAN] Goal already satisfied "
                    f"at timestep {current_step}"
                )
                break

            print(
                f"\n========== REPLANNING CYCLE {cycle} =========="
            )

            trajectory = self.plan(
                scenario,
                planning_problem,
                current_step=current_step,
                ego_state=current_state,
                debug_seed=seed + cycle,
                truncate_at_goal=False
            )

            print("\n========== REPLANNING LENGTH DEBUG ==========")
            print("Cycle:", cycle)
            print("Current timestep:", current_step)
            print("Predicted points:", len(trajectory))
            print("Requested execution steps:", replan_steps)
            print("=============================================\n")

            trajectory = np.asarray(trajectory, dtype=float)

            if trajectory.ndim != 2 or trajectory.shape[1] != 2:
                raise ValueError(
                    f"Unexpected trajectory shape: {trajectory.shape}"
                )

            if not np.all(np.isfinite(trajectory)):
                raise ValueError(
                    "Trajectory contains non-finite values."
                )

            if len(trajectory) == 0:
                print("[REPLAN] Empty trajectory.")
                break

            # Execute at most replan_steps future points.
            n = min(replan_steps, len(trajectory))
            segment = trajectory[:n]

            # ==========================================
            # FIND FIRST POINT NEAR GOAL TARGET
            # ==========================================

            for i, position in enumerate(segment):

                if self._is_destination_reached(
                        planning_problem,
                        position,
                        target_position=target_position,
                        tolerance=0.5
                ):
                    segment = segment[:i + 1]
                    n = len(segment)

                    print(
                        f"[REPLAN] Target predicted at "
                        f"segment index {i}"
                    )

                    break

            # ------------------------------------------
            # Collision validation
            # ------------------------------------------

            collision_free, collision_step, obstacle_id = (
                self._check_trajectory_collisions(
                    scenario,
                    segment,
                    start_step=current_step,
                    initial_position=current_state.position
                )
            )

            if not collision_free:
                print(
                    f"[REPLAN] Collision predicted at "
                    f"step {collision_step}, "
                    f"obstacle {obstacle_id}"
                )
                break

            # ------------------------------------------
            # Road-boundary validation
            # ------------------------------------------

            road_valid, invalid_index = (
                self._check_road_boundaries(
                    scenario,
                    segment,
                    initial_position=current_state.position
                )
            )

            if not road_valid:
                print(
                    f"[REPLAN] Road-boundary violation "
                    f"at segment index {invalid_index}"
                )
                break

            # ------------------------------------------
            # Update ego state
            # ------------------------------------------

            previous_position = (
                np.asarray(current_state.position, dtype=float)
                if n == 1
                else segment[-2]
            )

            new_position = segment[-1]

            direction = new_position - previous_position

            distance = float(np.linalg.norm(direction))

            if distance > 1e-6:
                new_heading = float(
                    np.arctan2(direction[1], direction[0])
                )
            else:
                new_heading = float(current_state.orientation)

            new_velocity = distance / scenario.dt

            current_step += n

            current_state = InitialState(
                position=np.asarray(new_position, dtype=float),
                orientation=new_heading,
                velocity=float(new_velocity),
                acceleration=0.0,
                yaw_rate=0.0,
                slip_angle=0.0,
                time_step=current_step
            )

            executed.extend(segment)

            print(f"[REPLAN] Current step: {current_step}")
            print(f"[REPLAN] Position: {new_position}")
            print(f"[REPLAN] Velocity: {new_velocity:.2f} m/s")

            # ==========================================
            # CHECK COMMONROAD GOAL SATISFACTION
            # ==========================================

            goal_reached = planning_problem.goal.is_reached(
                current_state
            )

            print(
                f"[REPLAN] Goal satisfied: {goal_reached}"
            )

            if goal_reached:
                print(
                    f"[REPLAN] Goal reached at timestep "
                    f"{current_step}"
                )
                break

        return np.asarray(executed, dtype=float)