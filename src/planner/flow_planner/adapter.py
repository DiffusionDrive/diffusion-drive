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

    def _evaluate_route_following(
            self,
            trajectory,
            reference
    ):
        """
        Measure how closely a trajectory follows
        the selected route centreline.
        """
        from shapely.geometry import LineString, Point

        route_line = LineString(reference)

        trajectory = np.asarray(trajectory, dtype=float)

        lateral_errors = np.array([
            route_line.distance(Point(float(p[0]), float(p[1])))
            for p in trajectory
        ])

        progress = np.array([
            route_line.project(Point(float(p[0]), float(p[1])))
            for p in trajectory
        ])

        results = {
            "mean_lateral_error": float(np.mean(lateral_errors)),
            "max_lateral_error": float(np.max(lateral_errors)),
            "route_progress": float(progress[-1] - progress[0]),
            "final_route_position": float(progress[-1]),
        }

        return results

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
        ego_position = np.asarray(
            self._map_ego_position,
            dtype=float
        )

        ranked_obstacles = []

        for obstacle in scenario.dynamic_obstacles:
            state = obstacle.state_at_time(current_time_step)

            if state is None:
                continue

            distance = np.linalg.norm(
                np.asarray(state.position, dtype=float)
                - ego_position
            )

            ranked_obstacles.append((distance, obstacle))

        ranked_obstacles.sort(key=lambda item: item[0])

        obstacles = [
            obstacle
            for _, obstacle in ranked_obstacles[:max_agents]
        ]

        num_agents = len(obstacles)

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

    def _get_route_centerline(
            self,
            scenario,
            planning_problem,
            spacing=0.5
    ):
        """
        Construct an ordered reference centreline from
        the CommonRoad navigation route.

        Returns:
            (N, 2) world-coordinate reference points.
        """
        from commonroad_route_planner.route_planner import RoutePlanner

        route_planner = RoutePlanner(scenario, planning_problem)
        candidate_routes = route_planner.plan_routes()
        routes = candidate_routes.retrieve_all_routes()

        if not routes:
            raise ValueError("No CommonRoad route found.")

        # Match the route currently used by the adapter.
        route = candidate_routes.retrieve_first_route()
        lanelet_ids = route.list_ids_lanelets

        points = []

        for lanelet_id in lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(
                lanelet_id
            )

            center = np.asarray(
                lanelet.center_vertices,
                dtype=float
            )

            if len(center) < 2:
                continue

            if points:
                # Avoid duplicate points at lanelet connections.
                if np.linalg.norm(center[0] - points[-1]) < 1e-3:
                    center = center[1:]

            points.extend(center)

        points = np.asarray(points, dtype=float)

        if len(points) < 2:
            raise ValueError("Route centreline has too few points.")

        # Remove consecutive duplicates.
        distances = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1
        )

        keep = np.concatenate([
            [True],
            distances > 1e-6
        ])

        points = points[keep]

        if len(points) < 2:
            raise ValueError("Route centreline has zero length.")

        # Resample at approximately uniform spacing.
        cumulative = np.concatenate([
            [0.0],
            np.cumsum(
                np.linalg.norm(np.diff(points, axis=0), axis=1)
            )
        ])

        total_length = cumulative[-1]

        n = max(2, int(np.ceil(total_length / spacing)) + 1)
        samples = np.linspace(0, total_length, n)

        x = np.interp(samples, cumulative, points[:, 0])
        y = np.interp(samples, cumulative, points[:, 1])

        reference = np.column_stack([x, y])

        print("\n========== REFERENCE ROUTE ==========")
        print("Lanelet IDs:", lanelet_ids)
        print("Reference points:", len(reference))
        print(f"Route length: {total_length:.2f} m")
        print("=====================================\n")

        return reference

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
                # EXPLICITLY mark traffic light as 'unknown' [0, 0, 0, 1]
                route_tensor[i, j, 8:12] = [0.0, 0.0, 0.0, 1.0]

        return route_tensor

    def _check_trajectory_collisions(self, scenario, trajectory, start_step,
                                     initial_position=None, ego_length=4.5,
                                     ego_width=2.0):
        """Validate predicted oriented footprints at CommonRoad time steps.

        trajectory must be (N, 3): world x, y, heading.
        This is a sampled-footprint check, not continuous swept collision checking.
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError("Collision checking requires (N, 3) [x, y, heading]")
        for i, state in enumerate(trajectory):
            future_step = start_step + i + 1
            ego_polygon = self._make_ego_polygon(
                state[:2], state[2], length=ego_length, width=ego_width
            )
            for obstacle in list(scenario.dynamic_obstacles) + list(scenario.static_obstacles):
                occupancy = obstacle.occupancy_at_time(future_step)
                if occupancy is None:
                    continue
                shape = occupancy.shape
                geometry = getattr(shape, "shapely_object", None)
                if geometry is None:
                    # Do not silently treat an unsupported obstacle shape as safe.
                    raise ValueError(f"Unsupported obstacle shape for {obstacle.obstacle_id}")
                if ego_polygon.intersects(geometry):
                    return False, future_step, obstacle.obstacle_id
        return True, None, None

    def _check_road_boundaries(self, scenario, trajectory,
                               initial_position=None, ego_length=4.5,
                               ego_width=2.0):
        """Check oriented ego footprints against the drivable lanelet union."""
        from shapely.ops import unary_union
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError("Road checking requires (N, 3) [x, y, heading]")
        road = unary_union([
            lanelet.polygon.shapely_object
            for lanelet in scenario.lanelet_network.lanelets
        ])
        if road.is_empty:
            raise ValueError("No drivable lanelet polygons")
        for i, state in enumerate(trajectory):
            footprint = self._make_ego_polygon(
                state[:2], state[2], length=ego_length, width=ego_width
            )
            if not road.covers(footprint):
                print(f"[ROAD] Rejected index={i}, xy={state[:2]}, "
                      f"heading={state[2]:.3f}, "
                      f"outside_area={footprint.difference(road).area:.3f} m^2")
                return False, i
        return True, None

    def _goal_region_contains(self, planning_problem, position):
        """Geometric goal membership; does not impose a speed requirement."""
        from shapely.geometry import Point
        p = Point(float(position[0]), float(position[1]))
        for state in planning_problem.goal.state_list:
            shape = getattr(state, "position", None)
            if shape is None:
                continue
            shapes = getattr(shape, "shapes", [shape])
            for part in shapes:
                geometry = getattr(part, "shapely_object", None)
                if geometry is not None and geometry.covers(p):
                    return True
        return False

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
        Select a target near the far end of the goal
        region using the route centreline.

        Supports straight and curved roads.
        """
        from shapely.geometry import Point
        from shapely.ops import unary_union

        reference = self._get_route_centerline(
            scenario,
            planning_problem
        )

        goal_shapes = []

        for goal_state in planning_problem.goal.state_list:
            position = getattr(goal_state, "position", None)

            if position is None:
                continue

            if hasattr(position, "shapely_object"):
                goal_shapes.append(position.shapely_object)

            elif hasattr(position, "shapes"):
                for shape in position.shapes:
                    if hasattr(shape, "shapely_object"):
                        goal_shapes.append(shape.shapely_object)

        if not goal_shapes:
            raise ValueError("No geometric goal region found.")

        goal_region = unary_union(goal_shapes)

        # Find reference points inside the goal region.
        inside_indices = [
            i for i, p in enumerate(reference)
            if goal_region.covers(Point(float(p[0]), float(p[1])))
        ]

        if not inside_indices:
            raise ValueError(
                "Reference route does not intersect the goal region. "
                "Check lanelet route selection."
            )

        # Select the last point inside the goal region
        # along the route's travel direction.
        last_index = inside_indices[-1]

        # Move backward along the reference route by end_margin.
        target_index = last_index
        remaining = float(end_margin)

        while target_index > 0 and remaining > 0:
            step_length = np.linalg.norm(
                reference[target_index] - reference[target_index - 1]
            )

            remaining -= step_length
            target_index -= 1

        # Prefer the point near the far end, but inside the goal.
        valid_indices = [
            i for i in inside_indices
            if i <= last_index
        ]

        target_index = min(
            valid_indices,
            key=lambda i: abs(i - target_index)
        )

        target = reference[target_index].copy()

        if not goal_region.covers(Point(float(target[0]), float(target[1]))):
            raise ValueError("Selected target is outside the goal.")

        print("\n========== ROUTE-AWARE GOAL TARGET ==========")
        print("Target:", target)
        print("Goal reference index:", target_index)
        print("=============================================\n")

        return target

    # =========================================================
    # CORE INFERENCE LOGIC
    # =========================================================
    def _extract_inputs(self, scenario, planning_problem, current_step=0, ego_state=None):
        current_ego_state = (
            ego_state
            if ego_state is not None
            else planning_problem.initial_state
        )

        self._map_ego_position = np.asarray(
            current_ego_state.position,
            dtype=np.float32
        )

        # Now calls its own internal private methods instead of importing them!
        lane_polylines, lanes_speed_limit, lanes_has_speed_limit = self._extract_map_tensors(scenario)
        neighbors_history = self._extract_neighbor_tensors(scenario, current_time_step=current_step)
        ego_state = self._extract_ego_state(planning_problem, current_time_step=current_step, ego_state=ego_state)
        ego_history = self._build_debug_ego_history(planning_problem, dt=scenario.dt, history_len=21, ego_state=current_ego_state)
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
            truncate_at_goal=True,
            return_full_state=False
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
        if planned_traj_ego.shape[1] < 4:
            raise ValueError("Model output needs [x, y, cos_heading, sin_heading]")
        local_heading = np.arctan2(planned_traj_ego[:, 3], planned_traj_ego[:, 2])
        world_heading = (local_heading + yaw + np.pi) % (2 * np.pi) - np.pi
        trajectory_world = np.column_stack((xy_world, world_heading))
        print(f"[HEADING] First local={local_heading[0]:.3f}, "
              f"world={world_heading[0]:.3f} rad; ego={yaw:.3f} rad")

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
                        trajectory_world = trajectory_world[:i + 1]
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

        return trajectory_world if return_full_state else xy_world


    def _build_route_speed_profile(self, reference, max_speed=13.0,
                                   max_lateral_accel=2.0, max_deceleration=3.0,
                                   min_curve_speed=1.0):
        """Curvature-based advisory speed profile with backward braking pass."""
        reference = np.asarray(reference, dtype=float)
        if len(reference) < 3:
            raise ValueError("Reference requires at least three points")
        ds = np.linalg.norm(np.diff(reference, axis=0), axis=1)
        if np.any(ds <= 1e-6):
            raise ValueError("Reference contains duplicate adjacent points")
        stations = np.r_[0.0, np.cumsum(ds)]
        dx = np.gradient(reference[:, 0], stations)
        dy = np.gradient(reference[:, 1], stations)
        ddx = np.gradient(dx, stations)
        ddy = np.gradient(dy, stations)
        curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-6)
        speed = np.clip(np.sqrt(max_lateral_accel / np.maximum(curvature, 1e-6)),
                        min_curve_speed, max_speed)
        for i in range(len(speed) - 2, -1, -1):
            speed[i] = min(speed[i], np.sqrt(speed[i+1]**2 + 2 * max_deceleration * ds[i]))
        return stations, curvature, speed

    def _evaluate_candidate_preview(self, trajectory, reference, stations,
                                    route_speeds, current_position, current_speed,
                                    dt, preview_steps=40, max_deceleration=3.0):
        """Advisory speed/acceleration diagnostics, not a feasibility certificate."""
        from shapely.geometry import LineString, Point
        preview = np.asarray(trajectory[:preview_steps], dtype=float)
        if len(preview) == 0:
            return {"speed_penalty": float("inf"), "max_predicted_speed": 0.0,
                    "max_braking": 0.0, "braking_required": False,
                    "stopping_distance": 0.0}
        line = LineString(reference)
        positions = np.vstack((np.asarray(current_position, dtype=float), preview[:, :2]))
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
        projected = np.array([line.project(Point(*p)) for p in preview[:, :2]])
        allowed = np.interp(projected, stations, route_speeds)
        excess = np.maximum(speeds - allowed, 0.0)
        accelerations = np.diff(np.r_[float(current_speed), speeds]) / dt
        current_s = line.project(Point(*np.asarray(current_position, dtype=float)))
        future_idx = np.where((stations > current_s) & (route_speeds < current_speed - 0.5))[0]
        braking_required = False
        if len(future_idx):
            idx = int(future_idx[0])
            available = max(0.0, stations[idx] - current_s)
            needed = max(0.0, (float(current_speed)**2 - route_speeds[idx]**2) / (2 * max_deceleration))
            braking_required = bool(needed > available)
        return {"speed_penalty": float(np.mean(excess**2)),
                "max_predicted_speed": float(np.max(speeds)),
                "max_braking": float(np.max(np.maximum(-accelerations, 0.0))),
                "braking_required": braking_required,
                "stopping_distance": float(current_speed)**2 / (2 * max_deceleration)}

    def _validate_preview_prefix(self, scenario, trajectory, current_step, current_position,
                                 preview_steps=40):
        """Return safe prefix length and rejection reason, using existing validators."""
        preview = np.asarray(trajectory[:preview_steps], dtype=float)
        safe_n, rejection = 0, "none"
        for k in range(1, len(preview) + 1):
            prefix = preview[:k]
            ok, time_step, obstacle_id = self._check_trajectory_collisions(
                scenario, prefix, start_step=current_step, initial_position=current_position)
            if not ok:
                rejection = f"collision obstacle={obstacle_id} step={time_step}"
                break
            ok, index = self._check_road_boundaries(
                scenario, prefix, initial_position=current_position)
            if not ok:
                rejection = f"road boundary index={index}"
                break
            safe_n = k
        return safe_n, rejection

    def _score_route_candidate(self, segment, reference, current_position):
        """Score route progress and centreline deviation for a validated prefix."""
        from shapely.geometry import LineString, Point

        route = LineString(np.asarray(reference, dtype=float))
        start = route.project(Point(*np.asarray(current_position, dtype=float)))
        positions = np.asarray(segment, dtype=float)[:, :2]
        stations = np.array([route.project(Point(*p)) for p in positions])
        errors = np.array([route.distance(Point(*p)) for p in positions])
        progress = float(stations[-1] - start)
        mean_error = float(np.mean(errors))
        backward_motion = float(np.sum(np.maximum(0.0, -np.diff(np.r_[start, stations]))))
        return progress, mean_error, backward_motion

    def _validate_dynamics_prefix(self, trajectory, current_state, dt,
                                  max_acceleration=4.0,
                                  max_deceleration=4.0,
                                  max_yaw_rate=0.65,
                                  max_lateral_acceleration=3.5,
                                  tolerance=1e-3):
        """Validate sampled kinematic continuity from the current ego state.

        Returns (valid_steps, reason). Uses XY finite-difference speed and
        predicted heading. This is a screening test, not a vehicle model.
        """
        states = np.asarray(trajectory, dtype=float)
        if states.ndim != 2 or states.shape[1] != 3 or dt <= 0:
            raise ValueError("Expected (N,3) states and positive dt")
        previous_xy = np.asarray(current_state.position, dtype=float)
        previous_heading = float(current_state.orientation)
        previous_speed = max(0.0, float(current_state.velocity))
        for i, row in enumerate(states):
            xy = row[:2]
            heading = float(row[2])
            speed = float(np.linalg.norm(xy - previous_xy) / dt)
            accel = (speed - previous_speed) / dt
            heading_change = (heading - previous_heading + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = abs(heading_change / dt)
            lateral_accel = 0.5 * (speed + previous_speed) * yaw_rate
            if accel > max_acceleration + tolerance:
                return i, f"acceleration={accel:.2f}m/s^2"
            if accel < -max_deceleration - tolerance:
                return i, f"deceleration={accel:.2f}m/s^2"
            if yaw_rate > max_yaw_rate + tolerance:
                return i, f"yaw_rate={yaw_rate:.2f}rad/s"
            if lateral_accel > max_lateral_acceleration + tolerance:
                return i, f"lateral_acceleration={lateral_accel:.2f}m/s^2"
            previous_xy, previous_heading, previous_speed = xy, heading, speed
        return len(states), "none"

    def _make_braking_trajectory(self, position, heading, speed, reference,
                                 dt, max_deceleration=3.0, max_yaw_rate=0.45,
                                 lookahead=5.0, max_steps=100):
        """Generate a conservative kinematic deceleration candidate.

        Uses bounded heading changes toward the reference line. This is a
        simulation fallback, not a certified vehicle-dynamics controller.
        Output: future (x, y, heading) samples, excluding the current state.
        """
        from shapely.geometry import LineString, Point
        line = LineString(np.asarray(reference, dtype=float))
        xy = np.asarray(position, dtype=float).copy()
        theta = float(heading)
        v = max(0.0, float(speed))
        out = []
        for _ in range(max_steps):
            if v <= 1e-3:
                break
            s = line.project(Point(float(xy[0]), float(xy[1])))
            dynamic_lookahead = max(float(lookahead), 1.2 * v)
            aim = line.interpolate(min(line.length, s + dynamic_lookahead))
            desired = np.arctan2(aim.y - xy[1], aim.x - xy[0])
            error = (desired - theta + np.pi) % (2 * np.pi) - np.pi
            # Also limit turning by lateral acceleration at the current speed.
            lateral_yaw_limit = 2.5 / max(v, 0.5)
            yaw_limit = min(max_yaw_rate, lateral_yaw_limit)
            turn = float(np.clip(error, -yaw_limit * dt, yaw_limit * dt))
            theta = (theta + turn + np.pi) % (2 * np.pi) - np.pi
            new_v = max(0.0, v - max_deceleration * dt)
            travel = 0.5 * (v + new_v) * dt
            xy = xy + travel * np.array([np.cos(theta), np.sin(theta)])
            out.append([xy[0], xy[1], theta])
            v = new_v
        return np.asarray(out, dtype=float).reshape(-1, 3)

    def _brake_is_valid(self, scenario, state, start_step, reference,
                        max_deceleration=3.0, wait_steps=10):
        """Validate a full braking maneuver and a short stationary hold.

        Returns (valid, trajectory, reason). Unknown obstacle occupancy is
        not evidence of future safety; use only for offline experiments.
        """
        brake = self._make_braking_trajectory(
            state.position, state.orientation, state.velocity,
            reference, scenario.dt, max_deceleration=max_deceleration)
        if len(brake) == 0:
            brake = np.empty((0, 3), dtype=float)
        hold = np.repeat(
            np.array([[float(brake[-1, 0]), float(brake[-1, 1]),
                       float(brake[-1, 2])]]) if len(brake) else
            np.array([[float(state.position[0]), float(state.position[1]),
                       float(state.orientation)]]),
            wait_steps, axis=0)
        full = np.vstack((brake, hold))
        dynamics_n, dynamics_reason = self._validate_dynamics_prefix(
            full, state, scenario.dt,
            max_acceleration=4.0,
            max_deceleration=max_deceleration + 0.5,
            max_yaw_rate=0.65,
            max_lateral_acceleration=3.5)
        if dynamics_n < len(full):
            return False, brake, (
                f"braking dynamics index={dynamics_n}: {dynamics_reason}")
        road_ok, idx = self._check_road_boundaries(
            scenario, full, initial_position=state.position)
        if not road_ok:
            return False, brake, f"braking road boundary index={idx}"
        collision_ok, t, oid = self._check_trajectory_collisions(
            scenario, full, start_step=start_step,
            initial_position=state.position)
        if not collision_ok:
            return False, brake, f"braking collision obstacle={oid} step={t}"
        return True, brake, "validated sampled braking + hold"

    def _rollout_route_controller(self, state, reference, stations, route_speeds,
                                  dt, steps=40, desired_speed=4.0,
                                  goal_target=None, goal_tolerance=0.5,
                                  wheelbase=2.7, max_steer=0.45,
                                  max_accel=2.0, max_brake=3.0):
        """Simulate a bounded bicycle-model route tracker (offline experiment).

        Returns future [x,y,heading] samples and corresponding speeds.
        Uses actual current velocity; never resets speed to the model prediction.
        """
        from shapely.geometry import LineString, Point
        route = LineString(np.asarray(reference, dtype=float))
        xy = np.asarray(state.position, dtype=float).copy()
        theta = float(state.orientation)
        v = max(0.0, float(state.velocity))
        result, velocities = [], []
        last_s = route.project(Point(float(xy[0]), float(xy[1])))
        goal_s = (route.project(Point(float(goal_target[0]), float(goal_target[1])))
                  if goal_target is not None else route.length)
        for _ in range(steps):
            station = route.project(Point(float(xy[0]), float(xy[1])))
            # Do not deliberately drive backwards along the route.
            station = max(station, last_s)
            last_s = station
            lookahead = max(3.0, min(10.0, 3.0 + 0.6 * v))
            target = route.interpolate(min(route.length, station + lookahead))
            alpha = (np.arctan2(target.y - xy[1], target.x - xy[0])
                     - theta + np.pi) % (2.0 * np.pi) - np.pi
            steer = float(np.clip(np.arctan2(2.0 * wheelbase * np.sin(alpha),
                                             lookahead), -max_steer, max_steer))
            # Speed advisory at current position and along braking-distance lookahead.
            ahead_s = min(route.length, station + v*v/(2.0*max_brake) + 2.0)
            ahead_mask = (stations >= station) & (stations <= ahead_s)
            curve_cap = float(np.min(route_speeds[ahead_mask])) if np.any(ahead_mask) else float(np.interp(station, stations, route_speeds))
            # Brake early enough to approach the selected target at low speed.
            # Never force a stationary state: speed evolves through acceleration.
            remaining_s = max(0.0, goal_s - station)
            # Smooth approach: brake early, but retain a low-speed creep mode
            # while the reference point remains outside the target tolerance.
            goal_cap = np.sqrt(2.0 * max_brake * max(0.0, remaining_s - 0.25))
            if remaining_s > goal_tolerance and v < 0.8:
                goal_cap = max(goal_cap, 0.6)
            command_speed = min(float(desired_speed), curve_cap, goal_cap)
            if goal_target is not None and np.linalg.norm(xy - goal_target) <= goal_tolerance:
                command_speed = 0.0
            acceleration = float(np.clip((command_speed - v)/dt,
                                         -max_brake, max_accel))
            v_next = max(0.0, v + acceleration * dt)
            v_mid = 0.5 * (v + v_next)
            yaw_rate = v_mid * np.tan(steer) / wheelbase
            # Bound lateral acceleration without discontinuously changing heading.
            max_yaw = 2.5 / max(v_mid, 0.5)
            yaw_rate = float(np.clip(yaw_rate, -max_yaw, max_yaw))
            mid_theta = theta + 0.5 * yaw_rate * dt
            xy = xy + v_mid * dt * np.array([np.cos(mid_theta), np.sin(mid_theta)])
            theta = (theta + yaw_rate * dt + np.pi) % (2*np.pi) - np.pi
            v = v_next
            result.append([xy[0], xy[1], theta])
            velocities.append(v)
        return np.asarray(result, dtype=float), np.asarray(velocities, dtype=float)

    def _route_controller_candidates(self, scenario, state, current_step,
                                     reference, stations, route_speeds,
                                     replan_steps, target, planning_problem):
        """Produce only fully preview-validated bicycle-model candidates.

        A future collision invalidates the entire rollout, rather than accepting
        a prefix which may leave no stopping room. Full validation is sampled,
        not continuous swept-volume verification.
        """
        choices = []
        # Include fine low-speed controls for final goal approach.
        for desired_speed in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            rollout, velocities = self._rollout_route_controller(
                state, reference, stations, route_speeds, scenario.dt,
                steps=40, desired_speed=desired_speed,
                goal_target=target, goal_tolerance=0.5)
            # Check full preview, including a feasible stop/hold when appropriate.
            road_ok, road_idx = self._check_road_boundaries(
                scenario, rollout, initial_position=state.position)
            if not road_ok:
                print(f"[LOCAL] target_speed={desired_speed:.1f} rejected=road index={road_idx}")
                continue
            collision_ok, time_step, obstacle_id = self._check_trajectory_collisions(
                scenario, rollout, start_step=current_step,
                initial_position=state.position)
            if not collision_ok:
                print(f"[LOCAL] target_speed={desired_speed:.1f} rejected=obstacle {obstacle_id} at {time_step}")
                continue
            # Ensure the trajectory remains physically continuous at the splice.
            valid_n, reason = self._validate_dynamics_prefix(
                rollout, state, scenario.dt,
                max_acceleration=2.1, max_deceleration=3.1,
                max_yaw_rate=0.65, max_lateral_acceleration=3.5)
            if valid_n < len(rollout):
                print(f"[LOCAL] target_speed={desired_speed:.1f} rejected=dynamics {reason}")
                continue
            segment = rollout[:replan_steps]
            for j, sample in enumerate(segment):
                if self._is_destination_reached(
                        planning_problem, sample[:2], target_position=target,
                        tolerance=0.5):
                    segment = segment[:j+1]
                    break
            progress, lateral, backward = self._score_route_candidate(
                segment, reference, state.position)
            # Require a meaningful preview safety reserve and prefer a controlled
            # speed near junctions, without rewarding unnecessary stationary holds.
            score = progress - 1.5*lateral - 2.0*backward
            # Near the goal, reward actual reduction of target-point error,
            # rather than only projected progress along the route.
            initial_error = float(np.linalg.norm(np.asarray(state.position) - target))
            final_error = float(np.linalg.norm(segment[-1, :2] - target))
            if initial_error < 12.0:
                score += 3.0 * (initial_error - final_error)
                score -= 0.2 * final_error
            # Avoid repeated stationary selection if a safe moving candidate exists.
            # A zero-speed candidate remains available for yielding to traffic.
            if desired_speed == 0.0:
                score -= 0.5
            if self._is_destination_reached(
                    planning_problem, segment[-1, :2], target_position=target,
                    tolerance=0.5):
                score += 1000.0
            choices.append((score, len(segment), f"local_{desired_speed:g}", segment))
            print(f"[LOCAL] target_speed={desired_speed:.1f} safe=40/40 progress={progress:.2f} score={score:.2f}")
        return choices

    def plan_receding_horizon(
            self, scenario, planning_problem, replan_steps=10,
            max_cycles=20, seed=42, num_candidates=5,
            max_deceleration=3.0, wait_steps=10,
            enforce_dynamics=False
    ):
        """Select safe Flow-Planner prefixes using route progress.

        Returns executed XY points (N, 2), including the initial position.
        Braking is an optional fallback, not a mandatory candidate gate.
        Dynamics checks are diagnostic by default because raw model predictions
        often disagree with the initial velocity at the first 0.1 s step.
        Set enforce_dynamics=True to reject such predictions. Preview
        safety is advisory and does not prove future stopping feasibility.
        Validation remains sampled-footprint based, not a full safety proof.
        """
        from commonroad.scenario.state import InitialState

        if replan_steps < 1 or max_cycles < 1 or num_candidates < 1:
            raise ValueError("replan_steps, max_cycles and num_candidates must be positive")

        if not enforce_dynamics:
            print("[SAFETY WARNING] Dynamics screening is diagnostic only; "
                  "executed trajectories may have infeasible acceleration.")
        current_state = planning_problem.initial_state
        current_step = int(current_state.time_step)
        target = self._get_goal_target(scenario, planning_problem)
        reference = self._get_route_centerline(scenario, planning_problem)
        stations, curvature, route_speeds = self._build_route_speed_profile(reference)
        print(f"[ROUTE SPEED] max curvature={np.max(curvature):.4f} 1/m; "
              f"advisory speed range={np.min(route_speeds):.2f}-{np.max(route_speeds):.2f} m/s")
        executed = [np.asarray(current_state.position, dtype=float)]
        arrival_tolerance = 0.5
        self.last_plan_status = "max_cycles_exceeded"
        self.last_goal_region_reached = False
        self.last_target_reached = False
        stagnant_cycles = 0

        for cycle in range(max_cycles):
            if self._is_destination_reached(
                    planning_problem, current_state.position,
                    target_position=target, tolerance=arrival_tolerance):
                self.last_plan_status = "target_reached"
                self.last_target_reached = True
                self.last_goal_region_reached = self._goal_region_contains(
                    planning_problem, current_state.position)
                break

            print(f"\n========== REPLANNING CYCLE {cycle} ========== ")
            choices = []
            for candidate_id in range(num_candidates):
                candidate_seed = seed + cycle * num_candidates + candidate_id
                trajectory = np.asarray(self.plan(
                    scenario, planning_problem, current_step=current_step,
                    ego_state=current_state, debug_seed=candidate_seed,
                    truncate_at_goal=False, return_full_state=True
                ), dtype=float)
                if trajectory.ndim != 2 or trajectory.shape[1] != 3 or not np.all(np.isfinite(trajectory)):
                    print(f"[CANDIDATE {candidate_id}] Invalid trajectory shape or values")
                    continue
                if len(trajectory) == 0:
                    continue

                proposed = trajectory[:min(replan_steps, len(trajectory))]
                # Stop the proposed prefix at the first point near the target.
                for j, state in enumerate(proposed):
                    if self._is_destination_reached(
                            planning_problem, state[:2], target_position=target,
                            tolerance=arrival_tolerance):
                        proposed = proposed[:j + 1]
                        break

                dynamics_n, dynamics_reason = self._validate_dynamics_prefix(
                    proposed, current_state, scenario.dt)
                if dynamics_n < len(proposed):
                    mode = "REJECT" if enforce_dynamics else "WARNING"
                    print(f"[DYNAMICS {mode}] candidate={candidate_id} "
                          f"valid_steps={dynamics_n}/{len(proposed)} "
                          f"reason={dynamics_reason}")
                # Compatibility mode preserves the previously working
                # geometric planning experiment. It is NOT dynamically
                # validated and must not be used for vehicle control.
                check_n = dynamics_n if enforce_dynamics else len(proposed)
                safe_n = 0
                rejection = "none"
                for k in range(1, check_n + 1):
                    prefix = proposed[:k]
                    collision_free, collision_step, obstacle_id = self._check_trajectory_collisions(
                        scenario, prefix, start_step=current_step,
                        initial_position=current_state.position
                    )
                    if not collision_free:
                        rejection = f"collision obstacle={obstacle_id} time={collision_step}"
                        break
                    road_valid, invalid_index = self._check_road_boundaries(
                        scenario, prefix, initial_position=current_state.position
                    )
                    if not road_valid:
                        rejection = f"road boundary index={invalid_index}"
                        break
                    safe_n = k

                if enforce_dynamics and safe_n == dynamics_n and dynamics_n < len(proposed):
                    rejection = f"dynamics {dynamics_reason} at index={dynamics_n}"
                if safe_n == 0:
                    print(f"[CANDIDATE {candidate_id}] safe=0 rejected={rejection}")
                    continue

                safe_segment = proposed[:safe_n]
                progress, lateral, backward = self._score_route_candidate(
                    safe_segment, reference, current_state.position
                )
                preview_metrics = self._evaluate_candidate_preview(
                    trajectory, reference, stations, route_speeds,
                    current_state.position, current_state.velocity, scenario.dt,
                    preview_steps=40)
                preview = trajectory[:40]
                preview_safe_n, preview_rejection = self._validate_preview_prefix(
                    scenario, preview, current_step, current_state.position,
                    preview_steps=40)
                preview_positions = np.vstack((
                    np.asarray(current_state.position, dtype=float),
                    preview[:preview_safe_n, :2]))
                safe_distance = float(np.linalg.norm(
                    np.diff(preview_positions, axis=0), axis=1).sum())
                print(f"[PREVIEW] candidate={candidate_id} "
                      f"safe_steps={preview_safe_n}/{len(preview)} "
                      f"safe_distance={safe_distance:.2f}m "
                      f"stopping_distance={preview_metrics['stopping_distance']:.2f}m "
                      f"speed_penalty={preview_metrics['speed_penalty']:.2f} "
                      f"braking_required={preview_metrics['braking_required']} "
                      f"rejection={preview_rejection}")
                # Route progress in metres; lateral and reverse motion penalized.
                # Prefer reaching the target, then progress; longer prefixes only
                # win when they also make useful route progress.
                reached = self._is_destination_reached(
                    planning_problem, safe_segment[-1, :2],
                    target_position=target, tolerance=arrival_tolerance
                )
                score = (progress - 1.5 * lateral - 2.0 * backward
                         - 1.0 * preview_metrics["speed_penalty"])
                if (preview_safe_n < len(preview)
                        and safe_distance < preview_metrics["stopping_distance"] + 1.0):
                    score -= 50.0
                if preview_metrics["braking_required"]:
                    score -= 10.0
                if reached:
                    score += 1000.0
                print(f"[CANDIDATE {candidate_id}] seed={candidate_seed} "
                      f"safe={safe_n}/{len(proposed)} progress={progress:.2f}m "
                      f"lateral={lateral:.2f}m backward={backward:.2f}m "
                      f"score={score:.2f} rejected={rejection}")
                # Do not require one specific braking rollout as a hard gate.
                # A sampled brake can intersect a moving obstacle even when the
                # Flow-Planner candidate has a valid preview. Braking feasibility
                # remains a diagnostic, not a proof of recoverability.
                choices.append((score, safe_n, candidate_id, safe_segment))

            # Add dynamically feasible, route-following alternatives. Unlike
            # the neural candidates, these use the current velocity explicitly.
            # A complete 40-step preview must pass before they are considered.
            local_choices = self._route_controller_candidates(
                scenario, current_state, current_step, reference,
                stations, route_speeds, replan_steps, target, planning_problem)
            # Favor the fully preview-validated controller when available;
            # neural candidates remain available if it cannot find a solution.
            if local_choices:
                # Prefer validated kinematic candidates, but avoid a stationary
                # hold when a safe positive-progress alternative is available.
                moving = [c for c in local_choices if c[2] != "local_0" and
                          c[0] > 0.05]
                eligible = moving if moving else local_choices
                choices = [(c[0] + 2.0, c[1], c[2], c[3])
                           for c in eligible]

            if not choices:
                # Only use the fallback when the COMPLETE stop and hold
                # have passed the sampled road and collision checks.
                brake_ok, brake, reason = self._brake_is_valid(
                    scenario, current_state, current_step, reference,
                    max_deceleration=max_deceleration,
                    wait_steps=wait_steps)
                if not brake_ok:
                    self.last_plan_status = "no_safe_candidate_or_brake"
                    print(f"[REPLAN] No safe candidate or braking fallback: {reason}")
                    break
                if len(brake):
                    segment = brake[:replan_steps]
                    n = len(segment)
                    chosen_id = "brake"
                    print(f"[REPLAN] Executing {n} validated braking steps")
                else:
                    # Already stopped. Wait only after validating the hold.
                    n = min(replan_steps, wait_steps)
                    segment = np.repeat(np.array([[
                        float(current_state.position[0]),
                        float(current_state.position[1]),
                        float(current_state.orientation)]]), n, axis=0)
                    chosen_id = "wait"
                    print(f"[REPLAN] Waiting {n} steps and replanning")
            else:
                # Stable deterministic tie-break.
                score, n, chosen_id, segment = max(
                    choices, key=lambda c: (c[0], c[1], str(c[2]))
                )
            previous_position = (
                np.asarray(current_state.position, dtype=float)
                if n == 1 else segment[-2, :2]
            )
            new_position = segment[-1, :2]
            new_heading = float(segment[-1, 2])
            new_velocity = float(np.linalg.norm(new_position - previous_position) / scenario.dt)
            current_step += n
            current_state = InitialState(
                position=np.asarray(new_position, dtype=float),
                orientation=new_heading, velocity=new_velocity,
                acceleration=0.0, yaw_rate=0.0, slip_angle=0.0,
                time_step=current_step
            )
            executed.extend(segment[:, :2])
            moved = float(np.linalg.norm(new_position - np.asarray(executed[-n-1])))
            stagnant_cycles = stagnant_cycles + 1 if moved < 0.02 else 0
            if stagnant_cycles >= 8:
                self.last_plan_status = "stalled_no_progress"
                print("[REPLAN] Stalled: no progress for 8 consecutive cycles")
                break
            print(f"[REPLAN] Selected candidate {chosen_id}; accepted={n}, "
                  f"step={current_step}, position={new_position}, "
                  f"speed={new_velocity:.2f}m/s")
            if self._is_destination_reached(
                    planning_problem, current_state.position,
                    target_position=target, tolerance=arrival_tolerance):
                self.last_plan_status = "target_reached"
                self.last_target_reached = True
                print(f"[REPLAN] Target reached at timestep {current_step}")
                break

        self.last_goal_region_reached = any(
            self._goal_region_contains(planning_problem, p) for p in executed)
        print(f"[REPLAN] Goal region reached: {self.last_goal_region_reached}; "
              f"target reached: {self.last_target_reached}")
        print(f"[REPLAN] Final status: {self.last_plan_status}")
        return np.asarray(executed, dtype=float)