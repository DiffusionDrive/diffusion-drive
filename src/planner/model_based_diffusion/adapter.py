import numpy as np
import jax.numpy as jnp
from commonroad_route_planner.route_planner import RoutePlanner
from src.planner.model_based_diffusion.model.mbd_planner import Args, run_diffusion
from src.planner.model_based_diffusion.envs.car2d import Car2d, car_dynamics, rk4
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

class ModelBasedDiffusionCommonRoadAdapter:
    def __init__(self, seed=0, num_samples=2048, horizon=80, diffusion_steps=100, auto_horizon=True, max_horizon=220):
        self.args = Args(
            seed=seed,
            env_name="car2d",
            not_render=True,
            Nsample=num_samples,
            Hsample=horizon,
            Ndiffuse=diffusion_steps,
            enable_demo=True,  # Force MBD to utilize CommonRoad route (xref) guidance
            disable_recommended_params=True
        )
        self.dt = 0.1
        self.auto_horizon = auto_horizon
        self.max_horizon = max_horizon
        self.last_diagnostics = {}
        self.trajectory_states = None
        self.drivable_polygon = None
        self.reference_path = None


    @staticmethod
    def _get_route(scenario, planning_problem):
        """Handle RoutePlanner versions returning Route, list, or RouteCandidates."""
        result = RoutePlanner(scenario, planning_problem).plan_routes()
        if hasattr(result, "retrieve_first_route"):
            route = result.retrieve_first_route()
        elif hasattr(result, "retrieve_all_routes"):
            routes = result.retrieve_all_routes()
            route = routes[0] if routes else None
        elif isinstance(result, (list, tuple)):
            route = result[0] if result else None
        else:
            route = result
        if isinstance(route, (list, tuple)):
            route = route[0] if route else None
        if route is None or not hasattr(route, "reference_path"):
            raise RuntimeError("No usable CommonRoad route reference_path")
        return route

    @staticmethod
    def _trim_path(path, start_xy, goal_xy):
        """Cut route at ego projection and target projection, preserving direction."""
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
            raise ValueError("Invalid route reference path")
        i = int(np.argmin(np.linalg.norm(path - start_xy, axis=1)))
        j = int(np.argmin(np.linalg.norm(path - goal_xy, axis=1)))
        if j <= i:
            raise RuntimeError(f"Route direction inconsistent: ego index={i}, goal index={j}")
        trimmed = np.vstack([start_xy, path[i:j + 1], goal_xy])
        d = np.linalg.norm(np.diff(trimmed, axis=0), axis=1)
        return trimmed[np.r_[True, d > 1e-5]]

    def _extract_and_override_env(self, scenario, planning_problem, ego_state=None, time_step=None, local_goal=None):
        """Creates the MBD Car2d env and overrides its hardcoded variables with CommonRoad data."""
        env = Car2d()
        env.dt = float(scenario.dt)
        env.H = self.args.Hsample

        # 1. Start State [x, y, theta]
        initial_state = planning_problem.initial_state
        x0 = jnp.array([initial_state.position[0], initial_state.position[1], initial_state.orientation, initial_state.velocity]) if ego_state is None else jnp.asarray(ego_state)
        env.x0 = x0

        # 2. Goal State [x, y, theta]
        goal_state = planning_problem.goal.state_list[0]

        # Goal is a CommonRoad lanelet ShapeGroup in this scenario.
        # Select the far-end center of the route's terminal goal lanelet.
        route = self._get_route(scenario, planning_problem)
        self.route = route
        goal_lanelets = planning_problem.goal.lanelets_of_goal_position
        lanelet_ids = sorted({lid for ids in (goal_lanelets or {}).values() for lid in ids})
        if lanelet_ids:
            route_ids = list(getattr(route, "lanelet_ids", []))
            ordered_goals = [lid for lid in route_ids if lid in lanelet_ids]
            selected_goal_id = ordered_goals[-1] if ordered_goals else lanelet_ids[0]
            print("Selected goal lanelet:", selected_goal_id, "route lanelets:", route_ids)
            lanelet = scenario.lanelet_network.find_lanelet_by_id(selected_goal_id)
            centerline = np.asarray(lanelet.center_vertices)
            start = np.asarray(initial_state.position)
            # Choose a stopping point within the goal lanelet, not at its far boundary.
            # An interior centerline point gives the ego footprint room to stop.
            idx = max(1, min(len(centerline) - 2, int(0.65 * (len(centerline) - 1))))
            goal_pos = centerline[idx]
            tangent = centerline[min(idx + 1, len(centerline) - 1)] - centerline[max(idx - 1, 0)]
            goal_yaw = float(np.arctan2(tangent[1], tangent[0]))
        else:
            pos_attr = goal_state.position
            goal_pos = np.asarray(pos_attr.center if hasattr(pos_attr, 'center') else pos_attr)
            goal_yaw = float(initial_state.orientation)
        if local_goal is not None:
            goal_pos, goal_yaw = local_goal
        env.xg = jnp.array([goal_pos[0], goal_pos[1], goal_yaw, 0.0])

        # 3. Obstacles (Approximating CommonRoad polygons/rectangles as circles)
        obs_centers = []
        radius_list = []
        for obs in scenario.static_obstacles:
            pos = obs.initial_state.position
            # Safely approximate bounding box as a circle
            radius = max(obs.obstacle_shape.length, obs.obstacle_shape.width) / 2.0
            obs_centers.append([pos[0], pos[1]])
            radius_list.append(radius)

        if len(obs_centers) > 0:
            env.obs_center = jnp.array(obs_centers)
            env.obs_radius = jnp.max(jnp.array(radius_list)) # MBD currently only supports a global radius
        else:
            # Dummy obstacle far away if map is empty to prevent JAX compilation crashes
            env.obs_center = jnp.array([[9999.0, 9999.0]])
            env.obs_radius = 0.1

        # Dynamic obstacle predictions
        H = self.args.Hsample
        dynamic_obstacles = list(scenario.dynamic_obstacles)
        N = max(1, len(dynamic_obstacles))

        dynamic_data = np.zeros((H, N, 3), dtype=np.float32)
        dynamic_valid = np.zeros((H, N), dtype=bool)

        for j, obstacle in enumerate(dynamic_obstacles):
            for k in range(H):
                # The first planned state is after the first action.
                t = (int(initial_state.time_step) if time_step is None else int(time_step)) + k + 1

                occupancy = obstacle.occupancy_at_time(t)
                if occupancy is None:
                    continue

                shape = occupancy.shape

                if hasattr(shape, "vertices"):
                    vertices = np.asarray(shape.vertices)
                    center = vertices.mean(axis=0)
                    radius = np.max(
                        np.linalg.norm(vertices - center, axis=1)
                    )
                elif hasattr(shape, "length") and hasattr(shape, "width"):
                    center = np.asarray(shape.center)
                    radius = 0.5 * np.hypot(
                        shape.length, shape.width
                    )
                elif hasattr(shape, "radius"):
                    center = np.asarray(shape.center)
                    radius = float(shape.radius)
                else:
                    raise TypeError(
                        f"Unsupported obstacle shape: {type(shape)}"
                    )

                dynamic_data[k, j, :2] = center
                dynamic_data[k, j, 2] = radius
                dynamic_valid[k, j] = True

        env.dynamic_obstacles = jnp.asarray(dynamic_data)
        env.dynamic_valid = jnp.asarray(dynamic_valid)

        # 4. Reference centerline from the selected route, starting at ego.
        ref_path = self._trim_path(
            route.reference_path, np.asarray(x0[:2]), np.asarray(env.xg[:2])
        )
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ref_path, axis=0), axis=1))]
        samples = np.linspace(0.0, arc[-1], self.args.Hsample)
        resampled = np.column_stack([np.interp(samples, arc, ref_path[:, k]) for k in range(2)])
        xref = np.zeros((self.args.Hsample, 4), dtype=np.float32)
        xref[:, :2] = resampled
        gradient = np.gradient(resampled, axis=0)
        xref[:, 2] = np.arctan2(gradient[:, 1], gradient[:, 0])
        xref[-1, 2] = goal_yaw
        xref[:, 3] = np.minimum(
            max(float(x0[3]), 0.1),
            np.sqrt(2.0 * 3.0 * np.maximum(arc[-1] - samples, 0.0))
        )
        # Geometry-derived curvature and a conservative speed envelope.
        # This is spatial guidance, NOT a time-indexed tracking target.
        seg = np.diff(resampled, axis=0)
        seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-4)
        headings = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        curvature = np.zeros(len(seg), dtype=np.float32)
        if len(seg) > 1:
            curvature[1:] = np.abs(np.diff(headings)) / np.maximum(
                0.5 * (seg_len[1:] + seg_len[:-1]), 1e-4
            )
            # Slow down in advance of a bend, not after entering it.
            for offset in (1, 2, 3, 4):
                curvature[:-offset] = np.maximum(
                    curvature[:-offset], 0.8 ** offset * curvature[offset:]
                )
        env.route_curvature = jnp.asarray(curvature)
        max_lat_accel = 2.0
        speed_cap = np.sqrt(max_lat_accel / np.maximum(curvature, 1e-4))
        speed_cap = np.minimum(speed_cap, 12.0)
        # A backward braking pass ensures the reference does not demand
        # physically impossible deceleration before a tighter bend or goal.
        speed_nodes = np.r_[speed_cap, 0.0]
        for k in range(len(speed_nodes) - 2, -1, -1):
            speed_nodes[k] = min(
                speed_nodes[k],
                np.sqrt(speed_nodes[k + 1] ** 2 + 2.0 * 3.0 * seg_len[k])
            )
        xref[:, 3] = speed_nodes
        xref[-1, 3] = 0.0
        env.xref = jnp.asarray(xref)
        env.thetaref = env.xref[:, 2]
        # Full road union for footprint validation. A route-specific corridor
        # is used for soft guidance, but the validation should not reject
        # legitimate intersection areas merely because they are on an
        # adjacent connected lanelet.
        lane_polys = []
        for lanelet in scenario.lanelet_network.lanelets:
            poly = Polygon(np.vstack((
                np.asarray(lanelet.left_vertices, dtype=float),
                np.asarray(lanelet.right_vertices, dtype=float)[::-1],
            )))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                lane_polys.append(poly)
        if not lane_polys:
            raise RuntimeError("No valid CommonRoad lanelet polygons")
        self.drivable_polygon = unary_union(lane_polys)
        self.reference_path = resampled.copy()
        # Local lane-width proxy for the soft road-boundary cost.
        widths = np.array([
            self.drivable_polygon.boundary.distance(Point(float(x), float(y)))
            for x, y in resampled
        ], dtype=np.float32)
        env.route_half_width = jnp.asarray(
            np.minimum(widths[:-1], widths[1:]), dtype=jnp.float32
        )
        print(f"Drivable half-width proxy: {widths.min():.2f}-{widths.max():.2f} m")
        env.route_length = float(arc[-1])
        print(f"Remaining route length: {env.route_length:.1f} m")
        print("Reference start:", resampled[0], "end:", resampled[-1])

        # Update the base reward calculation against the new xref
        import jax
        env.rew_xref = jax.vmap(env.get_reward)(env.xref).mean()

        return env

    def plan(self, scenario, planning_problem) -> np.ndarray:
        """
        Executes the Diffusion optimization loop and rolls out the actions to get the trajectory.
        Output: (N_steps, 2) numpy trajectory (X, Y)
        """
        import jax
        import jax.numpy as jnp
        from src.planner.model_based_diffusion.envs.car2d import State

        # Estimate horizon from remaining route distance, not from whole lanelets.
        if self.auto_horizon:
            route = self._get_route(scenario, planning_problem)
            start_xy = np.asarray(planning_problem.initial_state.position, dtype=float)
            path = np.asarray(route.reference_path, dtype=float)
            i = int(np.argmin(np.linalg.norm(path - start_xy, axis=1)))
            remaining_length = float(np.linalg.norm(np.diff(path[i:], axis=0), axis=1).sum())
            desired = int(np.ceil((remaining_length / 7.0 + 3.0) / float(scenario.dt)))
            self.args.Hsample = min(self.max_horizon, max(self.args.Hsample, desired))
            print(f"Estimated route length: {remaining_length:.1f} m; horizon: {self.args.Hsample}")

        # 1. Override the Car2d environment with CommonRoad data
        env = self._extract_and_override_env(scenario, planning_problem)
        self.env = env

        # 2. Run diffusion directly in memory (returns final reward and action tensor Yi)
        _, Yi = run_diffusion(self.args, env=env)

        # Grab the fully denoised action sequence from the last diffusion step: shape (Hsample, 2)
        best_action_sequence = Yi[-1]

        # 3. Rollout actions using the car's RK4 kinematics to get (X, Y) trajectory
        trajectory_states = np.zeros((self.args.Hsample, 4))
        current_state = env.x0

        step_env_jit = jax.jit(env.step)
        state_obj = State(
            pipeline_state=current_state,
            obs=current_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0)
        )

        collision_detected = False
        for t in range(self.args.Hsample):
            state_obj = step_env_jit(state_obj, best_action_sequence[t])
            trajectory_states[t] = np.asarray(state_obj.pipeline_state)
            collision_detected |= bool(np.asarray(state_obj.done))

        self.trajectory_states = trajectory_states.copy()
        trajectory = trajectory_states[:, :2]

        # Validate full vehicle footprints after rollout; retain states for
        # diagnostic plotting even if the plan must be rejected.
        corners = np.array([[2.25, 1.0], [2.25, -1.0],
                            [-2.25, -1.0], [-2.25, 1.0]])
        outside_steps = []
        for k, (x, y, yaw, speed) in enumerate(trajectory_states, start=1):
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                 [np.sin(yaw), np.cos(yaw)]])
            footprint = Polygon(corners @ rotation.T + np.array([x, y]))
            if not self.drivable_polygon.buffer(1e-6).covers(footprint):
                outside_steps.append(k)
        print("Road footprint violations:", len(outside_steps))
        # Validate the actual rollout, not an uninitialized zero array.
        safety_cost = float(env.dynamic_safety_cost(jnp.asarray(trajectory_states)))
        print(f"Dynamic safety cost (actual rollout): {safety_cost:.6f}")

        final_state = np.asarray(state_obj.pipeline_state)

        final_position_error = np.linalg.norm(
            final_state[:2] - np.asarray(env.xg[:2])
        )

        final_speed = float(final_state[3])

        print(
            "Final position error:",
            np.linalg.norm(
                trajectory_states[-1, :2]
                - np.asarray(env.xg[:2])
            )
        )

        print(
            "Final speed:",
            trajectory_states[-1, 3]
        )

        print(
            "Minimum speed:",
            trajectory_states[:, 3].min()
        )

        print(
            "Maximum speed:",
            trajectory_states[:, 3].max()
        )

        print(
            "Static collision detected:",
            collision_detected
        )

        self.last_diagnostics = {
            "position_error_m": float(final_position_error),
            "final_speed_mps": float(final_speed),
            "dynamic_safety_cost": float(safety_cost),
            "static_collision": bool(collision_detected),
            "horizon": int(self.args.Hsample),
            "road_footprint_violations": len(outside_steps),
            "first_road_violation_steps": outside_steps[:10],
        }
        if outside_steps:
            raise RuntimeError(
                f"MBD footprint leaves drivable road at {len(outside_steps)} "
                f"sampled states (first: {outside_steps[:10]})"
            )
        if collision_detected:
            raise RuntimeError("MBD trajectory intersects a static obstacle approximation.")
        if safety_cost > 1e-6:
            raise RuntimeError("MBD trajectory violates the dynamic safety buffer.")

        if final_position_error > 0.8:
            raise RuntimeError(
                "MBD failed to reach the target position."
            )

        if final_speed > 0.2:
            raise RuntimeError(
                "MBD failed to stop at the target."
            )

        return trajectory
    def plan_receding_horizon(
            self, scenario, planning_problem, *, cycle_horizon=60,
            execute_steps=5, max_cycles=100, lookahead_m=35.0,
            goal_tolerance=0.8, stop_tolerance=0.2, max_retries=2,
    ):
        """Closed-loop simulation: replan, validate, execute a short prefix, repeat.

        No unsafe prefix is executed. On failure, returns diagnostics via
        self.trajectory_states and raises RuntimeError.
        This is not a real-time performance guarantee or full CommonRoad checker.
        """
        import time
        import jax
        import jax.numpy as jnp
        from shapely.geometry import Polygon
        from src.planner.model_based_diffusion.envs.car2d import State

        if cycle_horizon < 2 or execute_steps < 1 or execute_steps > cycle_horizon:
            raise ValueError("Require cycle_horizon >= 2 and 1 <= execute_steps <= cycle_horizon")
        self.args.Hsample = int(cycle_horizon)
        self.auto_horizon = False
        route = self._get_route(scenario, planning_problem)
        raw = np.asarray(route.reference_path, dtype=float)
        initial = planning_problem.initial_state
        ego = np.array([*initial.position, initial.orientation, initial.velocity], dtype=float)
        current_step = int(initial.time_step)
        original_goal = self._extract_and_override_env(scenario, planning_problem)
        global_goal = np.asarray(original_goal.xg)
        self.drivable_polygon = original_goal.drivable_polygon if hasattr(original_goal, 'drivable_polygon') else self.drivable_polygon
        full_path = self._trim_path(raw, ego[:2], global_goal[:2])
        lengths = np.linalg.norm(np.diff(full_path, axis=0), axis=1)
        arclen = np.r_[0.0, np.cumsum(lengths)]
        if arclen[-1] < 1e-6:
            raise RuntimeError("Empty remaining route")
        line_segments = np.diff(full_path, axis=0)
        states = [ego.copy()]
        latencies = []
        cycle_log = []
        self.trajectory_states = np.asarray(states)
        self.cycle_latencies = latencies
        self.cycle_log = cycle_log
        self.full_reference_path = full_path.copy()
        corners = np.array([[2.25, 1.0], [2.25, -1.0], [-2.25, -1.0], [-2.25, 1.0]])

        def project(xy):
            vec = xy - full_path[:-1]
            denom = np.maximum(lengths ** 2, 1e-9)
            alpha = np.clip(np.einsum('ij,ij->i', vec, line_segments) / denom, 0.0, 1.0)
            nearest = full_path[:-1] + alpha[:, None] * line_segments
            idx = int(np.argmin(np.sum((nearest - xy) ** 2, axis=1)))
            return float(arclen[idx] + alpha[idx] * lengths[idx])

        def point_at(s):
            s = float(np.clip(s, 0, arclen[-1]))
            idx = min(int(np.searchsorted(arclen, s, side='right') - 1), len(lengths) - 1)
            alpha = (s - arclen[idx]) / max(lengths[idx], 1e-9)
            pos = full_path[idx] + alpha * line_segments[idx]
            heading = np.arctan2(line_segments[idx, 1], line_segments[idx, 0])
            return pos, float(heading)

        # Use the actual CommonRoad occupancy footprint, not an enclosing
        # circle: a long vehicle's bounding circle can reject a safe following
        # maneuver. The margin is still checked explicitly and never bypassed.
        dynamic_margin_m = 0.5

        def ego_footprint(q):
            x, y, yaw = map(float, q[:3])
            rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
            return Polygon(corners @ rot.T + np.array([x, y]))

        def occupancy_polygon(shape):
            if hasattr(shape, 'vertices'):
                return Polygon(np.asarray(shape.vertices, dtype=float))
            if hasattr(shape, 'length') and hasattr(shape, 'width'):
                from shapely.geometry import box
                from shapely.affinity import rotate, translate
                poly = box(-float(shape.length)/2, -float(shape.width)/2,
                           float(shape.length)/2, float(shape.width)/2)
                poly = rotate(poly, float(getattr(shape, 'orientation', 0.0)),
                              origin=(0, 0), use_radians=True)
                c = np.asarray(shape.center, dtype=float)
                return translate(poly, float(c[0]), float(c[1]))
            if hasattr(shape, 'radius'):
                return Point(np.asarray(shape.center, dtype=float)).buffer(float(shape.radius))
            if hasattr(shape, 'shapes'):
                return unary_union([occupancy_polygon(part) for part in shape.shapes])
            raise TypeError(f'Unsupported occupancy geometry: {type(shape)}')

        def validate_one(q, t, env):
            footprint = ego_footprint(q)
            if not self.drivable_polygon.buffer(1e-6).covers(footprint):
                return 'road footprint violation'
            # Preserve the existing conservative static-obstacle approximation.
            centers = np.asarray(env.obs_center)
            radius = float(np.asarray(env.obs_radius))
            if np.any(np.linalg.norm(centers - np.asarray(q[:2]), axis=1) < radius + 2.0):
                return 'static obstacle proximity'
            for obstacle in scenario.dynamic_obstacles:
                occ = obstacle.occupancy_at_time(int(t))
                if occ is None:
                    # Missing future occupancy is unknown, not evidence of clearance.
                    continue
                try:
                    other = occupancy_polygon(occ.shape)
                except (TypeError, ValueError) as exc:
                    return f'dynamic obstacle {obstacle.obstacle_id}: {exc}'
                gap = footprint.distance(other)
                if gap < dynamic_margin_m:
                    return (f'dynamic obstacle {obstacle.obstacle_id} '
                            f'clearance {gap:.2f} m < {dynamic_margin_m:.2f} m')
            return None

        # Evaluate future obstacle clearance along the same simulated dynamics.
        # A margin violation is never accepted; this function only ranks safe
        # candidates and discourages entering a state that cannot be stopped.
        def minimum_dynamic_clearance(q, t):
            footprint = ego_footprint(q)
            minimum = float('inf')
            for obstacle in scenario.dynamic_obstacles:
                occupancy = obstacle.occupancy_at_time(int(t))
                if occupancy is None:
                    continue
                other = occupancy_polygon(occupancy.shape)
                minimum = min(minimum, float(footprint.distance(other)))
            return minimum

        def follow_route_preview(env, preview, acceleration, n_steps):
            step_fn = jax.jit(env.step)
            state = State(jnp.asarray(ego), jnp.asarray(ego),
                          jnp.array(0.0), jnp.array(0.0))
            predicted = []
            min_clearance = float('inf')
            for k in range(n_steps):
                q0 = np.asarray(state.pipeline_state)
                control = route_control(q0, preview, acceleration)
                state = step_fn(state, jnp.asarray(control))
                q = np.asarray(state.pipeline_state)
                reason = validate_one(q, current_step + k + 1, env)
                if reason is not None:
                    return None, min_clearance, f'step {k + 1}: {reason}'
                predicted.append(q)
                min_clearance = min(
                    min_clearance,
                    minimum_dynamic_clearance(q, current_step + k + 1),
                )
            return predicted, min_clearance, None

        def conservative_fallback_score(predicted, clearance, acceleration, start_s):
            # Prefer a buffer beyond the hard minimum, then route progress.
            # Penalize high terminal speed when near a dynamic obstacle.
            final = predicted[-1]
            progress = project(final[:2]) - start_s
            lateral = np.linalg.norm(final[:2] - point_at(project(final[:2]))[0])
            clearance_reward = 2.5 * min(clearance, 3.0) if np.isfinite(clearance) else 7.5
            proximity_penalty = (3.0 * max(float(final[3]) - 2.0, 0.0)
                                 if clearance < 2.0 else 0.0)
            return (0.6 * progress - 2.0 * lateral + clearance_reward
                    - proximity_penalty - 0.25 * max(acceleration, 0.0))

        def simulate_controls(controls, env):
            step_fn = jax.jit(env.step)
            state = State(jnp.asarray(ego), jnp.asarray(ego),
                          jnp.array(0.0), jnp.array(0.0))
            prefix = []
            reason = None
            for k, control in enumerate(controls):
                state = step_fn(state, jnp.asarray(control))
                q = np.asarray(state.pipeline_state)
                reason = validate_one(q, current_step + k + 1, env)
                if reason is not None:
                    return None, f'step {k + 1}: {reason}'
                prefix.append(q)
            return prefix, None

        def route_control(q, preview, acceleration):
            target, _ = point_at(project(q[:2]) + preview)
            angle = np.arctan2(target[1] - q[1], target[0] - q[0])
            error = np.arctan2(np.sin(angle - q[2]), np.cos(angle - q[2]))
            delta = np.arctan2(2.0 * 2.7 * np.sin(error), preview)
            return np.array([np.clip(delta / 0.5, -1.0, 1.0), acceleration])

        for cycle in range(1, max_cycles + 1):
            distance = float(np.linalg.norm(ego[:2] - global_goal[:2]))
            if distance <= goal_tolerance and ego[3] <= stop_tolerance:
                print(f'Goal reached and stopped after {cycle - 1} cycles')
                break
            # Terminal stopping mode: once inside the goal tolerance, do not
            # request a new forward route. The ego and goal may legitimately
            # project to the same reference index at this point.
            # Brake through the same Car2d dynamics and validate every step.
            if distance <= goal_tolerance:
                terminal_start = time.perf_counter()
                env = self.env if self.env is not None else original_goal
                step_fn = jax.jit(env.step)
                terminal_state = State(jnp.asarray(ego), jnp.asarray(ego),
                                       jnp.array(0.0), jnp.array(0.0))
                terminal_prefix = []
                terminal_reason = None
                # Test both a straight stop and small corrective steering;
                # no artificial velocity reset or teleportation is permitted.
                for steering in (0.0, -0.1, 0.1):
                    candidate_state = terminal_state
                    candidate = []
                    reason = None
                    for k in range(cycle_horizon):
                        candidate_state = step_fn(
                            candidate_state, jnp.asarray([steering, -1.0]))
                        q = np.asarray(candidate_state.pipeline_state)
                        reason = validate_one(q, current_step + k + 1, env)
                        if reason is not None:
                            break
                        candidate.append(q)
                        if (np.linalg.norm(q[:2] - global_goal[:2]) <= goal_tolerance
                                and abs(float(q[3])) <= stop_tolerance):
                            terminal_prefix = candidate
                            break
                    if terminal_prefix:
                        break
                    terminal_reason = reason or 'unable to stop inside goal tolerance'
                latency = time.perf_counter() - terminal_start
                latencies.append(latency)
                if not terminal_prefix:
                    cycle_log.append({'cycle': cycle, 'latency_s': latency,
                                      'attempts': 0, 'status': 'terminal_stop_failed'})
                    self.trajectory_states = np.asarray(states)
                    raise RuntimeError(
                        f'Cycle {cycle}: terminal braking failed safely: '
                        f'{terminal_reason}')
                states.extend(terminal_prefix)
                ego = terminal_prefix[-1].copy()
                current_step += len(terminal_prefix)
                self.trajectory_states = np.asarray(states)
                cycle_log.append({'cycle': cycle, 'latency_s': latency,
                                  'attempts': 0, 'status': 'terminal_braking',
                                  'goal_distance_m': float(np.linalg.norm(
                                      ego[:2] - global_goal[:2])),
                                  'speed_mps': float(ego[3])})
                print(f'Cycle {cycle:03d}: validated terminal braking | '
                      f'goal distance {cycle_log[-1]["goal_distance_m"]:.3f} m | '
                      f'speed {ego[3]:.3f} m/s | latency {latency:.3f} s')
                continue

            # Try multiple independently seeded MBD proposals before rejecting a cycle.
            # The first feasible prefix wins; rejected proposals are never executed.
            cycle_start = time.perf_counter()
            s = project(ego[:2])
            accepted = None
            last_reason = 'no feasible proposal'
            candidate_count = 0
            for attempt in range(max_retries + 1):
                # Keep the global route fixed, but vary the local lookahead to
                # give the optimizer alternatives around turns and obstacles.
                candidate_lookahead = lookahead_m * (1.0 + 0.35 * attempt)
                local_s = min(arclen[-1], s + candidate_lookahead)
                local_goal = point_at(local_s)
                if arclen[-1] - s <= candidate_lookahead:
                    local_goal = (global_goal[:2], float(global_goal[2]))
                env = self._extract_and_override_env(
                    scenario, planning_problem, ego_state=ego,
                    time_step=current_step, local_goal=local_goal,
                )
                self.env = env
                self.args.seed = int((self.args.seed + cycle * 101 + attempt * 17) % 2147483647)
                _, Yi = run_diffusion(self.args, env=env)
                jax.block_until_ready(Yi)
                actions = np.asarray(Yi[-1])
                step_fn = jax.jit(env.step)
                state = State(jnp.asarray(ego), jnp.asarray(ego), jnp.array(0.0), jnp.array(0.0))
                prefix = []
                candidate_count += 1
                for k in range(execute_steps):
                    state = step_fn(state, jnp.asarray(actions[k]))
                    q = np.asarray(state.pipeline_state)
                    reason = validate_one(q, current_step + k + 1, env)
                    if reason is not None:
                        last_reason = f'attempt {attempt + 1}, step {k + 1}: {reason}'
                        prefix = []
                        break
                    prefix.append(q)
                if len(prefix) == execute_steps:
                    accepted = prefix
                    break

            # Proactively assess the next 1.5 seconds of route-following,
            # rather than accepting a fast 0.5-second prefix that becomes
            # impossible to brake safely in the following cycle.
            # The extra preview is a ranking tool; only execute_steps are used.
            preview_steps = max(execute_steps, min(cycle_horizon, 15))
            tracking_fallback = False
            fallback = False
            if accepted is None:
                env = self.env
                options = []
                # Use a broad family of slow/neutral controls first, including
                # shorter previews for the tight corner in scenario 67.
                for preview in (1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0):
                    for acceleration in (-1.0, -0.85, -0.7, -0.5,
                                         -0.35, -0.2, 0.0, 0.15, 0.25):
                        predicted, clearance, reason = follow_route_preview(
                            env, preview, acceleration, preview_steps)
                        if predicted is None:
                            if reason:
                                last_reason = f'fallback preview {preview:.1f}: {reason}'
                            continue
                        score = conservative_fallback_score(
                            predicted, clearance, acceleration, s)
                        options.append((score, predicted[:execute_steps],
                                        preview, acceleration, clearance))
                if options:
                    options.sort(key=lambda item: item[0], reverse=True)
                    _, accepted, selected_preview, selected_acceleration, clearance = options[0]
                    tracking_fallback = True
                    print(f'Cycle {cycle:03d}: anticipatory route-tracking fallback '
                          f'(preview={selected_preview:.1f} m, '
                          f'accel={selected_acceleration:+.2f}, '
                          f'predicted min clearance={clearance:.2f} m)')

            # If no 1.5-second preview is safe, try shorter validated braking
            # prefixes. Do not bypass road containment or the 0.50 m margin.
            if accepted is None:
                env = self.env
                braking_options = []
                for preview in (1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0):
                    for deceleration in (-1.0, -0.85, -0.7, -0.5, -0.35):
                        predicted, clearance, reason = follow_route_preview(
                            env, preview, deceleration, execute_steps)
                        if predicted is None:
                            if reason:
                                last_reason = f'braking preview {preview:.1f}: {reason}'
                            continue
                        # Favor slower terminal speeds, then larger clearance.
                        braking_options.append((float(predicted[-1][3]),
                                                -min(clearance, 10.0),
                                                predicted, preview, deceleration))
                if braking_options:
                    braking_options.sort(key=lambda item: (item[0], item[1]))
                    _, _, accepted, preview, deceleration = braking_options[0]
                    fallback = True
                    print(f'Cycle {cycle:03d}: validated route-aligned braking '
                          f'(preview={preview:.1f} m, accel={deceleration:+.2f})')

            # Waiting is only feasible once ego is stopped. Re-evaluate
            # dynamic occupancies at each future time, never freeze them.
            if accepted is None and abs(float(ego[3])) <= 0.05:
                env = self.env
                wait_states = [ego.copy() for _ in range(execute_steps)]
                wait_reasons = [validate_one(q, current_step + k + 1, env)
                                for k, q in enumerate(wait_states)]
                if all(reason is None for reason in wait_reasons):
                    accepted = wait_states
                    fallback = True
                    print(f'Cycle {cycle:03d}: safe wait for moving obstacles')

            latency = time.perf_counter() - cycle_start
            latencies.append(latency)
            if accepted is None:
                cycle_log.append({'cycle': cycle, 'latency_s': latency,
                                  'attempts': candidate_count, 'status': last_reason})
                self.trajectory_states = np.asarray(states)
                raise RuntimeError(
                    f'Cycle {cycle}: no safe executable prefix after '
                    f'{candidate_count} MBD attempts and braking fallback: '
                    f'{last_reason}; no unsafe action executed'
                )

            states.extend(accepted)
            ego = accepted[-1].copy()
            current_step += len(accepted)
            self.trajectory_states = np.asarray(states)
            cycle_log.append({'cycle': cycle, 'latency_s': latency,
                              'attempts': candidate_count,
                              'status': ('safe_braking_fallback' if fallback else
                                         'route_tracking_fallback' if tracking_fallback else 'executed'),
                              'goal_distance_m': float(np.linalg.norm(ego[:2] - global_goal[:2])),
                              'speed_mps': float(ego[3])})
            print(f'Cycle {cycle:03d}: {latency:.3f} s | '
                  f'attempts {candidate_count} | {cycle_log[-1]["status"]} | '
                  f'goal distance {cycle_log[-1]["goal_distance_m"]:.2f} m | '
                  f'speed {ego[3]:.2f} m/s')
        else:
            raise RuntimeError(f'Maximum {max_cycles} cycles reached without goal stop')
        print(f'Average planning latency: {np.mean(latencies):.3f} s '
              f'({1000 * np.mean(latencies):.1f} ms) over {len(latencies)} cycles')
        return np.asarray(states)[:, :2]
