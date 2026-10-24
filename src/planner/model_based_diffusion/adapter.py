import numpy as np
import jax.numpy as jnp
from commonroad_route_planner.route_planner import RoutePlanner
from src.planner.model_based_diffusion.model.mbd_planner import Args, run_diffusion
from src.planner.model_based_diffusion.envs.car2d import Car2d, car_dynamics, rk4

class ModelBasedDiffusionCommonRoadAdapter:
    def __init__(self, seed=0, num_samples=2048, horizon=80, diffusion_steps=100):
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

    def _extract_and_override_env(self, scenario, planning_problem):
        """Creates the MBD Car2d env and overrides its hardcoded variables with CommonRoad data."""
        env = Car2d()
        env.H = self.args.Hsample

        # 1. Start State [x, y, theta]
        initial_state = planning_problem.initial_state
        x0 = jnp.array([initial_state.position[0], initial_state.position[1], initial_state.orientation, initial_state.velocity])
        env.x0 = x0

        # 2. Goal State [x, y, theta]
        goal_state = planning_problem.goal.state_list[0]

        # Goal is a CommonRoad lanelet ShapeGroup in this scenario.
        # Select the far-end center of the route's terminal goal lanelet.
        goal_lanelets = planning_problem.goal.lanelets_of_goal_position
        lanelet_ids = sorted({lid for ids in (goal_lanelets or {}).values() for lid in ids})
        if lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_ids[0])
            centerline = np.asarray(lanelet.center_vertices)
            start = np.asarray(initial_state.position)
            # Choose the far end relative to the initial position.
            goal_pos = centerline[-1] if np.linalg.norm(centerline[-1]-start) > np.linalg.norm(centerline[0]-start) else centerline[0]
            tangent = (centerline[-1]-centerline[-2]) if np.all(goal_pos == centerline[-1]) else (centerline[0]-centerline[1])
            goal_yaw = float(np.arctan2(tangent[1], tangent[0]))
        else:
            pos_attr = goal_state.position
            goal_pos = np.asarray(pos_attr.center if hasattr(pos_attr, 'center') else pos_attr)
            goal_yaw = float(initial_state.orientation)
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
                t = int(initial_state.time_step) + k + 1

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

        # 4. Reference Trajectory (Routing)
        route_planner = RoutePlanner(scenario, planning_problem)
        candidate_routes = route_planner.plan_routes()
        if len(candidate_routes.retrieve_all_routes()) > 0:
            route = candidate_routes.retrieve_first_route()
            ref_path = route.reference_path

            # Trim the route at the chosen goal point, then resample by arc length.
            ref_path = np.asarray(ref_path, dtype=float)
            nearest = int(np.argmin(np.linalg.norm(ref_path - np.asarray(env.xg[:2]), axis=1)))
            ref_path = np.vstack([ref_path[:nearest + 1], np.asarray(env.xg[:2])])
            distances = np.linalg.norm(np.diff(ref_path, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(distances)])
            keep = np.concatenate([[True], np.diff(arc) > 1e-7])
            arc, ref_path = arc[keep], ref_path[keep]
            samples = np.linspace(0, arc[-1], self.args.Hsample)
            resampled_path = np.column_stack([np.interp(samples, arc, ref_path[:, k]) for k in range(2)])
            xref = np.zeros((self.args.Hsample, 4))
            xref[:, :2] = resampled_path
            diffs = np.gradient(resampled_path, axis=0)
            xref[:, 2] = np.arctan2(diffs[:, 1], diffs[:, 0])
            xref[-1, 2] = goal_yaw
            # Distance remaining to the destination
            remaining_distance = np.linalg.norm(
                resampled_path - np.asarray(env.xg[:2]),
                axis=1
            )

            # Maximum comfortable braking deceleration
            braking_acceleration = 3.0

            # Desired speed based on stopping distance
            reference_speed = np.sqrt(
                2.0 * braking_acceleration * remaining_distance
            )

            # Limit reference speed
            reference_speed = np.minimum(
                reference_speed,
                max(float(initial_state.velocity), 0.1)
            )

            reference_speed[-1] = 0.0

            xref[:, 3] = reference_speed

            env.xref = jnp.array(xref)
            env.thetaref = jnp.array(xref[:, 2])

            print("Reference start:", np.asarray(env.xref[0, :2]))
            print("Reference end:", np.asarray(env.xref[-1, :2]))
        else:
            # Fallback to straight line to goal if routing fails
            x_vals = np.linspace(x0[0], env.xg[0], self.args.Hsample)
            y_vals = np.linspace(x0[1], env.xg[1], self.args.Hsample)
            env.xref = jnp.column_stack((x_vals, y_vals, np.full(self.args.Hsample, goal_yaw)))
            env.thetaref = jnp.full(self.args.Hsample, goal_yaw)

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

        # Collect the complete rollout states, including yaw and speed.
        # Add this initialization before your existing rollout loop:
        # trajectory_states = np.zeros((self.args.Hsample, 4))
        #
        # Inside the loop, after each step:
        # trajectory_states[t] = np.asarray(state_obj.pipeline_state)

        safety_cost = float(
            env.dynamic_safety_cost(
                jnp.asarray(trajectory_states)
            )
        )

        print(f"Dynamic safety cost: {safety_cost:.6f}")

        if safety_cost > 1e-6:
            raise RuntimeError(
                "MBD trajectory violates the dynamic safety buffer."
            )

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

        trajectory = trajectory_states[:, :2]

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

        if final_position_error > 0.8:
            raise RuntimeError(
                "MBD failed to reach the target position."
            )

        if final_speed > 0.2:
            raise RuntimeError(
                "MBD failed to stop at the target."
            )

        return trajectory