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
        x0 = jnp.array([initial_state.position[0], initial_state.position[1], initial_state.orientation])
        env.x0 = x0

        # 2. Goal State [x, y, theta]
        goal_state = planning_problem.goal.state_list[0]

        # Safely extract target position
        if hasattr(goal_state, 'position'):
            pos_attr = goal_state.position
            if hasattr(pos_attr, 'center'):
                goal_pos = pos_attr.center
            elif hasattr(pos_attr, 'shapes') and len(pos_attr.shapes) > 0:
                goal_pos = pos_attr.shapes[0].center
            else:
                goal_pos = np.array(pos_attr)
        else:
            goal_pos = np.array([0.0, 0.0])

        # Safely extract target orientation (yaw) if specified, otherwise default to initial orientation
        if hasattr(goal_state, 'orientation'):
            orient_attr = goal_state.orientation
            if hasattr(orient_attr, 'start'):
                goal_yaw = float(orient_attr.start)
            elif hasattr(orient_attr, 'angle'):
                goal_yaw = float(orient_attr.angle)
            else:
                goal_yaw = float(orient_attr)
        else:
            goal_yaw = float(initial_state.orientation) if hasattr(initial_state, 'orientation') else 0.0

        env.xg = jnp.array([goal_pos[0], goal_pos[1], goal_yaw])

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

        # 4. Reference Trajectory (Routing)
        route_planner = RoutePlanner(scenario, planning_problem)
        candidate_routes = route_planner.plan_routes()
        if len(candidate_routes.retrieve_all_routes()) > 0:
            route = candidate_routes.retrieve_first_route()
            ref_path = route.reference_path

            # Resample CR reference path to match EXACTLY self.args.Hsample points
            indices = np.linspace(0, len(ref_path) - 1, self.args.Hsample).astype(int)
            resampled_path = ref_path[indices]

            # Reconstruct [x, y, theta] for xref
            xref = np.zeros((self.args.Hsample, 3))
            xref[:, :2] = resampled_path

            # Calculate heading (theta) based on path differences
            diffs = np.diff(resampled_path, axis=0)
            theta = np.arctan2(diffs[:, 1], diffs[:, 0])
            xref[:-1, 2] = theta
            xref[-1, 2] = theta[-1] if len(theta) > 0 else 0.0  # duplicate last heading

            env.xref = jnp.array(xref)
            env.thetaref = jnp.array(xref[:, 2])
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

        # 2. Run diffusion directly in memory (returns final reward and action tensor Yi)
        _, Yi = run_diffusion(self.args, env=env)

        # Grab the fully denoised action sequence from the last diffusion step: shape (Hsample, 2)
        best_action_sequence = Yi[-1]

        # 3. Rollout actions using the car's RK4 kinematics to get (X, Y) trajectory
        trajectory = np.zeros((self.args.Hsample, 2))
        current_state = env.x0

        step_env_jit = jax.jit(env.step)
        state_obj = State(
            pipeline_state=current_state,
            obs=current_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0)
        )

        for t in range(self.args.Hsample):
            state_obj = step_env_jit(state_obj, best_action_sequence[t])
            pos = state_obj.pipeline_state
            trajectory[t, 0] = float(pos[0])
            trajectory[t, 1] = float(pos[1])

        return trajectory