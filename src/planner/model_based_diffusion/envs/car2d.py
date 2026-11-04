import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

import src.planner.model_based_diffusion as mbd


def car_dynamics(x, u):
    """
    State:
        x = [px, py, yaw, velocity]

    Control:
        u = [normalized_steering, normalized_acceleration]
    """
    px, py, yaw, v = x

    wheelbase = 2.7
    max_steer = 0.5
    max_accel = 2.5
    max_brake = 5.0

    steering = jnp.clip(u[0], -1.0, 1.0) * max_steer
    accel_cmd = jnp.clip(u[1], -1.0, 1.0)

    acceleration = jnp.where(
        accel_cmd >= 0.0,
        accel_cmd * max_accel,
        accel_cmd * max_brake
    )

    return jnp.array([
        v * jnp.cos(yaw),
        v * jnp.sin(yaw),
        v * jnp.tan(steering) / wheelbase,
        acceleration
    ])


def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)

    x_new = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    return x_new.at[3].set(
        jnp.maximum(x_new[3], 0.0)
    )


def check_collision(x, obs_center, obs_radius):
    vehicle_radius = 1.5
    safety_margin = 0.5

    distances = jnp.linalg.norm(
        x[:2] - obs_center,
        axis=1
    )

    collision_distances = (
            obs_radius + vehicle_radius + safety_margin
    )

    return jnp.any(distances < collision_distances)


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class Car2d:
    def __init__(self):
        self.dt = 0.1
        self.H = 50
        self.goal_tolerance = 0.8
        self.target_speed = 0.0
        r_obs = 0.3
        self.obs_center = jnp.array(
            [
                [-r_obs * 3, r_obs * 2],
                [-r_obs * 2, r_obs * 2],
                [-r_obs * 1, r_obs * 2],
                [0.0, r_obs * 2],
                [0.0, r_obs * 1],
                [0.0, 0.0],
                [0.0, -r_obs * 1],
                [-r_obs * 3, -r_obs * 2],
                [-r_obs * 2, -r_obs * 2],
                [-r_obs * 1, -r_obs * 2],
                [0.0, -r_obs * 2],
            ]
        )
        self.obs_radius = r_obs  # Radius of the obstacle
        self.x0 = jnp.array([-0.5, 0.0, jnp.pi * 3 / 2, 0.0])
        self.xg = jnp.array([0.5, 0.0, 0.0, 0.0])
        self.xref = jnp.load(f"{mbd.__path__[0]}/assets/car2d_xref.npy")
        # self.xref = jnp.load(f"{mbd.__path__[0]}/../figure/car2d_xref.npy")
        xref_diff = jnp.diff(self.xref, axis=0)
        theta = jnp.arctan2(xref_diff[:, 0], xref_diff[:, 1])
        self.thetaref = jnp.append(theta, theta[-1])
        self.rew_xref = jax.vmap(self.get_reward)(self.xref).mean()
        self.dynamic_obstacles = jnp.zeros((self.H, 1, 3), dtype=jnp.float32)
        self.dynamic_valid = jnp.zeros((self.H, 1), dtype=bool)
        self.route_curvature = jnp.zeros((max(self.H - 1, 1),))
        self.route_half_width = jnp.full((max(self.H - 1, 1),), 2.0)

    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)

        q = state.pipeline_state
        q_new = rk4(car_dynamics, q, action, self.dt)

        collide = check_collision(
            q_new,
            self.obs_center,
            self.obs_radius
        )

        reward = self.get_reward(q_new)

        return state.replace(
            pipeline_state=q_new,
            obs=q_new,
            reward=reward,
            done=collide.astype(jnp.float32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        position_error = jnp.linalg.norm(
            q[:2] - self.xg[:2]
        )

        speed = q[3]

        return (
                -0.1 * position_error
                -0.02 * speed ** 2
        )

    @partial(jax.jit, static_argnums=(0,))
    def route_metrics(self, xs):
        """Project positions to ordered centerline segments: lateral error, progress."""
        p = xs[:, :2]
        a = self.xref[:-1, :2]
        b = self.xref[1:, :2]
        v = b - a
        lengths2 = jnp.maximum(jnp.sum(v * v, axis=1), 1e-8)
        fraction = jnp.clip(jnp.sum((p[:, None, :] - a[None, :, :]) * v[None, :, :], axis=-1) / lengths2[None, :], 0., 1.)
        projected = a[None, :, :] + fraction[..., None] * v[None, :, :]
        squared = jnp.sum((p[:, None, :] - projected) ** 2, axis=-1)
        nearest = jnp.argmin(squared, axis=1)
        lateral_sq = jnp.min(squared, axis=1)
        lengths = jnp.sqrt(lengths2)
        arc = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(lengths)])
        progress = jnp.take_along_axis(
            arc[:-1][None, :] + fraction * lengths[None, :],
            nearest[:, None], axis=1
        )[:, 0]
        return lateral_sq, progress, arc[-1]

    @partial(jax.jit, static_argnums=(0,))
    def curvature_reference_costs(self, xs):
        """Spatially indexed speed, heading, and road-tube costs.

        Uses nearest route segments. Unlike lookahead steering, this does not
        introduce an additional direction target that can conflict at turns.
        """
        p = xs[:, :2]
        a = self.xref[:-1, :2]
        b = self.xref[1:, :2]
        seg = b - a
        denom = jnp.maximum(jnp.sum(seg * seg, axis=1), 1e-8)
        frac = jnp.clip(
            jnp.sum((p[:, None, :] - a[None, :, :]) * seg[None, :, :], axis=-1)
            / denom[None, :], 0.0, 1.0
        )
        proj = a[None, :, :] + frac[..., None] * seg[None, :, :]
        d2 = jnp.sum((p[:, None, :] - proj) ** 2, axis=-1)
        idx = jnp.argmin(d2, axis=1)
        tangent = jnp.arctan2(seg[:, 1], seg[:, 0])
        # Avoid forcing a heading when the car is fully stopped.
        moving = jnp.clip(xs[:, 3] / 1.5, 0.0, 1.0)
        heading = jnp.mean(moving * (1.0 - jnp.cos(xs[:, 2] - tangent[idx])))
        desired_speed = self.xref[:-1, 3][idx]
        # Only penalize exceeding the curvature/braking envelope.
        speed = jnp.mean(jnp.maximum(xs[:, 3] - desired_speed, 0.0) ** 2)
        available = jnp.maximum(self.route_half_width[idx] - 1.0 - 0.15, 0.0)
        tube = jnp.mean(jnp.maximum(jnp.sqrt(jnp.min(d2, axis=1) + 1e-8) - available, 0.0) ** 2)
        return heading, speed, tube

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, xs):
        lateral_sq, _, _ = self.route_metrics(xs)
        return -jnp.mean(lateral_sq)

    @partial(jax.jit, static_argnums=(0,))
    def dynamic_safety_cost(self, xs):
        """
        xs: (H, 4), one planned vehicle state per timestep.

        Returns a scalar cost for proximity to dynamic obstacles.
        """
        ego_xy = xs[:, None, :2]            # (H, 1, 2)
        obs_xy = self.dynamic_obstacles[:, :, :2]
        obs_radius = self.dynamic_obstacles[:, :, 2]

        # Conservative circular approximation of ego footprint
        ego_radius = 0.5 * jnp.hypot(4.5, 2.0)

        # Extra safety buffer
        safety_margin = 1.0

        distances = jnp.linalg.norm(
            ego_xy - obs_xy,
            axis=-1
        )

        required_distance = (
                ego_radius + obs_radius + safety_margin
        )

        violation = jnp.maximum(
            required_distance - distances,
            0.0
        )

        violation = jnp.where(
            self.dynamic_valid,
            violation,
            0.0
        )

        # Sum over obstacles and average over timesteps
        return jnp.sum(violation ** 2) / xs.shape[0]

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 4

    def render(self, ax, xs: jnp.ndarray):
        # obstacles
        for i in range(self.obs_center.shape[0]):
            circle = plt.Circle(
                self.obs_center[i, :], self.obs_radius, color="k", fill=True, alpha=0.5
            )
            ax.add_artist(circle)
        # ax.quiver(
        #     xs[:, 0],
        #     xs[:, 1],
        #     jnp.sin(xs[:, 2]),
        #     jnp.cos(xs[:, 2]),
        #     range(self.H + 1),
        #     cmap="Reds",
        # )
        ax.scatter(xs[:, 0], xs[:, 1], c=range(self.H + 1), cmap="Reds")
        ax.plot(xs[:, 0], xs[:, 1], "r-", label="Car path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True)
        # ax.set_title("Car 2D")
