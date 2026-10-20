import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

import src.planner.model_based_diffusion as mbd


def car_dynamics(x, u):
    speed = u[1] * 15.0  # Scale speed for scenario distances
    return jnp.array([
        speed * jnp.cos(x[2]),  # x_dot = v * cos(theta)
        speed * jnp.sin(x[2]),  # y_dot = v * sin(theta)
        u[0] * jnp.pi / 2.0     # theta_dot (steering rate)
    ])


def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def check_collision(x, obs_center, obs_radius):
    dist2objs = jnp.linalg.norm(x[:2] - obs_center, axis=1)
    return jnp.any(dist2objs < obs_radius)


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
        self.x0 = jnp.array([-0.5, 0.0, jnp.pi*3/2])
        self.xg = jnp.array([0.5, 0.0, 0.0])
        self.xref = jnp.load(f"{mbd.__path__[0]}/assets/car2d_xref.npy")
        # self.xref = jnp.load(f"{mbd.__path__[0]}/../figure/car2d_xref.npy")
        xref_diff = jnp.diff(self.xref, axis=0)
        theta = jnp.arctan2(xref_diff[:, 0], xref_diff[:, 1])
        self.thetaref = jnp.append(theta, theta[-1])
        self.rew_xref = jax.vmap(self.get_reward)(self.xref).mean()

    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.clip(action, -1.0, 1.0)
        q = state.pipeline_state
        q_new = rk4(car_dynamics, state.pipeline_state, action, self.dt)
        collide = check_collision(q_new, self.obs_center, self.obs_radius)
        q = jnp.where(collide, q, q_new)
        reward = self.get_reward(q)
        return state.replace(pipeline_state=q, obs=q, reward=reward, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        # Scale goal distance metric for CommonRoad (e.g. max dist ~50m)
        dist = jnp.linalg.norm(q[:2] - self.xg[:2])
        reward = 1.0 - (jnp.clip(dist, 0.0, 50.0) / 50.0) ** 2
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, xs):
        xs_err = xs[:, :2] - self.xref[:, :2]
        # Rescale error clip to 10.0m for real-world road widths
        logpd = -((jnp.clip(jnp.linalg.norm(xs_err, axis=-1), 0.0, 10.0) / 10.0) ** 2).mean(axis=-1)
        return logpd

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 3

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
