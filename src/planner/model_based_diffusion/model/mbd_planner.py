import functools
import os
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt

import src.planner.model_based_diffusion as mbd

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False


def run_diffusion(args: Args, env=None):

    rng = jax.random.PRNGKey(seed=args.seed)

    ## setup env

    # recommended temperature for envs
    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
        "pushT": 0.2,
    }
    Ndiffuse_recommend = {
        "pushT": 200,
        "humanoidrun": 300,
    }
    Nsample_recommend = {
        "humanoidrun": 8192,
    }
    Hsample_recommend = {
        "pushT": 40,
    }
    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env_name, args.temp_sample)
        args.Ndiffuse = Ndiffuse_recommend.get(args.env_name, args.Ndiffuse)
        args.Nsample = Nsample_recommend.get(args.env_name, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env_name, args.Hsample)
        print(f"override temp_sample to {args.temp_sample}")
    if env is None:
        env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    # eval_us = jax.jit(functools.partial(mbd.utils.eval_us, step_env_jit))
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_us, step_env_jit))
    rollout_us_collision = jax.jit(functools.partial(mbd.utils.rollout_us_with_collision,step_env_jit))

    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset)

    ## run diffusion

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # esitimate mu_0tm1
        rewss, qs, collisions = jax.vmap(rollout_us_collision, in_axes=(None, 0))(state_init, Y0s)

        # Dynamic obstacle safety cost for every candidate trajectory
        dynamic_cost = jax.vmap(
            env.dynamic_safety_cost
        )(qs)

        # Terminal state
        terminal_state = qs[:, -1, :]

        # Position error
        terminal_distance = jnp.linalg.norm(
            terminal_state[:, :2] - env.xg[:2],
            axis=-1
        )

        # Heading error
        heading_error = jnp.arctan2(
            jnp.sin(terminal_state[:, 2] - env.xg[2]),
            jnp.cos(terminal_state[:, 2] - env.xg[2])
        )

        # Actual vehicle speed, not the control command
        terminal_speed = terminal_state[:, 3]

        # Smooth control sequence
        control_variation = jnp.square(
            jnp.diff(Y0s, axis=1)
        ).mean(axis=(1, 2))

        # Goal position
        position_cost = 8.0 * terminal_distance

        # Stopping objective
        stop_cost = 30.0 * terminal_speed ** 2

        # Goal orientation
        heading_cost = 1.0 * heading_error ** 2

        # Smooth controls
        smoothness_cost = 0.5 * control_variation

        # Collision penalty
        collision_cost = 1000.0 * jnp.any(
            collisions > 0, axis=1
        ).astype(jnp.float32)

        # Route guidance
        if args.enable_demo:
            route_cost = -jax.vmap(
                env.eval_xref_logpd
            )(qs)
        else:
            route_cost = jnp.zeros_like(terminal_distance)

        # Single consistent reward
        rews = (
                0.2 * rewss.mean(axis=-1)
                - position_cost
                - stop_cost
                - heading_cost
                - smoothness_cost
                - collision_cost
                - 50.0 * dynamic_cost
        )
        rew_std = rews.std()
        # Spatial centerline guidance
        if args.enable_demo:
            centerline_scores = jax.vmap(
                env.eval_xref_logpd
            )(qs)
        else:
            centerline_scores = jnp.zeros_like(rews)

        # Combine lane-centering with the existing objective
        rews = rews + 2.0 * centerline_scores

        # Normalize the combined reward
        rew_mean = jnp.mean(rews)
        rew_std = jnp.maximum(jnp.std(rews), 1e-4)

        logp0 = (
                (rews - rew_mean)
                / rew_std
                / args.temp_sample
        )

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), rews.mean()

    # run reverse
    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), rew = reverse_once(carry_once, None)
                Ybars.append(Yi)
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars)

    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp)
    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", Yi)
        if args.env_name == "car2d":
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            # rollout
            xs = jnp.array([state_init.pipeline_state])
            state = state_init
            for t in range(Yi.shape[1]):
                state = step_env_jit(state, Yi[-1, t])
                xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            env.render(ax, xs)
            if args.enable_demo:
                ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", label="RRT path")
            ax.legend()
            plt.savefig(f"{path}/rollout.png")
        else:
            render_us = functools.partial(
                mbd.utils.render_us,
                step_env_jit,
                env.sys.tree_replace({"opt.timestep": env.dt}),
            )
            webpage = render_us(state_init, Yi[-1])
            with open(f"{path}/rollout.html", "w") as f:
                f.write(webpage)
    rewss_final, _ = rollout_us(state_init, Yi[-1])
    rew_final = rewss_final.mean()

    return rew_final, Yi


if __name__ == "__main__":
    rew_final = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")
