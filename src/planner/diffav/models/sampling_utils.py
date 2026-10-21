"""Variance-reduction utilities for diffusion timestep sampling.

Diffusion training objectives are expectations over a uniformly drawn
timestep; estimating them with a handful of i.i.d. uniform draws leaves
substantial Monte Carlo variance in the loss and its gradients. The
stateless stratified sampler here is shared by every K-draw estimator in
the codebase: the DPO log-prob estimator spreads its K Monte Carlo draws
across the schedule, and the batched physics-informed trainer spreads the
per-scene draws of a B-scene batch the same way, so one optimizer step
always covers early, middle, and late noise levels instead of clustering
wherever independent uniform draws happen to land.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def stratified_timestep(
    t_key: jax.Array,
    sample_index: jax.Array | int,
    num_samples: int,
    num_timesteps: int,
) -> jax.Array:
    """Draw the ``sample_index``-th timestep of a K-draw stratified estimator.

    Partitions the unit interval into ``num_samples`` equal strata, draws
    uniformly within stratum ``sample_index``, and maps onto integer
    timesteps. Each draw is marginally uniform over its stratum and the K
    strata tile ``[0, num_timesteps)`` exactly, so the K-draw average stays
    an unbiased estimate of the uniform-timestep expectation while its
    variance can only decrease relative to K i.i.d. uniform draws
    (stratification). This is the stateless member of the timestep
    variance-reduction family: Nichol & Dhariwal (2021, section 3.3) reduce
    diffusion-loss gradient noise by replacing uniform timestep draws with
    importance sampling, and Kingma et al. (2021, appendix I.1) report a
    significant variance reduction from the equivalent low-discrepancy
    (stratified) sampler for continuous t. The stateful loss-aware
    alternative (``LossSecondMomentResampler`` in improved-diffusion's
    ``resample.py``) keeps a per-timestep loss history and is jit-hostile,
    so the stateless stratified draw is preferred here.

    Args:
        t_key: JAX random key for the within-stratum offset.
        sample_index: Which stratum to draw from, in ``[0, num_samples)``.
        num_samples: Number of Monte Carlo samples (K), i.e. strata count.
        num_timesteps: Total diffusion timesteps (T).

    Returns:
        Scalar ``int32`` timestep in ``[0, num_timesteps)``.
    """
    offset = jax.random.uniform(t_key)
    position = (sample_index + offset) / num_samples
    timestep = jnp.floor(position * num_timesteps).astype(jnp.int32)
    # Guard the open upper bound against float rounding at stratum edges.
    return jnp.minimum(timestep, num_timesteps - 1)
