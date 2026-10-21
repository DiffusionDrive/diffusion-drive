"""Trajectory diffusion model for multi-agent future prediction.

Composes artifex's NoiseSchedule with a factorized scene backbone
to implement denoising diffusion for trajectory generation. The model
generates multi-agent future trajectories conditioned on scene context
embeddings from the tokenizer.

Example:
    Create a model, compute a training loss, and sample trajectories::

        import jax
        import jax.numpy as jnp
        from flax import nnx

        from diffav.models.trajectory_diffusion import (
            TrajectoryDiffusionConfig,
            TrajectoryDiffusionModel,
        )

        config = TrajectoryDiffusionConfig(
            hidden_dim=128,
            num_blocks=2,
            num_heads=4,
            num_agents_max=32,
            future_steps=80,
            context_dim=64,
            num_timesteps=1000,
            noise_schedule_type="cosine",
        )
        model = TrajectoryDiffusionModel(
            config, rngs=nnx.Rngs(params=jax.random.key(0))
        )

        # Training: compute diffusion loss on ground-truth trajectories
        trajectories = jnp.zeros((32, 80, 4))   # [agents, steps, state]
        scene_ctx = jnp.zeros((32, 64))          # one context row per agent
        loss = model.compute_loss(
            trajectories, scene_ctx, key=jax.random.key(1)
        )

        # Inference: generate trajectories via reverse diffusion
        prediction = model.sample(scene_ctx, key=jax.random.key(2))
        assert prediction.trajectories.shape == (32, 80, 4)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from artifex.generative_models.core.configuration import NoiseScheduleConfig
from artifex.generative_models.core.noise_schedule import (
    create_noise_schedule,
    NoiseSchedule,
)
from flax import nnx

from diffav.core.config import (
    factorized_backbone_config_kwargs,
    FactorizedBackboneHyperparams,
    validate_factorized_backbone_config,
)
from diffav.core.constants import WOD_FUTURE_STEPS
from diffav.core.types import (
    LossType,
    NoiseScheduleType,
    PredictionType,
    TrajectoryPrediction,
)
from diffav.models.factorized_backbone import (
    FactorizedSceneBackbone,
    FactorizedSceneBackboneConfig,
)
from diffav.models.sampling_utils import stratified_timestep


@dataclass(frozen=True, slots=True, kw_only=True)
class GuidanceSpec:
    """Test-time x̂₀ guidance configuration for reverse-diffusion sampling.

    Bundles the three parameters that always travel together through the
    sampling call chain (``sample`` → ``p_sample_step`` → the map-conditioned
    wrapper), so they forward as one object rather than three parallel kwargs.

    Attributes:
        reward_fn: Differentiable scalar reward of the clean estimate x̂₀. The
            sampler gradient-ascends it, composed with the state denormalization
            so the chain rule stays exact in diffusion space.
        scale: Guidance step size η. Zero disables guidance.
        grad_clip: When set, the reward gradient's global L2 norm is clipped to
            this value before each guidance step — the CTG / MotionDiffuser
            stability guard that stops a large reward gradient from driving the
            sample off-distribution. ``None`` disables clipping.
    """

    reward_fn: Callable[[jax.Array], jax.Array]
    scale: float = 0.0
    grad_clip: float | None = None


# TODO(artifex-backbone-registry): This module composes artifex's
# NoiseSchedule with the custom FactorizedSceneBackbone instead of
# wrapping DDPMModel directly.
#
# Root cause: DDPMModel.__init__ calls create_backbone(config.backbone)
# which dispatches on a closed BackboneTypeLiteral ('unet', 'dit',
# 'uvit', 'unet2d_condition', 'unet_1d'). Custom backbones like
# FactorizedSceneBackbone cannot be registered without modifying artifex.
#
# Required artifex changes to enable DDPMModel wrapping:
#   1. Open backbone registry -- replace closed create_backbone()
#      factory with register_backbone(name, cls) + registry dict
#   2. Extensible BackboneConfig -- change backbone_type from
#      Literal[...] to open str for downstream subclassing
#   3. Backbone injection -- allow DiffusionModel/DDPMModel.__init__
#      to accept a pre-built backbone (backbone: nnx.Module | None)
#      instead of always constructing via create_backbone()
#
# Once resolved: replace q_sample/p_sample_step/sample/compute_loss
# with DDPMModel inheritance, gaining DDIM sampling, EMA, and
# distillation for free.


def _validate_state_normalization(
    offsets: tuple[float, ...] | None,
    scales: tuple[float, ...] | None,
    state_dim: int,
) -> None:
    """Validate the paired state normalization coefficients.

    Args:
        offsets: Per-dimension offsets, or ``None``.
        scales: Per-dimension positive scales, or ``None``.
        state_dim: Expected coefficient length.

    Raises:
        ValueError: If only one of the pair is set, lengths mismatch
            ``state_dim``, or any scale is non-positive.
    """
    if (offsets is None) != (scales is None):
        msg = "state_offsets and state_scales must be set together"
        raise ValueError(msg)
    if scales is None:
        return
    if len(scales) != state_dim or len(offsets or ()) != state_dim:
        msg = f"state_scales/state_offsets must have length state_dim={state_dim}"
        raise ValueError(msg)
    if any(scale <= 0.0 for scale in scales):
        msg = f"state_scales must be positive, got {scales}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionLossOutputs:
    """Structured outputs of one diffusion-loss evaluation.

    Attributes:
        loss: Scalar diffusion training loss (Min-SNR-weighted when the
            model config sets ``snr_gamma``).
        prediction: Predicted clean trajectories x̂₀ in raw data units,
            shaped like the input trajectories; carries model gradients.
        alpha_bar_t: Schedule value ᾱ_t of the drawn diffusion timestep —
            the model's trust in x̂₀ at that noise level. Downstream
            penalties on the reconstruction (e.g. the trainer's physics
            term) can scale by it, mirroring the reconstruction-guidance
            anneal the sampling path applies to x̂₀ guidance.
    """

    loss: jax.Array
    prediction: jax.Array
    alpha_bar_t: jax.Array


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryDiffusionConfig(FactorizedBackboneHyperparams):
    """Configuration for the trajectory diffusion model.

    Attributes:
        model_type: Generative model family (must be ``"diffusion"``).
        hidden_dim: Transformer hidden dimension.
        num_blocks: Number of factorized backbone blocks.
        num_temporal_layers: Temporal attention sublayers per block.
        num_social_layers: Social attention sublayers per block.
        use_social_interaction: Whether agents attend across one another.
        num_heads: Number of attention heads.
        future_steps: Prediction horizon in timesteps.
        noise_schedule_type: Noise schedule variant.
        num_timesteps: Number of diffusion timesteps.
        beta_start: Starting noise schedule value.
        beta_end: Ending noise schedule value.
        loss_type: Training loss (``"mse"`` or ``"l1"``).
        num_agents_max: Maximum number of agents.
        state_dim: Per-timestep state dimension.
        context_dim: Scene context embedding dimension.
        mlp_ratio: Feed-forward hidden dimension multiplier.
        dropout_rate: Dropout probability.
        x0_clip_bound: Optional bound for clamping the x̂₀ estimate during
            reverse diffusion. The model consumes trajectories at whatever
            scale the data pipeline emits — WOD scenarios are ego-centred
            metres (``AgentNormalizationOperator`` only re-centres, so 8 s
            horizons span hundreds of metres). Leave ``None`` (default) for
            such data; set it to the data's maximum absolute coordinate
            (e.g. 1.0 for unit-normalised inputs) to gain the standard DDPM
            stabilisation clamp.
        state_offsets: Optional per-dimension offsets for internal state
            normalization: diffusion runs on ``(x + offsets) / scales`` and
            every model boundary (loss inputs, predictions, samples,
            guidance rewards) stays in raw data units. Mirrors the
            add/div-coefficient scaling used by reference trajectory
            diffusers. Must be set together with ``state_scales``.
        state_scales: Optional per-dimension positive scales (see
            ``state_offsets``). With unit normalization, pair with
            ``x0_clip_bound=1.0`` for bounded, data-scale samples.
        snr_gamma: Optional Min-SNR-γ loss-weighting clamp (Hang et al.
            2023, arXiv 2303.09556). When set, the ε-prediction loss at
            the drawn timestep is scaled by ``min(SNR_t, γ) / SNR_t``
            with ``SNR_t = ᾱ_t / (1 - ᾱ_t)`` — the reference trainers'
            ε-parameterization form. This down-weights the easy
            high-SNR (low-noise) timesteps so optimization concentrates
            on the informative noise levels; the reference
            recommendation is ``5.0``. ``None`` (default) keeps the
            unweighted loss bit-identical to the prior behavior.
    """

    model_type: str = "diffusion"
    future_steps: int = WOD_FUTURE_STEPS
    noise_schedule_type: NoiseScheduleType = NoiseScheduleType.COSINE
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    loss_type: LossType = LossType.MSE
    prediction_type: PredictionType = PredictionType.EPSILON
    num_agents_max: int = 32
    state_dim: int = 4
    context_dim: int = 64
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    x0_clip_bound: float | None = None
    state_offsets: tuple[float, ...] | None = None
    state_scales: tuple[float, ...] | None = None
    snr_gamma: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.model_type != "diffusion":
            msg = f"model_type must be 'diffusion', got {self.model_type!r}"
            raise ValueError(msg)
        if self.x0_clip_bound is not None and self.x0_clip_bound <= 0.0:
            msg = f"x0_clip_bound must be positive when set, got {self.x0_clip_bound}"
            raise ValueError(msg)
        if self.snr_gamma is not None and self.snr_gamma <= 0.0:
            msg = f"snr_gamma must be positive when set, got {self.snr_gamma}"
            raise ValueError(msg)
        _validate_state_normalization(self.state_offsets, self.state_scales, self.state_dim)
        validate_factorized_backbone_config(
            self,
            extra_positive={
                "num_timesteps": self.num_timesteps,
            },
        )
        # StrEnum fields (LossType, NoiseScheduleType, PredictionType) validate
        # on construction; explicit checks kept for string inputs (e.g. YAML).
        LossType(self.loss_type)
        NoiseScheduleType(self.noise_schedule_type)
        PredictionType(self.prediction_type)


class TrajectoryDiffusionModel(nnx.Module):
    """Diffusion model for multi-agent trajectory generation.

    Composes a factorized scene backbone with artifex's NoiseSchedule
    to implement forward diffusion (noising) and reverse diffusion
    (denoising) for trajectory prediction conditioned on scene context.
    """

    def __init__(
        self,
        config: TrajectoryDiffusionConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize trajectory diffusion model.

        Args:
            config: Model configuration.
            rngs: Flax NNX random number generators.
        """
        self.config = config

        # Build the factorized temporal+social backbone
        backbone_config = FactorizedSceneBackboneConfig(
            **factorized_backbone_config_kwargs(config),
            dropout_rate=config.dropout_rate,
        )
        self.backbone = FactorizedSceneBackbone(backbone_config, rngs=rngs)

        # Build noise schedule from artifex
        ns_config = NoiseScheduleConfig(
            name="traj_noise",
            schedule_type=config.noise_schedule_type,
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        self.noise_schedule: NoiseSchedule = create_noise_schedule(ns_config)

    def q_sample(
        self,
        x_start: jax.Array,
        t: int | jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        """Forward diffusion: add noise to clean trajectories.

        Args:
            x_start: Clean trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            t: Diffusion timestep (scalar integer).
            noise: Noise to add, same shape as x_start.

        Returns:
            Noisy trajectories at timestep t.
        """
        t_batch = jnp.atleast_1d(jnp.asarray(t))
        return self.noise_schedule.q_sample(x_start, t_batch, noise)

    def predict_noise(
        self,
        noisy_traj: jax.Array,
        t: int | jax.Array,
        scene_context: jax.Array,
        *,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Predict noise in noisy trajectories.

        Args:
            noisy_traj: Noisy trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            t: Diffusion timestep (scalar).
            scene_context: Per-agent context (adaLN anchor),
                shape ``(num_agents, context_dim)``.
            scene_tokens: Fused scene tokens ``(num_tokens, scene_token_dim)``
                for the backbone map cross-attention; required when the config
                enables ``use_map_cross_attention`` and ``None`` otherwise.
            scene_token_valid: Optional ``(num_tokens,)`` boolean token mask.
            deterministic: If True, disable dropout.

        Returns:
            Predicted noise, same shape as noisy_traj.
        """
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        return self.backbone(
            noisy_traj,
            t_arr,
            scene_context,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
            deterministic=deterministic,
        )

    def _normalize_states(self, states: jax.Array) -> jax.Array:
        """Map raw-unit states into the model's diffusion space."""
        if self.config.state_scales is None:
            return states
        offsets = jnp.asarray(self.config.state_offsets)
        scales = jnp.asarray(self.config.state_scales)
        return (states + offsets) / scales

    def _denormalize_states(self, states: jax.Array) -> jax.Array:
        """Map diffusion-space states back to raw data units."""
        if self.config.state_scales is None:
            return states
        offsets = jnp.asarray(self.config.state_offsets)
        scales = jnp.asarray(self.config.state_scales)
        return states * scales - offsets

    def _prediction_target(self, x_0: jax.Array, noise: jax.Array, t: int | jax.Array) -> jax.Array:
        """The backbone's regression target for the configured parameterization.

        ``x_0`` and ``noise`` are in normalized diffusion space. Returns the
        injected noise (``EPSILON``), the clean signal (``X0``), or the velocity
        ``v = sqrt(abar)*noise - sqrt(1-abar)*x_0`` (``V``; Salimans & Ho).
        """
        if self.config.prediction_type == PredictionType.EPSILON:
            return noise
        if self.config.prediction_type == PredictionType.X0:
            return x_0
        alpha_bar = self.noise_schedule.alphas_cumprod[t]
        return jnp.sqrt(alpha_bar) * noise - jnp.sqrt(1.0 - alpha_bar) * x_0

    def _predict_x0_from_output(
        self, x_t: jax.Array, t: int | jax.Array, model_output: jax.Array
    ) -> jax.Array:
        """Reconstruct the clean estimate x-hat-0 from the backbone output.

        Dispatches on the parameterization so the sampling and loss paths share
        one x-hat-0 definition: invert the noise (``EPSILON``), pass the output
        through (``X0``), or ``sqrt(abar)*x_t - sqrt(1-abar)*output`` (``V``).
        """
        if self.config.prediction_type == PredictionType.EPSILON:
            return self.noise_schedule.predict_start_from_noise(
                x_t, jnp.atleast_1d(t), model_output
            )
        if self.config.prediction_type == PredictionType.X0:
            return model_output
        alpha_bar = self.noise_schedule.alphas_cumprod[t]
        return jnp.sqrt(alpha_bar) * x_t - jnp.sqrt(1.0 - alpha_bar) * model_output

    def p_sample_step(
        self,
        x_t: jax.Array,
        t: int | jax.Array,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        guidance: GuidanceSpec | None = None,
        temperature: float = 1.0,
    ) -> jax.Array:
        """Single reverse diffusion step: denoise x_t to x_{t-1}.

        Args:
            x_t: Noisy trajectories at timestep t.
            t: Current diffusion timestep.
            scene_context: Per-agent context (adaLN anchor).
            key: JAX random key for sampling.
            scene_tokens: Optional fused scene tokens for map cross-attention
                (see :meth:`predict_noise`).
            scene_token_valid: Optional per-token validity mask.
            guidance: Optional test-time x̂₀ guidance (:class:`GuidanceSpec`).
                When set with a non-zero ``scale``, the clean estimate takes a
                reward-gradient ascent step ``x̂₀ + η·ᾱ_t·∇R(x̂₀)`` before the
                posterior is formed — strong guidance only once x̂₀ is
                trustworthy. Non-finite reward gradients are skipped, and the
                gradient norm is clipped when ``grad_clip`` is set.
            temperature: Multiplier τ on the posterior noise (τ=1 standard
                ancestral sampling, 0≤τ<1 sharpens the conditional, τ=0 noise-free).

        Returns:
            Denoised trajectories at timestep t-1.
        """
        t_batch = jnp.atleast_1d(jnp.asarray(t))

        # Backbone output (epsilon, x0, or v per the parameterization).
        model_output = self.predict_noise(
            x_t,
            t,
            scene_context,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
        )

        # Estimate x_0 from the output; clamp only when the config declares a
        # data bound (WOD metres are unbounded — see config docs).
        x_0_pred = self._predict_x0_from_output(x_t, t, model_output)
        if self.config.x0_clip_bound is not None:
            bound = self.config.x0_clip_bound
            x_0_pred = jnp.clip(x_0_pred, -bound, bound)

        if guidance is not None and guidance.scale != 0.0:
            # The reward is a function of raw-unit states; composing with the
            # denormalization keeps the chain rule exact in diffusion space.
            grad_fn = jax.grad(lambda x0: guidance.reward_fn(self._denormalize_states(x0)))
            reward_grad: jax.Array = grad_fn(x_0_pred)
            reward_grad = jnp.where(
                jnp.all(jnp.isfinite(reward_grad)),
                reward_grad,
                jnp.zeros_like(reward_grad),
            )
            if guidance.grad_clip is not None:
                # Clip the gradient's global L2 norm so a large reward gradient
                # cannot push x̂₀ off-distribution in a single step; the
                # direction is preserved, only an over-large magnitude is capped.
                grad_norm = jnp.sqrt(jnp.sum(reward_grad**2) + 1e-12)
                reward_grad = reward_grad * jnp.minimum(1.0, guidance.grad_clip / grad_norm)
            annealing = self.noise_schedule.alphas_cumprod[jnp.asarray(t)]
            x_0_pred = x_0_pred + guidance.scale * annealing * reward_grad

        # Compute posterior q(x_{t-1} | x_t, x_0)
        mean, _, log_var = self.noise_schedule.q_posterior_mean_variance(x_0_pred, x_t, t_batch)

        # Sample (no noise at t=0). ``temperature`` scales the posterior noise:
        # tau < 1 sharpens the conditional (tighter samples), tau = 0 is noise-free.
        noise = jax.random.normal(key, x_t.shape)
        t_val = jnp.asarray(t)
        is_not_zero = (t_val > 0).astype(jnp.float32)
        return mean + temperature * is_not_zero * jnp.exp(0.5 * log_var) * noise

    def sample(
        self,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        guidance: GuidanceSpec | None = None,
        temperature: float = 1.0,
    ) -> TrajectoryPrediction:
        """Generate trajectories via full reverse diffusion.

        Starts from pure Gaussian noise and iteratively denoises
        through all timesteps to produce clean trajectory predictions.
        The number of agents is inferred from ``scene_context``: one
        trajectory is generated per context row.

        Args:
            scene_context: Per-agent context (adaLN anchor), shape
                ``(num_agents, context_dim)`` — one row per agent.
            key: JAX random key.
            scene_tokens: Optional fused scene tokens for map cross-attention
                (see :meth:`predict_noise`); required when the config enables
                ``use_map_cross_attention``.
            scene_token_valid: Optional per-token validity mask.
            guidance: Optional test-time x̂₀ guidance (:class:`GuidanceSpec`;
                see :meth:`p_sample_step`).
            temperature: Multiplier τ on the reverse-step posterior noise
                (τ=1 standard sampling, 0≤τ<1 sharpens/tightens the samples,
                τ=0 makes the reverse process noise-free). Must be non-negative.

        Returns:
            TrajectoryPrediction with generated trajectories.
        """
        if temperature < 0.0:
            msg = f"temperature must be non-negative, got {temperature}"
            raise ValueError(msg)
        num_agents = scene_context.shape[0]
        shape = (
            num_agents,
            self.config.future_steps,
            self.config.state_dim,
        )

        # Start from pure noise
        key, init_key = jax.random.split(key)
        x_init = jax.random.normal(init_key, shape)

        # Reverse diffusion via lax.scan: compile time stays constant in
        # num_timesteps (a Python loop unrolls into thousands of steps)
        timesteps = jnp.arange(self.config.num_timesteps - 1, -1, -1)

        def denoise_step(x_t: jax.Array, t_val: jax.Array) -> tuple[jax.Array, None]:
            step_key = jax.random.fold_in(key, t_val)
            next_x = self.p_sample_step(
                x_t,
                t_val,
                scene_context,
                key=step_key,
                scene_tokens=scene_tokens,
                scene_token_valid=scene_token_valid,
                guidance=guidance,
                temperature=temperature,
            )
            return next_x, None

        # Checkpoint the scan body so differentiating through sampling
        # (e.g. adversarial context perturbation) recomputes per-step
        # activations instead of storing all of them: memory stays O(1)
        # in num_timesteps. Primal-only sampling is unaffected.
        x_final, _ = jax.lax.scan(jax.checkpoint(denoise_step), x_init, timesteps)
        x_final = self._denormalize_states(x_final)

        agent_ids = tuple(f"agent_{i}" for i in range(num_agents))
        return TrajectoryPrediction(trajectories=x_final, agent_ids=agent_ids)

    def compute_loss_outputs(
        self,
        trajectories: jax.Array,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        valid_mask: jax.Array | None = None,
        timestep_stratum: tuple[jax.Array | int, int] | None = None,
        deterministic: bool = False,
    ) -> DiffusionLossOutputs:
        """Compute the diffusion loss, the x̂₀ prediction, and ᾱ_t.

        Samples a random timestep, adds noise to the clean trajectories,
        predicts the noise with the backbone, and reconstructs the model's
        clean-trajectory estimate x̂₀ from that prediction, clamped to
        ``x0_clip_bound`` when configured — the same data bound the
        sampling path enforces. The returned prediction carries model
        gradients, so downstream physics penalties applied to it shape the
        model (a future action-space rollout can satisfy the same seam).
        The drawn timestep's schedule value ᾱ_t is also returned so those
        penalties can anneal with the trustworthiness of x̂₀.

        When the config sets ``snr_gamma``, the loss is scaled by the
        Min-SNR-γ weight ``min(SNR_t, γ) / SNR_t`` of the drawn timestep
        (ε-prediction form of the reference trainers).

        Args:
            trajectories: Clean trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            scene_context: Scene context embeddings,
                shape ``(ctx_len, context_dim)``.
            key: JAX random key for timestep and noise sampling.
            scene_tokens: Optional fused scene tokens
                ``(num_tokens, scene_token_dim)`` for the backbone map
                cross-attention; required when the config enables
                ``use_map_cross_attention``.
            scene_token_valid: Optional ``(num_tokens,)`` boolean token mask.
            valid_mask: Optional per-step validity of shape
                ``(num_agents, future_steps)``. When given, the
                epsilon-reconstruction error is masked-averaged —
                ``sum(mask * sq_err) / (sum(mask) * state_dim)`` — so
                zero-padded agents and out-of-horizon steps contribute no
                training signal. This mirrors the official WOD motion
                tutorial, which weights its per-step loss by
                ``gt_future_is_valid`` rather than filtering to
                fully-valid agents. ``None`` (default) reproduces the
                unmasked mean exactly.
            timestep_stratum: Optional ``(stratum_index, stratum_count)``
                controlling the timestep draw. When set, the timestep is
                drawn from the ``stratum_index``-th of ``stratum_count``
                equal schedule strata via
                :func:`~diffav.models.sampling_utils.stratified_timestep`,
                so a batch of scenes calling with strata ``0..B-1`` covers
                the whole schedule every step instead of clustering wherever
                independent uniform draws land (variance reduction, VDM
                appendix I.1). ``None`` (default) draws a single uniform
                timestep — bit-identical to the prior behavior.
            deterministic: Disable backbone dropout. ``False`` (default) is the
                training draw; pass ``True`` for a dropout-free denoising score
                (e.g. a DPO log-prob estimate).

        Returns:
            :class:`DiffusionLossOutputs` with the scalar loss, the
            predicted clean trajectories (shaped like ``trajectories``),
            and the drawn timestep's ᾱ_t.
        """
        key_t, key_noise = jax.random.split(key)

        # Diffusion runs in normalized space when the config declares a
        # state normalizer; all boundaries stay in raw data units.
        trajectories = self._normalize_states(trajectories)

        # Sample the diffusion timestep: a plain uniform draw by default, or
        # a stratified draw when the caller assigns this scene a schedule
        # stratum (batched trainer spreads one stratum per scene).
        if timestep_stratum is None:
            t = jax.random.randint(key_t, (), 0, self.config.num_timesteps)
        else:
            stratum_index, stratum_count = timestep_stratum
            t = stratified_timestep(key_t, stratum_index, stratum_count, self.config.num_timesteps)

        # Sample noise
        noise = jax.random.normal(key_noise, trajectories.shape)

        # Forward diffusion
        noisy = self.q_sample(trajectories, t, noise)

        # Backbone output — epsilon, x0, or v per the parameterization.
        # Training draws dropout (deterministic=False); a log-prob / eval
        # estimate passes deterministic=True for a dropout-free score.
        model_output = self.predict_noise(
            noisy,
            t,
            scene_context,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
            deterministic=deterministic,
        )

        # Reconstruct the clean-trajectory estimate x-hat-0 from the output,
        # honouring the same x0_clip_bound as the sampling path (p_sample). For
        # EPSILON, this divides by sqrt(alpha_bar_t), which at high t amplifies
        # model error by orders of magnitude, so the data bound keeps downstream
        # physics penalties on the reconstruction finite (reference diffusers
        # clamp the x-hat-0 estimate to the data range for exactly this reason).
        x_0_pred = self._predict_x0_from_output(noisy, t, model_output)
        if self.config.x0_clip_bound is not None:
            bound = self.config.x0_clip_bound
            x_0_pred = jnp.clip(x_0_pred, -bound, bound)
        predicted = self._denormalize_states(x_0_pred)

        # The loss regresses the backbone output against the parameterization's
        # target (epsilon / x0 / v), all in normalized diffusion space.
        target = self._prediction_target(trajectories, noise, t)
        elementwise = (
            (model_output - target) ** 2
            if self.config.loss_type == "mse"
            else jnp.abs(model_output - target)
        )
        if valid_mask is None:
            loss = jnp.mean(elementwise)
        else:
            # Masked average over valid (agent, step) entries, mirroring the
            # WOD motion tutorial's ``sample_weight`` construction: numerator
            # sums the per-element error at valid steps, denominator counts
            # the valid scalar entries (steps * state_dim). Guard the
            # denominator so an all-invalid scene yields a finite zero loss.
            weights = valid_mask[..., None]
            numerator = jnp.sum(weights * elementwise)
            denominator = jnp.maximum(jnp.sum(valid_mask) * self.config.state_dim, 1.0)
            loss = numerator / denominator

        # ᾱ_t of the drawn timestep: the model's trust in x̂₀ at this
        # noise level (the sampling path uses the same factor to anneal
        # x̂₀ guidance).
        alpha_bar_t = self.noise_schedule.alphas_cumprod[t]

        # Min-SNR-γ weighting: scale the loss by min(SNR_t, γ) so easy
        # high-SNR timesteps stop dominating, converted to the parameterization
        # (epsilon: /SNR, v: /(SNR+1), x0: as-is). Reference: Hang et al. 2023;
        # diffusers/maxdiffusion snr_gamma paths.
        if self.config.snr_gamma is not None:
            snr = alpha_bar_t / jnp.maximum(1.0 - alpha_bar_t, 1e-12)
            capped = jnp.minimum(snr, self.config.snr_gamma)
            if self.config.prediction_type == PredictionType.EPSILON:
                weight = capped / snr
            elif self.config.prediction_type == PredictionType.V:
                weight = capped / (snr + 1.0)
            else:  # X0
                weight = capped
            loss = loss * weight

        return DiffusionLossOutputs(loss=loss, prediction=predicted, alpha_bar_t=alpha_bar_t)

    def compute_loss_and_prediction(
        self,
        trajectories: jax.Array,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        valid_mask: jax.Array | None = None,
        timestep_stratum: tuple[jax.Array | int, int] | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the diffusion loss and the predicted clean trajectories.

        Pair-shaped view of :meth:`compute_loss_outputs` for callers that
        do not need the drawn timestep's ᾱ_t.

        Args:
            trajectories: Clean trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            scene_context: Scene context embeddings,
                shape ``(ctx_len, context_dim)``.
            key: JAX random key for timestep and noise sampling.
            scene_tokens: Optional fused scene tokens for map cross-attention
                (see :meth:`compute_loss_outputs`).
            scene_token_valid: Optional per-token validity mask.
            valid_mask: Optional per-step validity of shape
                ``(num_agents, future_steps)`` (see
                :meth:`compute_loss_outputs`).
            timestep_stratum: Optional ``(stratum_index, stratum_count)``
                selecting a stratified timestep draw (see
                :meth:`compute_loss_outputs`).

        Returns:
            Tuple of ``(scalar loss, predicted clean trajectories)`` with
            the prediction shaped like ``trajectories``.
        """
        outputs = self.compute_loss_outputs(
            trajectories,
            scene_context,
            key=key,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
            valid_mask=valid_mask,
            timestep_stratum=timestep_stratum,
        )
        return outputs.loss, outputs.prediction

    def compute_loss(
        self,
        trajectories: jax.Array,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        valid_mask: jax.Array | None = None,
        timestep_stratum: tuple[jax.Array | int, int] | None = None,
    ) -> jax.Array:
        """Compute diffusion training loss.

        Samples a random timestep, adds noise to the clean trajectories,
        predicts the noise with the backbone, and returns the
        reconstruction error.

        Args:
            trajectories: Clean trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            scene_context: Scene context embeddings,
                shape ``(ctx_len, context_dim)``.
            key: JAX random key for timestep and noise sampling.
            scene_tokens: Optional fused scene tokens for map cross-attention
                (see :meth:`compute_loss_and_prediction`).
            scene_token_valid: Optional per-token validity mask.
            valid_mask: Optional per-step validity of shape
                ``(num_agents, future_steps)`` for masked-averaging the
                loss over valid entries (see
                :meth:`compute_loss_and_prediction`).
            timestep_stratum: Optional ``(stratum_index, stratum_count)``
                selecting a stratified timestep draw (see
                :meth:`compute_loss_and_prediction`). ``None`` (default)
                draws a single uniform timestep.

        Returns:
            Scalar loss value.
        """
        loss, _ = self.compute_loss_and_prediction(
            trajectories,
            scene_context,
            key=key,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
            valid_mask=valid_mask,
            timestep_stratum=timestep_stratum,
        )
        return loss


def create_trajectory_model(
    config: TrajectoryDiffusionConfig,
    *,
    rngs: nnx.Rngs,
) -> TrajectoryDiffusionModel:
    """Factory function to create a trajectory diffusion model.

    Args:
        config: Model configuration.
        rngs: Flax NNX random number generators.

    Returns:
        Initialized trajectory diffusion model.
    """
    return TrajectoryDiffusionModel(config, rngs=rngs)
