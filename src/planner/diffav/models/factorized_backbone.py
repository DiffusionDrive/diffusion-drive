"""Factorized temporal + social scene-diffusion backbone.

The denoiser treats a scene as a grid of ``(num_agents, future_steps)``
trajectory tokens and mixes them along the CTG++ scene-transformer
factorization:

* **temporal** attention lets each agent attend over its own future timesteps
  (the time axis, batched over agents);
* **map cross-attention** (optional, ``use_map_cross_attention``) lets each
  trajectory token attend over a shared, persistent set of fused scene tokens
  (map + agent + ego) — the CTG++/MotionDiffuser map-conditioning axis;
* **social** attention lets the agents at each timestep attend across one
  another (the agent axis, batched over time) — the sole channel through which
  one agent influences another.

Per-agent conditioning enters as adaptive layer norm (adaLN) modulation from
each agent's own ``scene_context`` row, so agent ``i``'s tokens are always
conditioned by context row ``i``. The diffusion timestep is embedded once and
added to the per-agent conditioning vector. When map cross-attention is
enabled the fused scene tokens are the sole map signal; the ``scene_context``
row then carries the agent's own identity/anchor rather than a pooled scene
summary.

Every attention kernel is :class:`flax.nnx.MultiHeadAttention`; the adaLN
modulation, timestep embedding, sinusoidal position embedding, and
feed-forward network are composed from ``artifex``. Only the attention masks
and axis reshapes — domain glue, not attention math — are assembled here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from artifex.generative_models.core.gradient_checkpointing import apply_remat
from artifex.generative_models.core.layers.transformers import FeedForwardNetwork
from artifex.generative_models.models.backbones.dit import (
    get_1d_sincos_pos_embed,
    modulate,
    TimestepEmbedder,
)
from flax import nnx

from diffav.core.config import FactorizedBackboneHyperparams, validate_positive


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorizedSceneBackboneConfig(FactorizedBackboneHyperparams):
    """Configuration for :class:`FactorizedSceneBackbone`.

    Extends :class:`~diffav.core.config.FactorizedBackboneHyperparams`
    (the shared ``hidden_dim`` / ``num_heads`` / block-count and attention
    toggles, including ``use_map_cross_attention`` and ``scene_token_dim``)
    with the backbone-specific dimensions below.

    Attributes:
        state_dim: Per-timestep trajectory state dimension.
        num_agents_max: Maximum agents per scene (for the caller's capacity
            planning; the backbone itself infers the agent count per call).
        future_steps: Prediction horizon in timesteps.
        context_dim: Per-agent scene-context row dimension.
        mlp_ratio: Feed-forward hidden-dimension multiplier.
        dropout_rate: Attention/FFN dropout probability.
    """

    state_dim: int = 4
    num_agents_max: int = 32
    future_steps: int = 80
    context_dim: int = 64
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0

    @property
    def resolved_scene_token_dim(self) -> int:
        """Fused-scene-token key/value width, defaulting to ``hidden_dim``."""
        return self.hidden_dim if self.scene_token_dim is None else self.scene_token_dim

    def __post_init__(self) -> None:
        """Validate positive dimensions and head divisibility."""
        for name in (
            "hidden_dim",
            "num_heads",
            "state_dim",
            "num_agents_max",
            "future_steps",
            "context_dim",
            "num_blocks",
            "num_temporal_layers",
            "num_social_layers",
        ):
            validate_positive(name, getattr(self, name))
        if self.scene_token_dim is not None:
            validate_positive("scene_token_dim", self.scene_token_dim)
        if self.hidden_dim % self.num_heads != 0:
            msg = (
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
            raise ValueError(msg)
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            msg = f"dropout_rate must be in [0, 1), got {self.dropout_rate}"
            raise ValueError(msg)


def _temporal_mask(time_valid: jax.Array | None, future_steps: int) -> jax.Array | None:
    """Boolean key-validity mask for the temporal (per-agent, over-time) pass.

    Args:
        time_valid: ``(num_agents, future_steps)`` boolean validity, or
            ``None`` when every timestep is valid.
        future_steps: The horizon length ``T``.

    Returns:
        A ``(num_agents, 1, T, T)`` boolean mask broadcastable to the
        attention weights ``(num_agents, heads, T, T)``, or ``None``. The
        diagonal is always kept so a fully-invalid agent never produces an
        all-masked (NaN) softmax row.
    """
    if time_valid is None:
        return None
    key_valid = time_valid[:, None, None, :]  # (A, 1, 1, T)
    mask = jnp.broadcast_to(key_valid, (time_valid.shape[0], 1, future_steps, future_steps))
    eye = jnp.eye(future_steps, dtype=bool)[None, None, :, :]
    return mask | eye


def _social_mask(
    agent_valid: jax.Array | None,
    num_agents: int,
    *,
    use_interaction: bool,
) -> jax.Array | None:
    """Boolean mask for the social (per-timestep, across-agents) pass.

    Args:
        agent_valid: ``(num_agents,)`` boolean validity, or ``None`` when
            every agent is real.
        num_agents: The agent count ``A``.
        use_interaction: When ``False`` the mask is the identity, so each
            agent attends only to itself (strict per-agent independence).

    Returns:
        A ``(1, 1, A, A)`` boolean mask broadcastable to the social attention
        weights ``(time, heads, A, A)``, or ``None`` when every agent is real
        and interaction is enabled (full attention). The diagonal is always
        kept to avoid all-masked softmax rows for padded agents.
    """
    if agent_valid is None and use_interaction:
        return None
    eye = jnp.eye(num_agents, dtype=bool)[None, None, :, :]
    if agent_valid is None:
        key_valid = jnp.ones((num_agents,), dtype=bool)
    else:
        key_valid = agent_valid
    allow = jnp.broadcast_to(key_valid[None, None, None, :], (1, 1, num_agents, num_agents))
    if not use_interaction:
        allow = allow & eye
    return allow | eye


def _scene_token_mask(scene_token_valid: jax.Array | None) -> jax.Array | None:
    """Boolean key-validity mask for the map cross-attention pass.

    Args:
        scene_token_valid: ``(num_tokens,)`` boolean validity, or ``None`` when
            every fused scene token is real.

    Returns:
        A ``(1, 1, 1, num_tokens)`` boolean mask broadcastable to the
        cross-attention weights, or ``None`` when unmasked. When every token
        is invalid the mask falls back to all-valid, so the softmax never sees
        an all-masked row (NaN-safe); every query shares one key mask, so the
        fallback is global.
    """
    if scene_token_valid is None:
        return None
    valid = jnp.asarray(scene_token_valid).astype(bool)
    valid = jnp.where(jnp.any(valid), valid, True)
    return valid[None, None, None, :]


class _MapCrossAttentionSublayer(nnx.Module):
    """adaLN-gated cross-attention from trajectory tokens to scene tokens.

    Each trajectory token (query) attends over the shared set of fused scene
    (map + agent + ego) tokens (keys/values), the CTG++/MotionDiffuser map
    conditioning mechanism. The keys/values may carry a different width than
    the trajectory tokens, handled by the attention kernel's ``in_kv_features``.
    """

    def __init__(
        self,
        hidden_dim: int,
        scene_token_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the query norm and the cross-attention kernel.

        Args:
            hidden_dim: Trajectory-token (query) hidden dimension.
            scene_token_dim: Fused-scene-token (key/value) dimension.
            num_heads: Attention heads.
            dropout_rate: Attention dropout probability.
            rngs: NNX random number generators.
        """
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            in_kv_features=scene_token_dim,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
        gate: jax.Array,
        scene_tokens: jax.Array,
        *,
        mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        """Modulate, cross-attend to the scene tokens, and residually gate.

        Args:
            x: Trajectory tokens ``(num_agents, future_steps, hidden_dim)``.
            shift: adaLN shift, broadcastable to ``x``.
            scale: adaLN scale, broadcastable to ``x``.
            gate: adaLN residual gate, broadcastable to ``x``.
            scene_tokens: Fused scene tokens ``(num_tokens, scene_token_dim)``.
            mask: Optional boolean key-validity mask.
            deterministic: Disable dropout when ``True``.

        Returns:
            The residual-updated trajectory tokens, same shape as ``x``.
        """
        num_agents, steps, hidden = x.shape
        normed = modulate(self.norm(x), shift, scale)
        queries = normed.reshape(1, num_agents * steps, hidden)
        tokens = scene_tokens[jnp.newaxis, ...]
        attended = self.attn(
            inputs_q=queries,
            inputs_k=tokens,
            inputs_v=tokens,
            mask=mask,
            deterministic=deterministic,
        )
        attended = attended.reshape(num_agents, steps, hidden)
        return x + gate * attended


class _AdaLNAttentionSublayer(nnx.Module):
    """One adaLN-gated multi-head attention over axis ``-2`` of the tokens.

    Axis-agnostic: the caller arranges the token grid so the axis to mix is
    ``-2`` and any batch axes lead. This is the reusable core of the
    factorized block — the temporal and social passes are the same sublayer
    applied to two different axis arrangements.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the norm and attention kernel.

        Args:
            hidden_dim: Token hidden dimension.
            num_heads: Attention heads.
            dropout_rate: Attention dropout probability.
            rngs: NNX random number generators.
        """
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
        gate: jax.Array,
        *,
        mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        """Modulate, attend over axis ``-2``, and residually gate.

        Args:
            x: Token grid ``(..., length, hidden_dim)``.
            shift: adaLN shift, broadcastable to ``x``.
            scale: adaLN scale, broadcastable to ``x``.
            gate: adaLN residual gate, broadcastable to ``x``.
            mask: Optional boolean attention mask.
            deterministic: Disable dropout when ``True``.

        Returns:
            The residual-updated token grid, same shape as ``x``.
        """
        normed = modulate(self.norm(x), shift, scale)
        attended = self.attn(inputs_q=normed, mask=mask, deterministic=deterministic)
        return x + gate * attended


class _FactorizedBlock(nnx.Module):
    """A block of temporal, [map cross-attention,] social, and FFN sublayers.

    All sublayers are adaLN-conditioned by the same per-agent conditioning
    vector, so agent ``i`` is modulated by context row ``i`` throughout. When
    ``use_map_cross_attention`` is set the trajectory tokens additionally
    cross-attend to a shared set of fused scene tokens between the temporal
    and social passes (CTG++/MotionDiffuser conditioning).
    """

    def __init__(self, config: FactorizedSceneBackboneConfig, *, rngs: nnx.Rngs) -> None:
        """Build the sublayers and the zero-initialised adaLN projection.

        Args:
            config: Backbone configuration.
            rngs: NNX random number generators.
        """
        hidden = config.hidden_dim
        self.temporal = nnx.List(
            [
                _AdaLNAttentionSublayer(hidden, config.num_heads, config.dropout_rate, rngs=rngs)
                for _ in range(config.num_temporal_layers)
            ]
        )
        self.map_cross: _MapCrossAttentionSublayer | None = (
            _MapCrossAttentionSublayer(
                hidden,
                config.resolved_scene_token_dim,
                config.num_heads,
                config.dropout_rate,
                rngs=rngs,
            )
            if config.use_map_cross_attention
            else None
        )
        self.social = nnx.List(
            [
                _AdaLNAttentionSublayer(hidden, config.num_heads, config.dropout_rate, rngs=rngs)
                for _ in range(config.num_social_layers)
            ]
        )
        self.ffn_norm = nnx.LayerNorm(num_features=hidden, rngs=rngs)
        self.ffn = FeedForwardNetwork(
            in_features=hidden,
            hidden_features=int(hidden * config.mlp_ratio),
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        # One (shift, scale, gate) triple per sublayer (temporal + [map] + social + FFN).
        num_map_layers = 1 if config.use_map_cross_attention else 0
        self._num_sublayers = (
            config.num_temporal_layers + num_map_layers + config.num_social_layers + 1
        )
        # Zero-initialised (DiT convention): every block starts as the identity.
        self.adaln = nnx.Linear(
            hidden,
            3 * self._num_sublayers * hidden,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        conditioning: jax.Array,
        *,
        temporal_mask: jax.Array | None,
        social_mask: jax.Array | None,
        scene_tokens: jax.Array | None,
        scene_mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        """Run temporal, [map], social, and feed-forward passes with adaLN gating.

        Args:
            x: Token grid ``(num_agents, future_steps, hidden_dim)``.
            conditioning: Per-agent conditioning ``(num_agents, hidden_dim)``.
            temporal_mask: Optional temporal attention mask.
            social_mask: Optional social attention mask.
            scene_tokens: Fused scene tokens ``(num_tokens, scene_token_dim)``
                for the map cross-attention pass; ``None`` when the block has
                no map sublayer.
            scene_mask: Optional key-validity mask for the map pass.
            deterministic: Disable dropout when ``True``.

        Returns:
            The updated token grid, same shape as ``x``.
        """
        triples = jnp.split(nnx.silu(self.adaln(conditioning)), 3 * self._num_sublayers, axis=-1)
        cursor = 0

        # Temporal pass: attend over the time axis (-2), batched over agents.
        for sublayer in self.temporal:
            shift, scale, gate = triples[cursor : cursor + 3]
            cursor += 3
            x = sublayer(
                x,
                shift[:, None, :],
                scale[:, None, :],
                gate[:, None, :],
                mask=temporal_mask,
                deterministic=deterministic,
            )

        # Map cross-attention pass: each trajectory token attends over the
        # shared fused-scene-token set (map + agent + ego). Per-agent adaLN
        # broadcasts over the time axis, as in the temporal pass.
        if self.map_cross is not None:
            if scene_tokens is None:
                msg = "map cross-attention block requires scene_tokens"
                raise ValueError(msg)
            shift, scale, gate = triples[cursor : cursor + 3]
            cursor += 3
            x = self.map_cross(
                x,
                shift[:, None, :],
                scale[:, None, :],
                gate[:, None, :],
                scene_tokens,
                mask=scene_mask,
                deterministic=deterministic,
            )

        # Social pass: swap agents to axis -2 so agents attend across each
        # other, batched over time; the per-agent conditioning broadcasts over
        # the time batch.
        xs = jnp.swapaxes(x, -3, -2)
        for sublayer in self.social:
            shift, scale, gate = triples[cursor : cursor + 3]
            cursor += 3
            xs = sublayer(
                xs,
                shift[None, :, :],
                scale[None, :, :],
                gate[None, :, :],
                mask=social_mask,
                deterministic=deterministic,
            )
        x = jnp.swapaxes(xs, -3, -2)

        # Feed-forward pass.
        shift, scale, gate = triples[cursor : cursor + 3]
        normed = modulate(self.ffn_norm(x), shift[:, None, :], scale[:, None, :])
        return x + gate[:, None, :] * self.ffn(normed, deterministic=deterministic)


class FactorizedSceneBackbone(nnx.Module):
    """Factorized temporal + social denoiser for multi-agent trajectories.

    Operates on a single scene ``(num_agents, future_steps, state_dim)``; the
    trainer vmaps over the scene batch. Each agent is conditioned on its own
    ``scene_context`` row (adaLN), and the social attention axis couples
    agents so one agent's motion can influence another.
    """

    def __init__(self, config: FactorizedSceneBackboneConfig, *, rngs: nnx.Rngs) -> None:
        """Build projections, embedders, and the factorized blocks.

        Args:
            config: Backbone configuration.
            rngs: NNX random number generators.
        """
        self.config = config
        hidden = config.hidden_dim
        self.input_proj = nnx.Linear(config.state_dim, hidden, rngs=rngs)
        self.context_proj = nnx.Linear(config.context_dim, hidden, rngs=rngs)
        self.timestep_embedder = TimestepEmbedder(hidden, rngs=rngs)
        self.blocks = nnx.List(
            [_FactorizedBlock(config, rngs=rngs) for _ in range(config.num_blocks)]
        )
        self.output_norm = nnx.LayerNorm(num_features=hidden, rngs=rngs)
        self.output_proj = nnx.Linear(hidden, config.state_dim, rngs=rngs)
        # Zero-initialised adaLN output head (DiT convention).
        self.output_adaln = nnx.Linear(
            hidden,
            2 * hidden,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def _validate_scene_tokens(self, scene_tokens: jax.Array | None) -> None:
        """Enforce the scene-token contract against the map-attention config.

        Args:
            scene_tokens: The fused scene tokens passed to ``__call__``, or
                ``None``.

        Raises:
            ValueError: If the presence, rank, or width of ``scene_tokens``
                does not match ``use_map_cross_attention`` and
                ``resolved_scene_token_dim``.
        """
        if not self.config.use_map_cross_attention:
            if scene_tokens is not None:
                msg = "map cross-attention is disabled; do not pass scene_tokens"
                raise ValueError(msg)
            return
        if scene_tokens is None:
            msg = "map cross-attention is enabled; scene_tokens is required"
            raise ValueError(msg)
        if scene_tokens.ndim != 2:
            msg = (
                f"scene_tokens must be 2-D (num_tokens, scene_token_dim), got {scene_tokens.shape}"
            )
            raise ValueError(msg)
        expected = self.config.resolved_scene_token_dim
        if scene_tokens.shape[1] != expected:
            msg = (
                f"scene_tokens dim {scene_tokens.shape[1]} != configured scene_token_dim {expected}"
            )
            raise ValueError(msg)

    def __call__(
        self,
        noisy_trajectories: jax.Array,
        timestep: jax.Array,
        scene_context: jax.Array,
        *,
        scene_tokens: jax.Array | None = None,
        scene_token_valid: jax.Array | None = None,
        agent_valid_mask: jax.Array | None = None,
        time_valid_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Predict noise conditioned on per-agent context and fused scene tokens.

        Args:
            noisy_trajectories: Noisy tokens ``(num_agents, future_steps,
                state_dim)``.
            timestep: Scalar diffusion timestep.
            scene_context: Per-agent context, strictly ``(num_agents,
                context_dim)`` — row ``i`` conditions agent ``i`` (adaLN).
            scene_tokens: Fused scene tokens ``(num_tokens, scene_token_dim)``
                the trajectory tokens cross-attend to. Required when the config
                enables ``use_map_cross_attention``; must be ``None`` otherwise.
            scene_token_valid: Optional ``(num_tokens,)`` boolean mask; invalid
                (``False``) tokens are excluded from the map cross-attention.
            agent_valid_mask: Optional ``(num_agents,)`` boolean mask; padded
                (``False``) agents are excluded from the social attention.
            time_valid_mask: Optional ``(num_agents, future_steps)`` boolean
                mask over valid future steps.
            deterministic: Disable dropout when ``True``.

        Returns:
            Predicted noise, same shape as ``noisy_trajectories``.

        Raises:
            ValueError: If ``scene_context`` is not ``(num_agents,
                context_dim)`` with one row per input trajectory, or if the
                ``scene_tokens`` contract does not match the map-cross-attention
                configuration.
        """
        num_agents = noisy_trajectories.shape[0]
        future_steps = noisy_trajectories.shape[1]
        if scene_context.ndim != 2:
            msg = f"scene_context must be 2-D (num_agents, context_dim), got {scene_context.shape}"
            raise ValueError(msg)
        if scene_context.shape[0] != num_agents:
            msg = (
                f"scene_context has {scene_context.shape[0]} rows but there are "
                f"{num_agents} agents; exactly one context row per agent is required"
            )
            raise ValueError(msg)
        if scene_context.shape[1] != self.config.context_dim:
            msg = (
                f"scene_context dim {scene_context.shape[1]} != configured "
                f"context_dim {self.config.context_dim}"
            )
            raise ValueError(msg)
        self._validate_scene_tokens(scene_tokens)

        x = self.input_proj(noisy_trajectories)
        x = x + get_1d_sincos_pos_embed(self.config.hidden_dim, future_steps)[None, :, :]

        time_embedding = self.timestep_embedder(jnp.atleast_1d(timestep))[0]
        conditioning = time_embedding[None, :] + self.context_proj(scene_context)

        temporal_mask = _temporal_mask(time_valid_mask, future_steps)
        social_mask = _social_mask(
            agent_valid_mask,
            num_agents,
            use_interaction=self.config.use_social_interaction,
        )
        scene_mask = _scene_token_mask(scene_token_valid)

        def make_block_forward(block_index: int) -> Callable[..., jax.Array]:
            # Close over the block (via ``self`` and a static index) rather than
            # passing it as an argument: nnx.remat would otherwise re-lift the
            # module's Param state through the remat boundary, leaving a stale
            # trace level the outer optimizer update rejects. Only the
            # differentiable tensors are arguments. (Pattern from artifex's
            # apply_remat usage.)
            def forward(tokens: jax.Array, cond: jax.Array, fused: jax.Array | None) -> jax.Array:
                return self.blocks[block_index](
                    tokens,
                    cond,
                    temporal_mask=temporal_mask,
                    social_mask=social_mask,
                    scene_tokens=fused,
                    scene_mask=scene_mask,
                    deterministic=deterministic,
                )

            return forward

        # Gradient checkpointing recomputes each block's activations in the
        # backward pass instead of storing them, trading ~1.3x compute for a
        # large activation saving so a bigger batch fits.
        for index in range(len(self.blocks)):
            forward = make_block_forward(index)
            if self.config.gradient_checkpointing:
                forward = apply_remat(forward)
            x = forward(x, conditioning, scene_tokens)

        shift, scale = jnp.split(nnx.silu(self.output_adaln(conditioning)), 2, axis=-1)
        x = modulate(self.output_norm(x), shift[:, None, :], scale[:, None, :])
        return self.output_proj(x)
