"""Shared configuration validators for the DiffAV pipeline.

Defines the split vocabulary and the positivity/divisibility validators
reused by config dataclasses across the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from diffav.core.constants import (
    DEFAULT_NUM_BLOCKS,
    DEFAULT_NUM_SOCIAL_LAYERS,
    DEFAULT_NUM_TEMPORAL_LAYERS,
)


VALID_SPLITS = frozenset({"train", "val", "test"})
"""Valid dataset split names shared across config modules."""


class FactorizedBackboneLikeConfig(Protocol):
    """Structural protocol for factorized-scene-backbone-compatible configs."""

    @property
    def hidden_dim(self) -> int:
        """Transformer hidden dimension."""
        ...

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        ...

    @property
    def num_blocks(self) -> int:
        """Number of factorized blocks."""
        ...

    @property
    def num_temporal_layers(self) -> int:
        """Temporal attention sublayers per block."""
        ...

    @property
    def num_social_layers(self) -> int:
        """Social attention sublayers per block."""
        ...

    @property
    def state_dim(self) -> int:
        """Per-timestep state dimension."""
        ...

    @property
    def num_agents_max(self) -> int:
        """Maximum number of agents."""
        ...

    @property
    def future_steps(self) -> int:
        """Future prediction horizon in timesteps."""
        ...

    @property
    def context_dim(self) -> int:
        """Scene context token dimension."""
        ...

    @property
    def mlp_ratio(self) -> float:
        """Feed-forward expansion ratio."""
        ...

    @property
    def use_social_interaction(self) -> bool:
        """Whether agents attend across one another in the social pass."""
        ...

    @property
    def use_map_cross_attention(self) -> bool:
        """Whether the denoiser cross-attends to fused scene tokens."""
        ...

    @property
    def scene_token_dim(self) -> int | None:
        """Fused-scene-token key/value width (``None`` mirrors hidden_dim)."""
        ...

    @property
    def gradient_checkpointing(self) -> bool:
        """Whether backbone blocks rematerialize activations to save memory."""
        ...


class FactorizedBackboneConfigKwargs(TypedDict):
    """Keyword arguments shared by factorized-scene-backbone constructors."""

    hidden_dim: int
    num_blocks: int
    num_temporal_layers: int
    num_social_layers: int
    num_heads: int
    state_dim: int
    num_agents_max: int
    future_steps: int
    context_dim: int
    mlp_ratio: float
    use_social_interaction: bool
    use_map_cross_attention: bool
    scene_token_dim: int | None
    gradient_checkpointing: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorizedBackboneHyperparams:
    """Architecture hyperparameters shared by factorized-backbone configs.

    Declared once so the model-level ``TrajectoryDiffusionConfig`` and the
    public-SDK ``MinerConfig`` expose an identical factorized
    temporal+social backbone surface without duplicating field defaults.

    Attributes:
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_blocks: Number of factorized backbone blocks.
        num_temporal_layers: Temporal attention sublayers per block.
        num_social_layers: Social attention sublayers per block.
        use_social_interaction: Whether agents attend across one another.
        use_map_cross_attention: Whether the denoiser cross-attends to a fused
            scene-token set (map conditioning); ``False`` keeps it map-blind.
        scene_token_dim: Fused-scene-token key/value width, or ``None`` to
            mirror ``hidden_dim``.
        gradient_checkpointing: Rematerialize each backbone block's activations
            in the backward pass instead of storing them, trading compute for a
            large activation-memory saving (so a bigger batch fits).
    """

    hidden_dim: int = 128
    num_heads: int = 4
    num_blocks: int = DEFAULT_NUM_BLOCKS
    num_temporal_layers: int = DEFAULT_NUM_TEMPORAL_LAYERS
    num_social_layers: int = DEFAULT_NUM_SOCIAL_LAYERS
    use_social_interaction: bool = True
    use_map_cross_attention: bool = False
    scene_token_dim: int | None = None
    gradient_checkpointing: bool = False


def validate_positive(field_name: str, value: int | float) -> None:
    """Validate that a numeric field is strictly positive.

    Args:
        field_name: Name of the field (for error messages).
        value: Value to validate.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        msg = f"{field_name} must be positive, got {value}"
        raise ValueError(msg)


def validate_transformer_fields(
    hidden_dim: int,
    num_heads: int,
    *,
    extra_positive: dict[str, int | float] | None = None,
) -> None:
    """Validate fields common to transformer-based configs.

    Checks ``hidden_dim`` and ``num_heads`` are positive and that
    ``hidden_dim`` is divisible by ``num_heads``.  Additional fields
    can be validated via *extra_positive*.

    Args:
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        extra_positive: Optional mapping of field name to value, each
            validated as strictly positive.

    Raises:
        ValueError: If any constraint is violated.
    """
    validate_positive("hidden_dim", hidden_dim)
    validate_positive("num_heads", num_heads)
    if extra_positive:
        for name, value in extra_positive.items():
            validate_positive(name, value)
    if hidden_dim % num_heads != 0:
        msg = f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        raise ValueError(msg)


def validate_factorized_backbone_fields(
    hidden_dim: int,
    num_heads: int,
    *,
    num_blocks: int,
    num_temporal_layers: int,
    num_social_layers: int,
    state_dim: int,
    num_agents_max: int,
    future_steps: int,
    context_dim: int,
    mlp_ratio: float,
    extra_positive: dict[str, int | float] | None = None,
) -> None:
    """Validate common fields shared by factorized-backbone model configs.

    Args:
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_blocks: Number of factorized blocks.
        num_temporal_layers: Temporal attention sublayers per block.
        num_social_layers: Social attention sublayers per block.
        state_dim: Per-step state dimension.
        num_agents_max: Maximum number of agents in a scene.
        future_steps: Number of future timesteps to model.
        context_dim: Scene context embedding dimension.
        mlp_ratio: Feed-forward width multiplier.
        extra_positive: Optional additional fields to validate as positive.
    """
    positive_fields: dict[str, int | float] = {
        "num_blocks": num_blocks,
        "num_temporal_layers": num_temporal_layers,
        "num_social_layers": num_social_layers,
        "state_dim": state_dim,
        "num_agents_max": num_agents_max,
        "future_steps": future_steps,
        "context_dim": context_dim,
        "mlp_ratio": mlp_ratio,
    }
    if extra_positive:
        positive_fields.update(extra_positive)

    validate_transformer_fields(
        hidden_dim,
        num_heads,
        extra_positive=positive_fields,
    )


def validate_factorized_backbone_config(
    config: FactorizedBackboneLikeConfig,
    *,
    extra_positive: dict[str, int | float] | None = None,
) -> None:
    """Validate a factorized-backbone-like config object."""
    kwargs = factorized_backbone_config_kwargs(config)
    validate_factorized_backbone_fields(
        hidden_dim=int(kwargs["hidden_dim"]),
        num_heads=int(kwargs["num_heads"]),
        num_blocks=int(kwargs["num_blocks"]),
        num_temporal_layers=int(kwargs["num_temporal_layers"]),
        num_social_layers=int(kwargs["num_social_layers"]),
        state_dim=int(kwargs["state_dim"]),
        num_agents_max=int(kwargs["num_agents_max"]),
        future_steps=int(kwargs["future_steps"]),
        context_dim=int(kwargs["context_dim"]),
        mlp_ratio=float(kwargs["mlp_ratio"]),
        extra_positive=extra_positive,
    )


def factorized_backbone_config_kwargs(
    config: FactorizedBackboneLikeConfig,
) -> FactorizedBackboneConfigKwargs:
    """Build kwargs shared by factorized-backbone-compatible configs."""
    return {
        "hidden_dim": config.hidden_dim,
        "num_blocks": config.num_blocks,
        "num_temporal_layers": config.num_temporal_layers,
        "num_social_layers": config.num_social_layers,
        "num_heads": config.num_heads,
        "state_dim": config.state_dim,
        "num_agents_max": config.num_agents_max,
        "future_steps": config.future_steps,
        "context_dim": config.context_dim,
        "mlp_ratio": config.mlp_ratio,
        "use_social_interaction": config.use_social_interaction,
        "use_map_cross_attention": config.use_map_cross_attention,
        "scene_token_dim": config.scene_token_dim,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
