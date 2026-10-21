"""Physics-informed loss functions for trajectory generation.

Composes opifex's adaptive weight scheduling with driving-domain residuals:
bicycle-model kinematics, collision avoidance, road-boundary penalties, and a
soft momentum-variation regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from opifex.core.physics.losses import AdaptiveWeightScheduler

from diffav.core.config import validate_positive
from diffav.core.geometry import RoadEdges, signed_distances_to_road_edges
from diffav.core.types import PhysicsScheduleType
from diffav.physics.kinematics import BicycleModelConfig, BicycleModelConstraint


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffAVPhysicsConfig:
    """Configuration for driving-domain physics losses.

    Attributes:
        kinematic_weight: Weight for bicycle model kinematic residual.
        collision_weight: Weight for pairwise collision penalty.
        collision_threshold: Minimum safe distance between agents (metres).
        road_boundary_weight: Weight for road boundary violation.
        offroad_chunk_size: Optional polyline-chunk size bounding peak memory of
            the boundary term's closest-segment search (see
            :func:`~diffav.core.geometry.signed_distance_to_polylines`). The
            penalty is independent of it; ``None`` evaluates every road-edge
            polyline at once. Set it (e.g. 8) when the exact full-resolution
            boundary would otherwise exceed device memory in a batched step.
        adaptive_weighting: Enable adaptive physics weight scheduling.
        schedule_type: Weight schedule type (``"exponential"`` or ``"linear"``).
        initial_physics_weight: Starting weight for physics loss.
        final_physics_weight: Final weight for physics loss.
        transition_epochs: Epochs over which weight transitions.
        momentum_variation_weight: Weight for the momentum-variation penalty, a
            soft regularizer that discourages large swings in the scene's total
            momentum over time. This is a smoothness prior, not a physical
            conservation law (traffic momentum is not conserved under tire and
            road forces).
    """

    kinematic_weight: float = 1.0
    collision_weight: float = 1.0
    collision_threshold: float = 2.0
    road_boundary_weight: float = 1.0
    offroad_chunk_size: int | None = None
    adaptive_weighting: bool = True
    schedule_type: PhysicsScheduleType = PhysicsScheduleType.EXPONENTIAL
    initial_physics_weight: float = 0.01
    final_physics_weight: float = 1.0
    transition_epochs: int = 100
    momentum_variation_weight: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.kinematic_weight < 0:
            msg = f"kinematic_weight must be non-negative, got {self.kinematic_weight}"
            raise ValueError(msg)
        if self.collision_weight < 0:
            msg = f"collision_weight must be non-negative, got {self.collision_weight}"
            raise ValueError(msg)
        validate_positive("collision_threshold", self.collision_threshold)
        if self.road_boundary_weight < 0:
            msg = f"road_boundary_weight must be non-negative, got {self.road_boundary_weight}"
            raise ValueError(msg)
        if self.offroad_chunk_size is not None and self.offroad_chunk_size <= 0:
            msg = f"offroad_chunk_size must be positive when set, got {self.offroad_chunk_size}"
            raise ValueError(msg)
        PhysicsScheduleType(self.schedule_type)
        validate_positive("initial_physics_weight", self.initial_physics_weight)
        validate_positive("final_physics_weight", self.final_physics_weight)
        validate_positive("transition_epochs", self.transition_epochs)
        if self.momentum_variation_weight < 0:
            msg = (
                f"momentum_variation_weight must be non-negative, "
                f"got {self.momentum_variation_weight}"
            )
            raise ValueError(msg)


class DiffAVPhysicsLoss:
    """Physics-informed loss combining kinematics, collision, and boundary terms.

    Combines :class:`BicycleModelConstraint`, a pairwise collision penalty, a
    signed-distance road-boundary penalty, and a soft momentum-variation
    regularizer into a single differentiable physics loss, scaled by opifex's
    :class:`AdaptiveWeightScheduler` for curriculum-style weighting during
    training.
    """

    def __init__(self, config: DiffAVPhysicsConfig | None = None) -> None:
        """Initialize the physics loss module.

        Args:
            config: Physics loss configuration. Uses defaults if ``None``.
        """
        self.config = config or DiffAVPhysicsConfig()
        self._bicycle = BicycleModelConstraint(BicycleModelConfig())

        # Adaptive weight scheduler (opifex) — the sole shared building block.
        self._scheduler: AdaptiveWeightScheduler | None = None
        if self.config.adaptive_weighting:
            self._scheduler = AdaptiveWeightScheduler(
                schedule_type=self.config.schedule_type,
                initial_physics_weight=self.config.initial_physics_weight,
                final_physics_weight=self.config.final_physics_weight,
                transition_epochs=self.config.transition_epochs,
            )

    def compute_loss(
        self,
        trajectories: jax.Array,
        epoch: jax.Array | int,
        road_edges: RoadEdges | None = None,
        valid_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute combined physics loss for generated trajectories.

        Args:
            trajectories: Generated trajectories,
                shape ``(num_agents, future_steps, 4)``
                with state ``[x, y, heading, velocity]``.
            epoch: Current training epoch (traced or static) for
                adaptive weighting.
            road_edges: Optional oriented road-edge polylines. If ``None``,
                boundary loss is zero.
            valid_mask: Optional per-step validity ``(num_agents,
                future_steps)``. When given, padded agents and out-of-horizon
                steps are excluded from the kinematic, collision, and boundary
                terms (masked averages over the valid entries). ``None``
                (default) averages over every entry.

        Returns:
            Tuple of (scalar loss, component dict) where the dict contains
            ``kinematic_loss``, ``collision_loss``, ``boundary_loss``,
            ``momentum_loss``, and ``total_loss``.
        """
        # Kinematic residual from bicycle model. A 0/1 float validity mask (as
        # the data pipeline emits) is accepted: the bicycle residual normalizes
        # it to bool, and the collision and boundary terms multiply by it.
        kinematic_loss = self._bicycle.compute_residuals(trajectories, valid_mask)

        # Collision penalty: pairwise soft penalty
        collision_loss = self._compute_collision_loss(trajectories, valid_mask)

        # Off-road boundary penalty
        boundary_loss = self._compute_boundary_loss(trajectories, road_edges, valid_mask)

        # Momentum conservation
        momentum_loss = self._compute_momentum_loss(trajectories)

        # Get physics weight for this epoch
        physics_weight = self.get_current_weight(epoch)

        # Compose weighted total
        total = physics_weight * (
            self.config.kinematic_weight * kinematic_loss
            + self.config.collision_weight * collision_loss
            + self.config.road_boundary_weight * boundary_loss
            + self.config.momentum_variation_weight * momentum_loss
        )

        components = {
            "kinematic_loss": kinematic_loss,
            "collision_loss": collision_loss,
            "boundary_loss": boundary_loss,
            "momentum_loss": momentum_loss,
            "total_loss": total,
        }
        return total, components

    def get_current_weight(self, epoch: jax.Array | int) -> jax.Array:
        """Get the adaptive physics weight for the given epoch.

        Args:
            epoch: Current training epoch (traced or static).

        Returns:
            Scalar weight value.
        """
        if self._scheduler is not None:
            # cast: opifex annotates get_weight(epoch: int) but converts to
            # a jax array internally, so traced epochs are runtime-safe.
            return self._scheduler.get_weight(cast(int, epoch))
        return jnp.array(1.0)

    def _compute_collision_loss(
        self,
        trajectories: jax.Array,
        valid_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute pairwise collision penalty.

        For each timestep, computes soft penalty for agent pairs closer
        than ``collision_threshold``: ``max(0, threshold - dist)^2``.

        Args:
            trajectories: Shape ``(num_agents, future_steps, 4)``.
            valid_mask: Optional per-step validity ``(num_agents,
                future_steps)``; a pair contributes at a step only when both
                agents are valid there.

        Returns:
            Scalar collision loss.
        """
        # Extract positions: (num_agents, future_steps, 2)
        positions = trajectories[:, :, :2]
        num_agents = positions.shape[0]

        if num_agents < 2:
            return jnp.array(0.0)

        # Pairwise distances: (num_agents, num_agents, future_steps)
        # positions[:, None] => (A, 1, T, 2), positions[None, :] => (1, A, T, 2)
        diff = positions[:, None, :, :] - positions[None, :, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

        # Soft penalty: max(0, threshold - dist)^2
        penalty = jnp.maximum(0.0, self.config.collision_threshold - dist) ** 2

        # Zero out self-pairs.
        self_mask = 1.0 - jnp.eye(num_agents)[:, :, None]
        penalty = penalty * self_mask

        if valid_mask is None:
            # Mean over all ordered pairs and timesteps.
            return jnp.sum(penalty) / (num_agents * (num_agents - 1) * positions.shape[1])

        # A pair contributes at step t only when both agents are valid there.
        agent_step = valid_mask.astype(penalty.dtype)
        pair_valid = agent_step[:, None, :] * agent_step[None, :, :] * self_mask
        return jnp.sum(penalty * pair_valid) / jnp.maximum(jnp.sum(pair_valid), 1.0)

    def _compute_boundary_loss(
        self,
        trajectories: jax.Array,
        road_edges: RoadEdges | None,
        valid_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute the off-road boundary violation penalty.

        Hinges on the signed distance to the oriented road edges: on-road
        positions contribute zero, off-road positions their squared
        distance to the boundary.

        Args:
            trajectories: Shape ``(num_agents, future_steps, 4)``.
            road_edges: Oriented road-edge polylines or ``None``.
            valid_mask: Optional per-step validity ``(num_agents,
                future_steps)``; invalid positions are excluded from the mean.

        Returns:
            Scalar boundary loss (zero if ``road_edges`` is ``None``).
        """
        if road_edges is None:
            return jnp.array(0.0)

        signed = signed_distances_to_road_edges(
            trajectories[:, :, :2], road_edges, chunk_size=self.config.offroad_chunk_size
        )
        penalty = jnp.maximum(signed, 0.0) ** 2
        if valid_mask is None:
            return jnp.mean(penalty)
        mask = valid_mask.reshape(-1).astype(penalty.dtype)
        return jnp.sum(penalty * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def _compute_momentum_loss(self, trajectories: jax.Array) -> jax.Array:
        """Compute the momentum-variation penalty.

        Penalizes the variance over time of the scene's total momentum — a soft
        smoothness prior discouraging large collective momentum swings. This is
        a regularizer, not a physical conservation law.

        Args:
            trajectories: Shape ``(num_agents, future_steps, 4)``.

        Returns:
            Scalar momentum-variation loss (zero when the weight is disabled).
        """
        if self.config.momentum_variation_weight <= 0:
            return jnp.array(0.0)

        # Momentum = velocity * heading_direction per agent
        velocity = trajectories[:, :, 3:4]  # (A, T, 1)
        heading = trajectories[:, :, 2:3]  # (A, T, 1)
        momentum = velocity * jnp.concatenate(
            [jnp.cos(heading), jnp.sin(heading)], axis=-1
        )  # (A, T, 2)

        # Total momentum per timestep: (T, 2)
        total_momentum = jnp.sum(momentum, axis=0)

        # Penalize variation over time in the scene's total momentum.
        return jnp.mean(jnp.var(total_momentum, axis=0))
