"""Bicycle model kinematic constraints for trajectory validation.

Provides differentiable, JIT-compatible kinematic constraint functions
built on the bicycle vehicle model. These constraints enforce physically
plausible trajectories by penalizing deviations from bicycle kinematics,
acceleration limits, steering limits, and velocity bounds.

The module includes:
- BicycleModelConstraint: residual and validation against bicycle kinematics.
- AckermanSteeringConstraint: minimum turning radius enforcement via Ackerman geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffav.core.config import validate_positive
from diffav.core.constants import (
    KINEMATIC_POSITION_RESIDUAL_TOLERANCE_M2,
    WOD_STEP_DURATION_SECONDS,
)
from diffav.core.geometry import wrap_angle


@dataclass(frozen=True, slots=True, kw_only=True)
class BicycleModelConfig:
    """Configuration for bicycle model kinematic constraints.

    Attributes:
        wheelbase: Distance between front and rear axles in metres.
        dt: Simulation timestep in seconds.
        max_acceleration: Maximum allowed acceleration magnitude in m/s^2.
        max_steering_angle: Maximum allowed steering angle in radians.
        max_velocity: Maximum allowed velocity in m/s.
    """

    wheelbase: float = 2.7
    dt: float = WOD_STEP_DURATION_SECONDS
    max_acceleration: float = 8.0
    max_steering_angle: float = 0.7
    max_velocity: float = 40.0

    def __post_init__(self) -> None:
        """Validate that all fields are strictly positive."""
        validate_positive("wheelbase", self.wheelbase)
        validate_positive("dt", self.dt)
        validate_positive("max_acceleration", self.max_acceleration)
        validate_positive("max_steering_angle", self.max_steering_angle)
        validate_positive("max_velocity", self.max_velocity)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, kw_only=True)
class KinematicValidationResult:
    """Detailed result of kinematic constraint validation.

    Registered as a JAX pytree so :meth:`BicycleModelConstraint.validate`
    traces under ``jit``. Each component is reported in its native unit;
    there is deliberately no aggregate across units.

    Attributes:
        is_valid: Scalar boolean array — True when every per-constraint
            check passes for every agent.
        position_residual: Per-agent mean squared per-step displacement
            deviation from the bicycle model (m^2).
        heading_residual: Per-agent mean squared heading change beyond
            the steering-feasible rate (rad^2).
        acceleration_violation: Per-agent mean acceleration excess (m/s^2).
        steering_violation: Per-agent mean steering-angle excess (rad).
        velocity_violation: Per-agent mean velocity excess (m/s).
    """

    is_valid: jax.Array
    position_residual: jax.Array
    heading_residual: jax.Array
    acceleration_violation: jax.Array
    steering_violation: jax.Array
    velocity_violation: jax.Array


def _compute_raw_residuals(
    trajectories: jax.Array,
    config: BicycleModelConfig,
) -> tuple[jax.Array, jax.Array]:
    """Compute per-agent position and heading residuals.

    The position residual is the squared error between finite-difference
    displacements and the bicycle-model prediction. The heading residual
    is steering-derived: any heading change achievable within
    ``max_steering_angle`` at the agent's speed is legitimate turning and
    incurs zero residual; only the excess beyond the feasible rate
    ``|v| / wheelbase * tan(max_steering_angle) * dt`` is penalized.

    Args:
        trajectories: Agent trajectories with shape
            ``(num_agents, future_steps, 4)`` where the state vector
            is ``[x, y, heading, velocity]``.
        config: Bicycle model parameters.

    Returns:
        Tuple of ``(position_sq_error, heading_sq_error)`` each with
        shape ``(num_agents, future_steps - 1)``.
    """
    x = trajectories[:, :, 0]
    y = trajectories[:, :, 1]
    heading = trajectories[:, :, 2]
    velocity = trajectories[:, :, 3]

    dx_actual = jnp.diff(x, axis=1)
    dy_actual = jnp.diff(y, axis=1)

    dx_model = velocity[:, :-1] * jnp.cos(heading[:, :-1]) * config.dt
    dy_model = velocity[:, :-1] * jnp.sin(heading[:, :-1]) * config.dt

    position_sq = (dx_actual - dx_model) ** 2 + (dy_actual - dy_model) ** 2

    dheading = jnp.diff(heading, axis=1)
    dheading_wrapped = jnp.arctan2(jnp.sin(dheading), jnp.cos(dheading))
    max_feasible_dheading = (
        jnp.abs(velocity[:, :-1])
        * config.dt
        / config.wheelbase
        * math.tan(config.max_steering_angle)
    )
    heading_excess = jnp.maximum(0.0, jnp.abs(dheading_wrapped) - max_feasible_dheading)
    heading_sq = heading_excess**2

    return position_sq, heading_sq


class BicycleModelConstraint:
    """Bicycle model kinematic constraint for trajectory evaluation.

    Computes residuals between observed trajectory finite differences and
    the predictions of a bicycle kinematic model. Supports both a scalar
    residual for loss computation and detailed per-agent validation.
    """

    def __init__(self, config: BicycleModelConfig | None = None) -> None:
        """Initialize with optional configuration.

        Args:
            config: Bicycle model parameters. Uses defaults if None.
        """
        self.config = config if config is not None else BicycleModelConfig()

    def compute_residuals(
        self,
        trajectories: jax.Array,
        valid_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute scalar kinematic residual for a batch of trajectories.

        Derives controls from finite differences between consecutive states
        and measures deviation from the bicycle kinematic model.

        Args:
            trajectories: Agent trajectories with shape
                ``(num_agents, future_steps, 4)`` where the state vector
                is ``[x, y, heading, velocity]``.
            valid_mask: Optional per-step validity ``(num_agents,
                future_steps)``. Each residual lives on a transition
                (step ``i`` to ``i+1``) and counts only when both endpoints
                are valid, so padded agents and out-of-horizon steps
                contribute nothing to the masked mean. ``None`` (default)
                averages over every transition.

        Returns:
            Scalar residual (sum of position and heading residuals).
        """
        position_sq, heading_sq = _compute_raw_residuals(trajectories, self.config)
        if valid_mask is None:
            return jnp.mean(position_sq) + jnp.mean(heading_sq)
        # A transition is valid only when both endpoints are; the boolean AND
        # requires a boolean mask, so accept the 0/1 float mask the data
        # pipeline emits.
        valid = valid_mask.astype(bool)
        transition_valid = (valid[:, :-1] & valid[:, 1:]).astype(position_sq.dtype)
        denom = jnp.maximum(jnp.sum(transition_valid), 1.0)
        weighted = jnp.sum(position_sq * transition_valid) + jnp.sum(heading_sq * transition_valid)
        return weighted / denom

    def validate(self, trajectories: jax.Array) -> KinematicValidationResult:
        """Perform detailed kinematic validation on trajectories.

        Computes per-agent residuals and constraint violations for
        acceleration, steering angle, and velocity limits. Validity is
        decided per constraint in its native unit — the hinge violations
        must be exactly zero (within limits) and the position residual
        below the bicycle-model consistency tolerance — and traces under
        ``jit`` (``is_valid`` is a scalar boolean array).

        Args:
            trajectories: Agent trajectories with shape
                ``(num_agents, future_steps, 4)`` where the state vector
                is ``[x, y, heading, velocity]``.

        Returns:
            Detailed validation result with per-agent violations.
        """
        cfg = self.config
        dt = cfg.dt

        position_sq, heading_sq = _compute_raw_residuals(trajectories, cfg)
        position_residual = jnp.mean(position_sq, axis=1)
        heading_residual = jnp.mean(heading_sq, axis=1)

        heading = trajectories[:, :, 2]
        velocity = trajectories[:, :, 3]

        # --- Acceleration violation ---
        dvelocity = jnp.diff(velocity, axis=1)
        acceleration = dvelocity / dt
        acceleration_violation = jnp.mean(
            jnp.maximum(0.0, jnp.abs(acceleration) - cfg.max_acceleration),
            axis=1,
        )

        # --- Steering violation ---
        dheading_wrapped = jnp.arctan2(
            jnp.sin(jnp.diff(heading, axis=1)),
            jnp.cos(jnp.diff(heading, axis=1)),
        )
        steering_angle = jnp.arctan(
            dheading_wrapped * cfg.wheelbase / (velocity[:, :-1] * dt + 1e-8)
        )
        steering_violation = jnp.mean(
            jnp.maximum(0.0, jnp.abs(steering_angle) - cfg.max_steering_angle),
            axis=1,
        )

        # --- Velocity violation ---
        velocity_violation = jnp.mean(
            jnp.maximum(0.0, velocity - cfg.max_velocity),
            axis=1,
        )

        # Per-constraint validity in native units: the hinge terms are
        # exactly zero when the trajectory stays within its limits.
        is_valid = (
            jnp.all(position_residual < KINEMATIC_POSITION_RESIDUAL_TOLERANCE_M2)
            & jnp.all(heading_residual == 0.0)
            & jnp.all(acceleration_violation == 0.0)
            & jnp.all(steering_violation == 0.0)
            & jnp.all(velocity_violation == 0.0)
        )

        return KinematicValidationResult(
            is_valid=is_valid,
            position_residual=position_residual,
            heading_residual=heading_residual,
            acceleration_violation=acceleration_violation,
            steering_violation=steering_violation,
            velocity_violation=velocity_violation,
        )


class AckermanSteeringConstraint:
    """Ackerman steering geometry constraint.

    Enforces minimum turning radius constraints derived from Ackerman
    steering geometry and the bicycle model wheelbase.
    """

    def __init__(self, config: BicycleModelConfig | None = None) -> None:
        """Initialize with optional configuration.

        Args:
            config: Bicycle model parameters. Uses defaults if None.
        """
        self.config = config if config is not None else BicycleModelConfig()

    def compute_min_turning_radius(self) -> float:
        """Compute the minimum turning radius from Ackerman geometry.

        Returns:
            Minimum turning radius in metres.
        """
        return self.config.wheelbase / math.tan(self.config.max_steering_angle)

    def compute_violation(self, trajectories: jax.Array) -> jax.Array:
        """Compute Ackerman steering violation for trajectories.

        Estimates curvature from consecutive displacement vectors and
        penalizes turning radii smaller than the Ackerman minimum.

        Args:
            trajectories: Agent trajectories with shape
                ``(num_agents, future_steps, 4)`` where the state vector
                is ``[x, y, heading, velocity]``.

        Returns:
            Scalar violation (mean excess curvature penalty).
        """
        positions = trajectories[:, :, :2]

        # Displacement vectors between consecutive positions
        dp = jnp.diff(positions, axis=1)
        dp1 = dp[:, :-1, :]  # (num_agents, steps-2, 2)
        dp2 = dp[:, 1:, :]  # (num_agents, steps-2, 2)

        # 2D cross product magnitude: |dp1 x dp2|
        cross = jnp.abs(dp1[:, :, 0] * dp2[:, :, 1] - dp1[:, :, 1] * dp2[:, :, 0])

        # |dp1|^3
        dp1_norm = jnp.sqrt(dp1[:, :, 0] ** 2 + dp1[:, :, 1] ** 2)
        dp1_norm_cubed = dp1_norm**3

        # Curvature: kappa = |cross(dp1, dp2)| / |dp1|^3
        kappa = cross / (dp1_norm_cubed + 1e-8)

        # Turning radius from curvature
        min_radius = 1.0 / (kappa + 1e-8)

        # Violation: radius smaller than Ackerman minimum
        min_turning_radius = self.compute_min_turning_radius()
        violation = jnp.mean(jnp.maximum(0.0, min_turning_radius - min_radius))

        return violation


# ---------------------------------------------------------------------------
# Central-difference kinematic features (official WOSAC reference mirror)
# ---------------------------------------------------------------------------


def central_diff(values: jax.Array, pad_value: float) -> jax.Array:
    """Central difference ``(f(x+h) - f(x-h)) / 2`` along the last axis.

    The two boundary steps cannot be estimated, so the result is prepended
    and appended with ``pad_value`` to keep the input shape.

    Args:
        values: Float array of shape ``(..., num_steps)``.
        pad_value: Value used for the two boundary steps.

    Returns:
        Central differences with shape ``(..., num_steps)``.
    """
    pad = jnp.full((*values.shape[:-1], 1), pad_value, dtype=values.dtype)
    diff = (values[..., 2:] - values[..., :-2]) / 2.0
    return jnp.concatenate([pad, diff, pad], axis=-1)


def central_logical_and(valid: jax.Array, pad_value: bool) -> jax.Array:
    """Central ``logical_and`` along the last axis.

    Element ``i`` of the result is True only when elements ``i-1`` and
    ``i+1`` of the input are both True — the validity counterpart of
    :func:`central_diff`.

    Args:
        valid: Boolean array of shape ``(..., num_steps)``.
        pad_value: Value used for the two boundary steps.

    Returns:
        Boolean array with shape ``(..., num_steps)``.
    """
    pad = jnp.full((*valid.shape[:-1], 1), pad_value, dtype=bool)
    both = valid[..., 2:] & valid[..., :-2]
    return jnp.concatenate([pad, both, pad], axis=-1)


def compute_kinematic_features(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    heading: jax.Array,
    seconds_per_step: float = WOD_STEP_DURATION_SECONDS,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Speeds and accelerations via central differences (reference mirror).

    Mirrors the official WOSAC ``compute_kinematic_features``: all steps are
    assumed valid (filter with :func:`compute_kinematic_validity`); speeds
    carry one NaN pad at each boundary and accelerations two. Heading deltas
    are scaled by two before wrapping and halved after, so the two-step
    central difference selects the acute one-step solution (bounding the
    measurable rate at pi/2 rad per step).

    Args:
        x: X coordinates, shape ``(..., num_steps)``.
        y: Y coordinates, shape ``(..., num_steps)``.
        z: Z coordinates, shape ``(..., num_steps)``.
        heading: Headings in radians, shape ``(..., num_steps)``.
        seconds_per_step: Duration of one step in seconds.

    Returns:
        Tuple of ``(linear_speed, linear_acceleration, angular_speed,
        angular_acceleration)``, each of shape ``(..., num_steps)``.
    """
    dpos = central_diff(jnp.stack([x, y, z], axis=0), pad_value=jnp.nan)
    linear_speed = jnp.linalg.norm(dpos, axis=0) / seconds_per_step
    linear_accel = central_diff(linear_speed, pad_value=jnp.nan) / seconds_per_step

    dh_step = wrap_angle(central_diff(heading, pad_value=jnp.nan) * 2.0) / 2.0
    angular_speed = dh_step / seconds_per_step
    d2h_step = wrap_angle(central_diff(dh_step, pad_value=jnp.nan) * 2.0) / 2.0
    angular_accel = d2h_step / (seconds_per_step**2)
    return linear_speed, linear_accel, angular_speed, angular_accel


def compute_kinematic_validity(valid: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Validity masks matching :func:`compute_kinematic_features`.

    A speed estimate is valid when both neighbouring steps are valid; an
    acceleration estimate applies the same rule to the speed validity.

    Args:
        valid: Boolean validity, shape ``(..., num_steps)``.

    Returns:
        Tuple of ``(speed_validity, acceleration_validity)``.
    """
    speed_validity = central_logical_and(valid, pad_value=False)
    acceleration_validity = central_logical_and(speed_validity, pad_value=False)
    return speed_validity, acceleration_validity
