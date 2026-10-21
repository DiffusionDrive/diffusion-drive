"""Physics-informed training loop for trajectory diffusion models.

Composes substrax's optimizer creation, opifex's error recovery and calibrax's
timing infrastructure to implement a training coordinator that combines
diffusion loss with physics-informed penalties.

The core computation (:meth:`TrajectoryTrainer.compute_train_step`) is
JIT-compatible via ``nnx.jit``.  Wrap it for compiled performance::

    jit_step = nnx.jit(trainer.compute_train_step)
    total_loss, aux = jit_step(
        trainer.model, trainer.optimizer,
        trajectories, scene_context, key, epoch,
    )

Python-level operations (timing, checkpointing, logging) are provided
by :meth:`TrajectoryTrainer.train_step`, which wraps the core and
returns :class:`TrainingMetrics`.

For distributed training across multiple devices, use
:meth:`TrajectoryTrainer.train_step_distributed`, which wraps
:meth:`compute_train_step` with ``nnx.jit`` and runs it inside a
:func:`jax.set_mesh` context for SPMD execution.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from opifex.core.training.components.recovery import ErrorRecoveryManager
from substrax.optim import create_optimizer, current_learning_rate, OptimizerConfig

from diffav.core.config import validate_positive
from diffav.core.constants import SCENE_BACKBONE_ARCHITECTURE_VERSION
from diffav.core.geometry import RoadEdges
from diffav.core.training_utils import nan_safe_gradients
from diffav.models.checkpointing import (
    CheckpointConfig,
    DiffAVCheckpointManager,
    TrainingState,
)
from diffav.models.trajectory_diffusion import TrajectoryDiffusionModel
from diffav.physics.losses import DiffAVPhysicsConfig, DiffAVPhysicsLoss


logger = logging.getLogger(__name__)


def _carries_learning_rate(optimizer: nnx.Optimizer) -> bool:
    """Whether ``optimizer``'s state holds the learning rate its updates apply.

    substrax's transformation injects the rate; a caller-supplied optax transformation
    usually does not, and then the rate is unknown to the trainer.
    """
    try:
        current_learning_rate(optimizer)
    except ValueError:
        return False
    return True


def _default_optimizer_config() -> OptimizerConfig:
    """Create default optimizer config for trajectory training."""
    return OptimizerConfig(
        optimizer_type="adam",
        learning_rate=1e-4,
        gradient_clip_norm=1.0,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainerConfig:
    """Configuration for the trajectory training loop.

    Attributes:
        optimizer_config: substrax's optimizer specification; a learning-rate schedule is
            an ``optax.Schedule`` passed as its ``learning_rate``.
        num_epochs: Number of training epochs.
        physics_config: Optional physics loss configuration. If ``None``,
            no physics penalty is applied.
        checkpoint_config: Optional checkpoint configuration. If ``None``,
            no checkpoints are saved.
        architecture_version: Backbone architecture revision stamped into
            saved checkpoints and required to match on restore.
        log_interval: Log metrics every N steps.
        nan_detection: When True, NaN gradients are zeroed in-step and the
            opifex ErrorRecoveryManager enforces the stability policy after
            every step (loss-explosion check, stable-state snapshots,
            rollback recovery, hard failure after max retries). When False,
            gradients are applied untouched and no stability machinery runs.
        loss_explosion_threshold: Loss value above which training is
            considered unstable (checked by the recovery manager).
        gradient_explosion_threshold: Global gradient norm above which
            training is considered unstable. A sanity bound, not a clip —
            per-step clipping stays with the optimizer's ``gradient_clip_norm``
            chain. Raw gradient norms on metre-scale WOD data routinely
            exceed small bounds early in training.
        profile_flops: Measure per-step FLOPs once via calibrax's
            FlopsCounter on the first step and report the value in
            ``TrainingMetrics.flops_per_step``. The count is XLA's cost
            analysis of the lowered step; a step containing a custom call
            with no cost model raises ``FlopsUnavailableError``. Off by
            default — the measurement adds one extra trace of the step.
        physics_x0_annealing: Scale the physics penalty on the x̂₀
            reconstruction by the drawn timestep's ᾱ_t. The
            reconstruction is only trustworthy where ᾱ_t is high —
            ``predict_start_from_noise`` divides by ``sqrt(ᾱ_t)``, so at
            high noise levels x̂₀ (and any physics score of it) is
            dominated by amplified model error. The same anneal already
            governs test-time x̂₀ guidance in
            :meth:`TrajectoryDiffusionModel.p_sample_step`
            (``x̂₀ + η·ᾱ_t·∇R``, reconstruction-guidance practice). Off
            by default — bit-identical to the unannealed penalty.
    """

    optimizer_config: OptimizerConfig = field(
        default_factory=_default_optimizer_config,
    )
    num_epochs: int = 100
    physics_config: DiffAVPhysicsConfig | None = None
    checkpoint_config: CheckpointConfig | None = None
    log_interval: int = 50
    nan_detection: bool = True
    loss_explosion_threshold: float = 1e6
    gradient_explosion_threshold: float = 1e4
    profile_flops: bool = False
    physics_x0_annealing: bool = False
    architecture_version: int = SCENE_BACKBONE_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        validate_positive("num_epochs", self.num_epochs)
        validate_positive("log_interval", self.log_interval)
        validate_positive("loss_explosion_threshold", self.loss_explosion_threshold)
        validate_positive("gradient_explosion_threshold", self.gradient_explosion_threshold)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingMetrics:
    """Immutable snapshot of metrics from a single training step.

    Attributes:
        step: Training step number.
        epoch: Current epoch number.
        total_loss: Combined diffusion + physics loss.
        diffusion_loss: Diffusion noise prediction loss.
        physics_loss: Physics constraint loss (zero if disabled).
        physics_weight: Adaptive physics weight for this epoch.
        grad_norm: Global gradient norm before optimizer update.
        learning_rate: The learning rate the step's update applied, read from the
            optimizer's state (the schedule's value when the configured rate is a
            schedule), or NaN when a caller-supplied optimizer carries no rate.
        has_nan: Whether NaN was detected in loss or gradients.
        wall_clock_sec: Wall-clock time for this step in seconds.
        flops_per_step: Per-step FLOPs, XLA's cost analysis of the step as
            calibrax's ``FlopsCounter`` reads it, when
            ``TrainerConfig.profile_flops`` is enabled; 0.0 when profiling
            is off.
    """

    step: int
    epoch: int
    total_loss: float
    diffusion_loss: float
    physics_loss: float
    physics_weight: float
    grad_norm: float
    learning_rate: float
    has_nan: bool
    wall_clock_sec: float
    flops_per_step: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class _StableTrainingState:
    """Snapshot of a stable training state for instability rollback.

    Attributes:
        model_state: NNX state of the model at snapshot time.
        optimizer_state: NNX state of the optimizer at snapshot time.
        step: Trainer step counter at snapshot time.
        epoch: Trainer epoch counter at snapshot time.
        recovery_state: Mutable scratch dict the opifex recovery manager
            annotates during recovery (e.g. reduced learning rate).
    """

    model_state: Any
    optimizer_state: Any
    step: int
    epoch: int
    recovery_state: dict[str, Any] = field(default_factory=dict)


class TrajectoryTrainer:
    """Physics-informed trainer for trajectory diffusion models.

    Composes :class:`TrajectoryDiffusionModel` with substrax's
    :func:`create_optimizer`, opifex's :class:`ErrorRecoveryManager`, calibrax's
    :class:`TimingCollector`, and diffav's :class:`DiffAVPhysicsLoss`
    and :class:`DiffAVCheckpointManager` to implement a training loop
    that combines diffusion loss with physics-informed penalties.

    The core computation (:meth:`compute_train_step`) is JIT-compatible
    via ``nnx.jit``.  Wrap it for compiled performance::

        jit_step = nnx.jit(trainer.compute_train_step)
        total_loss, aux = jit_step(
            trainer.model, trainer.optimizer,
            trajectories, scene_context, key, epoch,
        )

    The trainer's physics config is captured in the bound method's
    closure and treated as a static compile-time constant by JAX;
    ``epoch`` is a traced argument, so the adaptive physics weight
    advances across epochs under a single compiled function.
    """

    def __init__(
        self,
        model: TrajectoryDiffusionModel,
        config: TrainerConfig,
        *,
        optimizer: optax.GradientTransformation | None = None,
    ) -> None:
        """Initialize the trajectory trainer.

        Args:
            model: Trajectory diffusion model to train.
            config: Trainer configuration.
            optimizer: Optional custom optax optimizer. If ``None``,
                creates one from ``config.optimizer_config`` via substrax.
        """
        self.config = config
        self.model = model

        # Build the optimizer from substrax's specification, or wrap a custom one
        self.optimizer = (
            create_optimizer(model, config.optimizer_config)
            if optimizer is None
            else nnx.Optimizer(model, optimizer, wrt=nnx.Param)
        )
        # substrax's transformation injects its learning rate into the optimizer state,
        # so the step reads the applied rate on device; a caller-supplied optax
        # transformation carries none, and its rate is reported as NaN.
        self._reports_learning_rate = _carries_learning_rate(self.optimizer)

        # Physics loss (optional)
        self._physics: DiffAVPhysicsLoss | None = None
        if config.physics_config is not None:
            self._physics = DiffAVPhysicsLoss(config.physics_config)

        # Checkpoint manager (optional)
        self._checkpoint: DiffAVCheckpointManager | None = None
        if config.checkpoint_config is not None:
            self._checkpoint = DiffAVCheckpointManager(config.checkpoint_config)

        # Error recovery from opifex for NaN/instability detection
        self._recovery: ErrorRecoveryManager | None = None
        if config.nan_detection:
            self._recovery = ErrorRecoveryManager(
                {
                    "gradient_clip_threshold": config.gradient_explosion_threshold,
                    "loss_explosion_threshold": config.loss_explosion_threshold,
                }
            )

        # Compiled distributed step, built once (a fresh nnx.jit wrapper
        # per call would retrace and recompile on every step)
        self._jitted_step = nnx.jit(self.compute_train_step)

        # Lazily measured per-step FLOPs (profile_flops)
        self._flops_per_step: float | None = None

        # Internal state
        self._step = 0
        self._epoch = 0

    @property
    def physics_enabled(self) -> bool:
        """Whether a physics loss is applied during training."""
        return self._physics is not None

    @property
    def recovery_manager(self) -> ErrorRecoveryManager | None:
        """Access the opifex ErrorRecoveryManager (if enabled)."""
        return self._recovery

    @property
    def checkpoint_manager(self) -> DiffAVCheckpointManager | None:
        """Access the checkpoint manager (if configured)."""
        return self._checkpoint

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch number.

        Args:
            epoch: Epoch number to set.
        """
        self._epoch = epoch

    def restore_latest(self) -> TrainingState | None:
        """Resume model, optimizer, and step/epoch counters from the latest checkpoint.

        Restores in place and advances ``_step``/``_epoch`` to the values
        recorded at save time so training continues where it stopped.

        Returns:
            The restored :class:`TrainingState`, or ``None`` when no
            checkpoint manager is configured or no checkpoint exists yet.

        Raises:
            CheckpointCorruptError: If a checkpoint exists but cannot be
                restored.
        """
        if self._checkpoint is None:
            return None
        state = self._checkpoint.restore_latest(
            self.model,
            optimizer=self.optimizer,
            expected_architecture_version=self.config.architecture_version,
        )
        if state is not None:
            self._step = state.step
            self._epoch = state.epoch
        return state

    def compute_train_step(
        self,
        model: TrajectoryDiffusionModel,
        optimizer: nnx.Optimizer,
        trajectories: jax.Array,
        scene_context: jax.Array,
        key: jax.Array,
        epoch: jax.Array | int,
        road_edges: RoadEdges | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """JIT-compatible core of a training step.

        Computes combined diffusion + physics loss, gradients, NaN-safe
        optimizer update, and auxiliary metrics.  This method follows the
        artifex pattern: ``model`` and ``optimizer`` are explicit arguments
        (NNX-traced by ``nnx.jit``); the physics config is captured in the
        bound method's closure (static), while ``epoch`` is a traced
        argument so the adaptive physics-weight schedule advances across
        epochs without retracing.

        The physics penalty is applied to the model's clean-trajectory
        reconstruction (x̂₀) from
        :meth:`TrajectoryDiffusionModel.compute_loss_and_prediction`, so
        its gradients shape the model rather than scoring the fixed
        ground-truth batch.

        Wrap with ``nnx.jit`` for compiled performance::

            jit_step = nnx.jit(trainer.compute_train_step)
            total_loss, aux = jit_step(
                trainer.model, trainer.optimizer,
                trajectories, scene_context, key, epoch,
            )

        When called without JIT, the method works identically but without
        XLA compilation.

        Args:
            model: Trajectory diffusion model (NNX-traced).
            optimizer: Flax NNX optimizer (NNX-traced).
            trajectories: Ground-truth trajectories,
                shape ``(num_agents, future_steps, state_dim)``, or a
                stacked scene batch ``(batch, num_agents, future_steps,
                state_dim)`` whose per-scene losses are averaged.
            scene_context: Per-agent scene context, shape
                ``(num_agents, context_dim)`` — one row per agent, batched
                ``(batch, num_agents, context_dim)`` when trajectories are.
            key: JAX random key for diffusion sampling.
            epoch: Current training epoch (traced) for adaptive weighting.
            road_edges: Optional oriented road edges for the
                off-road boundary penalty.

        Returns:
            Tuple of ``(total_loss, aux_dict)`` where ``aux_dict``
            contains ``diffusion_loss``, ``physics_loss``,
            ``physics_weight``, ``grad_norm``, ``has_nan`` and, when the optimizer
            carries one, the applied ``learning_rate``.
        """
        # Physics config comes from self (static in JIT); epoch is traced.
        physics = self._physics

        def loss_fn(
            model: TrajectoryDiffusionModel,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            """Compute combined diffusion + physics loss for differentiation."""

            def scene_losses(
                scene_traj: jax.Array,
                scene_ctx: jax.Array,
                scene_key: jax.Array,
                timestep_stratum: tuple[jax.Array, int] | None,
            ) -> tuple[jax.Array, jax.Array]:
                outputs = model.compute_loss_outputs(
                    scene_traj,
                    scene_ctx,
                    key=scene_key,
                    timestep_stratum=timestep_stratum,
                )
                if physics is None:
                    return outputs.loss, jnp.array(0.0)
                physics_total, _ = physics.compute_loss(outputs.prediction, epoch, road_edges)
                if self.config.physics_x0_annealing:
                    # Trust the physics score of x̂₀ in proportion to ᾱ_t —
                    # the same anneal the sampling path applies to x̂₀
                    # guidance (config is static under JIT, so this is a
                    # compile-time branch).
                    physics_total = outputs.alpha_bar_t * physics_total
                return outputs.loss, physics_total

            # Batched (B, A, T, D) input: mean per-scene losses, physics
            # evaluated within each scene (reference practice trains on
            # ~100-scene batches; batch-mean smooths per-scene variance).
            # Each scene draws its diffusion timestep from its own schedule
            # stratum ``i`` of ``B``, so one optimizer step always spans
            # early, middle, and late noise levels — the dominant source of
            # step-to-step loss jitter otherwise (VDM appendix I.1).
            if trajectories.ndim == 4:
                batch_size = trajectories.shape[0]
                scene_keys = jax.random.split(key, batch_size)
                strata = jnp.arange(batch_size)
                diffusion_losses, physics_losses = jax.vmap(
                    scene_losses, in_axes=(0, 0, 0, (0, None))
                )(trajectories, scene_context, scene_keys, (strata, batch_size))
                diffusion_loss = jnp.mean(diffusion_losses)
                physics_loss = jnp.mean(physics_losses)
            else:
                # Single scene: no strata to spread, plain uniform draw.
                diffusion_loss, physics_loss = scene_losses(trajectories, scene_context, key, None)

            physics_weight = (
                physics.get_current_weight(epoch) if physics is not None else jnp.array(0.0)
            )
            total = diffusion_loss + physics_loss

            aux = {
                "diffusion_loss": diffusion_loss,
                "physics_loss": physics_loss,
                "physics_weight": physics_weight,
            }
            return total, aux

        (total_loss, aux), grads = nnx.value_and_grad(
            loss_fn,
            has_aux=True,
        )(model)

        # Compute gradient norm (JIT-compatible)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        squared_norms = [jnp.sum(g**2) for g in grad_leaves if hasattr(g, "shape")]
        grad_norm = jnp.sqrt(sum(squared_norms)) if squared_norms else jnp.array(0.0)

        safe_grads, has_nan = nan_safe_gradients(total_loss, grads)
        # Honest gating: zeroing only happens when nan_detection is on
        # (self.config is static under JIT, so this is a compile-time branch)
        grads_to_apply = safe_grads if self.config.nan_detection else grads

        optimizer.update(model, grads_to_apply)

        aux["grad_norm"] = grad_norm
        aux["has_nan"] = has_nan
        if self._reports_learning_rate:
            # Static under JIT: read with the loss, no extra transfer per step.
            aux["learning_rate"] = current_learning_rate(optimizer)

        return total_loss, aux

    def _snapshot(self) -> _StableTrainingState:
        """Capture the current model/optimizer state for rollback.

        ``nnx.state`` returns a live view that tracks later variable
        updates; tree-mapping detaches it into a fixed snapshot (arrays
        are immutable, so no device copy is needed).
        """
        detach = lambda leaf: leaf  # noqa: E731 - identity map detaches the live view
        return _StableTrainingState(
            model_state=jax.tree.map(detach, nnx.state(self.model)),
            optimizer_state=jax.tree.map(detach, nnx.state(self.optimizer)),
            step=self._step,
            epoch=self._epoch,
        )

    def _restore(self, snapshot: _StableTrainingState) -> None:
        """Restore model, optimizer, and counters from a snapshot.

        Args:
            snapshot: State captured by :meth:`_snapshot`.
        """
        nnx.update(self.model, snapshot.model_state)
        nnx.update(self.optimizer, snapshot.optimizer_state)
        self._step = snapshot.step
        self._epoch = snapshot.epoch

    def _enforce_stability(self, total_loss: float, grad_norm: float) -> None:
        """Run the opifex stability policy after a step.

        Checks loss explosion, gradient explosion, and NaN via
        ``ErrorRecoveryManager.check_training_stability`` (the gradient
        norm is passed as a single-leaf pytree whose norm equals the true
        global norm; in-step clipping is already enforced JIT-safely by
        the optax ``gradient_clip`` chain). Stable steps refresh the
        manager's stable snapshot; unstable steps roll back to it, and
        persistent instability raises ``RuntimeError`` from the manager.

        Args:
            total_loss: Materialized scalar loss of the step.
            grad_norm: Materialized global gradient norm of the step.
        """
        if self._recovery is None:
            return
        is_stable, issue = self._recovery.check_training_stability(
            total_loss,
            [jnp.asarray(grad_norm)],
            None,
        )
        if is_stable:
            self._recovery.update_stable_state(self._snapshot())
            return
        logger.warning(
            "Training instability '%s' at step %d; rolling back to last stable state",
            issue,
            self._step,
        )
        recovered = self._recovery.recover_from_instability(
            issue if issue is not None else "unknown",
            self._snapshot(),
        )
        self._restore(recovered)

    def _measure_flops_if_due(
        self,
        trajectories: jax.Array,
        scene_context: jax.Array,
        key: jax.Array,
        road_edges: RoadEdges | None,
    ) -> float:
        """Measure per-step FLOPs once via calibrax when profiling is enabled.

        Args:
            trajectories: Example trajectories for tracing.
            scene_context: Example scene context for tracing.
            key: Example random key for tracing.
            road_edges: Optional road edges matching the step call.

        Returns:
            The measured (cached) FLOP count, or 0.0 when profiling is off.
        """
        if not self.config.profile_flops:
            return 0.0
        if self._flops_per_step is None:
            from calibrax.profiling import FlopsCounter

            graphdef, state = nnx.split((self.model, self.optimizer))

            def pure_step(
                module_state: Any,
                traj: jax.Array,
                ctx: jax.Array,
                step_key: jax.Array,
            ) -> jax.Array:
                model, optimizer = nnx.merge(graphdef, module_state)
                loss, _ = self.compute_train_step(
                    model, optimizer, traj, ctx, step_key, 0, road_edges
                )
                return loss

            result = FlopsCounter().count(pure_step, state, trajectories, scene_context, key)
            self._flops_per_step = float(result.total_flops)
        return self._flops_per_step

    def train_step(
        self,
        trajectories: jax.Array,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        road_edges: RoadEdges | None = None,
    ) -> TrainingMetrics:
        """Execute a single training step with timing and logging.

        Runs the pre-built ``nnx.jit``-compiled step (compiled once per
        input shape) and wraps it with Python-level operations (wall-clock
        timing, NaN logging, checkpointing) that cannot run inside JIT.

        For custom compilation control, wrap :meth:`compute_train_step`
        directly.

        Args:
            trajectories: Ground-truth trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            scene_context: Per-agent scene context, shape
                ``(num_agents, context_dim)`` — one row per agent.
            key: JAX random key for diffusion sampling.
            road_edges: Optional oriented road edges for the
                off-road boundary penalty.

        Returns:
            TrainingMetrics for this step.
        """
        start = time.perf_counter()

        total_loss, aux = self._jitted_step(
            self.model,
            self.optimizer,
            trajectories,
            scene_context,
            key,
            self._epoch,
            road_edges,
        )

        flops = self._measure_flops_if_due(trajectories, scene_context, key, road_edges)
        return self._finalize_step(total_loss, aux, start, flops)

    def _finalize_step(
        self,
        total_loss: jax.Array,
        aux: dict[str, jax.Array],
        start: float,
        flops: float,
    ) -> TrainingMetrics:
        """Shared Python-level tail of a training step.

        Materializes step outputs, logs NaN events, checkpoints if due,
        enforces the stability policy, and builds the metrics record.

        Args:
            total_loss: Scalar loss from the step computation.
            aux: Auxiliary metrics dict from the step computation.
            start: ``time.perf_counter()`` value at step start.
            flops: Per-step FLOPs to report (0.0 when profiling is off).

        Returns:
            TrainingMetrics for this step.
        """
        has_nan = bool(aux["has_nan"])
        loss_value = float(total_loss)
        grad_norm = float(aux["grad_norm"])

        if has_nan:
            logger.warning(
                "NaN detected at step %d%s",
                self._step,
                ", grads zeroed" if self.config.nan_detection else "",
            )

        step = self._step
        self._step += 1

        # Checkpoint if due (Python-level I/O, outside JIT). A checkpoint is labelled with
        # the updates it holds, so a resumed run continues at the next step and never
        # saves over the one it restored.
        if self._checkpoint is not None:
            self._checkpoint.save_if_due(
                self.model,
                self._step,
                loss_value,
                optimizer=self.optimizer,
                epoch=self._epoch,
                architecture_version=self.config.architecture_version,
            )

        wall_clock = time.perf_counter() - start

        metrics = TrainingMetrics(
            step=step,
            epoch=self._epoch,
            total_loss=loss_value,
            diffusion_loss=float(aux["diffusion_loss"]),
            physics_loss=float(aux["physics_loss"]),
            physics_weight=float(aux["physics_weight"]),
            grad_norm=grad_norm,
            learning_rate=float(aux["learning_rate"]) if "learning_rate" in aux else math.nan,
            has_nan=has_nan,
            wall_clock_sec=wall_clock,
            flops_per_step=flops,
        )
        self._enforce_stability(loss_value, grad_norm)
        return metrics

    def train_step_distributed(
        self,
        trajectories: jax.Array,
        scene_context: jax.Array,
        *,
        key: jax.Array,
        mesh: jax.sharding.Mesh,
        road_edges: RoadEdges | None = None,
    ) -> TrainingMetrics:
        """Execute a distributed training step using SPMD via ``nnx.jit``.

        Compiles :meth:`compute_train_step` with ``nnx.jit`` and executes it
        inside a :func:`jax.set_mesh` context.  XLA's SPMD partitioner
        automatically replicates parameters and all-reduces gradients across
        the mesh — functionally identical to :meth:`train_step` on a
        single-device mesh.

        Reports per-step FLOPs in ``metrics.flops_per_step`` when
        ``TrainerConfig.profile_flops`` is enabled (measured once via
        calibrax and cached).

        Args:
            trajectories: Ground-truth trajectories,
                shape ``(num_agents, future_steps, state_dim)``.
            scene_context: Per-agent scene context, shape
                ``(num_agents, context_dim)`` — one row per agent.
            key: JAX random key for diffusion sampling.
            mesh: JAX device mesh, for data parallelism
                ``substrax.mesh.DeviceMeshManager.create_device_mesh({"data": n})``.
            road_edges: Optional oriented road edges for the
                off-road boundary penalty.

        Returns:
            Training metrics for this step, including FLOP count in
            ``metrics.flops_per_step`` when profiling is enabled.
        """
        start = time.perf_counter()

        # The compiled step is built once in __init__; XLA handles data
        # placement and AllReduce automatically on multi-device meshes.
        with jax.set_mesh(mesh):
            total_loss, aux = self._jitted_step(
                self.model,
                self.optimizer,
                trajectories,
                scene_context,
                key,
                self._epoch,
                road_edges,
            )

        flops = self._measure_flops_if_due(trajectories, scene_context, key, road_edges)
        return self._finalize_step(total_loss, aux, start, flops)

    def train_epoch(
        self,
        data_iterator: Iterator[tuple[jax.Array, jax.Array]],
        *,
        key: jax.Array,
        road_edges: RoadEdges | None = None,
    ) -> list[TrainingMetrics]:
        """Train for one epoch over a data iterator.

        Each element from the iterator should be a tuple of
        ``(trajectories, scene_context)``.

        Args:
            data_iterator: Iterator yielding (trajectories, scene_context) pairs.
            key: JAX random key (split per step).
            road_edges: Optional oriented road edges for the off-road
                boundary penalty.

        Returns:
            List of TrainingMetrics, one per batch.
        """
        metrics_list: list[TrainingMetrics] = []

        for i, (trajectories, scene_context) in enumerate(data_iterator):
            step_key = jax.random.fold_in(key, i)
            metrics = self.train_step(
                trajectories,
                scene_context,
                key=step_key,
                road_edges=road_edges,
            )
            metrics_list.append(metrics)

            if self.config.log_interval > 0 and self._step % self.config.log_interval == 0:
                logger.info(
                    "Step %d | loss=%.6f | diff=%.6f | phys=%.6f | grad=%.4f",
                    metrics.step,
                    metrics.total_loss,
                    metrics.diffusion_loss,
                    metrics.physics_loss,
                    metrics.grad_norm,
                )

        self._epoch += 1
        return metrics_list

    def train(
        self,
        data_fn: Callable[[], Iterator[tuple[jax.Array, jax.Array]]],
        *,
        key: jax.Array,
        road_edges: RoadEdges | None = None,
    ) -> list[TrainingMetrics]:
        """Run the full training loop for ``num_epochs`` epochs.

        Args:
            data_fn: Callable that returns a fresh data iterator for each
                epoch. Each iterator yields ``(trajectories, scene_context)``.
            key: JAX random key.
            road_edges: Optional oriented road edges for the off-road
                boundary penalty.

        Returns:
            Flat list of TrainingMetrics across all epochs.
        """
        all_metrics: list[TrainingMetrics] = []

        for epoch in range(self.config.num_epochs):
            epoch_key = jax.random.fold_in(key, epoch)
            data_iterator = data_fn()
            epoch_metrics = self.train_epoch(
                data_iterator,
                key=epoch_key,
                road_edges=road_edges,
            )
            all_metrics.extend(epoch_metrics)
            logger.info(
                "Epoch %d complete | steps=%d | last_loss=%.6f",
                epoch,
                len(epoch_metrics),
                epoch_metrics[-1].total_loss if epoch_metrics else float("nan"),
            )

        return all_metrics

    def close(self) -> None:
        """Release resources held by the trainer."""
        if self._checkpoint is not None:
            self._checkpoint.close()
