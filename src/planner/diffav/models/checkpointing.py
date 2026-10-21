"""Checkpoint management for trajectory model training.

Composes substrax's :class:`OrbaxCheckpointStore` (checkpoint format 3: named items beside a
JSON record) with diffav-specific training state. The model and, when given, the optimizer
are the ``model`` and ``optimizer`` items; the loss and metrics, the epoch, diffav as the
producer and the backbone architecture version go into the record, so training can resume
exactly where it stopped and a restore can refuse incompatible weights before loading any.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Self, TYPE_CHECKING

from flax import nnx
from substrax.checkpoint import (
    CheckpointMetadata,
    OrbaxCheckpointStore,
    Producer,
    UnsupportedCheckpointError,
)

from diffav.core.config import validate_positive


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

MODEL_ITEM = "model"
OPTIMIZER_ITEM = "optimizer"
LOSS_METRIC = "loss"
ARCHITECTURE_VERSION_KEY = "architecture_version"

_PRODUCER = Producer(name="diffav", version=version("diffav"))


class CheckpointCorruptError(RuntimeError):
    """Raised when a checkpoint exists on disk but cannot be restored.

    Distinguishes unreadable/mismatched checkpoints from the benign
    no-checkpoint-yet case (which restore reports as ``None``).
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointConfig:
    """Configuration for checkpoint management.

    Attributes:
        checkpoint_dir: Directory path for storing checkpoints, named by the caller: a
            default under the working directory would let two runs collide at the same
            steps, which the store refuses to overwrite.
        save_interval_steps: Save a checkpoint every N training steps.
        max_to_keep: Maximum number of recent checkpoints to retain.
    """

    checkpoint_dir: str
    save_interval_steps: int = 1000
    max_to_keep: int = 5

    def __post_init__(self) -> None:
        """Validate field constraints."""
        validate_positive("save_interval_steps", self.save_interval_steps)
        validate_positive("max_to_keep", self.max_to_keep)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingState:
    """Immutable snapshot of training state restored from a checkpoint.

    Attributes:
        step: Training step the checkpoint was saved at.
        epoch: Training epoch the checkpoint was saved at.
        model_state: PyTree of restored model parameters.
        optimizer_state: PyTree of restored optimizer state, or ``None``
            when the checkpoint was saved without an optimizer.
        best_loss: Loss value recorded at save time.
        metrics: Dictionary of metric name to value recorded at save time.
        architecture_version: Backbone architecture revision the checkpoint
            was saved under, or ``None`` for checkpoints written before
            versioning existed.
    """

    step: int
    epoch: int
    model_state: Any
    optimizer_state: Any
    best_loss: float
    metrics: Mapping[str, float]
    architecture_version: int | None = None


class DiffAVCheckpointManager:
    """Checkpoint manager for diffav trajectory model training.

    Wraps substrax's :class:`OrbaxCheckpointStore` with periodic-save gating
    and full training-state persistence (model + optimizer + step/epoch).
    Doubles as a context manager so backend resources are released
    deterministically::

        with DiffAVCheckpointManager(config) as manager:
            manager.save(model, step, loss, optimizer=optimizer, epoch=epoch)
    """

    def __init__(self, config: CheckpointConfig) -> None:
        """Initialize the checkpoint manager.

        The store opens on first use and creates the directory on the first save.

        Args:
            config: Checkpoint configuration.
        """
        self.config = config
        self._store = OrbaxCheckpointStore(
            config.checkpoint_dir,
            max_to_keep=config.max_to_keep,
        )

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the underlying store on context exit."""
        self.close()

    @staticmethod
    def _items(model: nnx.Module, optimizer: nnx.Optimizer | None) -> dict[str, Any]:
        """The checkpoint items: the model's state, and the optimizer's when given."""
        items: dict[str, Any] = {MODEL_ITEM: nnx.state(model)}
        if optimizer is not None:
            items[OPTIMIZER_ITEM] = nnx.state(optimizer)
        return items

    def save(
        self,
        model: nnx.Module,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        *,
        optimizer: nnx.Optimizer | None = None,
        epoch: int = 0,
        architecture_version: int | None = None,
    ) -> Path:
        """Save a checkpoint at the given step.

        Args:
            model: Flax NNX model to checkpoint.
            step: Current training step number.
            loss: Current loss value, recorded as the ``loss`` metric.
            metrics: Optional dictionary of additional metrics.
            optimizer: Optional NNX optimizer whose state is persisted
                alongside the model.
            epoch: Current training epoch, recorded in the record.
            architecture_version: Backbone architecture revision to stamp
                into the record. Pass the running code's version so a later
                restore can reject incompatible weights; ``None`` writes an
                unversioned checkpoint.

        Returns:
            The directory of the saved checkpoint.
        """
        extra = (
            None
            if architecture_version is None
            else {ARCHITECTURE_VERSION_KEY: architecture_version}
        )
        checkpoint_path = self._store.save(
            step,
            self._items(model, optimizer),
            epoch=epoch,
            metrics={**(metrics or {}), LOSS_METRIC: loss},
            producer=_PRODUCER,
            extra=extra,
        )
        logger.info("Saved checkpoint at step %d: %s", step, checkpoint_path)
        return checkpoint_path

    def save_if_due(
        self,
        model: nnx.Module,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        *,
        optimizer: nnx.Optimizer | None = None,
        epoch: int = 0,
        architecture_version: int | None = None,
    ) -> Path | None:
        """Save a checkpoint only if the step aligns with the save interval.

        A checkpoint is saved when ``step > 0`` and ``step`` is a multiple
        of ``config.save_interval_steps``.

        Args:
            model: Flax NNX model to checkpoint.
            step: Current training step number.
            loss: Current loss value.
            metrics: Optional dictionary of additional metrics.
            optimizer: Optional NNX optimizer whose state is persisted
                alongside the model.
            epoch: Current training epoch, recorded in the record.
            architecture_version: Backbone architecture revision stamped
                into the record (see :meth:`save`).

        Returns:
            The checkpoint's directory if one was saved, None otherwise.
        """
        if step > 0 and step % self.config.save_interval_steps == 0:
            return self.save(
                model,
                step,
                loss,
                metrics,
                optimizer=optimizer,
                epoch=epoch,
                architecture_version=architecture_version,
            )
        return None

    def _corrupt(self, step: int, reason: object) -> CheckpointCorruptError:
        """The error for a checkpoint at ``step`` that exists but cannot be restored."""
        return CheckpointCorruptError(
            f"Checkpoint at step {step} in {self.config.checkpoint_dir!r} exists "
            f"but could not be restored: {reason}"
        )

    def _record(self, step: int) -> CheckpointMetadata:
        """Read the record of the checkpoint at ``step``, or raise it as corrupt."""
        try:
            return self._store.read_metadata(step)
        except (KeyError, ValueError, OSError, UnsupportedCheckpointError) as error:
            raise self._corrupt(step, error) from error

    def _architecture_version(
        self, step: int, record: CheckpointMetadata, expected: int | None
    ) -> int | None:
        """The record's architecture version, refused when malformed or not ``expected``.

        Raises:
            CheckpointCorruptError: If the stored version is present but not an integer,
                or ``expected`` is set and the stored version differs from it.
        """
        stored = record.extra.get(ARCHITECTURE_VERSION_KEY)
        # bool is an int subclass, and a JSON true is no architecture version.
        if stored is not None and (isinstance(stored, bool) or not isinstance(stored, int)):
            raise self._corrupt(step, f"its architecture version {stored!r} is not an integer")
        if expected is not None and stored != expected:
            raise CheckpointCorruptError(
                f"Checkpoint at step {step} in {self.config.checkpoint_dir!r} was saved "
                f"under backbone architecture version {stored!r}, but the running "
                f"code expects {expected!r}. Retrain the checkpoint "
                "for the current architecture."
            )
        return stored

    def restore_latest(
        self,
        model: nnx.Module,
        *,
        optimizer: nnx.Optimizer | None = None,
        expected_architecture_version: int | None = None,
    ) -> TrainingState | None:
        """Restore model (and optionally optimizer) from the latest checkpoint.

        The model and optimizer are updated **in place**; the returned
        :class:`TrainingState` carries the bookkeeping needed to resume
        training (step, epoch, loss, metrics). The record is read, and the
        architecture version checked, before any weights are loaded.

        Args:
            model: Flax NNX model to restore into.
            optimizer: Optional NNX optimizer to restore into. Must be
                provided when the checkpoint was saved with an optimizer
                and vice versa — the items have to match.
            expected_architecture_version: When set, the checkpoint's stored
                architecture version must equal it or the restore is rejected
                **before** any weights are loaded. Guards against silently
                loading stale weights into an incompatible backbone that
                happens to share leaf names. ``None`` skips the check.

        Returns:
            The restored :class:`TrainingState`, or ``None`` when no
            checkpoint exists yet.

        Raises:
            CheckpointCorruptError: If the latest checkpoint exists on disk
                but cannot be restored (unreadable data, items whose
                structure does not match ``model``/``optimizer``, an
                architecture version that is not an integer, or one that
                does not match ``expected_architecture_version``).
        """
        latest = self._store.latest_step()
        if latest is None:
            return None

        record = self._record(latest)
        stored_version = self._architecture_version(latest, record, expected_architecture_version)
        try:
            checkpoint = self._store.restore(latest, templates=self._items(model, optimizer))
        except (KeyError, ValueError, OSError, UnsupportedCheckpointError) as error:
            # A missing item, an array whose tree differs from the template, or
            # unreadable data on disk.
            raise self._corrupt(latest, error) from error

        nnx.update(model, checkpoint.items[MODEL_ITEM])
        optimizer_state = checkpoint.items.get(OPTIMIZER_ITEM)
        if optimizer is not None and optimizer_state is not None:
            nnx.update(optimizer, optimizer_state)

        metrics = {name: value for name, value in record.metrics.items() if name != LOSS_METRIC}
        return TrainingState(
            step=checkpoint.step,
            epoch=record.epoch or 0,
            model_state=checkpoint.items[MODEL_ITEM],
            optimizer_state=optimizer_state,
            best_loss=float(record.metrics.get(LOSS_METRIC, float("inf"))),
            metrics=metrics,
            architecture_version=stored_version,
        )

    def latest_step(self) -> int | None:
        """Return the step number of the latest checkpoint.

        Returns:
            Latest checkpoint step, or None if no checkpoints exist.
        """
        return self._store.latest_step()

    def list_steps(self) -> list[int]:
        """Return all retained checkpoint steps in ascending order."""
        return sorted(self._store.list_steps())

    def close(self) -> None:
        """Close the underlying checkpoint store and release resources."""
        self._store.close()
