"""Trajectory model wrappers built on artifex generative models."""

from diffav.models.checkpointing import (
    CheckpointConfig,
    CheckpointCorruptError,
    DiffAVCheckpointManager,
    TrainingState,
)
from diffav.models.factorized_backbone import (
    FactorizedSceneBackbone,
    FactorizedSceneBackboneConfig,
)
from diffav.models.trainer import (
    TrainerConfig,
    TrainingMetrics,
    TrajectoryTrainer,
)
from diffav.models.trajectory_diffusion import (
    create_trajectory_model,
    DiffusionLossOutputs,
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)


__all__ = [
    "CheckpointConfig",
    "CheckpointCorruptError",
    "DiffusionLossOutputs",
    "FactorizedSceneBackbone",
    "FactorizedSceneBackboneConfig",
    "DiffAVCheckpointManager",
    "TrainerConfig",
    "TrainingMetrics",
    "TrainingState",
    "TrajectoryDiffusionConfig",
    "TrajectoryDiffusionModel",
    "TrajectoryTrainer",
    "create_trajectory_model",
]
