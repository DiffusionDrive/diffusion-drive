"""Core domain types for the DiffAV evaluation engine.

Defines immutable data containers for agent states, map features,
scene contexts, trajectory predictions, and evaluation results.
All types use frozen dataclasses with slots for performance and safety.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

import jax


class AgentType(enum.StrEnum):
    """Classification of traffic participants."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"


class MapFeatureType(enum.StrEnum):
    """Classification of map elements."""

    LANE = "lane"
    CROSSWALK = "crosswalk"
    TRAFFIC_SIGNAL = "traffic_signal"


class LossType(enum.StrEnum):
    """Diffusion training loss function."""

    MSE = "mse"
    L1 = "l1"


class PredictionType(enum.StrEnum):
    """Diffusion parameterization: the quantity the backbone regresses.

    ``EPSILON`` predicts the injected noise (the DDPM default). ``X0`` predicts
    the clean trajectory directly, and ``V`` predicts the velocity
    ``v = sqrt(abar)*eps - sqrt(1-abar)*x0`` (Salimans & Ho). For smooth,
    bounded trajectory data the clean-signal targets (``X0``/``V``) are better
    conditioned across noise levels than ``EPSILON``, whose x-hat-0
    reconstruction amplifies error at high noise.
    """

    EPSILON = "epsilon"
    X0 = "x0"
    V = "v"


class NoiseScheduleType(enum.StrEnum):
    """Noise schedule variant for diffusion models."""

    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"
    SQRT = "sqrt"


class PhysicsScheduleType(enum.StrEnum):
    """Weight schedule for adaptive physics loss."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class FusionStrategy(enum.StrEnum):
    """Cross-modal fusion method for scene tokenization."""

    CROSS_ATTENTION = "cross_attention"
    EARLY = "early"
    ADDITIVE = "additive"


class DatasetMode(enum.StrEnum):
    """WOD data source loading mode."""

    EAGER = "eager"
    STREAMING = "streaming"


class Density(enum.StrEnum):
    """Agent density level for SDK scenario generation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ScenarioType(enum.StrEnum):
    """Scene template vocabulary for SDK scenario generation."""

    FORWARD = "forward"
    LANE_CHANGE = "lane_change"
    UNPROTECTED_LEFT_TURN = "unprotected_left_turn"
    ADVERSARIAL = "adversarial"


class Modality(enum.StrEnum):
    """Scene modality encoded by the tokenization pipeline."""

    AGENT = "agent"
    MAP = "map"
    EGO = "ego"
    LIDAR = "lidar"
    CAMERA = "camera"


class ModalityMode(enum.StrEnum):
    """Capability-negotiation mode for a tokenizer modality.

    ``AUTO`` enables the modality when the data source provides its
    required keys, ``REQUIRED`` fails construction when they are missing,
    and ``OFF`` excludes the modality unconditionally.
    """

    AUTO = "auto"
    REQUIRED = "required"
    OFF = "off"


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentState:
    """Immutable snapshot of a single agent's kinematic state.

    Attributes:
        position: Agent position as [x, y] coordinates, shape (2,).
        heading: Orientation in radians.
        velocity: Scalar speed (m/s).
        acceleration: Scalar acceleration (m/s^2).
        agent_type: Traffic participant classification.
    """

    position: jax.Array
    heading: float
    velocity: float
    acceleration: float
    agent_type: AgentType


@dataclass(frozen=True, slots=True, kw_only=True)
class MapFeature:
    """Immutable representation of a map element as a polyline.

    Attributes:
        polyline_points: Ordered coordinates, shape (num_points, 2).
        feature_type: Map element classification.
    """

    polyline_points: jax.Array
    feature_type: MapFeatureType


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneContext:
    """Immutable snapshot of a complete driving scene.

    Bundles the ego vehicle, surrounding agents, map features,
    and associated timestamps into a single container.

    Attributes:
        ego_state: Current state of the ego vehicle.
        agent_states: States of surrounding traffic participants.
        map_features: Relevant map polylines near the scene.
        timestamps: Time values for history steps, shape (num_steps,).
    """

    ego_state: AgentState
    agent_states: tuple[AgentState, ...]
    map_features: tuple[MapFeature, ...]
    timestamps: jax.Array


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryPrediction:
    """Immutable container for predicted future trajectories.

    Attributes:
        trajectories: Predicted states, shape (num_agents, future_steps, 4).
            State dimensions are [x, y, heading, velocity].
        agent_ids: Identifier for each predicted agent.
    """

    trajectories: jax.Array
    agent_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioMetadata:
    """Immutable identification and tagging for a driving scenario.

    Attributes:
        scenario_id: Unique scenario identifier.
        source_dataset: Origin dataset name (e.g. "waymo_open_dataset").
        tags: Optional descriptive tags for scenario classification.
    """

    scenario_id: str
    source_dataset: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricsReport:
    """Immutable aggregation of evaluation metric values.

    Attributes:
        metric_values: Named metric scores (e.g. {"ade": 1.5, "fde": 3.2}).
        per_scenario_breakdown: Optional per-scenario-type metric breakdown.
        kinematic_violation_summary: Optional per-violation-type rates.
    """

    metric_values: Mapping[str, float]
    per_scenario_breakdown: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    kinematic_violation_summary: Mapping[str, float] = field(default_factory=dict)
