"""Canonical Waymo Open Motion Dataset and WOSAC constants.

Single source of truth for scenario timing, submission dimensions, and the
2025 WOSAC challenge kinematic envelope. Values mirror the official
waymo-open-dataset references: ``utils/sim_agents/submission_specs.py``
(timing and rollout dimensions) and
``wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto``
(kinematic-feature histogram support ranges).

This module has no dependencies and sits at the bottom of the layering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from diffav.core.types import Modality


# ---------------------------------------------------------------------------
# Scenario timing — WOMD scenarios are sampled at 10 Hz.
# ---------------------------------------------------------------------------

WOD_STEP_DURATION_SECONDS: float = 0.1
"""Duration of one scenario step in seconds (10 Hz sampling)."""

WOD_CURRENT_TIME_INDEX: int = 10
"""Index of the `current` step (the 11th step when 1-indexed)."""

WOD_HISTORY_STEPS: int = WOD_CURRENT_TIME_INDEX + 1
"""Number of history steps, indices 0..current inclusive (1.1 s)."""

WOD_FUTURE_STEPS: int = 80
"""Number of simulated future steps in a sim-agents submission (8 s)."""

WOD_FULL_SCENARIO_STEPS: int = WOD_HISTORY_STEPS + WOD_FUTURE_STEPS
"""Total steps in a full scenario (history + future = 91)."""


# ---------------------------------------------------------------------------
# WOSAC submission dimensions.
# ---------------------------------------------------------------------------

WOSAC_N_ROLLOUTS: int = 32
"""Number of parallel rollouts required for a valid WOSAC submission."""


# ---------------------------------------------------------------------------
# Road-edge geometry (official map metric features implementation).
# ---------------------------------------------------------------------------

EXTREMELY_LARGE_DISTANCE: float = 1e10
"""Sentinel distance for invalid segments; reduced out by the minimum."""

CYCLIC_POLYLINE_TOLERANCE_M2: float = 1.0
"""Max squared endpoint gap (m^2) for a polyline to count as a closed loop."""

ROAD_EDGE_Z_STRETCH: float = 3.0
"""Vertical-distance scaling for closest-segment association at over-/under-passes."""

ROADGRAPH_TYPE_ROAD_EDGE_BOUNDARY: int = 15
"""Roadgraph sample type of a drivable-area boundary edge (official mapping)."""

ROADGRAPH_TYPE_ROAD_EDGE_MEDIAN: int = 16
"""Roadgraph sample type of a road-edge median (official mapping)."""

ROAD_EDGE_TYPES: tuple[int, int] = (
    ROADGRAPH_TYPE_ROAD_EDGE_BOUNDARY,
    ROADGRAPH_TYPE_ROAD_EDGE_MEDIAN,
)
"""Roadgraph types the official map metric treats as road edges.

The WOSAC map metric computes offroad features against road-edge features
only; note type 17 is STOP_SIGN, not a road edge.
"""

MAX_ROAD_EDGE_POLYLINES: int = 128
"""Fixed number of road-edge polylines per scene for jit-batched off-road.

Covers the measured WOD distribution (~40 median, ~110 at the 99th percentile,
117 observed max over 50 training scenes) with margin; scenes with more edges
are truncated by :func:`~diffav.data.fixed_shape_road_edges_from_wod_dict`.
"""

MAX_ROAD_EDGE_POINTS: int = 768
"""Fixed number of points per road-edge polyline for jit-batched off-road.

Covers the measured longest road edge (~600 at the 99th percentile, 608 max)
with margin; longer polylines are truncated. Peak memory is bounded not by
these caps but by the ``chunk_size`` knob of
:func:`~diffav.core.geometry.signed_distance_to_polylines`.
"""


# ---------------------------------------------------------------------------
# Kinematic validation tolerance (diffav-chosen, not a WOSAC value).
# ---------------------------------------------------------------------------

KINEMATIC_POSITION_RESIDUAL_TOLERANCE_M2: float = 1e-3
"""Max mean squared per-step displacement deviation (m^2) from the bicycle
model for a trajectory to count as kinematically valid (~3 cm per step)."""


# ---------------------------------------------------------------------------
# 2025 challenge kinematic envelope.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class KinematicEnvelope:
    """Support ranges of the WOSAC kinematic-feature histograms.

    The challenge scores kinematic realism with per-feature histogram
    log-likelihoods over these ranges; feasibility penalties and the
    realism metametric share them as the canonical envelope.

    Attributes:
        linear_speed_min: Lower edge of the linear-speed histogram (m/s).
        linear_speed_max: Upper edge of the linear-speed histogram (m/s).
        linear_acceleration_min: Lower edge of linear acceleration (m/s^2).
        linear_acceleration_max: Upper edge of linear acceleration (m/s^2).
        angular_speed_min: Lower edge of angular speed (rad/s).
        angular_speed_max: Upper edge of angular speed (rad/s).
        angular_acceleration_min: Lower edge of angular acceleration (rad/s^2).
        angular_acceleration_max: Upper edge of angular acceleration (rad/s^2).
    """

    linear_speed_min: float
    linear_speed_max: float
    linear_acceleration_min: float
    linear_acceleration_max: float
    angular_speed_min: float
    angular_speed_max: float
    angular_acceleration_min: float
    angular_acceleration_max: float


WOSAC_2025_KINEMATIC_ENVELOPE = KinematicEnvelope(
    linear_speed_min=0.0,
    linear_speed_max=25.0,
    linear_acceleration_min=-12.0,
    linear_acceleration_max=12.0,
    angular_speed_min=-0.628,
    angular_speed_max=0.628,
    angular_acceleration_min=-3.14,
    angular_acceleration_max=3.14,
)
"""The 2025 WOSAC challenge kinematic envelope."""


# ---------------------------------------------------------------------------
# 2025 challenge metametric estimator configuration
# (challenge_2025_sim_agents_config.textproto).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class HistogramEstimateConfig:
    """Histogram estimator specification for one metametric feature.

    Attributes:
        min_val: Lower edge of the histogram support.
        max_val: Upper edge of the histogram support.
        num_bins: Number of equal-width bins.
        additive_smoothing_pseudocount: Count added to every bin before
            normalization.
    """

    min_val: float
    max_val: float
    num_bins: int
    additive_smoothing_pseudocount: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureEstimateConfig:
    """Estimator specification for one metametric feature.

    A feature is scored with a histogram estimator when ``histogram`` is
    set, and with a bernoulli estimator otherwise (mirroring the proto's
    estimator oneof).

    Attributes:
        histogram: Histogram spec, or ``None`` for a bernoulli feature.
        bernoulli_pseudocount: Additive smoothing for bernoulli features.
        independent_timesteps: Pool rollout and step samples into one
            distribution per agent (True for all 2025 time-series features).
        metametric_weight: Weight of this feature in the metametric.
    """

    histogram: HistogramEstimateConfig | None = None
    bernoulli_pseudocount: float = 0.1
    independent_timesteps: bool = False
    metametric_weight: float


@dataclass(frozen=True, slots=True, kw_only=True)
class WosacMetametricConfig:
    """Per-feature estimator specs of the 2025 WOSAC metametric.

    Field names and values mirror the official
    ``challenge_2025_sim_agents_config.textproto``.
    """

    linear_speed: FeatureEstimateConfig
    linear_acceleration: FeatureEstimateConfig
    angular_speed: FeatureEstimateConfig
    angular_acceleration: FeatureEstimateConfig
    distance_to_nearest_object: FeatureEstimateConfig
    collision_indication: FeatureEstimateConfig
    time_to_collision: FeatureEstimateConfig
    distance_to_road_edge: FeatureEstimateConfig
    offroad_indication: FeatureEstimateConfig
    traffic_light_violation: FeatureEstimateConfig


WOSAC_2025_METAMETRIC_CONFIG = WosacMetametricConfig(
    linear_speed=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=0.0, max_val=25.0, num_bins=10, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.05,
    ),
    linear_acceleration=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=-12.0, max_val=12.0, num_bins=11, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.05,
    ),
    angular_speed=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=-0.628, max_val=0.628, num_bins=11, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.05,
    ),
    angular_acceleration=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=-3.14, max_val=3.14, num_bins=11, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.05,
    ),
    distance_to_nearest_object=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=-5.0, max_val=40.0, num_bins=10, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.1,
    ),
    collision_indication=FeatureEstimateConfig(
        bernoulli_pseudocount=0.1,
        metametric_weight=0.25,
    ),
    time_to_collision=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=0.0, max_val=5.0, num_bins=10, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.1,
    ),
    distance_to_road_edge=FeatureEstimateConfig(
        histogram=HistogramEstimateConfig(
            min_val=-20.0, max_val=40.0, num_bins=10, additive_smoothing_pseudocount=0.1
        ),
        independent_timesteps=True,
        metametric_weight=0.05,
    ),
    offroad_indication=FeatureEstimateConfig(
        bernoulli_pseudocount=0.1,
        metametric_weight=0.25,
    ),
    traffic_light_violation=FeatureEstimateConfig(
        bernoulli_pseudocount=0.1,
        metametric_weight=0.05,
    ),
)
"""The 2025 WOSAC challenge metametric configuration (all ten features)."""

# =============================================================================
# WOD scenario dictionary keys
# =============================================================================
# Scenarios travel through the data pipeline as flat dicts; these constants
# name the raw keys the WOD Motion schema provides and the derived keys the
# tokenization pipeline adds, so no consumer hardcodes the strings.

# Agent state (past/current/future aggregated into state/all/*)
STATE_X = "state/all/x"
STATE_Y = "state/all/y"
STATE_Z = "state/all/z"
STATE_BBOX_YAW = "state/all/bbox_yaw"
STATE_VELOCITY_X = "state/all/velocity_x"
STATE_VELOCITY_Y = "state/all/velocity_y"
STATE_VALID = "state/all/valid"
STATE_TIMESTAMP_MICROS = "state/all/timestamp_micros"

# Per-object metadata (no temporal axis)
STATE_ID = "state/id"
STATE_TYPE = "state/type"
STATE_IS_SDC = "state/is_sdc"
STATE_TRACKS_TO_PREDICT = "state/tracks_to_predict"
STATE_OBJECTS_OF_INTEREST = "state/objects_of_interest"

# Scenario identifier (string feature; excluded from numeric element specs)
SCENARIO_ID = "scenario/id"

# Roadgraph samples
ROADGRAPH_XYZ = "roadgraph_samples/xyz"
ROADGRAPH_DIR = "roadgraph_samples/dir"
ROADGRAPH_TYPE = "roadgraph_samples/type"
ROADGRAPH_ID = "roadgraph_samples/id"
ROADGRAPH_VALID = "roadgraph_samples/valid"

# Sensor modalities (absent from WOD Motion; present in sensor datasets)
LIDAR_POINTS = "lidar/points"
CAMERA_IMAGES = "camera/images"

# Keys derived by the tokenization pipeline
STACKED_HISTORY = "stacked_history"
AGENT_EMBEDDING = "agent_emb"
MAP_EMBEDDING = "map_emb"
EGO_EMBEDDING = "ego_emb"
LIDAR_EMBEDDING = "lidar_emb"
CAMERA_EMBEDDING = "camera_emb"
SCENE_EMBEDDING = "scene_embedding"
SCENE_CONTEXT = "scene_context"
SCENE_TOKEN_VALID = "scene_token_valid"

# Per-token validity each modality writes alongside its embedding, so fusion
# can mark padded tokens (invalid agents, empty polyline slots) for exclusion
# from the backbone's map cross-attention.
AGENT_TOKEN_VALID = "agent_token_valid"
MAP_TOKEN_VALID = "map_token_valid"
EGO_TOKEN_VALID = "ego_token_valid"
LIDAR_TOKEN_VALID = "lidar_token_valid"
CAMERA_TOKEN_VALID = "camera_token_valid"

# Keys the ego-frame preprocessing backbone reads; every state-derived
# modality (agent, ego, map) depends on them.
STATE_BACKBONE_KEYS = frozenset(
    {
        STATE_IS_SDC,
        STATE_X,
        STATE_Y,
        STATE_BBOX_YAW,
        STATE_VELOCITY_X,
        STATE_VELOCITY_Y,
    }
)

# The full roadgraph sample group the map encoder consumes.
ROADGRAPH_KEYS = frozenset(
    {
        ROADGRAPH_XYZ,
        ROADGRAPH_DIR,
        ROADGRAPH_TYPE,
        ROADGRAPH_ID,
        ROADGRAPH_VALID,
    }
)

# Embedding key each modality contributes to fusion, in token order.
MODALITY_EMBEDDING_KEYS: dict[Modality, str] = {
    Modality.AGENT: AGENT_EMBEDDING,
    Modality.MAP: MAP_EMBEDDING,
    Modality.EGO: EGO_EMBEDDING,
    Modality.LIDAR: LIDAR_EMBEDDING,
    Modality.CAMERA: CAMERA_EMBEDDING,
}

# Per-token validity key each modality contributes to fusion, in token order.
MODALITY_TOKEN_VALID_KEYS: dict[Modality, str] = {
    Modality.AGENT: AGENT_TOKEN_VALID,
    Modality.MAP: MAP_TOKEN_VALID,
    Modality.EGO: EGO_TOKEN_VALID,
    Modality.LIDAR: LIDAR_TOKEN_VALID,
    Modality.CAMERA: CAMERA_TOKEN_VALID,
}


# =============================================================================
# Sensor simulation constants
# =============================================================================

LIDAR_HIT_WEIGHT_THRESHOLD_FACTOR: float = 0.5
"""A LiDAR beam registers a hit when its total absorbed weight exceeds this
fraction of the configured density threshold."""

FOG_EXTINCTION_COEFFICIENT: float = 3.0
"""Beer-Lambert extinction coefficient at full fog intensity: transmission
is ``exp(-intensity * FOG_EXTINCTION_COEFFICIENT)``, ~5% at intensity 1."""

NERF_CAMERA_HORIZONTAL_FOV_RAD: float = math.pi / 3.0
"""Horizontal field of view (60°) of the pinhole camera model used by the
NeRF renderer's ray generator."""

# ── ScenarioMiner state normalization ─────────────────────────────────────────

MINER_STATE_OFFSETS: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
"""Per-dimension offsets for the miner's internal diffusion normalization.

The miner's synthetic scene templates are ego-centred, so no re-centring is
needed; offsets exist so a checkpoint trained on differently-centred data can
declare its own.
"""

MINER_STATE_SCALES: tuple[float, float, float, float] = (150.0, 80.0, 6.3, 25.0)
"""Per-dimension scales mapping trajectory states to roughly unit range.

Diffusion runs on ``(x + offsets) / scales`` (the add/div-coefficient scheme
used by reference trajectory diffusers) and every model boundary stays in
metres. Values are derived from recentred WOD Motion statistics (99th
percentiles with headroom: x ≈ 127 m, y ≈ 67 m, unwrapped heading to 2π,
speed ≈ 20 m/s), so one coefficient set serves both WOD-trained checkpoints
and the synthetic template family, which sits well inside these bounds.
"""

AGENT_LOCAL_STATE_OFFSETS: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
"""Per-dimension offsets for the per-agent local-frame diffusion normalization.

Trajectories are re-expressed in each agent's own reference frame (see
:func:`~diffav.core.geometry.to_agent_frame`), which already centres every
agent's future at the origin, so no additional offset is applied.
"""

AGENT_LOCAL_STATE_SCALES: tuple[float, float, float, float] = (22.5, 4.6, 0.3, 5.4)
"""Per-dimension scales for the per-agent local-frame diffusion normalization.

Applies to trajectories already transformed into each agent's own frame
(translate to the agent's current pose, rotate to its heading), the agent-centric
convention of the reference trajectory models. These are the per-dimension
**standard deviations** of the local-frame target distribution over an 8 s
horizon, measured on the WOD Motion training split restricted to agents valid at
the current step (forward std ``≈ 22.5`` m, lateral ``≈ 4.6`` m, heading
``≈ 0.30`` rad, speed ``≈ 5.4`` m/s). Normalizing by the std makes the diffusion
target ~unit variance — the regime the cosine/x0 noise schedule assumes
(MotionDiffuser's unit-variance whitening rationale). The earlier p99 tail maxima
``(150, 80, π, 25)`` left the target at ~0.15 std, so signal exceeded noise only
in the top few percent of the schedule and most of the training gradient fell on
noise-dominated timesteps.
"""

# ---------------------------------------------------------------------------
# Factorized scene-diffusion backbone architecture.
# ---------------------------------------------------------------------------

SCENE_BACKBONE_ARCHITECTURE_VERSION: int = 3
"""Backbone architecture revision persisted in every checkpoint.

Restoring a checkpoint whose stored version differs from the running code's
version is rejected fail-fast, because the factorized backbone shares leaf
names (input/context/output projections) with prior denoisers — without the
guard, stale weights would restore silently into an incompatible model. Bump
this on any change to the denoiser's weight structure or its semantics that
invalidates previously trained weights.
"""

DEFAULT_NUM_BLOCKS: int = 2
"""Factorized backbone blocks; each block runs temporal, social, then FFN
sublayers (CTG++ scene denoiser uses two encoder/decoder blocks)."""

DEFAULT_NUM_TEMPORAL_LAYERS: int = 2
"""Temporal attention sublayers per block — each agent attends over its own
time axis (CTG++ ``TransformerEncoder(num_layers=2)`` temporal pass)."""

DEFAULT_NUM_SOCIAL_LAYERS: int = 1
"""Social attention sublayers per block — at each timestep agents attend
across each other (CTG++ ``TransformerEncoder(num_layers=1)`` social pass)."""
