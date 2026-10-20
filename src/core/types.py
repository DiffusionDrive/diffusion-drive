import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    velocity: float
    acceleration: float = 0.0
    length: float = 5.18  # Default nuPlan vehicle length
    width: float = 2.30  # Default nuPlan vehicle width

@dataclass
class Trajectory:
    ego_states: np.ndarray
    neighbor_predictions: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    cost: Optional[float] = None

@dataclass
class PlannerInput:
    """Standardized input payload mirroring NuPlan/Diffusion Planner normalizer keys."""
    ego_current_state: np.ndarray          # Shape: (10,)
    neighbor_agents_past: np.ndarray       # Shape: (32, 21, 11)
    lanes: np.ndarray                      # Shape: (70, 20, 12)
    lanes_speed_limit: np.ndarray          # Shape: (70, 1)
    lanes_has_speed_limit: np.ndarray      # Shape: (70, 1)
    static_objects: np.ndarray             # Shape: (5, 10)
    route_lanes: np.ndarray                # Shape: (25, 20, 12)
    route_lanes_speed_limit: np.ndarray    # Shape: (25, 1)
    route_lanes_has_speed_limit: np.ndarray # Shape: (25, 1)

    platoon_leader_trajectory: Optional[Trajectory] = None