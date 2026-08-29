import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VehicleState:
    """
    Represents the full kinematic state of a vehicle.
    Note: The Diffusion Planner intentionally excludes ego velocity and acceleration
    from the initial state constraint to prevent shortcut learning.
    """

    x: float
    y: float
    yaw: float
    velocity: float
    acceleration: float = 0.0
    length: float = 5.18  # Default nuPlan vehicle length
    width: float = 2.30  # Default nuPlan vehicle width


@dataclass
class Trajectory:
    """
    The output of the Diffusion Planner, which jointly predicts the ego vehicle's
    path and the future trajectories of neighboring vehicles.
    """

    # Expected shape: (T, 4) -> [x, y, cos(yaw), sin(yaw)]
    # where T is typically 80 steps (8 seconds at 10Hz)
    ego_states: np.ndarray

    # Expected shape: (M, T, 4) -> Predictions for the M nearest neighboring vehicles
    neighbor_predictions: Optional[np.ndarray] = None

    timestamps: Optional[np.ndarray] = None
    cost: Optional[float] = None


@dataclass
class PlannerInput:
    """
    The standardized input payload expected by the Diffusion Planner.
    All dimensions match the MLP-Mixer and DiT requirements.
    """

    # 1. Ego Vehicle State
    # Expected shape: (4,) -> [x, y, cos(yaw), sin(yaw)]
    ego_current_state: np.ndarray

    # 2. Neighboring Vehicles (History)
    # Expected shape: (M, L, D_neighbor) -> e.g., (32, 21, 11)
    # M: Nearest neighbors (e.g., 10)
    # L: Past timestamps over 2 seconds (e.g., 21 at 10Hz)
    # D_neighbor: Features including coordinates, heading, velocity, size, category
    neighbors_history: np.ndarray

    # 3. Map & Lane Information
    # Expected shape: (Num_Lanes, P, D_lane) -> e.g., (70, 20, 12)
    # P: Points per polyline (e.g., 20)
    # D_lane: Features including coordinates, traffic light status, speed limits
    lane_polylines: np.ndarray

    # 4. Static Objects (Updated to 10 dimensions)
    # Expected shape: (Num_Static, D_static)
    # D_static: Features including coordinates, heading, size, category
    static_objects: np.ndarray

    # 5. Navigation / Route Guidance
    # Expected shape: (K, P, D_route) -> e.g., (25, 20, 4)
    # K: Number of route lanes
    # D_route: Coordinate information guiding the intended route
    navigation_route: np.ndarray
    lanes_speed_limit: np.ndarray  # Shape: (70, 1)
    lanes_has_speed_limit: np.ndarray  # Shape: (70, 1) - Boolean mask

    # For Student D's platooning module (Optional extension for capstone)
    platoon_leader_trajectory: Optional[Trajectory] = None
