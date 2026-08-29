import torch
from src.core.types import PlannerInput


def map_input_to_tensors(inputs: PlannerInput, device: torch.device) -> dict:
    return {
        "ego_current_state": torch.tensor(
            inputs.ego_current_state, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "neighbor_agents_past": torch.tensor(
            inputs.neighbors_history, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "lanes": torch.tensor(
            inputs.lane_polylines, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "lanes_speed_limit": torch.tensor(
            inputs.lanes_speed_limit, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "lanes_has_speed_limit": torch.tensor(
            inputs.lanes_has_speed_limit, dtype=torch.bool, device=device
        ).unsqueeze(0),
        "static_objects": torch.tensor(
            inputs.static_objects, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "route_lanes": torch.tensor(
            inputs.navigation_route, dtype=torch.float32, device=device
        ).unsqueeze(0),
    }
