import torch
from dataclasses import dataclass
from typing import Any, Optional
from src.core.types import PlannerInput, Trajectory
from src.planner.adapter import map_input_to_tensors
from src.planner.normalizer import ObservationNormalizer, StateNormalizer
from src.planner.models.diffusion_planner import Diffusion_Planner


@dataclass
class PlannerConfig:
    """Configuration mapping for Diffusion Planner encoders and decoders."""

    # --- Decoder Configs ---
    decoder_drop_path_rate: float = 0.3
    predicted_neighbor_num: int = 10
    future_len: int = 80
    route_num: int = 25
    lane_len: int = 20
    decoder_depth: int = 3
    num_heads: int = 6
    diffusion_model_type: str = "x_start"
    state_normalizer: StateNormalizer = StateNormalizer()
    observation_normalizer: ObservationNormalizer = ObservationNormalizer()
    guidance_fn: Optional[Any] = None

    # --- Encoder Configs ---
    hidden_dim: int = 192
    encoder_drop_path_rate: float = 0.3
    encoder_depth: int = 3
    agent_num: int = 32
    static_objects_num: int = 5
    lane_num: int = 70
    time_len: int = 21
    static_objects_state_dim: int = 10
    device: str = "cpu"


class DiffusionPlannerNode:
    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = PlannerConfig()

        self.model = Diffusion_Planner(self.config).to(self.device)

        if checkpoint_path:
            # 1. Load the raw dictionary from the .pth file
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 2. Extract the actual state dict (prioritize EMA weights for inference)
            if "ema_state_dict" in checkpoint:
                state_dict = checkpoint["ema_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint  # Fallback if it's already a raw state_dict

            # 3. Clean up potential prefix mismatches from training wrappers
            clean_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes from training frameworks
                new_key = k.replace("module.", "").replace("_orig_mod.", "")
                clean_state_dict[new_key] = v

            # 4. Load the cleaned weights into the model
            # strict=False allows it to load even if some non-critical keys (like normalizers) are missing
            self.model.load_state_dict(clean_state_dict, strict=False)

        self.model.eval()
        print(f"Diffusion Planner initialized on {self.device}")

    def plan(self, inputs: PlannerInput) -> Trajectory:
        tensor_inputs = map_input_to_tensors(inputs, self.device)

        with torch.no_grad():
            raw_output = self.model(tensor_inputs)

            # Handle the tuple return from the master Diffusion_Planner wrapper
            if isinstance(raw_output, tuple):
                # Find the dictionary containing the 'prediction' key
                decoder_outputs = next(
                    item
                    for item in raw_output
                    if isinstance(item, dict) and "prediction" in item
                )
            else:
                decoder_outputs = raw_output

            trajectory_tensor = decoder_outputs["prediction"]

        return Trajectory(
            ego_states=trajectory_tensor[0, 0].cpu().numpy(),
            neighbor_predictions=trajectory_tensor[0, 1:].cpu().numpy(),
        )


if __name__ == "__main__":
    import numpy as np
    import os
    from commonroad.common.file_reader import CommonRoadFileReader

    # Import the new adapters
    from src.planner.adapter import (
        extract_map_tensors,
        extract_neighbor_tensors,
        extract_ego_state,
        extract_static_objects,
        extract_navigation_route,
    )

    print("Initializing Full Integration Test...")

    # 1. Instantiate the planner node
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "model.pth")
    planner = DiffusionPlannerNode(checkpoint_path=ckpt_path, device="cpu")

    # 2. Load the CommonRoad Scenario
    file_path = os.path.join(
        ROOT_DIR, "data", "scenarios", "DEU_Flensburg-41_1_T-1.xml"
    )
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    ego_id = list(planning_problem_set.planning_problem_dict.keys())[0]
    planning_problem = planning_problem_set.planning_problem_dict[ego_id]

    # 3. Extract Live Tensors
    print("Extracting live tensors from scenario...")
    polylines, speed_limit, has_speed = extract_map_tensors(scenario)
    neighbors = extract_neighbor_tensors(scenario, current_time_step=0)
    ego_state = extract_ego_state(planning_problem, current_time_step=0)
    static_objs = extract_static_objects(scenario)
    nav_route = extract_navigation_route(scenario, planning_problem)

    # 4. Pack into PlannerInput contract
    live_input = PlannerInput(
        ego_current_state=ego_state,
        neighbors_history=neighbors,
        lane_polylines=polylines,
        lanes_speed_limit=speed_limit,
        lanes_has_speed_limit=has_speed,
        static_objects=static_objs,
        navigation_route=nav_route,
    )

    # 5. Run the Neural Network
    print("Running Diffusion Planner forward pass...")
    output_trajectory = planner.plan(live_input)

    print(
        f"✅ Success! Generated Ego Trajectory Shape: {output_trajectory.ego_states.shape}"
    )
    print(
        f"First 3 future waypoints (x, y, cos, sin):\n{np.round(output_trajectory.ego_states[:3], 2)}"
    )
