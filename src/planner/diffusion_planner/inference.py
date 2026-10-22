import torch
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from src.core.types import PlannerInput, Trajectory
from src.planner.diffusion_planner.adapter import map_input_to_tensors
from src.planner.diffusion_planner.normalizer import (
    ObservationNormalizer,
    StateNormalizer,
)
from src.planner.diffusion_planner.models.diffusion_planner import Diffusion_Planner
from src.planner.diffusion_planner.guidance.guidance_wrapper import GuidanceWrapper

@dataclass
class PlannerConfig:
    """Configuration mapping for Diffusion Planner encoders and decoders."""

    # --- Decoder Configs ---
    decoder_drop_path_rate: float = 0.1
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
    encoder_drop_path_rate: float = 0.1
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
        if checkpoint_path:
            args_path = Path(checkpoint_path).with_name("args.json")
            if args_path.exists():
                with args_path.open(encoding="utf-8") as f:
                    saved_args = json.load(f)
                for key in ("encoder_drop_path_rate", "decoder_drop_path_rate",
                            "future_len", "route_num", "lane_len", "decoder_depth",
                            "num_heads", "hidden_dim", "encoder_depth", "agent_num",
                            "static_objects_num", "lane_num", "time_len",
                            "predicted_neighbor_num", "diffusion_model_type"):
                    if key in saved_args:
                        setattr(self.config, key, saved_args[key])

        self.config.guidance_fn = None # GuidanceWrapper()
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
            # Load pretrained model weights
            load_result = self.model.load_state_dict(
                clean_state_dict,
                strict=False
            )

            # Normalization statistics may be initialized directly
            # inside normalizer.py rather than saved in the checkpoint.
            allowed_missing_keys = {
                "decoder.decoder._state_normalizer.mean",
                "decoder.decoder._state_normalizer.std",
                "decoder.decoder._observation_normalizer.ego_mean",
                "decoder.decoder._observation_normalizer.ego_std",
                "decoder.decoder._observation_normalizer.neigh_mean",
                "decoder.decoder._observation_normalizer.neigh_std",
                "decoder.decoder._observation_normalizer.lane_mean",
                "decoder.decoder._observation_normalizer.lane_std",
                "decoder.decoder._observation_normalizer.stat_mean",
                "decoder.decoder._observation_normalizer.stat_std",
            }

            # Identify genuinely incompatible checkpoint parameters
            critical_missing_keys = [
                key for key in load_result.missing_keys
                if key not in allowed_missing_keys
            ]

            if critical_missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint compatibility error:\n"
                    f"Missing model parameters: {critical_missing_keys}\n"
                    f"Unexpected parameters: {load_result.unexpected_keys}"
                )

            if load_result.missing_keys:
                print(
                    "[INFO] Normalization buffers not found in checkpoint. "
                    "Using values initialized by normalizer.py."
                )

            print("[INFO] Pretrained Diffusion-Planner weights loaded successfully.")
            print(f"Loaded {len(clean_state_dict)} checkpoint tensors successfully")

        self.model.eval()
        print(f"Diffusion Planner initialized on {self.device}")

    def plan(self, inputs: PlannerInput) -> Trajectory:
        # Convert dataclass to tensor dictionary
        tensor_inputs = map_input_to_tensors(inputs, self.device)

        # 1. Normalize observations for the encoder/attention layers
        normalized_inputs = self.config.observation_normalizer(tensor_inputs)

        with torch.no_grad():
            # 2. Forward pass through neural network
            raw_output = self.model(normalized_inputs)

            if isinstance(raw_output, tuple):
                decoder_outputs = next(
                    item for item in raw_output if isinstance(item, dict) and "prediction" in item
                )
            else:
                decoder_outputs = raw_output

            trajectory_tensor = decoder_outputs["prediction"]

        # 3. Extract physical trajectory directly (the model sampler already denormalized it)
        ego_states = trajectory_tensor[0, 0].cpu().numpy()
        neighbor_preds = trajectory_tensor[0, 1:].cpu().numpy()

        return Trajectory(
            ego_states=ego_states,
            neighbor_predictions=neighbor_preds,
        )
