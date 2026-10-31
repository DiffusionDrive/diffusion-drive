import os
import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# 1. Add nuplan-devkit to path so the model unpickles correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(project_root, "src", "planner", "diffusion_es", "nuplan_devkit"))

# Discover all CommonRoad scenarios
SCENARIO_DIR = Path(project_root) / "data" / "scenarios"

SCENARIO_FILES = sorted(SCENARIO_DIR.glob("*.xml"))

if not SCENARIO_FILES:
    raise RuntimeError(
        f"No CommonRoad XML scenarios found in {SCENARIO_DIR}"
    )

from src.planner.diffusion_es.adapter import extract_vector_map, extract_ego_history, extract_agent_history

class MockAgentHistory:
    def __init__(self, ego_history, agent_data, agent_masks):
        self.ego = ego_history
        self.data = agent_data
        self.mask = agent_masks
        self.batch_size = ego_history.shape[0]

    def repeat_interleave(self, repeats, dim=0):
        """Duplicates the tensors along the batch dimension for multimodal inference."""
        return MockAgentHistory(
            self.ego.repeat_interleave(repeats, dim),
            self.data.repeat_interleave(repeats, dim),
            self.mask.repeat_interleave(repeats, dim)
        )

class MockVectorMap:
    def __init__(self, coords, masks, traffic_lights):
        self.coords = coords # The (x, y) coordinates for each map element
        self.traffic_light_data = traffic_lights # One-hot encoding of traffic light status
        self.availabilities = masks # Boolean indicator of whether feature data is available or zero-padded

import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("[SEED] Random seed:", seed)


@pytest.mark.parametrize(
    "scenario_path",
    SCENARIO_FILES,
    ids=lambda path: path.stem
)
def test_model(scenario_path):

    scenario_name = scenario_path.stem

    # Create plot output directory
    plot_dir = Path(project_root) / "plots" / "diffusion_es"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / f"{scenario_name}_trajectory.png"

    print(f"\n1. Loading {scenario_name}...")

    scenario, pp_set = CommonRoadFileReader(
        str(scenario_path)
    ).open()

    planning_problem = list(pp_set.planning_problem_dict.values())[0]

    print("2. Extracting Vector Tensors...")
    map_coords, map_avails, map_tl = extract_vector_map(scenario, planning_problem)
    ego_history = extract_ego_history(planning_problem)
    agent_data, agent_masks = extract_agent_history(scenario)

    print("3. Transforming Global -> Ego-Centric...")
    ego_x, ego_y = planning_problem.initial_state.position
    ego_yaw = planning_problem.initial_state.orientation
    cos_inv, sin_inv = np.cos(-ego_yaw), np.sin(-ego_yaw)

    def to_ego_centric_map(coords_dict, masks_dict):
        for key in coords_dict.keys():
            coords = coords_dict[key]
            valid = masks_dict[key]

            # Isolate the translated coordinates so they aren't overwritten early
            dx = coords[valid, 0] - ego_x
            dy = coords[valid, 1] - ego_y

            coords[valid, 0] = dx * cos_inv - dy * sin_inv
            coords[valid, 1] = dx * sin_inv + dy * cos_inv
        return coords_dict

    def to_ego_centric_kinematics(tensor):
        # Translate X, Y
        dx = tensor[..., 0] - ego_x
        dy = tensor[..., 1] - ego_y
        tensor[..., 0] = dx * cos_inv - dy * sin_inv
        tensor[..., 1] = dx * sin_inv + dy * cos_inv

        # Rotate Yaw
        tensor[..., 2] -= ego_yaw

        # Isolate Velocities
        dvx = tensor[..., 3].clone()
        dvy = tensor[..., 4].clone()
        tensor[..., 3] = dvx * cos_inv - dvy * sin_inv
        tensor[..., 4] = dvx * sin_inv + dvy * cos_inv

        # Isolate Accelerations
        dax = tensor[..., 5].clone()
        day = tensor[..., 6].clone()
        tensor[..., 5] = dax * cos_inv - day * sin_inv
        tensor[..., 6] = dax * sin_inv + day * cos_inv

        return tensor

    map_coords = to_ego_centric_map(map_coords, map_avails)
    ego_history = to_ego_centric_kinematics(ego_history)
    agent_data = to_ego_centric_kinematics(agent_data)

    agent_data = torch.where(
        agent_masks.unsqueeze(-1),
        agent_data,
        torch.zeros_like(agent_data)
    )

    print("4. Loading Model and Weights...")
    from omegaconf import OmegaConf
    from src.planner.diffusion_es.nuplan_devkit.nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

    # 1. Load Hydra config saved from HPC training
    cfg_path = os.path.join(project_root, "checkpoints", "diffusion_es", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config.yaml at {cfg_path}. Ensure it was copied from HPC.")

    config = OmegaConf.load(cfg_path)

    # 2. Build model architecture matching HPC parameters exactly
    model = build_torch_module_wrapper(config.model)

    # 3. Load Checkpoint weights
    ckpt_path = os.path.join(project_root, "checkpoints", "diffusion_es", "model_cond.ckpt")

    import pickle
    class DummyClass: pass
    class MockUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("nuplan.planning.training.callbacks"): return DummyClass
            return super().find_class(module, name)
    class MockPickleModule:
        Unpickler = MockUnpickler; load = pickle.load; loads = pickle.loads

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=MockPickleModule)

    # Clean state dict keys (strip 'model.' prefix if saved by PyTorch Lightning)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    clean_state_dict = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    result = model.load_state_dict(
        clean_state_dict,
        strict=False
    )

    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    model.eval()

    # Enable evolutionary sampling with multiple candidates
    model.predictions_per_sample = 64

    print(
        "[DIFFUSION-ES] Predictions per sample:",
        model.predictions_per_sample
    )

    print("5. Running Forward Pass...")

    # Extract Goal Center in Global Coordinates
    goal_state = planning_problem.goal.state_list[0]
    goal_center_global = np.mean(goal_state.position.shapes[0].vertices, axis=0) if hasattr(goal_state.position, 'shapes') else np.mean(goal_state.position.vertices, axis=0)

    # Transform Goal Center to Ego-Centric Frame
    dx_g = goal_center_global[0] - ego_x
    dy_g = goal_center_global[1] - ego_y
    goal_x_ego = dx_g * cos_inv - dy_g * sin_inv
    goal_y_ego = dx_g * sin_inv + dy_g * cos_inv
    goal_tensor = torch.tensor([goal_x_ego, goal_y_ego], dtype=torch.float32)


    constraint_calls = {"count": 0}
    # Goal-Reaching Cost for Diffusion-ES Guidance
    def goal_constraint(trajectory):
        if trajectory.ndim == 2:
            trajectory = trajectory.reshape(
                trajectory.shape[0], -1, 3
            )

        batch_size = trajectory.shape[0]

        final_poses = trajectory[:, -1, :2]

        dist_to_goal = torch.linalg.norm(
            final_poses - goal_tensor.to(
                device=trajectory.device,
                dtype=trajectory.dtype
            ),
            dim=-1
        )

        cost = dist_to_goal * 10.0

        constraint_calls["count"] += 1

        if constraint_calls["count"] <= 3:
            print(
                f"[CONSTRAINT] Call {constraint_calls['count']}: "
                f"Candidates={batch_size}, "
                f"Min cost={cost.min().item():.3f}, "
                f"Mean cost={cost.mean().item():.3f}, "
                f"Max cost={cost.max().item():.3f}"
            )

        info = {
            "traj_sim": torch.zeros(
                (batch_size, 3),
                device=trajectory.device
            )
        }

        return cost, info

    features = {
        "agent_history": MockAgentHistory(ego_history, agent_data, agent_masks),
        "vector_set_map": MockVectorMap(map_coords, map_avails, map_tl),
        "constraints": [goal_constraint]
    }

    with torch.no_grad():
        output = model.run_diffusion_es(features)

    print(
        "[CONSTRAINT] Total evaluations:",
        constraint_calls["count"]
    )

    raw_trajectories = (
        output["multimodal_trajectories"]
        .data
        .detach()
        .cpu()
        .numpy()
    )

    multimodal_trajectories = raw_trajectories.reshape(-1, 16, 3)

    # Evaluate every returned candidate against the goal
    goal_xy = goal_tensor.detach().cpu().numpy()

    final_positions = multimodal_trajectories[:, -1, :2]

    goal_distances = np.linalg.norm(
        final_positions - goal_xy[None, :],
        axis=1
    )

    # Select candidate with minimum final goal distance
    best_idx = int(np.argmin(goal_distances))

    local_traj = multimodal_trajectories[best_idx]

    print("\n========== CANDIDATE SELECTION ==========")
    print("Number of candidates:", len(multimodal_trajectories))
    print("Best candidate index:", best_idx)
    print("Best goal distance:", goal_distances[best_idx])
    print("Mean goal distance:", goal_distances.mean())
    print("Worst goal distance:", goal_distances.max())
    print("=========================================\n")

    print("\n========== LOCAL TRAJECTORY DIAGNOSTICS ==========")
    print("Raw output shape:", raw_trajectories.shape)
    print("Reshaped output shape:", multimodal_trajectories.shape)
    print("Selected candidate index:", best_idx)
    print("Selected trajectory shape:", local_traj.shape)
    print("Selected first pose:", local_traj[0])
    print("Selected final pose:", local_traj[-1])
    print(
        "Selected final heading (wrapped):",
        np.arctan2(
            np.sin(local_traj[-1, 2]),
            np.cos(local_traj[-1, 2])
        )
    )
    print("Local goal:", goal_tensor.numpy())

    final_xy = local_traj[-1, :2]
    goal_xy = goal_tensor.detach().cpu().numpy()

    selected_goal_distance = np.linalg.norm(
        final_xy - goal_xy
    )

    print("\n========== SELECTED TRAJECTORY ==========")
    print("Selected endpoint:", final_xy)
    print("Goal:", goal_xy)
    print("Selected goal distance:", selected_goal_distance)
    print("Selected goal cost:", selected_goal_distance * 10.0)
    print("=========================================\n")

    # Measure progress and lateral deviation in the ego frame
    forward_displacement = (
            local_traj[-1, 0] - local_traj[0, 0]
    )

    lateral_displacement = (
            local_traj[-1, 1] - local_traj[0, 1]
    )

    goal_distance = np.linalg.norm(
        local_traj[-1, :2] - goal_tensor.numpy()
    )

    print("Forward displacement:", forward_displacement)
    print("Lateral displacement:", lateral_displacement)
    print("Final distance to goal:", goal_distance)
    print("==================================================\n")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        local_traj[:, 0],
        local_traj[:, 1],
        "r-o",
        label="Predicted trajectory"
    )

    ax.scatter(
        goal_x_ego,
        goal_y_ego,
        c="orange",
        s=150,
        marker="X",
        label="Goal center",
        zorder=10
    )

    ax.arrow(
        0, 0, 5, 0,
        head_width=0.5,
        color="green"
    )

    ax.set_xlabel("Local X (forward, m)")
    ax.set_ylabel("Local Y (left, m)")
    ax.set_title("Diffusion-ES Predicted Trajectory (Ego Frame)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()

    print("6. Transforming Ego-Centric -> Global and Plotting Multimodal Spray...")
    cos_y, sin_y = np.cos(ego_yaw), np.sin(ego_yaw)

    rnd = MPRenderer(figsize=(12, 12))
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()

    # Extract all generated trajectories and reshape to [Samples, Horizon, 3]
    raw_trajectories = output["multimodal_trajectories"].data.cpu().numpy()
    multimodal_trajectories = raw_trajectories.reshape(-1, 16, 3)

    # Plot all candidate trajectories in blue
    for idx, pred in enumerate(multimodal_trajectories):

        global_x = (
                pred[:, 0] * cos_y
                - pred[:, 1] * sin_y
                + ego_x
        )

        global_y = (
                pred[:, 0] * sin_y
                + pred[:, 1] * cos_y
                + ego_y
        )

        plt.plot(
            global_x,
            global_y,
            color="blue",
            linewidth=1,
            alpha=0.12,
            zorder=20
        )

    # Highlight the selected trajectory in red
    best_pred = multimodal_trajectories[best_idx]

    best_global_x = (
            best_pred[:, 0] * cos_y
            - best_pred[:, 1] * sin_y
            + ego_x
    )

    best_global_y = (
            best_pred[:, 0] * sin_y
            + best_pred[:, 1] * cos_y
            + ego_y
    )

    plt.plot(
        best_global_x,
        best_global_y,
        color="red",
        linewidth=3,
        alpha=1.0,
        label="Selected trajectory",
        zorder=40
    )

    plt.legend()

    plt.title(f"Diffusion-ES Trajectory - {scenario_name}")

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[PLOT] Saved trajectory to: {plot_path}")

    plt.show()
    plt.close()

if __name__ == "__main__":
    test_model()