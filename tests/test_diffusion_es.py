import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# 1. Add nuplan-devkit to path so the model unpickles correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(project_root, "src", "planner", "diffusion_es", "nuplan_devkit"))

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

def test_model():
    print("1. Loading DEU_Flensburg-41_1_T-1.xml...")
    xml_path = os.path.join(project_root, "data", "scenarios", "DEU_Flensburg-41_1_T-1.xml")
    scenario, pp_set = CommonRoadFileReader(xml_path).open()
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

    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

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

    # Goal-Reaching Cost for Diffusion-ES Guidance
    def goal_constraint(trajectory):
        """
        trajectory: [Batch_Size, 48] or [Batch_Size, Horizon, 3]
        """
        # Ensure tensor is reshaped to [Batch_Size, Horizon, 3]
        if trajectory.ndim == 2:
            trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)

        batch_size = trajectory.shape[0]

        # Distance from final predicted pose (timestep -1) to target box center
        final_poses = trajectory[:, -1, :2]  # [Batch_Size, 2]
        dist_to_goal = torch.norm(final_poses - goal_tensor.to(trajectory.device), dim=-1)

        # Scale cost weight to guide sampling optimization
        cost = dist_to_goal * 10.0

        info = {"traj_sim": torch.zeros((batch_size, 3), device=trajectory.device)}
        return cost, info

    features = {
        "agent_history": MockAgentHistory(ego_history, agent_data, agent_masks),
        "vector_set_map": MockVectorMap(map_coords, map_avails, map_tl),
        "constraints": [goal_constraint]
    }

    with torch.no_grad():
        output = model.run_diffusion_es(features)

    print("6. Transforming Ego-Centric -> Global and Plotting Multimodal Spray...")
    cos_y, sin_y = np.cos(ego_yaw), np.sin(ego_yaw)

    rnd = MPRenderer(figsize=(12, 12))
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()

    # Extract all generated trajectories and reshape to [Samples, Horizon, 3]
    raw_trajectories = output["multimodal_trajectories"].data.cpu().numpy()
    multimodal_trajectories = raw_trajectories.reshape(-1, 16, 3)

    for idx in range(multimodal_trajectories.shape[0]):
        pred = multimodal_trajectories[idx]
        global_x = pred[:, 0] * cos_y - pred[:, 1] * sin_y + ego_x
        global_y = pred[:, 0] * sin_y + pred[:, 1] * cos_y + ego_y

        # Plot the entire distribution spray generated by the diffusion model
        color = 'blue' if idx > 0 else 'red'
        alpha = 0.3 if idx > 0 else 0.8
        plt.plot(global_x, global_y, color=color, linewidth=2, alpha=alpha, zorder=30)

    plt.title("CommonRoad + Conditional Diffusion-ES Multimodal Output")
    plt.show()

if __name__ == "__main__":
    test_model()