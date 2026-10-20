import os
import glob
import torch
from commonroad.common.file_reader import CommonRoadFileReader
from src.planner.flow_planner.adapter import FlowPlannerCommonRoadAdapter

def test_run():
    # 1. Locate files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # Update paths to be absolute based on project root
    scenario_search_path = os.path.join(project_root, "data", "scenarios", "*.xml")
    scenario_files = glob.glob(scenario_search_path)
    if not scenario_files:
        raise FileNotFoundError("No .xml scenario files found in data/scenarios/")
    scenario_path = scenario_files[0]
    print(f"Testing on scenario: {scenario_path}")

    config_path = "checkpoints/flow_planner/model_config.yaml"
    checkpoint_path = "checkpoints/flow_planner/model.pth"

    # 2. Load CommonRoad scenario and planning problem
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # 3. Initialize Adapter
    adapter = FlowPlannerCommonRoadAdapter(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 4. Generate Trajectory
    print("Generating trajectory with Flow-Planner...")
    planned_trajectory = adapter.plan(scenario, planning_problem, current_step=0)
    print(f"Trajectory generated successfully! Shape: {planned_trajectory.shape}")
    print(f"Start pos: {planned_trajectory[0]}, End pos: {planned_trajectory[-1]}")

    # 5. Visualize
    print(f"Trajectory generated successfully! Length: {planned_trajectory.shape[0]} steps")

    import matplotlib.pyplot as plt
    from commonroad.visualization.mp_renderer import MPRenderer

    # Initialize CommonRoad Renderer
    plt.figure(figsize=(12, 12))
    rnd = MPRenderer()
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()

    # Plot our Flow-Planner numpy trajectory on top!
    plt.plot(planned_trajectory[:, 0], planned_trajectory[:, 1],
             color='red', linewidth=2.5, marker='o', markersize=3,
             label='Flow-Planner Output', zorder=20)

    plt.legend()
    plt.title("Flow-Planner")
    plt.show()

if __name__ == "__main__":
    test_run()