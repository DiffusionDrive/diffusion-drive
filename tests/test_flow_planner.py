import os
import glob
import torch
import numpy as np
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
    scenario_normal_path = scenario_files[0]
    print(f"Testing on scenario: {scenario_normal_path}")

    config_path = "checkpoints/flow_planner/model_config.yaml"
    checkpoint_path = "checkpoints/flow_planner/model.pth"

    # 2. Load CommonRoad scenario and planning problem
    scenario, planning_problem_set = CommonRoadFileReader(scenario_normal_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # 3. Initialize Adapter
    adapter = FlowPlannerCommonRoadAdapter(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================================
    # COLLISION-AWARE RECEDING-HORIZON TEST
    # ==========================================

    print("\n========== CLOSED-LOOP TEST ==========")

    closed_loop_trajectory = adapter.plan_receding_horizon(
        scenario=scenario,
        planning_problem=planning_problem,
        replan_steps=10,
        max_cycles=30,
        seed=42
    )

    target_position = adapter._get_goal_target(
        scenario,
        planning_problem
    )

    final_position = closed_loop_trajectory[-1]

    distance_to_target = np.linalg.norm(
        final_position - target_position
    )

    print("\n========== TARGET ARRIVAL ==========")
    print("Target position:", target_position)
    print("Final position:", final_position)
    print(f"Distance to target: {distance_to_target:.3f} m")
    print("====================================\n")
    '''
    assert distance_to_target <= 0.5, (
        f"Target not reached. Remaining distance: "
        f"{distance_to_target:.3f} m"
    )
    '''
    print("\n========== CLOSED-LOOP RESULTS ==========")
    print("Trajectory shape:", closed_loop_trajectory.shape)
    print("Start position:", closed_loop_trajectory[0])
    print("Final position:", closed_loop_trajectory[-1])
    print("========================================\n")

    assert closed_loop_trajectory.ndim == 2
    assert closed_loop_trajectory.shape[1] == 2
    assert np.all(np.isfinite(closed_loop_trajectory))

    import matplotlib.pyplot as plt
    from commonroad.visualization.mp_renderer import MPRenderer

    plt.figure(figsize=(12, 12))

    renderer = MPRenderer()
    scenario.draw(renderer)
    planning_problem.draw(renderer)
    renderer.render()

    plt.plot(
        closed_loop_trajectory[:, 0],
        closed_loop_trajectory[:, 1],
        color="red",
        linewidth=2.5,
        label="Executed receding-horizon trajectory",
        zorder=20
    )

    plt.scatter(
        closed_loop_trajectory[0, 0],
        closed_loop_trajectory[0, 1],
        color="green",
        s=70,
        label="Start",
        zorder=21
    )

    plt.scatter(
        closed_loop_trajectory[-1, 0],
        closed_loop_trajectory[-1, 1],
        color="red",
        s=70,
        label="Final position",
        zorder=21
    )

    target_position = adapter._get_goal_target(
        scenario,
        planning_problem
    )

    plt.scatter(
        target_position[0],
        target_position[1],
        color="purple",
        marker="x",
        s=120,
        linewidths=3,
        label="Goal target",
        zorder=30
    )

    plt.axis("equal")
    plt.legend()
    plt.title("Flow-Planner Receding-Horizon Trajectory")
    plt.show()

if __name__ == "__main__":
    test_run()