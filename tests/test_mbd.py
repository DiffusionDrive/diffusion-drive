import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from src.planner.model_based_diffusion.adapter import ModelBasedDiffusionCommonRoadAdapter


def test_run():
    # 1. Locate files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    scenario_search_path = os.path.join(project_root, "data", "scenarios", "*.xml")
    scenario_files = glob.glob(scenario_search_path)
    if not scenario_files:
        raise FileNotFoundError("No .xml scenario files found in data/scenarios/")

    scenario_path = scenario_files[0]
    print(f"Testing MBD on scenario: {scenario_path}")

    # 2. Load CommonRoad scenario and planning problem
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # 3. Initialize Adapter
    adapter = ModelBasedDiffusionCommonRoadAdapter(
        seed=0,
        num_samples=2048,
        horizon=100,
        diffusion_steps=100
    )

    # 4. Generate Trajectory
    print("Generating trajectory with Model-Based Diffusion (MBD)...")
    planned_trajectory = adapter.plan(scenario, planning_problem)

    print(f"Trajectory generated successfully! Shape: {planned_trajectory.shape}")
    print(f"Start pos: {planned_trajectory[0]}, End pos: {planned_trajectory[-1]}")

    # 5. Visualize
    plt.figure(figsize=(12, 12))
    rnd = MPRenderer()
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()

    # Plot planned MBD trajectory in red
    plt.plot(
        planned_trajectory[:, 0],
        planned_trajectory[:, 1],
        color='red',
        linewidth=2.5,
        marker='o',
        markersize=3,
        label='MBD Output',
        zorder=20
    )

    plt.plot(
        np.asarray(adapter.env.xref[:, 0]),
        np.asarray(adapter.env.xref[:, 1]),
        "g--",
        label="Reference centerline"
    )

    plt.legend()
    plt.title("Model-Based Diffusion (MBD) - CommonRoad")
    plt.show()


if __name__ == "__main__":
    test_run()