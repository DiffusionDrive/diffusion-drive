import os
import glob
import numpy as np
import pytest
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from src.planner.model_based_diffusion.adapter import ModelBasedDiffusionCommonRoadAdapter


# Project directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

SCENARIO_DIR = os.path.join(PROJECT_ROOT, "data", "scenarios")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "mbd")

# Create plot directory if it does not exist
os.makedirs(PLOT_DIR, exist_ok=True)

# Find all XML scenario files
SCENARIO_FILES = sorted(
    glob.glob(os.path.join(SCENARIO_DIR, "*.xml"))
)

if not SCENARIO_FILES:
    raise FileNotFoundError(
        f"No XML scenario files found in {SCENARIO_DIR}"
    )


@pytest.mark.parametrize(
    "scenario_path",
    SCENARIO_FILES,
    ids=lambda path: os.path.splitext(os.path.basename(path))[0]
)
def test_run(scenario_path):

    scenario_name = os.path.splitext(
        os.path.basename(scenario_path)
    )[0]

    print(f"\nTesting MBD on scenario: {scenario_name}")

    # Load CommonRoad scenario
    scenario, planning_problem_set = (
        CommonRoadFileReader(scenario_path).open()
    )

    planning_problem = list(
        planning_problem_set.planning_problem_dict.values()
    )[0]

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

    # Save plot using the scenario filename
    plot_path = os.path.join(
        PLOT_DIR,
        f"{scenario_name}_trajectory.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Trajectory plot saved to: {plot_path}")

    plt.legend()
    plt.title("Model-Based Diffusion (MBD) - CommonRoad")
    plt.show()
    plt.close()

if __name__ == "__main__":
    test_run()