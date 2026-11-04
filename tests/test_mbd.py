import os
import glob
import time
import numpy as np
import pytest
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from src.planner.model_based_diffusion.adapter import ModelBasedDiffusionCommonRoadAdapter


# Project directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))

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


@pytest.fixture(scope="session")
def latency_results():
    results = []
    yield results
    if results:
        values = [x['latency'] for x in results]
        print("\n========== MBD PER-CYCLE LATENCY SUMMARY ==========")
        for x in results:
            print(f"{x['scenario']} | cycle {x['cycle']:03d}: {x['latency']:.3f} s")
        print(f"Average across {len(values)} cycles: {np.mean(values):.3f} s")
        print(f"Median: {np.median(values):.3f} s | Min: {min(values):.3f} s | Max: {max(values):.3f} s")
        print("Includes setup, compilation and optimization per cycle; first cycle may be cold.")


@pytest.mark.parametrize(
    "scenario_path",
    SCENARIO_FILES,
    ids=lambda path: os.path.splitext(os.path.basename(path))[0]
)
def test_run(scenario_path, latency_results):

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

    # Closed-loop replanning: 60-step optimization horizon, execute 5 steps
    # (0.5 seconds at scenario.dt=0.1) before replanning.
    print("Generating receding-horizon MBD trajectory...")
    planning_error = None
    try:
        planned_trajectory = adapter.plan_receding_horizon(
            scenario, planning_problem,
            cycle_horizon=60,
            execute_steps=5,
            max_cycles=100,
            lookahead_m=35.0,
            max_retries=2,
        )
    except RuntimeError as exc:
        planning_error = exc
        if adapter.trajectory_states is None:
            raise
        planned_trajectory = adapter.trajectory_states[:, :2]
        print(f"Plan rejected, saving diagnostic plot: {exc}")
    finally:
        for i, latency in enumerate(getattr(adapter, "cycle_latencies", []), start=1):
            latency_results.append({
                "scenario": scenario_name,
                "cycle": i,
                "latency": latency,
            })
        if getattr(adapter, "cycle_latencies", []):
            print(f"Average planning latency for {scenario_name}: "
                  f"{np.mean(adapter.cycle_latencies):.3f} s/cycle")

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

    reference = getattr(adapter, "full_reference_path", None)
    if reference is None:
        reference = np.asarray(adapter.env.xref[:, :2])
    plt.plot(reference[:, 0], reference[:, 1], "g--", label="Full reference centerline")

    # Save plot using the scenario filename
    plot_path = os.path.join(
        PLOT_DIR,
        f"{scenario_name}_trajectory.png"
    )

    # Overlay the same drivable polygon used by the adapter's road validation.
    drivable = getattr(adapter, "drivable_polygon", None)
    if drivable is not None:
        polygons = list(drivable.geoms) if hasattr(drivable, "geoms") else [drivable]
        for poly in polygons:
            if poly.geom_type == "Polygon":
                xy = np.asarray(poly.exterior.coords)
                plt.plot(xy[:, 0], xy[:, 1], color="purple", alpha=0.35,
                         linewidth=0.6, zorder=8)

    plt.legend()
    plt.title(f"MBD - {scenario_name}" + (" (FAILED)" if planning_error else ""))
    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Trajectory plot saved to: {plot_path}")

    plt.close()
    if planning_error is not None:
        pytest.fail(str(planning_error))

