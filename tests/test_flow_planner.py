"""Parameterized CommonRoad evaluation for the pretrained Flow-Planner adapter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from src.planner.flow_planner.adapter import FlowPlannerCommonRoadAdapter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = PROJECT_ROOT / "data" / "scenarios"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "flow_planner"
SCENARIO_FILES = sorted(SCENARIO_DIR.glob("*.xml"))


@pytest.fixture(scope="module")
def adapter():
    """Load the pretrained model once for all scenarios in this test module."""
    return FlowPlannerCommonRoadAdapter(
        config_path=str(PROJECT_ROOT / "checkpoints" / "flow_planner" / "model_config.yaml"),
        checkpoint_path=str(PROJECT_ROOT / "checkpoints" / "flow_planner" / "model.pth"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@pytest.mark.parametrize(
    "scenario_path",
    SCENARIO_FILES,
    ids=lambda path: path.stem,
)
def test_run(scenario_path, adapter):
    """Evaluate one scenario and save its plot, even if the goal is not reached."""
    scenario_name = scenario_path.stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n========== SCENARIO: {scenario_name} ==========")
    print(f"[TEST] Scenario file: {scenario_path}")

    scenario, planning_problem_set = CommonRoadFileReader(str(scenario_path)).open()
    planning_problems = planning_problem_set.planning_problem_dict
    assert planning_problems, f"{scenario_name}: no planning problem in XML"
    planning_problem = next(iter(planning_problems.values()))

    closed_loop_trajectory = np.asarray(
        adapter.plan_receding_horizon(
            scenario=scenario,
            planning_problem=planning_problem,
            replan_steps=10,
            max_cycles=30,
            seed=42,
            num_candidates=5
        ),
        dtype=float
    )

    print(
        f"[TEST] Planning status: "
        f"{adapter.last_plan_status}"
    )

    reference = adapter._get_route_centerline(
        scenario,
        planning_problem
    )

    route_metrics = adapter._evaluate_route_following(
        closed_loop_trajectory,
        reference
    )

    print("\n========== ROUTE FOLLOWING ==========")
    print(
        f"Mean lateral error: "
        f"{route_metrics['mean_lateral_error']:.3f} m"
    )
    print(
        f"Maximum lateral error: "
        f"{route_metrics['max_lateral_error']:.3f} m"
    )
    print(
        f"Route progress: "
        f"{route_metrics['route_progress']:.3f} m"
    )
    print("=====================================\n")

    assert closed_loop_trajectory.ndim == 2, (
        f"{scenario_name}: unexpected trajectory dimensions: {closed_loop_trajectory.shape}"
    )
    assert closed_loop_trajectory.shape[1] == 2, (
        f"{scenario_name}: expected (N, 2), got {closed_loop_trajectory.shape}"
    )
    assert len(closed_loop_trajectory) > 0, f"{scenario_name}: empty trajectory"
    assert np.all(np.isfinite(closed_loop_trajectory)), (
        f"{scenario_name}: non-finite trajectory coordinates"
    )

    target_position = np.asarray(
        adapter._get_goal_target(scenario, planning_problem), dtype=float
    )
    distance_to_target = float(
        np.linalg.norm(closed_loop_trajectory[-1] - target_position)
    )

    print("\n========== SCENARIO RESULTS ==========")
    print("Scenario:", scenario_name)
    print("Trajectory points:", len(closed_loop_trajectory))
    print("Final position:", closed_loop_trajectory[-1])
    print("Goal target:", target_position)
    print(f"Target distance: {distance_to_target:.3f} m")
    print("Goal region reached:", adapter.last_goal_region_reached)
    print("Target reached:", adapter.last_target_reached)
    print("======================================")

    fig, ax = plt.subplots(figsize=(12, 12))
    try:
        renderer = MPRenderer(ax=ax)
        scenario.draw(renderer)
        planning_problem.draw(renderer)
        renderer.render()

        # ==========================================
        # REFERENCE ROUTE VISUALIZATION
        # ==========================================

        reference = adapter._get_route_centerline(
            scenario,
            planning_problem
        )

        ax.plot(
            reference[:, 0],
            reference[:, 1],
            color="limegreen",
            linewidth=2.0,
            linestyle="--",
            label="CommonRoad reference centreline",
            zorder=15
        )

        ax.plot(
            closed_loop_trajectory[:, 0],
            closed_loop_trajectory[:, 1],
            color="red",
            linewidth=2.5,
            label="Executed receding-horizon trajectory",
            zorder=20,
        )
        ax.scatter(
            closed_loop_trajectory[0, 0],
            closed_loop_trajectory[0, 1],
            color="green", s=70, label="Start", zorder=21,
        )
        ax.scatter(
            closed_loop_trajectory[-1, 0],
            closed_loop_trajectory[-1, 1],
            color="red", s=70, label="Final position", zorder=21,
        )
        ax.scatter(
            target_position[0],
            target_position[1],
            color="purple", marker="x", s=120,
            linewidths=3, label="Goal target", zorder=30,
        )
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"Flow-Planner: {scenario_name}")

        output_path = OUTPUT_DIR / f"{scenario_name}_trajectory.png"
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        print(f"[SAVED] {output_path}")
    finally:
        plt.close(fig)

    assert adapter.last_goal_region_reached, (
        f"{scenario_name}: goal region not reached; "
        f"target distance: {distance_to_target:.3f} m; "
        f"status: {adapter.last_plan_status}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
