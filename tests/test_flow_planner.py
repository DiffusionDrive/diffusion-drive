import os
import glob
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from src.planner.flow_planner.adapter import FlowPlannerCommonRoadAdapter
from commonroad.visualization.mp_renderer import MPRenderer

def test_run():

    # ==========================================
    # PROJECT PATHS
    # ==========================================

    project_root = Path(__file__).resolve().parent.parent

    scenario_dir = project_root / "data" / "scenarios"
    output_dir = project_root / "plots" / "flow_planner"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    scenario_files = sorted(
        scenario_dir.glob("*.xml")
    )

    if not scenario_files:
        raise FileNotFoundError(
            f"No XML scenarios found in {scenario_dir}"
        )

    print(
        f"\nFound {len(scenario_files)} CommonRoad scenario(s)."
    )

    # ==========================================
    # INITIALIZE FLOW-PLANNER ONCE
    # ==========================================

    adapter = FlowPlannerCommonRoadAdapter(
        config_path=str(
            project_root / "checkpoints" / "flow_planner"
            / "model_config.yaml"
        ),
        checkpoint_path=str(
            project_root / "checkpoints" / "flow_planner"
            / "model.pth"
        ),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================================
    # RUN ALL SCENARIOS
    # ==========================================

    for scenario_path in scenario_files:

        scenario_name = scenario_path.stem

        print(
            f"\n========== SCENARIO: {scenario_name} =========="
        )

        scenario, planning_problem_set = (
            CommonRoadFileReader(
                str(scenario_path)
            ).open()
        )

        planning_problem = next(
            iter(
                planning_problem_set.planning_problem_dict.values()
            )
        )

        # --------------------------------------
        # RECEDING-HORIZON PLANNING
        # --------------------------------------

        closed_loop_trajectory = adapter.plan_receding_horizon(
            scenario=scenario,
            planning_problem=planning_problem,
            replan_steps=10,
            max_cycles=30,
            seed=42
        )

        assert closed_loop_trajectory.ndim == 2
        assert closed_loop_trajectory.shape[1] == 2
        assert np.all(np.isfinite(closed_loop_trajectory))

        # --------------------------------------
        # GOAL TARGET
        # --------------------------------------

        target_position = adapter._get_goal_target(
            scenario,
            planning_problem
        )

        distance_to_target = np.linalg.norm(
            closed_loop_trajectory[-1] - target_position
        )

        print("\n========== SCENARIO RESULTS ==========")
        print("Scenario:", scenario_name)
        print("Trajectory points:", len(closed_loop_trajectory))
        print("Final position:", closed_loop_trajectory[-1])
        print("Goal target:", target_position)
        print(f"Target distance: {distance_to_target:.3f} m")
        print("======================================")

        # --------------------------------------
        # SAVE PLOT
        # --------------------------------------

        fig, ax = plt.subplots(figsize=(12, 12))

        renderer = MPRenderer(ax=ax)

        scenario.draw(renderer)
        planning_problem.draw(renderer)
        renderer.render()

        ax.plot(
            closed_loop_trajectory[:, 0],
            closed_loop_trajectory[:, 1],
            color="red",
            linewidth=2.5,
            label="Executed receding-horizon trajectory",
            zorder=20
        )

        ax.scatter(
            closed_loop_trajectory[0, 0],
            closed_loop_trajectory[0, 1],
            color="green",
            s=70,
            label="Start",
            zorder=21
        )

        ax.scatter(
            closed_loop_trajectory[-1, 0],
            closed_loop_trajectory[-1, 1],
            color="red",
            s=70,
            label="Final position",
            zorder=21
        )

        ax.scatter(
            target_position[0],
            target_position[1],
            color="purple",
            marker="x",
            s=120,
            linewidths=3,
            label="Goal target",
            zorder=30
        )

        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(
            f"Flow-Planner: {scenario_name}"
        )

        output_path = (
                output_dir / f"{scenario_name}_trajectory.png"
        )

        fig.savefig(
            str(output_path),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"[SAVED] {output_path}")

        # --------------------------------------
        # GOAL ARRIVAL ASSERTION
        # --------------------------------------

        assert distance_to_target <= 0.5, (
            f"{scenario_name}: target not reached. "
            f"Remaining distance: {distance_to_target:.3f} m"
        )

if __name__ == "__main__":
    test_run()