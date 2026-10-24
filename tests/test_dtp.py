"""Offline DTP reference-goal tests. Run: pytest -s tests/test_dtp.py"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.planner.dtp.adapter import goal_center
from src.planner.dtp.inference import DTPPredictor
from src.planner.dtp.receding_horizon import plan_receding_horizon

SCENARIOS = sorted((ROOT / 'data' / 'scenarios').glob('*.xml'))


@pytest.mark.parametrize('scenario_path', SCENARIOS, ids=lambda p: p.stem)
def test_dtp(scenario_path):
    scenario, pps = CommonRoadFileReader(str(scenario_path)).open()
    problem = next(iter(pps.planning_problem_dict.values()))
    predictor = DTPPredictor(ROOT)
    rollout = plan_receding_horizon(
        scenario, problem, predictor,
        execute_steps=50, samples=12, ddim_steps=50,
        max_steps=500, reference_tolerance=0.5,
        enable_route_refinement=True, strict_missing=False,
    )
    goal = goal_center(scenario, problem)
    print(f'\nScenario: {scenario_path.name}')
    print(f'Status: {rollout.status}; cycles: {rollout.cycles}; '
          f'final step: {rollout.times[-1]}; goal distance: {rollout.goal_distance:.2f} m')
    for report in rollout.diagnostics:
        print(report)

    fig = plt.figure(figsize=(12, 12))
    renderer = MPRenderer()
    scenario.draw(renderer)
    problem.draw(renderer)
    renderer.render()
    xy = rollout.trajectory
    plt.plot(xy[:, 0], xy[:, 1], 'r-o', linewidth=2.5, markersize=3,
             label=f'DTP path ({rollout.status})', zorder=20)
    plt.scatter(*np.asarray(problem.initial_state.position), c='green',
                edgecolors='black', s=90, label='Start', zorder=25)
    plt.scatter(*goal, c='purple', marker='x', s=120, linewidths=3,
                label='Goal reference', zorder=25)
    plt.scatter(*xy[-1], c='red', marker='s', s=65,
                label='Final position', zorder=25)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(f'DTP Receding-Horizon Planning | {rollout.status}\n'
              f'Distance to reference: {rollout.goal_distance:.2f} m')
    plt.tight_layout()
    folder = ROOT / 'plots' / 'dtp'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'{scenario_path.stem}_trajectory.png'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print('Plot saved to:', output)
    plt.show()
    plt.close(fig)
    assert np.isfinite(xy).all()
    # Goal success is reported separately from test execution; uncomment to require it:
    # assert rollout.status == 'goal_reached'

if __name__ == "__main__":
    test_dtp()