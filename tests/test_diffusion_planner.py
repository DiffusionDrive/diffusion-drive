"""Diffusion-only closed-loop evaluation with per-cycle diagnostics.

No centreline trajectory substitution and no relaxation of collision checking.
"""
from pathlib import Path
import inspect
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from src.planner.diffusion_planner.inference import DiffusionPlannerNode
from src.planner.diffusion_planner.receding_horizon import plan_receding_horizon

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / 'data' / 'scenarios').glob('*.xml'))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _call_options():
    """Enable diagnostic kwargs only when the installed planner supports them."""
    supported = inspect.signature(plan_receding_horizon).parameters
    options = dict(replan_steps=5, max_cycles=100,
                   seeds_per_cycle=4, max_planning_time=25.0)
    for key in ('goal_progress_debug', 'diagnostic_mode'):
        if key in supported:
            options[key] = True
    if 'recovery_seeds_per_cycle' in supported:
        options['recovery_seeds_per_cycle'] = 20
    return options


@pytest.mark.parametrize('scenario_path', SCENARIOS, ids=lambda p: p.stem)
def test_receding(scenario_path):
    scenario, problems = CommonRoadFileReader(str(scenario_path)).open()
    problem = next(iter(problems.planning_problem_dict.values()))
    start = np.asarray(problem.initial_state.position, dtype=float)
    planner = DiffusionPlannerNode(
        checkpoint_path=str(ROOT / 'checkpoints' / 'diffusion_planner' / 'model.pth'),
        device='cuda' if torch.cuda.is_available() else 'cpu')

    options = _call_options()
    print(f'[CONFIG] {scenario_path.stem}: {options}', flush=True)
    begin = time.perf_counter()
    result = plan_receding_horizon(planner, scenario, problem, **options)
    elapsed = time.perf_counter() - begin

    output = ROOT / 'plots' / 'diffusion_planner'
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    renderer = MPRenderer(ax=ax)
    scenario.draw(renderer)
    problem.draw(renderer)
    renderer.render()
    ax.scatter(*start, color='green', label='Start', zorder=20)
    xy = np.asarray([s.position for s in result.states], dtype=float) if result.states else np.empty((0, 2))
    if len(xy):
        ax.plot(xy[:, 0], xy[:, 1], 'r-', linewidth=2,
                label='Executed diffusion predictions', zorder=21)
        ax.scatter(*xy[-1], color='red', label='Final position', zorder=22)
    ax.set_aspect('equal')
    ax.set_title(f'{scenario_path.stem} - {result.reason}')
    ax.legend()
    fig.savefig(output / f'{scenario_path.stem}_trajectory.png', dpi=170,
                bbox_inches='tight')
    plt.close(fig)

    diagnostics_dir = ROOT / 'diagnostics' / 'diffusion_planner' / scenario_path.stem
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cycles = list(result.diagnostics or [])
    (diagnostics_dir / 'replanning.json').write_text(
        json.dumps(cycles, indent=2, default=_json_default), encoding='utf-8')

    for i, cycle in enumerate(cycles):
        if not isinstance(cycle, dict):
            continue
        top = cycle.get('scored_candidates') or []
        rejects = cycle.get('rejections') or {}
        print('[CYCLE SUMMARY]', json.dumps({
            'cycle': cycle.get('cycle', i),
            'route_lanelets': cycle.get('route_lanelets'),
            'selected_kind': cycle.get('selected_kind'),
            'selected_end_speed_mps': cycle.get('selected_end_speed_mps'),
            'selected_goal_distance_m': cycle.get('selected_goal_distance_m'),
            'top_candidate': top[0] if top else None,
            'top_rejections': sorted(rejects.items(), key=lambda x: -x[1])[:10],
            'failure_diagnostics': cycle.get('failure_diagnostics'),
            'turn_speed_cap_mps': cycle.get('route_turn_speed_cap_mps'),
        }, default=_json_default), flush=True)

    distance = float(np.linalg.norm(xy[-1] - start)) if len(xy) else 0.0
    summary = {
        'scenario': scenario_path.stem,
        'success': bool(result.success), 'reason': str(result.reason),
        'safety_certified': False,  # Simulation checks do not constitute certification.
        'cycles': int(result.cycles), 'states': len(result.states),
        'start_xy': start.tolist(),
        'final_xy': xy[-1].tolist() if len(xy) else start.tolist(),
        'displacement_m': distance,
        'planning_wall_time_s': round(elapsed, 3),
        'average_wall_time_per_cycle_s': round(elapsed / max(int(result.cycles), 1), 3),
        'diagnostic_options': options,
    }
    (diagnostics_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print('[FINAL SUMMARY]', json.dumps(summary), flush=True)
    assert result.success, result.reason


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-s', '-v']))
