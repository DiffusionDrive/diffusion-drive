"""Run: python tests/test_opttrajdiff.py [--max-cycles 150] [--samples 12]."""
from __future__ import annotations

import argparse
import sys
import traceback
import platform
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/planner/opttrajdiff'))
from adapter import OptTrajDiffAdapter
from lane_guidance import LaneGuide, CollisionChecker, _goal_shapes


def goal_target(problem, scenario):
    region = problem.goal
    shape = _goal_shapes(problem)
    if shape is not None and not shape.is_empty:
        return np.asarray(shape.centroid.coords[0], dtype=float)
    for state in region.state_list:
        pos = getattr(state, 'position', None)
        if pos is not None:
            if hasattr(pos, 'center'):
                center = pos.center
                if hasattr(center, 'x') and hasattr(center, 'y'):
                    return np.array([center.x, center.y], dtype=float)
                return np.asarray(center, dtype=float)[:2]
            if hasattr(pos, 'vertices'):
                return np.asarray(pos.vertices, dtype=float)[:, :2].mean(axis=0)
    for lanelet_id in getattr(region, 'lanelets_of_goal_position', {}) or {}:
        try:
            ll = scenario.lanelet_network.find_lanelet_by_id(int(lanelet_id))
            return np.asarray(ll.center_vertices[-1], dtype=float)[:2]
        except (KeyError, ValueError):
            continue
    raise ValueError('Goal position is not a supported point/shape; cannot select a target')


def goal_reached(problem, state):
    try:
        return bool(problem.goal.is_reached(_state_for_goal(state)))
    except (AttributeError, TypeError, ValueError):
        return False


def plot_scenario(scenario, problem, trajectory, output, reached, reason):
    """Match the Diffusion Planner test style, including rendered goal/obstacles."""
    fig, ax = plt.subplots(figsize=(10, 10))
    renderer = MPRenderer(ax=ax)
    scenario.draw(renderer)
    problem.draw(renderer)
    renderer.render()
    p = np.asarray(trajectory, dtype=float)
    ax.scatter(*p[0], color='green', s=55, label='Start', zorder=20)
    if len(p) > 1:
        ax.plot(p[:, 0], p[:, 1], 'r-', linewidth=2,
                label='Executed trajectory', zorder=21)
    ax.scatter(*p[-1], color='red', s=55, label='Final position', zorder=22)
    ax.set_aspect('equal')
    ax.set_title(f'{output.stem.replace("_trajectory", "")} - {reason}')
    ax.legend(loc='best')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


def _state_for_goal(ego):
    """Construct a CommonRoad-compatible state for the goal checker."""
    from commonroad.scenario.state import CustomState
    return CustomState(position=np.asarray(ego.position, dtype=float),
                       orientation=float(ego.orientation),
                       velocity=float(ego.velocity),
                       time_step=int(ego.time_step))


def run_one(adapter, xml, outdir, max_cycles, execute_steps, tolerance, clearance, max_wait_cycles=100):
    scenario, planning_problem_set = CommonRoadFileReader(str(xml)).open()
    if not planning_problem_set.planning_problem_dict:
        raise ValueError('Scenario contains no planning problem')
    problem = next(iter(planning_problem_set.planning_problem_dict.values()))
    init = problem.initial_state
    print(f'\n[DIAG] scenario={xml.name}, dt={scenario.dt}, lanelets={len(scenario.lanelet_network.lanelets)}, obstacles={len(scenario.obstacles)}', flush=True)
    print(f'[DIAG] planning_problem_id={problem.planning_problem_id} goal_states={len(problem.goal.state_list)}', flush=True)
    for idx, state in enumerate(problem.goal.state_list):
        print(f'[DIAG] goal_state[{idx}] position_type={type(getattr(state, "position", None)).__name__} position={str(getattr(state, "position", None))[:250]}', flush=True)
    target = goal_target(problem, scenario)
    print(f'[DIAG] goal_target={target.tolist()}', flush=True)
    guide = LaneGuide(scenario, init.position, target, clearance=clearance, problem=problem)
    ego = SimpleNamespace(position=np.asarray(init.position, dtype=float)[:2],
                          orientation=float(getattr(init, 'orientation', 0.0)),
                          velocity=float(getattr(init, 'velocity', 0.0)),
                          time_step=int(init.time_step))
    # Use the route guide's exact goal centroid as the inference target.
    target = guide.goal.copy()
    collision = CollisionChecker(scenario, ego_radius=1.1, margin=0.4)
    history = [[*ego.position, ego.orientation, ego.velocity]]
    trajectory = [ego.position.copy()]
    latencies = []
    diagnostics = []
    road_violations = 0
    controller_counts = {}
    reached = False  # Require goal-center arrival and stopping, not first region entry
    best_distance = float(np.linalg.norm(ego.position - target))
    stagnant_cycles = 0
    reason = "max_cycles"
    center_reached = False
    collision_wait_steps = 0
    consecutive_wait_cycles = 0
    total_executed_steps = 0
    dt = float(scenario.dt)
    # Receding horizon: replan from the latest executed ego state each cycle.
    for cycle in range(max_cycles):
        if reached:
            reason = "goal_region_reached"
            break
        print(f'[DIAG] cycle={cycle} timestep={ego.time_step} ego={ego.position.tolist()} speed={ego.velocity:.3f} history_len={len(history)}', flush=True)
        plan = adapter.plan(scenario, ego, ego.time_step, target, history)
        print(f'[DIAG] plan returned shape={plan.points.shape} candidates={plan.candidate_count} latency={plan.latency:.3f}s', flush=True)
        latencies.append(plan.latency)
        candidates = plan.candidates
        if candidates is None or len(candidates) == 0:
            reason = 'no_diffusion_candidates'
            break
        blocked = False
        moved_this_cycle = False
        # Execute only a short prefix, then re-observe and replan.
        for step_index in range(min(execute_steps, len(candidates[0]))):
            old = ego.position.copy()
            # Keep step lengths bounded; this does not constitute full dynamics validation.
            remaining = float(np.linalg.norm(old - target))
            # Brake progressively toward the center (2.5 m/s^2 nominal).
            # This bounds displacement but is not a complete dynamics model.
            approach_speed = np.sqrt(max(0.0, 2.0 * 2.5 * max(remaining - tolerance * 0.5, 0.0)))
            speed_limit = min(15.0, approach_speed)
            max_step = min(1.5, max(0.0, ego.velocity * dt + 0.25), speed_limit * dt)
            max_step = min(max_step, remaining)
            if remaining <= tolerance:
                max_step = 0.0
            if remaining <= tolerance:
                # Hold position to satisfy the stop condition, but only if
                # the hold is road-compliant and occupancy-screened.
                if guide.valid_point(old) and collision.safe_segment(old, old, ego.time_step + 1):
                    xy, source = old.copy(), 'goal_hold'
                else:
                    xy, source = old.copy(), 'blocked_collision_or_route'
            else:
                xy, source = guide.choose_step(
                    ego, candidates, dt, step_index, max_step, lookahead=2.5,
                    collision_checker=lambda a, b: collision.safe_segment(a, b, ego.time_step + 1))
            controller_counts[source] = controller_counts.get(source, 0) + 1
            if source in ('blocked_route_corridor', 'blocked_collision_or_route'):
                blocked = True
                reason = 'no_collision_free_road_step'
                print('[ROAD] No compliant candidate or route-following step; stopping', flush=True)
                break
            if not guide.valid_segment(old, xy):
                road_violations += 1
                reason = 'road_violation_detected'
                blocked = True
                break
            d = np.asarray(xy) - old
            if source in ('safe_wait', 'goal_hold'):
                collision_wait_steps += 1
            if np.linalg.norm(d) > 1e-5:
                moved_this_cycle = True
            total_executed_steps += 1
            ego.position = np.asarray(xy, dtype=float)
            ego.velocity = float(np.linalg.norm(d) / dt)
            if np.linalg.norm(d) > 1e-4:
                ego.orientation = float(np.arctan2(d[1], d[0]))
            ego.time_step += 1
            history.append([*ego.position, ego.orientation, ego.velocity])
            trajectory.append(ego.position.copy())
            center_distance = float(np.linalg.norm(ego.position - target))
            if center_distance <= tolerance and ego.velocity <= 0.5 and goal_reached(problem, ego):
                reached = True
                center_reached = True
                reason = 'goal_center_stopped'
                break
        if blocked:
            print(f'[ROAD] halted at {ego.position.tolist()} reason={reason}', flush=True)
            break
        distance_now = float(np.linalg.norm(ego.position - target))
        diagnostics.append(dict(cycle=cycle, time_step=ego.time_step,
                                final_xy=ego.position.tolist(),
                                distance_to_target_m=distance_now,
                                planning_latency_s=plan.latency,
                                candidate_count=plan.candidate_count,
                                goal_region_reached=bool(reached),
                                route_progress_m=guide.progress(ego.position),
                                controller_counts=dict(controller_counts),
                                road_violations=road_violations))
        print(f'[CYCLE SUMMARY] cycle={cycle} goal_distance={distance_now:.2f}m '
              f'goal_region_reached={reached} latency={plan.latency:.3f}s', flush=True)
        if distance_now < best_distance - 0.05:
            best_distance = distance_now
            stagnant_cycles = 0
        elif moved_this_cycle:
            stagnant_cycles += 1
        # Intentional waiting does not count as failure to make progress.
        if moved_this_cycle:
            consecutive_wait_cycles = 0
        else:
            consecutive_wait_cycles += 1
        if reached:
            break
        if consecutive_wait_cycles >= max_wait_cycles:
            reason = f'waiting_timeout_{max_wait_cycles}_cycles'
            break
        if stagnant_cycles >= 90:
            reason = 'no_goal_progress_90_moving_cycles'
            break
    output = outdir / f'{xml.stem}_trajectory.png'
    plot_scenario(scenario, problem, trajectory, output, reached, reason)
    distance = float(np.linalg.norm(ego.position-target))
    avg = float(np.mean(latencies)) if latencies else 0.0
    diagnostics_dir = ROOT / 'diagnostics' / 'opttrajdiff' / xml.stem
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / 'replanning.json').write_text(json.dumps(diagnostics, indent=2))
    summary = dict(scenario=xml.stem, success=bool(reached),
                   center_reached_and_stopped=bool(center_reached), reason=reason,
                   cycles=len(latencies), final_distance_to_target_m=distance,
                   final_xy=ego.position.tolist(),
                   average_planning_latency_s=avg,
                   goal_tolerance_m=tolerance,
                   safety_certified=False, road_violations=road_violations,
                   collision_rejections=collision.rejections,
                   unknown_obstacle_occupancies=collision.unknown_count,
                   unsupported_occupancy_types=sorted(collision.unknown_types),
                   replanning_mode='closed_loop_receding_horizon',
                   executed_steps_per_cycle=execute_steps,
                   safe_wait_steps=collision_wait_steps,
                   executed_steps=total_executed_steps,
                   final_speed_mps=float(ego.velocity),
                   goal_center_route_progress_m=float(guide.progress(target)),
                   max_wait_cycles=max_wait_cycles,
                   collision_check='swept-circle against available discrete-time obstacle occupancies',
                   road_check='center-point segments inside eroded directed-route corridor',
                   controller_counts=controller_counts, route_lanelets=guide.route_ids,
                   plot=str(output))
    (diagnostics_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('[FINAL SUMMARY]', json.dumps(summary), flush=True)
    print(f'{xml.stem}: reached={reached}, distance={distance:.2f}m, cycles={len(latencies)}, '
          f'avg_latency={avg:.3f}s, plot={output}')
    return avg, reached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cycles', type=int, default=150)
    parser.add_argument('--execute-steps', type=int, default=5)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--goal-tolerance', type=float, default=1.5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--road-clearance', type=float, default=0.45,
                        help='Minimum center-point distance from drivable boundary (m)')
    parser.add_argument('--scenario', default=None, help='Optional scenario filename or stem; default: all XML files')
    parser.add_argument('--max-wait-cycles', type=int, default=100,
                        help='Maximum consecutive replanning cycles without movement')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at first error for a focused traceback')
    args = parser.parse_args()
    scenarios = sorted((ROOT / 'data/scenarios').glob('*.xml'))
    if not scenarios:
        raise FileNotFoundError(f'No XML scenarios found at {ROOT / "data/scenarios"}')
    if args.scenario:
        scenarios = [p for p in scenarios if p.name == args.scenario or p.stem == args.scenario]
        if not scenarios:
            raise FileNotFoundError(f'No scenario matches {args.scenario!r}')
    print(f'[DIAG] Python={platform.python_version()} project={ROOT} scenario_count={len(scenarios)}', flush=True)
    print(f'[DIAG] args={vars(args)}', flush=True)
    adapter = OptTrajDiffAdapter(ROOT, device=args.device, samples=args.samples)
    results = []
    for xml in scenarios:
        stale_plot = ROOT / 'plots/opttrajdiff' / f'{xml.stem}_trajectory.png'
        if stale_plot.exists():
            stale_plot.unlink()
        try:
            results.append(run_one(adapter, xml, ROOT / 'plots/opttrajdiff',
                                   args.max_cycles, args.execute_steps, args.goal_tolerance,
                                   args.road_clearance, args.max_wait_cycles))
        except Exception as exc:
            import traceback
            print(f'[FAILED] {xml.name}: {exc}', file=sys.stderr)
            traceback.print_exc()
            failure_dir = ROOT / 'diagnostics/opttrajdiff' / xml.stem
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / 'summary.json').write_text(json.dumps({'scenario': xml.stem, 'success': False, 'reason': 'exception', 'error': str(exc)}, indent=2))
            results.append((0.0, False))
            if args.fail_fast:
                raise
    if results:
        print(f'\nSuccessful: {sum(reached for _, reached in results)}/{len(scenarios)} scenarios')
        print(f'Average scenario planning latency: {np.mean([lat for lat, _ in results]):.3f}s')


if __name__ == '__main__':
    main()
