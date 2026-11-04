"""Five-scenario DTP benchmark. Run: pytest -s tests/test_dtp.py"""
from pathlib import Path
import sys
import time
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.planner.dtp.inference import DTPPredictor
from src.planner.dtp.receding_horizon import plan_receding_horizon, astar_route, goal_region_contains
from src.planner.dtp.adapter import goal_center

SCENARIOS = sorted((ROOT / "data/scenarios").glob("*.xml"))
if not SCENARIOS:
    raise RuntimeError("No CommonRoad scenarios found")
RESULTS = []

@pytest.mark.parametrize("scenario_path", SCENARIOS, ids=lambda p: p.stem)
def test_dtp(scenario_path):
    scenario, pps = CommonRoadFileReader(str(scenario_path)).open()
    problem = next(iter(pps.planning_problem_dict.values()))
    predictor = DTPPredictor(ROOT)
    start = time.perf_counter()
    rollout = plan_receding_horizon(
        scenario, problem, predictor,
        execute_steps=50, samples=6, ddim_steps=20,
        max_steps=500, reference_tolerance=0.5,
        enable_route_refinement=True, strict_missing=False,
        enable_lateral_offsets=False,
    )
    elapsed = time.perf_counter() - start
    xy = np.asarray(rollout.trajectory, dtype=float)
    assert xy.ndim == 2 and xy.shape[1] == 2 and len(xy) >= 1
    assert np.isfinite(xy).all()

    # Reconstruct the same goal reference used by the planner.
    from shapely.ops import unary_union
    goal_position = next(
        state.position for state in problem.goal.state_list
        if getattr(state, "position", None) is not None
    )
    if isinstance(goal_position, np.ndarray):
        preliminary_goal = np.asarray(goal_position, dtype=float)
    elif hasattr(goal_position, "shapely_object"):
        centroid = goal_position.shapely_object.centroid
        preliminary_goal = np.array([centroid.x, centroid.y])
    elif hasattr(goal_position, "shapes"):
        shapes = [shape.shapely_object for shape in goal_position.shapes
                  if hasattr(shape, "shapely_object")]
        if not shapes:
            raise ValueError("Goal ShapeGroup contains no usable geometry")
        centroid = unary_union(shapes).centroid
        preliminary_goal = np.array([centroid.x, centroid.y])
    else:
        preliminary_goal = np.asarray(goal_position.center, dtype=float)

    _, route = astar_route(scenario, xy[0], preliminary_goal)
    goal_xy = np.asarray(
        goal_center(scenario, problem, end_margin=1.0, route=route),
        dtype=float
    )
    # Match the planner's near-entry reference, rather than the far end.
    goal_geometries = []
    for gs in problem.goal.state_list:
        pos = getattr(gs, 'position', None)
        if hasattr(pos, 'shapely_object'):
            goal_geometries.append(pos.shapely_object)
        elif hasattr(pos, 'shapes'):
            goal_geometries.extend(shape.shapely_object for shape in pos.shapes
                                   if hasattr(shape, 'shapely_object'))
    if goal_geometries:
        goal_area = unary_union(goal_geometries)
        stations = np.linspace(0.0, float(route.length),
                               max(2, int(np.ceil(route.length / 0.20)) + 1))
        inside = [float(s) for s in stations if goal_area.covers(route.interpolate(float(s)))]
        if inside:
            pt = route.interpolate(inside[0] + 0.50 * (inside[-1] - inside[0]))
            goal_xy = np.array([pt.x, pt.y], dtype=float)

    final_distance = float(np.linalg.norm(xy[-1] - goal_xy))
    traveled = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    from shapely.geometry import Point
    max_route_deviation = max(float(route.distance(Point(*point))) for point in xy)
    in_goal_region = bool(goal_region_contains(problem, xy[-1]))
    near_interior_target = final_distance <= 0.5 + 1e-5
    print(f"[DTP VALIDATION] in_goal_region={in_goal_region}, "
          f"max_route_deviation={max_route_deviation:.3f} m")

    print(f"\\n{scenario_path.stem}: status={rollout.status}, cycles={rollout.cycles}, "
          f"avg_planning={rollout.planning_latency_average:.3f}s, "
          f"total_planning={rollout.planning_latency_total:.3f}s, "
          f"wall={elapsed:.1f}s")
    print(f"[DTP PLOT] points={len(xy)}, travelled={traveled:.2f} m, "
          f"goal_distance={final_distance:.3f} m, "
          f"reported_goal_distance={rollout.goal_distance:.3f} m")
    if rollout.status == "goal_reached" and final_distance > 0.5 + 1e-5:
        print("[DTP WARNING] Goal-reached status disagrees with plotted final position!")

    RESULTS.append((scenario_path.stem, rollout.planning_latency_average,
                    rollout.planning_latency_total, len(rollout.planning_latencies)))

    fig, ax = plt.subplots(figsize=(12, 10))
    renderer = MPRenderer(ax=ax)
    scenario.draw(renderer)
    pps.draw(renderer)
    renderer.render()

    # Plot AFTER rendering the map; otherwise the road polygons hide the path.
    ax.plot(xy[:, 0], xy[:, 1], color="red", linewidth=3,
            marker="o", markersize=2, zorder=100,
            label=f"DTP executed path ({rollout.status})")
    ax.scatter(xy[-1, 0], xy[-1, 1], c="red", marker="s",
               s=85, zorder=111, label="Final position")
    ax.set_title(f"DTP Receding-Horizon Planning | {rollout.status}\\n"
                 f"Distance to reference: {final_distance:.2f} m")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()

    plot_dir = ROOT / "plots/dtp"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output = plot_dir / f"{scenario_path.stem}_trajectory.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Plot saved to: {output}")
    # The benchmark must fail when navigation did not succeed.
    # A successful Python execution is not a successful driving task.
    from src.planner.dtp.receding_horizon import goal_reached_check
    assert rollout.status == 'goal_reached' and in_goal_region and near_interior_target, (
        f'{scenario_path.stem}: planner status={rollout.status}, '
        f'final distance={final_distance:.3f} m')
    assert goal_reached_check(problem, xy[-1], goal_xy, 0.5), (
        f'{scenario_path.stem}: final position not in goal region and tolerance')



@pytest.fixture(scope='session', autouse=True)
def print_latency_summary():
    yield
    if RESULTS:
        print('\n=== DTP PLANNING LATENCY SUMMARY ===')
        for name, avg, total, calls in RESULTS:
            print(f'{name}: {avg:.3f}s/call, {calls} calls, {total:.2f}s total')
        calls = sum(row[3] for row in RESULTS)
        total = sum(row[2] for row in RESULTS)
        print(f'OVERALL weighted average: {total/max(calls,1):.3f}s/call; total={total:.2f}s')
