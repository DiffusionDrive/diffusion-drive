import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from src.core.types import PlannerInput
from src.planner.diffusion_planner.inference import DiffusionPlannerNode
from src.planner.diffusion_planner.goal_stop import stop_at_goal
from src.planner.diffusion_planner.adapter import (
    extract_map_tensors, extract_neighbor_tensors, extract_navigation_route,
    extract_static_objects, extract_ego_state, normalize_to_ego_frame,
    map_input_to_tensors,
)
from src.planner.diffusion_planner.trajectory_validation import (
    local_to_global_states,
    check_vehicle_dynamics,
    check_goal_satisfaction,
    check_collision_and_road,
)

def valid_lanes(a):
    """Ignore zero-padded polylines; retain genuine local (0,0) points."""
    return a[np.any(np.linalg.norm(a[..., :2], axis=-1) > 1e-6, axis=1)]


def plot_lanes(ax, lanes, color, label, alpha=0.7, linewidth=1.0):
    for i, lane in enumerate(valid_lanes(lanes)):
        ax.plot(lane[:, 0], lane[:, 1], color=color, alpha=alpha,
                linewidth=linewidth, label=label if i == 0 else None)


def stats(log, name, array):
    a = np.asarray(array)
    if a.dtype == np.bool_:
        log.append(f'{name}: shape={a.shape}, true={int(a.sum())}')
        return
    finite = np.isfinite(a)
    if finite.any():
        log.append(f'{name}: shape={a.shape}, finite={finite.sum()}/{a.size}, '
                   f'min={a[finite].min():.5g}, max={a[finite].max():.5g}, '
                   f'mean={a[finite].mean():.5g}, zeros={(a == 0).sum()}')
    else:
        log.append(f'{name}: shape={a.shape}, NO FINITE VALUES')


def guidance_target(route, ego_speed, horizon=8.0):
    """Replicate the current guidance target selection for B=1 (diagnostic only)."""
    points = route[..., :2].reshape(-1, 2)
    points = points[np.abs(points).sum(axis=1) > 1e-5]
    if not len(points):
        return None
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-3]
    points = points[keep]
    start = np.argmin(np.linalg.norm(points, axis=1))
    points = np.vstack([np.zeros((1, 2)), points[start:]])
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(distances)]
    travel = np.clip(ego_speed * horizon, 3.0, 50.0)
    idx = np.searchsorted(cumulative, travel)
    if idx >= len(points):
        return points[-1]
    if idx == 0:
        return points[0]
    fraction = np.clip((travel - cumulative[idx-1]) /
                       max(cumulative[idx] - cumulative[idx-1], 1e-6), 0, 1)
    return points[idx-1] + fraction * (points[idx] - points[idx-1])


def test_run():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / 'diagnostics'
    output_dir.mkdir(exist_ok=True)
    scenario_path = project_root / 'data/scenarios/DEU_Flensburg-41_1_T-1.xml'
    assert scenario_path.exists(), f'Missing scenario: {scenario_path}'
    scenario, problem_set = CommonRoadFileReader(str(scenario_path)).open()
    planning_problem = next(iter(problem_set.planning_problem_dict.values()))

    log = [f'Scenario: {scenario_path}', f'Scenario dt: {scenario.dt}']
    torch.manual_seed(0)
    np.random.seed(0)
    planner = DiffusionPlannerNode(
        checkpoint_path=str(project_root / 'checkpoints/diffusion_planner/model.pth'),
        device='cuda' if torch.cuda.is_available() else 'cpu')
    log.append(f'Device: {planner.device}')
    log.append('Checkpoint mismatches: see console output from DiffusionPlannerNode; '
               'normalization-buffer omissions are allowed by the earlier patch.')

    # Log guidance configuration safely
    guidance = planner.config.guidance_fn

    if guidance is None:
        log.append("Guidance functions: DISABLED")
    else:
        guidance_names = [
            fn.__name__
            for fn in guidance._guidance_fns
        ]
        log.append(f"Guidance functions: {guidance_names}")


    polylines, speed, has_speed = extract_map_tensors(scenario, planning_problem)
    neighbors = extract_neighbor_tensors(scenario, current_time_step=0)
    static_objs = extract_static_objects(scenario)
    nav_route, route_speed, route_has_speed = extract_navigation_route(scenario, planning_problem)
    ego_state = extract_ego_state(planning_problem)
    ego_pose = ego_state[:4]
    local_poly, local_neigh, local_route, local_stat = normalize_to_ego_frame(
        ego_pose, polylines, neighbors, nav_route, static_objs)

    vel = float(np.linalg.norm(ego_state[4:6]))
    accel = float(np.linalg.norm(ego_state[6:8]))
    local_ego = np.array([0, 0, 1, 0, vel, 0, accel, 0, 2.30, 5.18], dtype=np.float32)
    live_input = PlannerInput(
        ego_current_state=local_ego, neighbor_agents_past=local_neigh,
        lanes=local_poly, lanes_speed_limit=speed,
        lanes_has_speed_limit=has_speed, static_objects=local_stat,
        route_lanes=local_route, route_lanes_speed_limit=route_speed,
        route_lanes_has_speed_limit=route_has_speed)

    log.append(f'Ego world pose [x,y,cos,sin]: {ego_pose.tolist()}')
    log.append(f'Ego speed={vel:.3f} m/s, acceleration magnitude={accel:.3f} m/s^2')
    log.append(f'Valid local lanes={len(valid_lanes(local_poly))}; '
               f'valid route slots={len(valid_lanes(local_route))} '
               '(may include duplicated padding)')
    target = guidance_target(local_route, vel)
    log.append(f'Approximate guidance target (local XY): {None if target is None else target.tolist()}')
    log.append('NOTE: guidance target is independently recomputed, not captured inside sampler.')

    for name, value in vars(live_input).items():
        if value is not None:
            stats(log, f'input.{name}', value)
    tensors = map_input_to_tensors(live_input, planner.device)
    with torch.no_grad():
        normalized = planner.config.observation_normalizer(tensors)
    for name, value in normalized.items():
        stats(log, f'normalized.{name}', value.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_lanes(ax, local_poly, 'gray', 'Local lane centerlines')
    plot_lanes(ax, local_route, 'tab:blue', 'Local route', 0.9, 2)
    ax.scatter([0], [0], color='green', s=70, label='Ego (0,0)', zorder=10)
    ax.arrow(0, 0, 8, 0, width=0.2, color='green', length_includes_head=True)
    if target is not None:
        ax.scatter([target[0]], [target[1]], color='orange', marker='x', s=90,
                   label='Approx. guidance target')
    ax.set(title='01 - Local input map', xlabel='Local X (m)', ylabel='Local Y (m)')
    ax.axis('equal'); ax.grid(True); ax.legend()
    fig.savefig(output_dir / '01_local_input_map.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    trajectory = planner.plan(live_input)

    goal_shape = planning_problem.goal.state_list[0].position

    stopped_states = stop_at_goal(
        ego_states=trajectory.ego_states,
        goal_shape=goal_shape,
        ego_pose=ego_pose,
        initial_speed=vel,
        dt=scenario.dt,
        max_deceleration=3.0,
    )

    # Extend the stopped trajectory through the goal time interval.
    goal_states = planning_problem.goal.state_list

    goal_end_step = max(
        int(gs.time_step.end)
        for gs in goal_states
        if hasattr(gs, "time_step")
    )

    initial_step = int(planning_problem.initial_state.time_step)

    required_samples = goal_end_step - initial_step

    if len(stopped_states) < required_samples:
        extra_samples = required_samples - len(stopped_states)

        # Hold the final position and heading.
        stationary_states = np.repeat(
            stopped_states[-1:, :],
            extra_samples,
            axis=0
        )

        stopped_states = np.vstack([
            stopped_states,
            stationary_states
        ])

    print(
        f"[GOAL] Extended trajectory to "
        f"{len(stopped_states)} samples"
    )

    # ==================================================
    # VALIDATION OF STOPPED DIFFUSION-PLANNER TRAJECTORY
    # ==================================================

    global_states = local_to_global_states(
        local_states=stopped_states,
        ego_pose=ego_pose,
        dt=scenario.dt,
        initial_time_step=initial_step,
    )

    print("\n========== GOAL CONSTRAINT DETAILS ==========")

    for i, goal_state in enumerate(planning_problem.goal.state_list):
        print(f"\nGoal state {i}:")

        for attribute in (
                "position",
                "orientation",
                "velocity",
                "time_step",
        ):
            if hasattr(goal_state, attribute):
                print(
                    f"  {attribute}: "
                    f"{getattr(goal_state, attribute)}"
                )

    print("\nFinal predicted state:")
    final_state = global_states[-1]

    print(f"  Position: {final_state.position}")
    print(f"  Orientation: {final_state.orientation:.4f} rad")
    print(f"  Velocity: {final_state.velocity:.4f} m/s")
    print(f"  Time step: {final_state.time_step}")
    print(f"  Time: {final_state.time_step * scenario.dt:.2f} s")

    print("=============================================")

    from commonroad.geometry.shape import Rectangle
    from commonroad.scenario.trajectory import Trajectory
    from commonroad.prediction.prediction import TrajectoryPrediction
    from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
        create_collision_object,
    )

# 1. Vehicle dynamics
    dynamics = check_vehicle_dynamics(
        states=global_states,
        dt=scenario.dt,
        initial_speed=vel,
        max_acceleration=3.0,
        max_deceleration=3.0,
        max_jerk=5.0,
    )

    # 2. CommonRoad goal satisfaction
    goal_results = check_goal_satisfaction(
        states=global_states,
        planning_problem=planning_problem,
    )

    # 3. Collision and road containment
    safety = check_collision_and_road(
        scenario=scenario,
        states=global_states,
        vehicle_length=5.18,
        vehicle_width=2.30,
    )

    print("\n========== TRAJECTORY VALIDATION ==========")

    print("\n[1] VEHICLE DYNAMICS")
    for key, value in dynamics.items():
        print(f"{key}: {value}")

    print("\n[2] COMMONROAD GOAL")
    for key, value in goal_results.items():
        print(f"{key}: {value}")

    print("\n[3] SAFETY")
    for key, value in safety.items():
        print(f"{key}: {value}")

    print("==========================================")

    # Save results for analysis
    validation_path = output_dir / "validation.txt"

    with open(validation_path, "w", encoding="utf-8") as f:
        for section_name, results in [
            ("VEHICLE DYNAMICS", dynamics),
            ("COMMONROAD GOAL", goal_results),
            ("SAFETY", safety),
        ]:
            f.write(f"\n[{section_name}]\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    print(f"Validation saved to: {validation_path}")

    stopped_xy = np.asarray(stopped_states[:, :2])

    # Calculate the actual stopping time.
    step_distances = np.linalg.norm(
        np.diff(
            np.vstack([np.zeros((1, 2)), stopped_xy]),
            axis=0
        ),
        axis=1
    )

    speeds = step_distances / scenario.dt

    print("\n=== STOPPING VERIFICATION ===")
    print(f"Original prediction samples: {len(trajectory.ego_states)}")
    print(f"Stopped trajectory samples: {len(stopped_states)}")
    print(f"Final local XY: {stopped_xy[-1]}")
    print(f"Final speed: {speeds[-1]:.5f} m/s")
    print(f"Maximum speed: {speeds.max():.3f} m/s")

    # Convert final local position to global.
    x0, y0, c, s = ego_pose
    rotation = np.array([[c, -s], [s, c]])

    final_global = stopped_xy[-1] @ rotation.T + [x0, y0]

    print(f"Final global XY: {final_global}")
    print(
        f"Final position inside goal: "
        f"{goal_shape.contains_point(final_global)}"
    )

    local_xy = stopped_states[:, :2]

    # Inspect predicted trajectory displacement over time
    print("\n=== TRAJECTORY TEMPORAL DIAGNOSTICS ===")

    dt = 0.1
    initial_speed = 7.831

    previous_xy = np.array([0.0, 0.0])

    for i in [0, 1, 4, 9, 19, 39, 59, 79]:
        current_xy = local_xy[i]

        # Average speed from the initial ego position
        elapsed_time = (i + 1) * dt
        displacement = np.linalg.norm(current_xy)
        avg_speed = displacement / elapsed_time

        print(
            f"t={elapsed_time:.1f}s | "
            f"x={current_xy[0]:.2f} | "
            f"y={current_xy[1]:.2f} | "
            f"avg_speed={avg_speed:.2f} m/s"
        )

    # Calculate speed between consecutive predicted states,
    # including the current ego position at t=0.
    trajectory_with_origin = np.vstack([
        np.zeros((1, 2)),
        local_xy
    ])

    step_distances = np.linalg.norm(
        np.diff(trajectory_with_origin, axis=0),
        axis=1
    )

    predicted_speeds = step_distances / dt

    print(f"\nInitial ego speed: {initial_speed:.2f} m/s")
    print(f"First predicted speed: {predicted_speeds[0]:.2f} m/s")
    print(f"Mean predicted speed: {predicted_speeds.mean():.2f} m/s")
    print(f"Maximum predicted speed: {predicted_speeds.max():.2f} m/s")
    print(f"Final predicted speed: {predicted_speeds[-1]:.2f} m/s")


    stats(log, 'prediction.local_xy', local_xy)
    log.append(f'Local first 5 XY: {local_xy[:5].tolist()}')
    log.append(f'Local last XY: {local_xy[-1].tolist()}')

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_lanes(ax, local_poly, 'gray', 'Local lane centerlines')
    plot_lanes(ax, local_route, 'tab:blue', 'Local route', 0.9, 2)
    ax.plot(local_xy[:, 0], local_xy[:, 1], 'r.-', markersize=3, label='Model prediction')
    ax.scatter([0], [0], color='green', s=70, label='Ego')
    if target is not None:
        ax.scatter([target[0]], [target[1]], color='orange', marker='x', s=90,
                   label='Approx. guidance target')
    ax.set(title='02 - Local prediction vs route', xlabel='Local X (m)', ylabel='Local Y (m)')
    ax.axis('equal'); ax.grid(True); ax.legend()
    fig.savefig(output_dir / '02_local_prediction.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    ego_x, ego_y, cos_yaw, sin_yaw = ego_pose
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    global_xy = local_xy @ rotation.T + np.array([ego_x, ego_y])
    stats(log, 'prediction.global_xy', global_xy)
    log.append(f'Global last XY: {global_xy[-1].tolist()}')
    goal_shape = planning_problem.goal.state_list[0].position
    inside = [i for i, p in enumerate(global_xy) if goal_shape.contains_point(p)]
    log.append(f'Goal reached in predicted samples: {bool(inside)}; first index={inside[0] if inside else None}')

    fig, ax = plt.subplots(figsize=(10, 10))
    rnd = MPRenderer(ax=ax)
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()
    ax.plot(global_xy[:, 0], global_xy[:, 1], 'r.-', markersize=3,
            linewidth=2, label='Full predicted trajectory', zorder=20)
    ax.scatter([ego_x], [ego_y], color='green', s=50, label='Ego', zorder=21)
    ax.set(title='03 - Global prediction (not truncated)', xlabel='Global X (m)', ylabel='Global Y (m)')
    ax.axis('equal'); ax.legend()
    fig.savefig(output_dir / '03_global_prediction.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    (output_dir / 'diagnostics.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    print(f'Diagnostics saved to: {output_dir}')
    print('\n'.join(log[:10]))


if __name__ == '__main__':
    test_run()
