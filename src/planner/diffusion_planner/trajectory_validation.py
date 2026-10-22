
import numpy as np

from commonroad.scenario.state import KSState
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_object,
)
from commonroad_dc.collision.trajectory_queries import (
    trajectory_queries,
)


def local_to_global_states(
        local_states,
        ego_pose,
        dt,
        initial_time_step=0,
):
    """
    Convert local (x,y,cos,sin) trajectory into global
    CommonRoad KSState objects, including velocity.

    Returns states at time steps 1..N.
    """
    local_states = np.asarray(local_states, dtype=float)

    x0, y0, c, s = ego_pose
    R = np.array([[c, -s], [s, c]])

    local_xy = local_states[:, :2]
    global_xy = local_xy @ R.T + [x0, y0]

    # Include current ego position to calculate velocity.
    points = np.vstack([[x0, y0], global_xy])

    displacement = np.diff(points, axis=0)
    velocities = np.linalg.norm(displacement, axis=1) / dt

    headings = np.arctan2(
        local_states[:, 3],
        local_states[:, 2],
    ) + np.arctan2(s, c)

    headings = (headings + np.pi) % (2 * np.pi) - np.pi

    states = []

    for i in range(len(global_xy)):
        states.append(
            KSState(
                position=global_xy[i],
                orientation=float(headings[i]),
                velocity=float(velocities[i]),
                time_step=initial_time_step + i + 1,
            )
        )

    return states


def check_vehicle_dynamics(
        states,
        dt,
        initial_speed,
        max_acceleration=3.0,
        max_deceleration=3.0,
        max_jerk=5.0,
):
    """
    Evaluate longitudinal speed, acceleration and jerk.
    """
    speeds = np.array(
        [initial_speed] + [s.velocity for s in states],
        dtype=float,
        )

    acceleration = np.diff(speeds) / dt
    jerk = np.diff(acceleration) / dt

    results = {
        "initial_predicted_speed": float(speeds[0]),
        "final_speed": float(speeds[-1]),
        "maximum_speed": float(np.max(speeds)),
        "maximum_acceleration": float(
            np.max(acceleration)
        ) if len(acceleration) else 0.0,
        "maximum_deceleration": float(
            -np.min(acceleration)
        ) if len(acceleration) else 0.0,
        "maximum_absolute_jerk": float(
            np.max(np.abs(jerk))
        ) if len(jerk) else 0.0,
    }

    results["acceleration_valid"] = (
            np.all(acceleration <= max_acceleration + 1e-6)
            and np.all(acceleration >= -max_deceleration - 1e-6)
    )

    results["jerk_valid"] = (
        np.all(np.abs(jerk) <= max_jerk + 1e-6)
    )

    results["stopped"] = speeds[-1] < 0.1

    return results


def check_goal_satisfaction(states, planning_problem):
    """
    Evaluate CommonRoad goal constraints using its
    built-in goal-state checking functionality.
    """
    goal = planning_problem.goal

    satisfied_steps = []

    for state in states:
        try:
            if goal.is_reached(state):
                satisfied_steps.append(state.time_step)
        except Exception as exc:
            raise RuntimeError(
                f"Goal evaluation failed at step "
                f"{state.time_step}: {exc}"
            ) from exc

    return {
        "goal_reached": bool(satisfied_steps),
        "first_goal_step": (
            satisfied_steps[0] if satisfied_steps else None
        ),
        "final_state_satisfies_goal": bool(
            goal.is_reached(states[-1])
        ),
    }


def check_collision_and_road(
        scenario,
        states,
        vehicle_length=5.18,
        vehicle_width=2.30,
):
    """
    Check collision against CommonRoad obstacles and
    perform a basic lanelet containment check.
    """
    ego_shape = Rectangle(
        length=vehicle_length,
        width=vehicle_width,
    )

    trajectory = Trajectory(
        initial_time_step=states[0].time_step,
        state_list=states,
    )

    prediction = TrajectoryPrediction(
        trajectory=trajectory,
        shape=ego_shape,
    )

    ego_collision = create_collision_object(prediction)

    collisions = []

    for obstacle in scenario.obstacles:
        obstacle_collision = create_collision_object(obstacle)

        if ego_collision.collide(obstacle_collision):
            collisions.append(obstacle.obstacle_id)

    # Basic road check: vehicle center point in lanelet network.
    # This does NOT guarantee that the full vehicle fits.
    outside_road_steps = []

    for state in states:
        lanelet_ids = (
            scenario.lanelet_network.find_lanelet_by_position(
                [state.position]
            )[0]
        )

        if not lanelet_ids:
            outside_road_steps.append(state.time_step)

    return {
        "collision_free": len(collisions) == 0,
        "colliding_obstacle_ids": collisions,
        "ego_center_on_road": len(outside_road_steps) == 0,
        "outside_road_steps": outside_road_steps,
    }
