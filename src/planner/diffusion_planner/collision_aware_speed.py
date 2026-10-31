
import numpy as np

from src.planner.diffusion_planner.trajectory_validation import (
    local_to_global_states,
    check_vehicle_dynamics,
    check_goal_satisfaction,
    check_collision_and_road,
)
from src.planner.diffusion_planner.trajectory_validation import (
    get_collision_details,
)


def _quintic_coefficients(distance, speed, duration):
    """s(0)=0, s'(0)=speed, s''(0)=0,
    s(T)=distance, s'(T)=0, s''(T)=0."""
    T = duration

    A = np.array([
        [T**3, T**4, T**5],
        [3*T**2, 4*T**3, 5*T**4],
        [6*T, 12*T**2, 20*T**3],
    ], dtype=float)

    b = np.array([
        distance - speed*T,
        -speed,
        0.0,
        ])

    a3, a4, a5 = np.linalg.solve(A, b)

    return np.array([0, speed, 0, a3, a4, a5])


def _motion(coeff, t):
    a0, a1, a2, a3, a4, a5 = coeff

    s = (a0 + a1*t + a2*t**2 + a3*t**3
         + a4*t**4 + a5*t**5)

    v = (a1 + 2*a2*t + 3*a3*t**2
         + 4*a4*t**3 + 5*a5*t**4)

    a = (2*a2 + 6*a3*t + 12*a4*t**2
         + 20*a5*t**3)

    j = 6*a3 + 24*a4*t + 60*a5*t**2

    return s, v, a, j


def get_far_edge_goal_target(
        goal_shape,
        reference_world_path,
        margin=0.8,
):
    """
    Find the far-edge midpoint of a CommonRoad goal
    rectangle relative to the path's direction of travel.

    Returns a target inside the goal region.
    """
    import numpy as np

    shapes = (
        list(goal_shape.shapes)
        if hasattr(goal_shape, "shapes")
        else [goal_shape]
    )

    rectangles = [
        shape for shape in shapes
        if hasattr(shape, "vertices")
           and hasattr(shape, "center")
    ]

    if not rectangles:
        raise ValueError(
            "No rectangular goal shape was found."
        )

    # Select the rectangle intersected by the path.
    intersected = [
        shape for shape in rectangles
        if any(
            shape.contains_point(p)
            for p in reference_world_path
        )
    ]

    if not intersected:
        raise ValueError(
            "Reference path does not intersect a goal rectangle."
        )

    rectangle = intersected[0]

    vertices = np.asarray(
        rectangle.vertices, dtype=float
    )[:4]

    # Determine the direction of travel near the goal.
    inside_indices = [
        i for i, p in enumerate(reference_world_path)
        if rectangle.contains_point(p)
    ]

    mid_idx = inside_indices[len(inside_indices) // 2]

    i0 = max(0, mid_idx - 2)
    i1 = min(len(reference_world_path) - 1, mid_idx + 2)

    direction = (
            reference_world_path[i1]
            - reference_world_path[i0]
    )
    direction /= max(np.linalg.norm(direction), 1e-8)

    # Identify the rectangle edge furthest along travel.
    edges = [
        (vertices[i], vertices[(i + 1) % 4])
        for i in range(4)
    ]

    # Calculate edge lengths and midpoints.
    edge_lengths = np.array([
        np.linalg.norm(p1 - p0)
        for p0, p1 in edges
    ], dtype=float)

    edge_midpoints = np.array([
        (p0 + p1) / 2.0
        for p0, p1 in edges
    ])

    # For a non-square rectangle, choose the two
    # shortest edges. For a square, use all four.
    sorted_indices = np.argsort(edge_lengths)

    if np.isclose(
            edge_lengths.max(),
            edge_lengths.min(),
            rtol=1e-3,
            atol=1e-5,
    ):
        candidate_indices = range(4)
    else:
        candidate_indices = sorted_indices[:2]

    # Choose the candidate edge whose midpoint is
    # farthest along the direction of travel.
    rectangle_center = np.asarray(
        rectangle.center, dtype=float
    )

    far_idx = max(
        candidate_indices,
        key=lambda i: np.dot(
            edge_midpoints[i] - rectangle_center,
            direction
        )
    )

    far_edge = edges[far_idx]

    far_midpoint = (
                           far_edge[0] + far_edge[1]
                   ) / 2

    # Move inward from the far-edge midpoint toward
    # the geometric center of the goal rectangle.
    rectangle_center = np.asarray(
        rectangle.center,
        dtype=np.float64
    )

    inward_direction = rectangle_center - far_midpoint
    inward_distance = np.linalg.norm(inward_direction)

    if inward_distance < 1e-8:
        raise ValueError(
            "Invalid goal rectangle: zero inward distance."
        )

    inward_direction /= inward_distance

    # Keep the target inside the rectangle.
    # Do not move beyond the rectangle center.
    safe_margin = min(
        float(margin),
        0.9 * inward_distance
    )

    target = far_midpoint + safe_margin * inward_direction

    if not rectangle.contains_point(target):
        raise ValueError(
            f"Calculated target is outside goal rectangle: {target}"
        )

    print("\n========== GOAL EDGE DEBUG ==========")
    print(f"Edge lengths: {edge_lengths}")
    print(f"Selected edge index: {far_idx}")
    print(f"Selected far-edge midpoint: {far_midpoint}")
    print(f"Target: {target}")
    print("=====================================")

    return target


def _cruise_then_brake(
        times,
        initial_speed,
        goal_distance,
        cruise_time,
):
    """
    Cruise at initial speed, then smoothly decelerate
    to zero using a cubic smoothstep velocity profile.

    Returns:
        distance, velocity, acceleration, jerk
    """
    import numpy as np

    v0 = float(initial_speed)
    tc = float(cruise_time)

    remaining_distance = goal_distance - v0 * tc

    if remaining_distance <= 0:
        raise ValueError(
            "Cruise distance exceeds goal distance."
        )

    # Cubic smoothstep velocity profile:
    # v(u) = v0 * (1 - 3u^2 + 2u^3)
    #
    # Its average velocity over braking is v0 / 2.
    brake_time = 2.0 * remaining_distance / v0

    stop_time = tc + brake_time

    t = np.asarray(times, dtype=float)

    distance = np.empty_like(t)
    velocity = np.empty_like(t)
    acceleration = np.empty_like(t)
    jerk = np.empty_like(t)

    cruise = t <= tc
    braking = (t > tc) & (t < stop_time)
    stopped = t >= stop_time

    # Cruise phase
    distance[cruise] = v0 * t[cruise]
    velocity[cruise] = v0
    acceleration[cruise] = 0.0
    jerk[cruise] = 0.0

    # Smooth braking phase
    u = (t[braking] - tc) / brake_time

    distance[braking] = (
            v0 * tc
            + v0 * brake_time
            * (u - u**3 + 0.5 * u**4)
    )

    velocity[braking] = (
            v0 * (1 - 3*u**2 + 2*u**3)
    )

    acceleration[braking] = (
            v0 / brake_time * (-6*u + 6*u**2)
    )

    jerk[braking] = (
            v0 / brake_time**2 * (-6 + 12*u)
    )

    # Fully stopped phase
    distance[stopped] = goal_distance
    velocity[stopped] = 0.0
    acceleration[stopped] = 0.0
    jerk[stopped] = 0.0

    return distance, velocity, acceleration, jerk, stop_time

def collision_aware_stop(
        reference_states,
        goal_shape,
        ego_pose,
        initial_speed,
        scenario,
        planning_problem,
        dt=0.1,
        max_acceleration=3.0,
        max_deceleration=3.0,
        max_jerk=5.0,
        vehicle_length=5.18,
        vehicle_width=2.30,
        max_planning_time=30.0,
):
    """
    Search collision-free stopping profiles along a fixed
    diffusion-predicted reference path.

    Returns:
        stopped_states, validation_results

    Raises:
        RuntimeError if no candidate passes validation.
    """
    reference_states = np.asarray(
        reference_states, dtype=float
    )

    path = np.vstack([
        np.zeros((1, 2)),
        reference_states[:, :2],
    ])

    segments = np.linalg.norm(
        np.diff(path, axis=0), axis=1
    )

    arc = np.concatenate([
        [0.0], np.cumsum(segments)
    ])

    # Remove duplicate arc-length coordinates.
    keep = np.concatenate([
        [True], np.diff(arc) > 1e-7
    ])
    arc = arc[keep]
    path = path[keep]

    if len(arc) < 2:
        raise RuntimeError("Reference path is degenerate.")

    x0, y0, c, s = ego_pose
    R = np.array([[c, -s], [s, c]])

    # Densify the reference path for more accurate goal selection.
    dense_arc = np.linspace(
        0.0,
        arc[-1],
        max(2, int(np.ceil(arc[-1] / 0.1)) + 1)
    )

    dense_path = np.column_stack([
        np.interp(dense_arc, arc, path[:, 0]),
        np.interp(dense_arc, arc, path[:, 1]),
    ])

    arc = dense_arc
    path = dense_path

    world_path = path @ R.T + [x0, y0]

    # ==================================================
    # SELECT FAR-END STOPPING POINT ALONG REFERENCE PATH
    # ==================================================

    # Keep the vehicle reference point slightly inside
    # the far boundary of the goal region.
    goal_margin = 0.8

    # Find every reference-path point inside the goal.
    inside_mask = np.array([
        goal_shape.contains_point(point)
        for point in world_path
    ], dtype=bool)

    inside_indices = np.flatnonzero(inside_mask)

    if len(inside_indices) == 0:
        raise RuntimeError(
            "Diffusion reference path does not enter "
            "the CommonRoad goal region."
        )

    # Identify the first contiguous passage through the goal.
    entry_idx = int(inside_indices[0])
    exit_idx = entry_idx

    while (
            exit_idx + 1 < len(world_path)
            and inside_mask[exit_idx + 1]
    ):
        exit_idx += 1

    # Use the last point inside the goal, then move
    # backward along the path by the desired margin.
    exit_arc = float(arc[exit_idx])
    target_arc = max(
        float(arc[entry_idx]),
        exit_arc - goal_margin
    )

    # Select the reference-path point closest to target_arc.
    candidate_indices = np.arange(
        entry_idx,
        exit_idx + 1
    )

    goal_idx = int(candidate_indices[
                       np.argmin(
                           np.abs(arc[candidate_indices] - target_arc)
                       )
                   ])

    goal_distance = float(arc[goal_idx])
    target_world = world_path[goal_idx]

    print("\n========== FAR-END GOAL TARGET ==========")
    print(f"Goal entry: {world_path[entry_idx]}")
    print(f"Goal exit: {world_path[exit_idx]}")
    print(f"Selected target: {target_world}")
    print(f"Distance along path: {goal_distance:.3f} m")
    print(f"Margin from exit: {exit_arc - goal_distance:.3f} m")
    print("=========================================")


    # No CommonRoad arrival-time constraint.
    # Time is still needed for dynamic obstacle predictions.
    initial_step = int(planning_problem.initial_state.time_step)

    horizon_steps = int(np.ceil(max_planning_time / dt))

    horizon = horizon_steps * dt
    times = np.arange(1, horizon_steps + 1) * dt

    print(f"[PLANNER] Maximum planning duration: {horizon:.1f}s")
    print("[PLANNER] CommonRoad goal deadline: ignored")


    if horizon_steps <= 0:
        raise RuntimeError(
            "Goal time interval is already past."
        )

    # Search stopping times in increments of dt.
    # Candidate profiles must stop before the goal
    # deadline and remain stopped afterward.
    stop_times = np.arange(
        dt, horizon + dt * 0.5, dt
    )

    best = None

    rejection_counts = {
        "negative_velocity": 0,
        "acceleration_limit": 0,
        "deceleration_limit": 0,
        "jerk_limit": 0,
        "discrete_dynamics": 0,
        "goal_not_satisfied": 0,
        "collision": 0,
        "road_containment": 0,
    }

    for stop_time in stop_times:
        coeff = _quintic_coefficients(
            goal_distance,
            initial_speed,
            stop_time,
        )

        # Evaluate dynamics densely, including t=0.
        dense_t = np.linspace(
            0, stop_time,
            max(101, int(np.ceil(stop_time / 0.02)) + 1),
        )

        _, v, a, j = _motion(coeff, dense_t)


        if np.min(v) < -1e-5:
            rejection_counts["negative_velocity"] += 1
            continue

        if np.max(a) > max_acceleration + 1e-5:
            rejection_counts["acceleration_limit"] += 1
            continue

        if np.min(a) < -max_deceleration - 1e-5:
            rejection_counts["deceleration_limit"] += 1
            continue

        if np.max(np.abs(j)) > max_jerk + 1e-5:
            rejection_counts["jerk_limit"] += 1
            continue


        # Sample motion at CommonRoad timesteps.
        sample_t = np.minimum(times, stop_time)
        distances, _, _, _ = _motion(
            coeff, sample_t
        )

        distances = np.clip(
            distances, 0, goal_distance
        )

        xy = np.column_stack([
            np.interp(distances, arc, path[:, 0]),
            np.interp(distances, arc, path[:, 1]),
        ])

        # Determine local heading from the path tangent.
        tangent = np.gradient(path, arc, axis=0)
        tangent_norm = np.linalg.norm(
            tangent, axis=1, keepdims=True
        )
        tangent /= np.maximum(tangent_norm, 1e-8)

        headings = np.column_stack([
            np.interp(distances, arc, tangent[:, 0]),
            np.interp(distances, arc, tangent[:, 1]),
        ])
        headings /= np.maximum(
            np.linalg.norm(headings, axis=1, keepdims=True),
            1e-8,
        )

        candidate = np.column_stack([
            xy, headings
        ]).astype(np.float32)

        global_states = local_to_global_states(
            local_states=candidate,
            ego_pose=ego_pose,
            dt=dt,
            initial_time_step=initial_step,
        )

        dynamics = check_vehicle_dynamics(
            states=global_states,
            dt=dt,
            initial_speed=initial_speed,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration,
            max_jerk=max_jerk,
        )

        # Check goal position without enforcing the original
        # CommonRoad arrival-time interval.
        final_position = np.asarray(
            global_states[-1].position,
            dtype=float
        )

        goal_position_reached = bool(
            goal_shape.contains_point(final_position)
        )

        goal = {
            "goal_reached": goal_position_reached,
            "final_state_satisfies_goal": goal_position_reached,
            "arrival_time": float(
                (global_states[-1].time_step - initial_step) * dt
            ),
        }


        safety = check_collision_and_road(
            scenario=scenario,
            states=global_states,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
        )


        if not (
                dynamics["acceleration_valid"]
                and dynamics["jerk_valid"]
                and dynamics["stopped"]
        ):
            rejection_counts["discrete_dynamics"] += 1

        if not goal["final_state_satisfies_goal"]:
            rejection_counts["goal_not_satisfied"] += 1

        if not safety["collision_free"]:
            rejection_counts["collision"] += 1

        if not safety["ego_center_on_road"]:
            rejection_counts["road_containment"] += 1


        accepted = (
                dynamics["acceleration_valid"]
                and dynamics["jerk_valid"]
                and dynamics["stopped"]
                and goal["final_state_satisfies_goal"]
                and safety["collision_free"]
                and safety["ego_center_on_road"]
        )

        if accepted:
            # Prefer a feasible profile with a shorter
            # stopping time. Change this objective later
            # if comfort or energy is more important.
            if best is None or stop_time < best[0]:
                best = (
                    stop_time,
                    candidate,
                    {
                        "stop_time": float(stop_time),
                        "goal_distance": goal_distance,
                        "dynamics": dynamics,
                        "goal": goal,
                        "safety": safety,
                    },
                )


    print("\n========== SPEED PROFILE REJECTIONS ==========")
    print(f"Target distance: {goal_distance:.3f} m")
    print(f"Initial speed: {initial_speed:.3f} m/s")
    print(f"Available horizon: {horizon:.2f} s")
    print(f"Candidates tested: {len(stop_times)}")

    for reason, count in rejection_counts.items():
        print(f"{reason}: {count}")

    print("==============================================")

    # Additional candidate profiles: cruise then brake.
    cruise_times = np.arange(
        0.0,
        min(goal_distance / initial_speed, horizon),
        dt,
    )

    cruise_rejections = {
        "invalid_distance": 0,
        "horizon": 0,
        "negative_velocity": 0,
        "acceleration": 0,
        "deceleration": 0,
        "jerk": 0,
        "discrete_dynamics": 0,
        "goal": 0,
        "collision": 0,
        "road": 0,
        "accepted": 0,
    }

    for cruise_time in cruise_times:
        try:
            (
                distances,
                velocities,
                accelerations,
                jerks,
                stop_time,
            ) = _cruise_then_brake(
                times=times,
                initial_speed=initial_speed,
                goal_distance=goal_distance,
                cruise_time=cruise_time,
            )
        except ValueError:
            continue

        if stop_time > horizon:
            cruise_rejections["horizon"] += 1
            continue

        # Validate continuous-profile dynamics on a dense grid,
        # not only at CommonRoad sampling times.
        dense_times = np.linspace(
            0.0,
            stop_time,
            max(101, int(np.ceil(stop_time / 0.02)) + 1),
        )

        _, dense_v, dense_a, dense_j, _ = _cruise_then_brake(
            times=dense_times,
            initial_speed=initial_speed,
            goal_distance=goal_distance,
            cruise_time=cruise_time,
        )

        if np.min(dense_v) < -1e-5:
            rejection_counts["negative_velocity"] += 1
            continue

        if np.max(dense_a) > max_acceleration + 1e-5:
            rejection_counts["acceleration_limit"] += 1
            continue

        if np.min(dense_a) < -max_deceleration - 1e-5:
            rejection_counts["deceleration_limit"] += 1
            continue

        if np.max(np.abs(dense_j)) > max_jerk + 1e-5:
            cruise_rejections["jerk"] += 1
            continue

        # ==========================================
        # CONSTRUCT CRUISE-THEN-BRAKE TRAJECTORY
        # ==========================================

        distances = np.clip(
            distances,
            0.0,
            goal_distance
        )

        # Interpolate positions along the reference path.
        xy = np.column_stack([
            np.interp(distances, arc, path[:, 0]),
            np.interp(distances, arc, path[:, 1]),
        ])

        # Calculate local heading along the reference path.
        tangent = np.gradient(path, arc, axis=0)

        tangent_norm = np.linalg.norm(
            tangent,
            axis=1,
            keepdims=True
        )

        tangent /= np.maximum(tangent_norm, 1e-8)

        headings = np.column_stack([
            np.interp(distances, arc, tangent[:, 0]),
            np.interp(distances, arc, tangent[:, 1]),
        ])

        headings /= np.maximum(
            np.linalg.norm(headings, axis=1, keepdims=True),
            1e-8
        )

        candidate = np.column_stack([
            xy,
            headings
        ]).astype(np.float32)

        # ==========================================
        # VALIDATE CANDIDATE
        # ==========================================

        global_states = local_to_global_states(
            local_states=candidate,
            ego_pose=ego_pose,
            dt=dt,
            initial_time_step=initial_step,
        )

        dynamics = check_vehicle_dynamics(
            states=global_states,
            dt=dt,
            initial_speed=initial_speed,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration,
            max_jerk=max_jerk,
        )

        goal = check_goal_satisfaction(
            states=global_states,
            planning_problem=planning_problem,
        )

        safety = check_collision_and_road(
            scenario=scenario,
            states=global_states,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
        )

        accepted = (
                dynamics["acceleration_valid"]
                and dynamics["jerk_valid"]
                and dynamics["stopped"]
                and goal["final_state_satisfies_goal"]
                and safety["collision_free"]
                and safety["ego_center_on_road"]
        )

        if not dynamics["acceleration_valid"] or not dynamics["jerk_valid"]:
            cruise_rejections["discrete_dynamics"] += 1

        if not goal["final_state_satisfies_goal"]:
            cruise_rejections["goal"] += 1

        if not safety["collision_free"]:
            cruise_rejections["collision"] += 1

            # Print details for a small number of candidates
            # to avoid excessive diagnostic output.
            if cruise_rejections["collision"] <= 3:
                details = get_collision_details(
                    scenario=scenario,
                    states=global_states,
                    vehicle_length=vehicle_length,
                    vehicle_width=vehicle_width,
                )

                print(
                    f"\n[CRUISE-BRAKE COLLISION] "
                    f"cruise_time={cruise_time:.2f}s, "
                    f"stop_time={stop_time:.2f}s"
                )

                for item in details[:10]:
                    print(
                        f"  t={item['time_seconds']:.2f}s, "
                        f"obstacle={item['obstacle_id']}, "
                        f"ego={item['ego_position']}"
                    )

        if not safety["ego_center_on_road"]:
            cruise_rejections["road"] += 1

        if accepted:
            cruise_rejections["accepted"] += 1

        if accepted:
            print(
                f"[CRUISE-BRAKE] Valid candidate found: "
                f"cruise={cruise_time:.2f}s, "
                f"stop={stop_time:.2f}s"
            )

            if best is None or stop_time < best[0]:
                best = (
                    stop_time,
                    candidate,
                    {
                        "stop_time": float(stop_time),
                        "cruise_time": float(cruise_time),
                        "profile_type": "cruise_then_brake",
                        "goal_distance": goal_distance,
                        "dynamics": dynamics,
                        "goal": goal,
                        "safety": safety,
                    },
                )

    print("\n========== CRUISE-BRAKE REJECTIONS ==========")
    for reason, count in cruise_rejections.items():
        print(f"{reason}: {count}")
    print("============================================")

    print("\n========== FINAL PROFILE SEARCH ==========")

    if best is None:
        print("[FAILED] No feasible stopping profile found.")
        print(f"Target distance: {goal_distance:.3f} m")
        print(f"Initial speed: {initial_speed:.3f} m/s")
        print(f"Available time: {horizon:.2f} s")
        print("==========================================")

        raise RuntimeError(
            "No feasible stopping profile found for the "
            "far-end goal target. Both quintic and "
            "cruise-then-brake candidates failed validation."
        )

    print(f"[SUCCESS] Stop time: {best[0]:.2f}s")
    print(
        f"Profile: "
        f"{best[2].get('profile_type', 'quintic')}"
    )
    print(f"Goal distance: {goal_distance:.3f} m")
    print("==========================================")

    return best[1], best[2]