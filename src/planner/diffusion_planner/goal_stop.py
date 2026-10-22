
import numpy as np


def stop_at_goal(
        ego_states,
        goal_shape,
        ego_pose,
        initial_speed,
        dt=0.1,
        max_deceleration=3.0,
):
    """
    Retimes a local-frame trajectory to stop inside a
    CommonRoad goal position region.

    ego_states: (N, 4) -> local x, y, cos(yaw), sin(yaw)
    ego_pose: global x, y, cos(yaw), sin(yaw)

    Returns:
        (N, 4) stopped local-frame trajectory.

    Raises ValueError when no reachable stopping point
    exists along the predicted path.
    """
    states = np.asarray(ego_states, dtype=np.float64)
    xy = states[:, :2]

    x0, y0, c, s = ego_pose
    rotation = np.array([[c, -s], [s, c]])

    # Include current ego position as the first path point.
    path = np.vstack([np.zeros((1, 2)), xy])
    world_path = path @ rotation.T + [x0, y0]

    # Find points inside the CommonRoad goal region.
    candidates = [
        i for i in range(1, len(path))
        if goal_shape.contains_point(world_path[i])
    ]

    if not candidates:
        raise ValueError(
            "Predicted trajectory does not intersect the goal region."
        )

    # Support CommonRoad ShapeGroup and individual shapes.
    if hasattr(goal_shape, "shapes"):
        goal_shapes = list(goal_shape.shapes)
    else:
        goal_shapes = [goal_shape]

    # Extract centers of supported goal shapes.
    goal_centers = [
        np.asarray(shape.center, dtype=np.float64)
        for shape in goal_shapes
        if hasattr(shape, "center")
    ]

    if not goal_centers:
        raise ValueError(
            "No goal shape with a center was found."
        )

    # Choose a stopping point inside the goal region that
    # is closest to the center of a shape containing it.
    def goal_center_distance(index):
        point = world_path[index]

        distances = [
            np.linalg.norm(point - center)
            for shape, center in zip(goal_shapes, goal_centers)
            if shape.contains_point(point)
        ]

        return min(distances) if distances else float("inf")


    goal_index = min(
        candidates,
        key=goal_center_distance
    )

    # Identify the center associated with the selected stop point.
    selected_point = world_path[goal_index]

    containing_shapes = [
        (shape, np.asarray(shape.center, dtype=np.float64))
        for shape in goal_shapes
        if hasattr(shape, "center")
           and shape.contains_point(selected_point)
    ]

    goal_center = min(
        (center for _, center in containing_shapes),
        key=lambda center: np.linalg.norm(
            selected_point - center
        )
    )

    print(f"[GOAL] Selected goal center: {goal_center.tolist()}")
    print(f"[GOAL] Selected stop point (global): {selected_point.tolist()}")
    print(f"[GOAL] Selected stop point (local): {path[goal_index].tolist()}")

    print(
        f"[GOAL] Center (global): {goal_center.tolist()}"
    )
    print(
        f"[GOAL] Selected stop point (global): "
        f"{world_path[goal_index].tolist()}"
    )
    print(
        f"[GOAL] Selected stop point (local): "
        f"{path[goal_index].tolist()}"
    )

    goal_center_local = (
                                goal_center - np.asarray(ego_pose[:2])
                        ) @ np.array([
        [ego_pose[2], -ego_pose[3]],
        [ego_pose[3], ego_pose[2]]
    ])

    # ---------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        goal_center_local[0],
        goal_center_local[1],
        marker="*",
        s=200,
        color="purple",
        label="CommonRoad goal center",
        zorder=10
    )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axis("equal")  # Keeps distances geometrically proportional
    ax.legend(loc="upper left")

    plt.show()
    # -------------------

    if not candidates:
        raise ValueError(
            "Predicted trajectory does not enter the goal region."
        )

    # Arc length along the predicted path.
    segment_lengths = np.linalg.norm(
        np.diff(path, axis=0), axis=1
    )
    cumulative = np.concatenate([
        [0.0], np.cumsum(segment_lengths)
    ])

    v0 = max(float(initial_speed), 0.0)
    braking_distance = v0**2 / (2 * max_deceleration)

    goal_distance = cumulative[goal_index]

    if goal_distance < braking_distance:
        raise ValueError(
            f"Insufficient stopping distance: "
            f"available={goal_distance:.2f} m, "
            f"required={braking_distance:.2f} m."
        )

    print(
        f"[GOAL] Distance along path: "
        f"{goal_distance:.2f} m"
    )

    # Constant deceleration selected to stop at goal.
    required_deceleration = (
        v0**2 / (2 * goal_distance)
        if goal_distance > 0 else 0.0
    )

    if v0 > 0:
        stop_time = v0 / required_deceleration
    else:
        stop_time = 0.0

    # Allow the stopping trajectory to extend beyond the
    # original diffusion model prediction horizon.
    prediction_horizon = len(states) * dt

    num_steps = max(
        len(states),
        int(np.ceil(stop_time / dt))
    )

    result = np.zeros(
        (num_steps, states.shape[1]),
        dtype=np.float64
    )

    print(
        f"[GOAL] Required stopping time: {stop_time:.2f}s"
    )
    print(
        f"[GOAL] Original prediction horizon: "
        f"{prediction_horizon:.2f}s"
    )
    print(
        f"[GOAL] Extended trajectory horizon: "
        f"{num_steps * dt:.2f}s"
    )

    for i in range(num_steps):
        t = (i + 1) * dt

        if t >= stop_time:
            distance = goal_distance
        else:
            distance = (
                    v0 * t
                    - 0.5 * required_deceleration * t**2
            )

        distance = np.clip(distance, 0, goal_distance)

        # Interpolate along the original trajectory geometry.
        result[i, 0] = np.interp(
            distance, cumulative, path[:, 0]
        )
        result[i, 1] = np.interp(
            distance, cumulative, path[:, 1]
        )

        # Interpolate heading consistently along arc length.
        idx = min(
            np.searchsorted(cumulative, distance),
            len(path) - 1
        )
        idx = max(idx, 1)

        direction = path[idx] - path[idx - 1]
        norm = np.linalg.norm(direction)

        if norm > 1e-8:
            result[i, 2:4] = direction / norm
        elif i > 0:
            result[i, 2:4] = result[i - 1, 2:4]
        else:
            result[i, 2:4] = [1.0, 0.0]

    return result.astype(np.float32)
