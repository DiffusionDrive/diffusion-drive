"""Shared geometry utilities for trajectory evaluation.

Provides the WOSAC-conformant signed distance to oriented road-edge
polylines — the shared primitive behind the physics boundary loss, the
safety boundary reward, and offroad evaluation — together with the
:class:`RoadEdges` container and the off-road hinge penalty built on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from diffav.core.constants import (
    CYCLIC_POLYLINE_TOLERANCE_M2,
    EXTREMELY_LARGE_DISTANCE,
    ROAD_EDGE_Z_STRETCH,
)


def _dot_2d(a: jax.Array, b: jax.Array) -> jax.Array:
    """Dot product over the last (2D) axis."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _cross_2d(a: jax.Array, b: jax.Array) -> jax.Array:
    """Signed magnitude of the 2D cross product over the last axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _divide_no_nan(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Elementwise division returning zero where the denominator is zero."""
    safe = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(denominator == 0.0, 0.0, numerator / safe)


def wrap_angle(angle: jax.Array) -> jax.Array:
    """Wrap angles into the principal range ``[-pi, pi]``.

    Args:
        angle: Angles in radians, any shape.

    Returns:
        Wrapped angles with the same shape.
    """
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def to_agent_frame(trajectories: jax.Array, reference_pose: jax.Array) -> jax.Array:
    """Transform trajectories into each agent's own local frame.

    Re-expresses every agent's ``[x, y, heading, speed]`` trajectory relative to
    that agent's reference pose: translate by the reference position, then rotate
    by the negative reference heading so forward is ``+x_local``. This is the
    agent-centric (roto-translation) convention used by the reference trajectory
    models; predicting in it makes each agent's target a small local displacement
    around the origin rather than a large absolute offset. The scalar speed
    channel is rotation-invariant and passes through unchanged. Inverted by
    :func:`from_agent_frame`.

    Args:
        trajectories: ``(..., T, 4)`` with channels ``[x, y, heading, speed]``.
        reference_pose: ``(..., 3)`` per-agent reference ``[x0, y0, yaw0]``,
            broadcasting over the trajectory's step axis.

    Returns:
        Local-frame trajectories with the same shape as ``trajectories``.
    """
    x0 = reference_pose[..., 0, None]
    y0 = reference_pose[..., 1, None]
    yaw0 = reference_pose[..., 2, None]
    cos_yaw = jnp.cos(yaw0)
    sin_yaw = jnp.sin(yaw0)
    delta_x = trajectories[..., 0] - x0
    delta_y = trajectories[..., 1] - y0
    local_x = delta_x * cos_yaw + delta_y * sin_yaw
    local_y = -delta_x * sin_yaw + delta_y * cos_yaw
    local_heading = wrap_angle(trajectories[..., 2] - yaw0)
    return jnp.stack([local_x, local_y, local_heading, trajectories[..., 3]], axis=-1)


def from_agent_frame(local: jax.Array, reference_pose: jax.Array) -> jax.Array:
    """Invert :func:`to_agent_frame`, recovering the shared (world/ego) frame.

    Rotate each agent's local ``[x, y, heading, speed]`` trajectory by the
    positive reference heading, then translate by the reference position. The
    scalar speed channel passes through unchanged.

    Args:
        local: ``(..., T, 4)`` local-frame trajectories ``[x, y, heading, speed]``.
        reference_pose: ``(..., 3)`` per-agent reference ``[x0, y0, yaw0]``.

    Returns:
        Trajectories in the reference pose's frame, same shape as ``local``.
    """
    x0 = reference_pose[..., 0, None]
    y0 = reference_pose[..., 1, None]
    yaw0 = reference_pose[..., 2, None]
    cos_yaw = jnp.cos(yaw0)
    sin_yaw = jnp.sin(yaw0)
    local_x = local[..., 0]
    local_y = local[..., 1]
    world_x = local_x * cos_yaw - local_y * sin_yaw + x0
    world_y = local_x * sin_yaw + local_y * cos_yaw + y0
    world_heading = wrap_angle(local[..., 2] + yaw0)
    return jnp.stack([world_x, world_y, world_heading, local[..., 3]], axis=-1)


def _signed_distance_chunk(
    xyzs: jax.Array,
    polylines: jax.Array,
    is_polyline_cyclic: jax.Array,
    z_stretch: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Closest-segment reduction over one chunk of polylines.

    The per-query search core of :func:`signed_distance_to_polylines`, kept
    separate so it can be tiled over polyline chunks. Selects, per query point,
    the segment of minimum ``z_stretch``-scaled 3D distance and reports that
    minimum distance alongside the WOSAC sign and planar 2D distance at it; the
    sign follows the adjacent pair's local convexity when a point projects
    beyond a segment end, exactly as in the reference.

    Args:
        xyzs: Query points, shape ``(num_points, 3)``.
        polylines: Padded polylines, shape ``(num_polylines, num_segments+1,
            4)`` (x, y, z, validity).
        is_polyline_cyclic: Boolean flags, shape ``(num_polylines,)``.
        z_stretch: Factor scaling vertical distances during association.

    Returns:
        A triple of ``(min_distance_3d, sign, distance_2d)``, each shape
        ``(num_points,)``: the minimum scaled 3D distance and, at that segment,
        the signed convention and the planar 2D distance.
    """
    num_points = xyzs.shape[0]
    num_polylines = polylines.shape[0]
    num_segments = polylines.shape[1] - 1

    # shape: (num_polylines, num_segments+1)
    is_point_valid = polylines[:, :, 3].astype(bool)
    # shape: (num_polylines, num_segments)
    is_segment_valid = is_point_valid[:, :-1] & is_point_valid[:, 1:]

    # shape: (num_points, num_polylines, num_segments, 3)
    xyz_starts = polylines[jnp.newaxis, :, :-1, :3]
    xyz_ends = polylines[jnp.newaxis, :, 1:, :3]
    start_to_point = xyzs[:, jnp.newaxis, jnp.newaxis, :3] - xyz_starts
    start_to_end = xyz_ends - xyz_starts

    # Relative coordinate of the point projection on each segment.
    # shape: (num_points, num_polylines, num_segments)
    rel_t = _divide_no_nan(
        _dot_2d(start_to_point[..., :2], start_to_end[..., :2]),
        _dot_2d(start_to_end[..., :2], start_to_end[..., :2]),
    )

    # Negative on the port side of a segment, positive on starboard.
    n = jnp.sign(_cross_2d(start_to_point[..., :2], start_to_end[..., :2]))

    # Distance to each segment: 3D (z-stretched) for association, 2D for the
    # returned magnitude.
    segment_to_point = start_to_point - (start_to_end * jnp.clip(rel_t, 0.0, 1.0)[..., jnp.newaxis])
    stretch_vector = jnp.array([1.0, 1.0, z_stretch], dtype=xyzs.dtype)
    distance_to_segment_3d = jnp.linalg.norm(segment_to_point * stretch_vector, axis=-1)
    distance_to_segment_2d = jnp.linalg.norm(segment_to_point[..., :2], axis=-1)

    # Local convexity of consecutive segment pairs, with edge segments
    # wrapped so the first/last pair is well defined.
    # shape: (num_points, num_polylines, num_segments+2, 2)
    start_to_end_padded = jnp.concatenate(
        [
            start_to_end[:, :, -1:, :2],
            start_to_end[..., :2],
            start_to_end[:, :, :1, :2],
        ],
        axis=-2,
    )
    # shape: (num_points, num_polylines, num_segments+1)
    is_locally_convex = (
        _cross_2d(start_to_end_padded[:, :, :-1], start_to_end_padded[:, :, 1:]) > 0.0
    )

    # Shifted views of `n` and segment validity: rolled for cyclic
    # polylines, edge-padded otherwise.
    cyclic_pn = is_polyline_cyclic[jnp.newaxis, :, jnp.newaxis]
    n_prior = jnp.concatenate(
        [jnp.where(cyclic_pn, n[:, :, -1:], n[:, :, :1]), n[:, :, :-1]], axis=-1
    )
    n_next = jnp.concatenate(
        [n[:, :, 1:], jnp.where(cyclic_pn, n[:, :, :1], n[:, :, -1:])], axis=-1
    )
    cyclic_p = is_polyline_cyclic[:, jnp.newaxis]
    is_prior_segment_valid = jnp.concatenate(
        [
            jnp.where(cyclic_p, is_segment_valid[:, -1:], is_segment_valid[:, :1]),
            is_segment_valid[:, :-1],
        ],
        axis=-1,
    )
    is_next_segment_valid = jnp.concatenate(
        [
            is_segment_valid[:, 1:],
            jnp.where(cyclic_p, is_segment_valid[:, :1], is_segment_valid[:, -1:]),
        ],
        axis=-1,
    )

    # Sign selection: projection within the segment keeps `n`; beyond an end
    # it follows the convexity of the adjacent pair.
    sign_if_before = jnp.where(
        is_locally_convex[:, :, :-1], jnp.maximum(n, n_prior), jnp.minimum(n, n_prior)
    )
    sign_if_after = jnp.where(
        is_locally_convex[:, :, 1:], jnp.maximum(n, n_next), jnp.minimum(n, n_next)
    )
    sign_to_segment = jnp.where(
        (rel_t < 0.0) & is_prior_segment_valid,
        sign_if_before,
        jnp.where((rel_t > 1.0) & is_next_segment_valid, sign_if_after, n),
    )

    # Flatten polylines together and mask invalid segments out of the search.
    flat = (num_points, num_polylines * num_segments)
    distance_to_segment_3d = distance_to_segment_3d.reshape(flat)
    distance_to_segment_2d = distance_to_segment_2d.reshape(flat)
    sign_to_segment = sign_to_segment.reshape(flat)
    is_segment_valid_flat = is_segment_valid.reshape(num_polylines * num_segments)

    # cast: the jax 0.9 stubs annotate three-argument jnp.where with the
    # one-argument tuple return included; the three-argument form is an Array.
    distance_to_segment_3d = cast(
        jax.Array,
        jnp.where(
            is_segment_valid_flat[jnp.newaxis], distance_to_segment_3d, EXTREMELY_LARGE_DISTANCE
        ),
    )
    distance_to_segment_2d = cast(
        jax.Array,
        jnp.where(
            is_segment_valid_flat[jnp.newaxis], distance_to_segment_2d, EXTREMELY_LARGE_DISTANCE
        ),
    )

    # Closest segment by (z-stretched) 3D distance; signed 2D result.
    closest_segment_index = jnp.argmin(distance_to_segment_3d, axis=-1)
    min_distance_3d = jnp.take_along_axis(
        distance_to_segment_3d, closest_segment_index[:, jnp.newaxis], axis=-1
    )[:, 0]
    distance_sign = jnp.take_along_axis(
        sign_to_segment, closest_segment_index[:, jnp.newaxis], axis=-1
    )[:, 0]
    distance_2d = jnp.take_along_axis(
        distance_to_segment_2d, closest_segment_index[:, jnp.newaxis], axis=-1
    )[:, 0]
    return min_distance_3d, distance_sign, distance_2d


def signed_distance_to_polylines(
    xyzs: jax.Array,
    polylines: jax.Array,
    is_polyline_cyclic: jax.Array | None = None,
    z_stretch: float = 1.0,
    chunk_size: int | None = None,
) -> jax.Array:
    """Signed 2D distance to the boundary defined by oriented polylines.

    Mirrors the official WOSAC map metric: negative distances are inside the
    boundary (on-road), positive outside (off-road). Polylines must be oriented
    with counterclockwise winding, so their port side is on-road. The closest
    segment per query point is selected by 3D distance with the vertical
    component scaled by ``z_stretch``; the returned magnitude is the planar 2D
    distance, and beyond a segment end the sign follows the local convexity of
    the adjacent segment pair, exactly as in the reference implementation.

    Args:
        xyzs: Query points, shape ``(num_points, 3)``.
        polylines: Padded polylines, shape ``(num_polylines, num_segments+1,
            4)`` holding x, y, z, and a validity flag per point (see
            :func:`stack_polylines`). Degenerate (zero-length) segments cause
            undefined behaviour, as in the reference.
        is_polyline_cyclic: Boolean flags, shape ``(num_polylines,)``. When
            ``None``, all polylines are treated as open.
        z_stretch: Factor scaling vertical distances during closest-segment
            association. Road-edge evaluation uses
            :data:`diffav.core.constants.ROAD_EDGE_Z_STRETCH`.
        chunk_size: When set, tile the closest-segment search over polyline
            chunks of this many polylines via :func:`jax.lax.scan`, bounding
            peak memory to ``num_points * chunk_size * num_segments`` instead of
            the full product. The running minimum over chunks reproduces the
            monolithic ``argmin`` exactly, so the result is independent of
            ``chunk_size``; ``None`` (default) evaluates every polyline at once.

    Returns:
        Signed planar distances, shape ``(num_points,)``.
    """
    num_polylines = polylines.shape[0]
    if is_polyline_cyclic is None:
        is_polyline_cyclic = jnp.zeros((num_polylines,), dtype=bool)

    if chunk_size is None or chunk_size >= num_polylines:
        _, distance_sign, distance_2d = _signed_distance_chunk(
            xyzs, polylines, is_polyline_cyclic, z_stretch
        )
        return distance_sign * distance_2d

    # Pad the polyline axis to a whole number of chunks with all-invalid rows
    # (masked out of the search), then fold a running minimum over the chunks.
    max_length = polylines.shape[1]
    pad = (-num_polylines) % chunk_size
    if pad:
        polylines = jnp.concatenate(
            [polylines, jnp.zeros((pad, max_length, 4), dtype=polylines.dtype)], axis=0
        )
        is_polyline_cyclic = jnp.concatenate(
            [is_polyline_cyclic, jnp.zeros((pad,), dtype=bool)], axis=0
        )
    num_chunks = polylines.shape[0] // chunk_size
    polyline_chunks = polylines.reshape(num_chunks, chunk_size, max_length, 4)
    cyclic_chunks = is_polyline_cyclic.reshape(num_chunks, chunk_size)

    num_points = xyzs.shape[0]
    init = (
        jnp.full((num_points,), EXTREMELY_LARGE_DISTANCE, dtype=xyzs.dtype),
        jnp.zeros((num_points,), dtype=xyzs.dtype),
        jnp.zeros((num_points,), dtype=xyzs.dtype),
    )

    def _fold(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        chunk: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        best_3d, best_sign, best_2d = carry
        chunk_polylines, chunk_cyclic = chunk
        min_3d, sign, dist_2d = _signed_distance_chunk(
            xyzs, chunk_polylines, chunk_cyclic, z_stretch
        )
        is_closer = min_3d < best_3d
        return (
            jnp.where(is_closer, min_3d, best_3d),
            jnp.where(is_closer, sign, best_sign),
            jnp.where(is_closer, dist_2d, best_2d),
        ), None

    (_, distance_sign, distance_2d), _ = jax.lax.scan(_fold, init, (polyline_chunks, cyclic_chunks))
    return distance_sign * distance_2d


def _write_polyline(destination: np.ndarray, index: int, polyline: np.ndarray) -> bool:
    """Write one polyline into ``destination[index]`` and report its cyclicity.

    Fills the leading points with ``(x, y, z, validity=1)``, leaving the
    remaining slots at their zero (invalid) initialization.

    Args:
        destination: Pre-zeroed array of shape ``(num_polylines, max_length,
            4)`` written in place.
        index: Row to write into.
        polyline: Points of shape ``(num_points, 3)``; ``num_points`` must not
            exceed ``destination.shape[1]``.

    Returns:
        ``True`` when the polyline's endpoints lie within
        :data:`CYCLIC_POLYLINE_TOLERANCE_M2` of each other (a closed loop).
    """
    length = polyline.shape[0]
    destination[index, :length, :3] = polyline
    destination[index, :length, 3] = 1.0
    gap_sq = float(np.sum((polyline[0] - polyline[-1]) ** 2))
    return gap_sq < CYCLIC_POLYLINE_TOLERANCE_M2


def stack_polylines(
    polylines: Sequence[np.ndarray],
) -> tuple[jax.Array, jax.Array]:
    """Pack variable-length polylines into the padded tensor format.

    Mirrors the reference tensorization: polylines shorter than two points
    are dropped, the rest are zero-padded (validity 0) to the longest
    length, and a polyline whose endpoints lie within
    :data:`CYCLIC_POLYLINE_TOLERANCE_M2` of each other is flagged cyclic.

    Args:
        polylines: Sequence of arrays with shape ``(num_points_i, 3)``.

    Returns:
        A pair of the padded polylines array with shape
        ``(num_polylines, max_length, 4)`` (x, y, z, validity) and the
        boolean cyclic flags with shape ``(num_polylines,)``.

    Raises:
        ValueError: If no polyline has at least two points.
    """
    usable = [np.asarray(p, dtype=np.float32) for p in polylines if len(p) >= 2]
    if not usable:
        raise ValueError("No usable polyline with at least two points was provided.")

    max_length = max(p.shape[0] for p in usable)
    stacked = np.zeros((len(usable), max_length, 4), dtype=np.float32)
    cyclic = np.zeros((len(usable),), dtype=bool)
    for i, polyline in enumerate(usable):
        cyclic[i] = _write_polyline(stacked, i, polyline)
    return jnp.asarray(stacked), jnp.asarray(cyclic)


def stack_polylines_fixed(
    polylines: Sequence[np.ndarray],
    *,
    max_polylines: int,
    max_length: int,
) -> tuple[jax.Array, jax.Array]:
    """Pack polylines into a global fixed ``(max_polylines, max_length, 4)`` tensor.

    Unlike :func:`stack_polylines`, the output shape is independent of the
    input, so a jitted consumer compiles once and reuses the trace across
    every scene. Excess polylines and points beyond the caps are dropped,
    shorter inputs are zero-padded (validity 0), and an empty input yields an
    all-invalid tensor rather than raising — an edge-less scene contributes no
    boundary. To preserve a long polyline's full extent, resample it to
    ``max_length`` before packing; this function truncates.

    Args:
        polylines: Sequence of arrays with shape ``(num_points_i, 3)``.
        max_polylines: Fixed number of polyline rows.
        max_length: Fixed number of points per polyline.

    Returns:
        A pair of the padded polylines array with shape ``(max_polylines,
        max_length, 4)`` and the boolean cyclic flags ``(max_polylines,)``.
    """
    usable = [np.asarray(p, dtype=np.float32)[:max_length] for p in polylines if len(p) >= 2]
    usable = usable[:max_polylines]
    stacked = np.zeros((max_polylines, max_length, 4), dtype=np.float32)
    cyclic = np.zeros((max_polylines,), dtype=bool)
    for i, polyline in enumerate(usable):
        cyclic[i] = _write_polyline(stacked, i, polyline)
    return jnp.asarray(stacked), jnp.asarray(cyclic)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, kw_only=True)
class RoadEdges:
    """Oriented road-edge polylines in the padded tensor format.

    Registered as a JAX pytree so instances trace through ``jit``,
    ``grad``, and ``vmap``. Polylines follow the WOSAC convention:
    counterclockwise winding, so the port side is on-road (negative
    signed distance) and starboard off-road (positive).

    Attributes:
        polylines: Padded polylines, shape ``(num_polylines,
            max_length, 4)`` holding x, y, z, and a validity flag per
            point (see :func:`stack_polylines`).
        is_cyclic: Boolean flags, shape ``(num_polylines,)``, marking
            closed-loop polylines.
    """

    polylines: jax.Array
    is_cyclic: jax.Array

    @classmethod
    def from_polylines(cls, polylines: Sequence[np.ndarray]) -> RoadEdges:
        """Build road edges from variable-length raw polylines.

        Args:
            polylines: Sequence of arrays with shape ``(num_points_i, 3)``.

        Returns:
            The packed road edges.

        Raises:
            ValueError: If no polyline has at least two points.
        """
        stacked, cyclic = stack_polylines(polylines)
        return cls(polylines=stacked, is_cyclic=cyclic)


def signed_distances_to_road_edges(
    positions: jax.Array, road_edges: RoadEdges, chunk_size: int | None = None
) -> jax.Array:
    """Signed road-edge distance at planar positions.

    Query points are placed at ``z = 0``, and closest-segment association
    uses the standard
    :data:`~diffav.core.constants.ROAD_EDGE_Z_STRETCH` vertical scaling.

    Args:
        positions: Planar positions, shape ``(..., 2)``; leading axes
            (agents, timesteps) are flattened.
        road_edges: Oriented road-edge polylines.
        chunk_size: Optional polyline-chunk size bounding peak memory of the
            closest-segment search (see
            :func:`signed_distance_to_polylines`). The result is independent of
            it; ``None`` (default) evaluates every polyline at once.

    Returns:
        Signed distances (positive = off-road), shape ``(num_positions,)``.
    """
    flat = positions.reshape(-1, 2)
    xyzs = jnp.concatenate([flat, jnp.zeros((flat.shape[0], 1), dtype=flat.dtype)], axis=-1)
    return signed_distance_to_polylines(
        xyzs,
        road_edges.polylines,
        road_edges.is_cyclic,
        z_stretch=ROAD_EDGE_Z_STRETCH,
        chunk_size=chunk_size,
    )


def mean_offroad_penalty(positions: jax.Array, road_edges: RoadEdges) -> jax.Array:
    """Mean squared hinge on the signed distance to road edges.

    On-road positions (negative signed distance) contribute zero;
    off-road positions contribute their squared distance to the nearest
    boundary. The penalty is differentiable, so its gradient pushes
    off-road positions back inside the drivable area.

    Args:
        positions: Planar positions, shape ``(..., 2)``; leading axes
            (agents, timesteps) are flattened.
        road_edges: Oriented road-edge polylines.

    Returns:
        Scalar mean penalty over all positions.
    """
    signed = signed_distances_to_road_edges(positions, road_edges)
    return jnp.mean(jnp.maximum(signed, 0.0) ** 2)
