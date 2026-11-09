"""Directed CommonRoad lanelet guidance for hybrid OptTrajDiff evaluation.

A route-following fallback is explicitly reported; this is NOT diffusion-only.
Checks concern ego center-point segments, not full vehicle footprint or collisions.
"""
from __future__ import annotations
import heapq
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union


def _poly(ll):
    return Polygon(np.vstack((ll.left_vertices[:, :2], ll.right_vertices[::-1, :2]))).buffer(0)


def _goal_shapes(problem):
    """Extract actual CommonRoad goal position geometry where available."""
    shapes = []
    for state in problem.goal.state_list:
        pos = getattr(state, 'position', None)
        if pos is None:
            continue
        items = getattr(pos, 'occupancies', None)
        if items is None:
            items = [pos]
        for item in items:
            obj = getattr(item, 'polygon', item)
            if hasattr(obj, 'geom_type'):
                shapes.append(obj)
            elif hasattr(obj, 'vertices'):
                shapes.append(Polygon(np.asarray(obj.vertices)[:, :2]).buffer(0))
            elif hasattr(obj, 'center'):
                c = obj.center
                xy = (c.x, c.y) if hasattr(c, 'x') else np.asarray(c)[:2]
                shapes.append(Point(xy).buffer(0.05))
    return unary_union(shapes) if shapes else None


class LaneGuide:
    def __init__(self, scenario, start, goal, clearance=0.45, problem=None):
        self.network = scenario.lanelet_network
        self.lanes = {int(ll.lanelet_id): ll for ll in self.network.lanelets}
        self.polygons = {i: _poly(ll) for i, ll in self.lanes.items()}
        self.clearance = float(clearance)
        self.start = np.asarray(start, dtype=float)[:2]
        self.goal = np.asarray(goal, dtype=float)[:2]
        self.goal_shape = _goal_shapes(problem) if problem is not None else None
        # Aim at the actual goal geometry centroid, not an arbitrary lanelet endpoint.
        if self.goal_shape is not None and not self.goal_shape.is_empty:
            self.goal = np.asarray(self.goal_shape.centroid.coords[0], dtype=float)
        self.scenario = scenario
        self.route_ids = self._route_ids(problem)
        self.corridor = unary_union([self.polygons[i] for i in self.route_ids]).buffer(0)
        self.drivable = self.corridor.buffer(-self.clearance)
        if self.drivable.is_empty:
            raise ValueError('Selected route has no interior at requested clearance')
        self.centers = [LineString(np.asarray(self.lanes[i].center_vertices)[:, :2]) for i in self.route_ids]
        self.lengths = np.asarray([c.length for c in self.centers], dtype=float)
        self.offsets = np.r_[0., np.cumsum(self.lengths[:-1])]
        self.route_length = float(self.lengths.sum())
        print(f'[ROUTE] goal_center={self.goal.tolist()} center_inside_corridor={self.valid_point(self.goal)}', flush=True)
        print(f'[ROUTE] lanelets={self.route_ids} length={self.route_length:.1f}m '
              f'clearance={clearance:.2f}m goal_candidates={self.goal_candidates}', flush=True)

    def _start_candidates(self):
        p = Point(self.start)
        inside = [i for i, poly in self.polygons.items() if poly.covers(p)]
        ids = inside or sorted(self.lanes, key=lambda i: self.polygons[i].distance(p))[:4]
        heading = None
        return sorted(ids, key=lambda i: self.polygons[i].distance(p))

    def _goal_candidates(self, problem):
        ids = set()
        if problem is not None:
            goal_lanes = getattr(problem.goal, 'lanelets_of_goal_position', None)
            if isinstance(goal_lanes, dict):
                for values in goal_lanes.values():
                    for v in values:
                        if int(v) in self.lanes:
                            ids.add(int(v))
            if self.goal_shape is not None and not self.goal_shape.is_empty:
                ids.update(i for i, p in self.polygons.items() if p.intersects(self.goal_shape))
        if not ids:
            point = Point(self.goal)
            ids.update(i for i, p in self.polygons.items() if p.covers(point))
        if not ids:
            ids.update(sorted(self.lanes, key=lambda i: self.polygons[i].distance(Point(self.goal)))[:5])
        return sorted(ids)

    def _route_ids(self, problem):
        starts = self._start_candidates()
        self.goal_candidates = self._goal_candidates(problem)
        best = None
        for start in starts:
            q = [(0., start)]
            dist, parent = {start: 0.}, {}
            while q:
                cost, cur = heapq.heappop(q)
                if cost > dist[cur] + 1e-8:
                    continue
                lane = self.lanes[cur]
                neighbors = [(int(n), 1.) for n in (lane.successor or [])]
                for attr, same in (('adj_left', 'adj_left_same_direction'), ('adj_right', 'adj_right_same_direction')):
                    n = getattr(lane, attr, None)
                    if n is not None and bool(getattr(lane, same, False)):
                        neighbors.append((int(n), 3.))
                for nxt, factor in neighbors:
                    if nxt not in self.lanes:
                        continue
                    length = float(np.linalg.norm(np.diff(self.lanes[nxt].center_vertices[:, :2], axis=0), axis=1).sum())
                    nc = cost + factor * max(length, 1.)
                    if nc < dist.get(nxt, float('inf')):
                        dist[nxt], parent[nxt] = nc, cur
                        heapq.heappush(q, (nc, nxt))
            for end in self.goal_candidates:
                if end not in dist:
                    continue
                # Prefer a reachable lane intersecting the goal, not a nearby unreachable lane.
                # Strongly prefer a lane containing the CENTER of the goal, not merely
                # one touching the near edge of a long goal region.
                center_penalty = self.polygons[end].distance(Point(self.goal))
                total = dist[end] + 50. * center_penalty
                if best is None or total < best[0]:
                    route = [end]
                    while route[-1] != start:
                        route.append(parent[route[-1]])
                    best = (total, route[::-1])
        if best is None:
            raise ValueError(f'No directed route from start candidates {starts} to goal candidates {self.goal_candidates}')
        return best[1]

    def progress(self, xy):
        p = Point(np.asarray(xy, dtype=float)[:2])
        choices = [(line.distance(p), float(offset + line.project(p)))
                   for line, offset in zip(self.centers, self.offsets)]
        return min(choices, key=lambda x: x[0])[1]

    def target(self, xy, lookahead=4.):
        s = min(self.route_length, self.progress(xy) + lookahead)
        for line, off, length in zip(self.centers, self.offsets, self.lengths):
            if s <= off + length:
                return np.asarray(line.interpolate(max(0., s-off)).coords[0], dtype=float)
        return np.asarray(self.centers[-1].coords[-1], dtype=float)

    def valid_point(self, xy):
        return bool(self.drivable.covers(Point(np.asarray(xy, dtype=float)[:2])))

    def valid_segment(self, a, b):
        a = np.asarray(a, dtype=float)[:2]
        b = np.asarray(b, dtype=float)[:2]
        if np.linalg.norm(b-a) < 1e-9:
            return self.valid_point(a)
        return bool(self.drivable.covers(LineString([a, b])))

    def choose_step(self, ego, candidates, dt, step_index, max_step, lookahead=4., collision_checker=None):
        """Choose a short, route-compliant step without passing the goal station.

        Diffusion candidates remain preferred when suitable. Near the goal,
        longitudinal overshoot is prohibited and distance to the actual centroid
        takes priority over route progress. The fallback is reported separately.
        """
        origin = np.asarray(ego.position, dtype=float)[:2]
        base = self.progress(origin)
        goal_s = self.progress(self.goal)
        remaining_s = goal_s - base
        distance_goal = float(np.linalg.norm(origin - self.goal))
        near_goal = distance_goal <= 20.0
        desired = self.goal.copy() if near_goal and self.valid_point(self.goal) else self.target(origin, lookahead)
        # If the centroid is already reached, request a stationary hold. The
        # test runner must still check speed and the CommonRoad goal state.
        if distance_goal <= 0.05:
            return origin.copy(), 'goal_hold'

        def acceptable(xy):
            xy = np.asarray(xy, dtype=float)[:2]
            if not np.isfinite(xy).all() or not self.valid_segment(origin, xy):
                return False
            advance = self.progress(xy) - base
            if advance < -0.08 or advance > max_step * 2.0 + 0.2:
                return False
            # Prevent advancing beyond the goal's longitudinal position.
            # Small tolerance accounts for centerline projection noise.
            if remaining_s >= -0.2 and self.progress(xy) > goal_s + 0.12:
                return False
            return collision_checker is None or bool(collision_checker(origin, xy))

        options = []
        for idx, path in enumerate(candidates):
            if step_index >= len(path):
                continue
            delta = np.asarray(path[step_index], dtype=float)[:2] - origin
            norm = float(np.linalg.norm(delta))
            if not np.isfinite(norm) or norm < 1e-8:
                continue
            xy = origin + delta * min(1.0, max_step / norm)
            if acceptable(xy):
                advance = self.progress(xy) - base
                score = (-10.0 * np.linalg.norm(xy - self.goal) + 0.1 * advance
                         if near_goal else 6.0 * advance - np.linalg.norm(xy - desired))
                options.append((score, xy, f'diffusion_{idx}'))

        delta = desired - origin
        norm = float(np.linalg.norm(delta))
        if norm > 1e-8:
            fallback = origin + delta * min(1.0, max_step / norm)
            if acceptable(fallback):
                advance = self.progress(fallback) - base
                score = (-10.0 * np.linalg.norm(fallback - self.goal) + 0.1 * advance
                         if near_goal else 6.0 * advance - np.linalg.norm(fallback - desired))
                options.append((score, fallback, 'route_fallback'))

        if options:
            _, xy, source = max(options, key=lambda item: item[0])
            return np.asarray(xy, dtype=float), source
        if collision_checker is None or collision_checker(origin, origin):
            return origin.copy(), 'safe_wait'
        return origin.copy(), 'blocked_collision_or_route'


def _xy(center):
    if hasattr(center, 'x') and hasattr(center, 'y'):
        return np.array([center.x, center.y], dtype=float)
    return np.asarray(center, dtype=float).reshape(-1)[:2]


def _shape_geometry(shape):
    """Convert CommonRoad occupancy wrappers and shape objects to Shapely.

    Supports both older shape-based occupancies and newer RectOccupancy /
    PolygonOccupancy representations. Never assume occupancy.shape exists.
    """
    if shape is None:
        return None
    if hasattr(shape, 'shape'):
        return _shape_geometry(shape.shape)
    if hasattr(shape, 'occupancies'):
        geoms = [_shape_geometry(item) for item in shape.occupancies]
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        return unary_union(geoms) if geoms else None
    if hasattr(shape, 'shapes'):
        geoms = [_shape_geometry(item) for item in shape.shapes]
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        return unary_union(geoms) if geoms else None
    if hasattr(shape, 'polygon'):
        return _shape_geometry(shape.polygon)
    if hasattr(shape, 'geom_type'):
        return shape
    if hasattr(shape, 'vertices'):
        vertices = np.asarray(shape.vertices, dtype=float)
        if vertices.ndim == 2 and vertices.shape[0] >= 3:
            return Polygon(vertices[:, :2]).buffer(0)
    if hasattr(shape, 'center') and hasattr(shape, 'radius'):
        return Point(_xy(shape.center)).buffer(float(shape.radius))
    if hasattr(shape, 'center') and hasattr(shape, 'length') and hasattr(shape, 'width'):
        from shapely.geometry import box
        from shapely.affinity import rotate, translate
        geom = box(-float(shape.length)/2, -float(shape.width)/2,
                   float(shape.length)/2, float(shape.width)/2)
        geom = rotate(geom, float(getattr(shape, 'orientation', 0.0)), use_radians=True)
        center = _xy(shape.center)
        return translate(geom, float(center[0]), float(center[1]))
    # RectOccupancy may expose a rectangle via a different field name.
    for attr in ('rectangle', 'rect', 'geometry'):
        if hasattr(shape, attr):
            return _shape_geometry(getattr(shape, attr))
    return None


class CollisionChecker:
    """Discrete-time swept-circle collision screening for CommonRoad tests.

    Checks each executed segment against occupancies at both the previous and
    next time steps. This is conservative but not continuous-time certification.
    Unknown occupancy is logged; it is not evidence of clearance.
    """
    def __init__(self, scenario, ego_radius=1.1, margin=0.4):
        self.scenario = scenario
        self.radius = float(ego_radius) + float(margin)
        self.unknown_count = 0
        self.rejections = 0
        self.unknown_types = set()

    def safe_segment(self, a, b, next_time_step):
        a = np.asarray(a, dtype=float)[:2]
        b = np.asarray(b, dtype=float)[:2]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            self.rejections += 1
            return False
        sweep = (LineString([a, b]).buffer(self.radius)
                 if np.linalg.norm(b-a) > 1e-9 else Point(a).buffer(self.radius))
        for obstacle in self.scenario.obstacles:
            for t in (int(next_time_step) - 1, int(next_time_step)):
                occupancy = obstacle.occupancy_at_time(t)
                if occupancy is None:
                    self.unknown_count += 1
                    continue
                geom = _shape_geometry(occupancy)
                if geom is None or geom.is_empty:
                    self.unknown_count += 1
                    self.unknown_types.add(type(occupancy).__name__)
                    continue
                if sweep.intersects(geom):
                    self.rejections += 1
                    return False
        return True
