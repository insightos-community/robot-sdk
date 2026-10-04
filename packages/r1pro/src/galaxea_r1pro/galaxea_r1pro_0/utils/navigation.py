"""Offline collision geometry projection and A*. No simulator imports or stepping."""

from __future__ import annotations

import hashlib
import copy
import itertools
import heapq
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from scipy.ndimage import distance_transform_edt


class NavigationError(ValueError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


def fingerprint(snapshot):
    objects = [o for o in snapshot["objects"] if o["name"] not in snapshot["excluded_names"]]
    data = dict(
        version=6,
        scene=snapshot["scene"],
        objects=objects,
        navigation_footprint=snapshot.get("navigation_footprint"),
        frame_id=snapshot.get("frame_id", "world"),
        odom_origin_world=snapshot.get("odom_origin_world"),
    )
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def geometry_in_odom(snapshot, origin_world):
    """Temporary truth-map adapter; uses a FIXED startup alignment, never current pose."""
    from scipy.spatial.transform import Rotation

    result = copy.deepcopy(snapshot)
    transform = np.linalg.inv(origin_world)

    def points(values):
        p = np.asarray(values, dtype=float)
        return p @ transform[:3, :3].T + transform[:3, 3]

    def bounds(values):
        corners = points(list(itertools.product(*np.asarray(values).T)))
        return np.stack([corners.min(0), corners.max(0)]).tolist()

    for obj in result["objects"]:
        if "position" in obj:
            obj["position"] = points(obj["position"]).tolist()
            rotation = transform[:3, :3] @ Rotation.from_quat(obj["orientation_xyzw"]).as_matrix()
            obj["orientation_xyzw"] = Rotation.from_matrix(rotation).as_quat().tolist()
        for mesh in obj["colliders"]:
            vertices = mesh.pop("vertices_world", [])
            if len(vertices):
                p = points(vertices)
                mesh["vertices"] = p.tolist()
                mesh["aabb"] = np.stack([p.min(0), p.max(0)]).tolist()
            else:
                mesh["aabb"] = bounds(mesh["aabb"])
        if "collision_aabb" in obj:
            meshes = obj["colliders"]
            obj["collision_aabb"] = (
                [
                    np.min([m["aabb"][0] for m in meshes], axis=0).tolist(),
                    np.max([m["aabb"][1] for m in meshes], axis=0).tolist(),
                ]
                if meshes
                else bounds(obj["collision_aabb"])
            )
            size = np.diff(obj["collision_aabb"], axis=0)[0]
            obj.update(bbox_size_m=size.tolist(), bbox_volume_m3=float(np.prod(size)))
    result.update(
        frame_id="odom",
        odom_origin_world=np.asarray(origin_world).tolist(),
        floor_z=snapshot["floor_z"] - origin_world[2, 3],
        robot_top_z=snapshot["robot_top_z"] - origin_world[2, 3],
    )
    return result


def _paint(grid, polygon, origin, cell, *, support=False):
    """Conservative triangle/polygon coverage; thin walls also occupy cells."""
    p = np.asarray(polygon, dtype=float)[:, :2]
    low = np.maximum(np.floor((p.min(0) - origin) / cell).astype(int) - 1, 0)
    high = np.minimum(np.ceil((p.max(0) - origin) / cell).astype(int) + 1, grid.shape[::-1])
    if np.any(high <= low):
        return
    x, y = np.meshgrid(
        origin[0] + (np.arange(low[0], high[0]) + 0.5) * cell,
        origin[1] + (np.arange(low[1], high[1]) + 0.5) * cell,
    )
    inside = np.zeros(x.shape, bool)
    near = np.zeros(x.shape, bool)
    boundary = np.zeros(x.shape, bool)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        inside ^= ((a[1] > y) != (b[1] > y)) & (
            x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-30) + a[0]
        )
        delta = b - a
        t = np.clip(
            ((x - a[0]) * delta[0] + (y - a[1]) * delta[1]) / max(float(delta @ delta), 1e-30), 0, 1
        )
        distance2 = (x - a[0] - t * delta[0]) ** 2 + (y - a[1] - t * delta[1]) ** 2
        near |= distance2 <= (cell * math.sqrt(2) / 2) ** 2
        boundary |= distance2 <= 1e-20
    # Include shared triangle edges without expanding floor support by half a cell.
    grid[low[1] : high[1], low[0] : high[0]] |= (inside | boundary) if support else (inside | near)


def _project(grid, mesh, origin, cell, zmin, zmax, *, support=False):
    bounds = np.asarray(mesh["aabb"])
    if bounds[1, 2] < zmin or bounds[0, 2] > zmax:
        return
    vertices = np.asarray(mesh.get("vertices", mesh.get("vertices_world", [])))
    faces = mesh.get("faces", [])
    if len(vertices) and len(faces):
        for face in faces:
            polygon = vertices[face]
            if polygon[:, 2].max() < zmin or polygon[:, 2].min() > zmax:
                continue
            # Floor support comes only from horizontal surfaces on this floor.
            if support and np.ptp(polygon[:, 2]) > 0.025:
                continue
            _paint(grid, polygon, origin, cell, support=support)
    elif not support:
        a, b = bounds[:, :2]
        _paint(grid, [a, [b[0], a[1]], b, [a[0], b[1]]], origin, cell)


class NavigationMap:
    def __init__(self, occupancy, origin, resolution, metadata):
        self.occupancy = np.asarray(occupancy, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)
        self.resolution = float(resolution)
        self.metadata = metadata
        # Padding makes map edges non-traversable after footprint inflation.
        padded = np.pad(~self.occupancy, 1, constant_values=False)
        self.clearance = (
            distance_transform_edt(padded)[1:-1, 1:-1] * resolution - resolution * math.sqrt(2) / 2
        )

    def cell(self, xy):
        x, y = np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        return int(y), int(x)

    def point(self, cell):
        return self.origin + (np.array(cell[::-1]) + 0.5) * self.resolution

    def free(self, radius, margin):
        return (~self.occupancy) & (self.clearance >= radius + margin)

    def valid(self, point, free):
        y, x = self.cell(point)
        return 0 <= y < free.shape[0] and 0 <= x < free.shape[1] and bool(free[y, x])

    def segment_free(self, a, b, free):
        """Supercover grid traversal: includes both neighbors at a corner crossing."""
        a, b = [(np.asarray(p) - self.origin) / self.resolution for p in (a, b)]
        x, y = np.floor(a).astype(int)
        endx, endy = np.floor(b).astype(int)
        dx, dy = b - a
        sx, sy = int(np.sign(dx)), int(np.sign(dy))
        tx = ((x + (sx > 0) - a[0]) / dx) if dx else math.inf
        ty = ((y + (sy > 0) - a[1]) / dy) if dy else math.inf
        stepx, stepy = abs(1 / dx) if dx else math.inf, abs(1 / dy) if dy else math.inf

        def valid(ix, iy):
            return 0 <= iy < free.shape[0] and 0 <= ix < free.shape[1] and free[iy, ix]

        for _ in range(abs(endx - x) + abs(endy - y) + 3):
            if not valid(x, y):
                return False
            if (x, y) == (endx, endy):
                return True
            if abs(tx - ty) < 1e-12:
                if not valid(x + sx, y) or not valid(x, y + sy):
                    return False
                x += sx
                y += sy
                tx += stepx
                ty += stepy
            elif tx < ty:
                x += sx
                tx += stepx
            else:
                y += sy
                ty += stepy
        return False

    def plan(self, start, goal, radius, margin):
        started = time.monotonic()
        free = self.free(radius, margin)
        for name, point in [("start", start), ("goal", goal)]:
            y, x = self.cell(point)
            if not (0 <= y < free.shape[0] and 0 <= x < free.shape[1]):
                raise NavigationError(name + "_out_of_bounds", name + " is outside map")
            if not free[y, x]:
                raise NavigationError(name + "_occupied", name + " overlaps inflated obstacles")
        source, target = self.cell(start), self.cell(goal)
        queue = [(0.0, source)]
        costs, parent, closed = {source: 0.0}, {}, set()

        def heuristic(p):
            d = np.abs(np.array(p) - target)
            return float(max(d) + (math.sqrt(2) - 1) * min(d))

        while queue:
            _, current = heapq.heappop(queue)
            if current in closed:
                continue
            if current == target:
                break
            closed.add(current)
            if time.monotonic() - started > 20:
                raise NavigationError("planning_timeout", "A* exceeded 20 wall-clock seconds")
            y, x = current
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]) or not free[ny, nx]:
                    continue
                if dx and dy and not (free[y, nx] and free[ny, x]):
                    continue
                node = ny, nx
                cost = costs[current] + math.hypot(dx, dy)
                if cost < costs.get(node, math.inf):
                    costs[node], parent[node] = cost, current
                    heapq.heappush(queue, (cost + heuristic(node), node))
        else:
            raise NavigationError("no_path", "No collision-free grid path")
        cells = [target]
        while cells[-1] != source:
            cells.append(parent[cells[-1]])
        raw = [np.asarray(start), *[self.point(c) for c in reversed(cells)], np.asarray(goal)]
        path, i = [raw[0]], 0
        while i < len(raw) - 1:
            j = len(raw) - 1
            while j > i:
                if time.monotonic() - started > 20:
                    raise NavigationError(
                        "planning_timeout", "Path planning exceeded 20 wall-clock seconds"
                    )
                if self.segment_free(raw[i], raw[j], free):
                    break
                j -= 1
            if j == i:
                raise NavigationError(
                    "endpoint_connection_blocked", "Exact endpoint cannot connect to grid"
                )
            if np.linalg.norm(raw[j] - path[-1]) > 1e-9:
                path.append(raw[j])
            i = j
        return dict(
            path=np.asarray(path),
            free=free,
            planning_seconds=time.monotonic() - started,
            expanded_cells=len(closed),
        )


def build_map(snapshot, cache_dir, resolution=0.05, refresh=False):
    started = time.monotonic()
    floor_z, top_z = snapshot["floor_z"], snapshot["robot_top_z"]
    key = hashlib.sha256(
        f"{fingerprint(snapshot)}:{resolution}:{floor_z:.3f}:{top_z:.3f}".encode()
    ).hexdigest()
    cache = Path(cache_dir).expanduser() / key
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "map.npz"
    if path.exists() and not refresh:
        try:
            with path.open("rb") as stream, np.load(stream, allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata"]))
                occupancy, origin = saved["occupancy"], saved["origin"]
                if (
                    meta["map_id"] != key
                    or meta["resolution_m"] != resolution
                    or meta["frame_id"] != snapshot.get("frame_id", "world")
                    or occupancy.dtype != np.bool_
                    or occupancy.ndim != 2
                    or occupancy.shape != tuple(meta["shape_yx"])
                    or not occupancy.size
                    or origin.shape != (2,)
                    or not np.isfinite(origin).all()
                    or not np.array_equal(origin, meta["origin_xy"])
                ):
                    raise ValueError("Navigation cache metadata mismatch")
                navmap = NavigationMap(occupancy, origin, resolution, meta)
            navmap.metadata = dict(meta, cache_hit=True, load_seconds=time.monotonic() - started)
            return navmap
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
            pass  # Rebuild corrupt/incomplete cache from this validated snapshot.
    floors = [
        m
        for o in snapshot["objects"]
        if o["category"] == "floors"
        for m in o["colliders"]
        if m["aabb"][0][2] <= floor_z + 0.1 and m["aabb"][1][2] >= floor_z - 0.1
    ]
    if not floors:
        raise NavigationError("floor_geometry_unavailable", "No measured support floor")
    origin = np.floor(np.min([m["aabb"][0][:2] for m in floors], axis=0) / resolution) * resolution
    upper = np.max([m["aabb"][1][:2] for m in floors], axis=0)
    shape = np.ceil((upper - origin) / resolution).astype(int)[::-1]
    if np.prod(shape) > 4_000_000:
        raise NavigationError("map_too_large", "Map exceeds four million cells")
    floor = np.zeros(shape, bool)
    obstacles = np.zeros(shape, bool)
    for mesh in floors:
        _project(floor, mesh, origin, resolution, floor_z - 0.04, floor_z + 0.04, support=True)
    if not floor.any():
        raise NavigationError(
            "floor_geometry_unavailable", "No horizontal floor surface at robot height"
        )
    for obj in snapshot["objects"]:
        if obj["name"] in snapshot["excluded_names"] or obj["category"] == "floors":
            continue
        for mesh in obj["colliders"]:
            _project(obstacles, mesh, origin, resolution, floor_z + 0.025, top_z + 0.05)
    occupancy = obstacles | ~floor
    meta = dict(
        map_id=key,
        frame_id=snapshot.get("frame_id", "world"),
        origin_xy=origin.tolist(),
        resolution_m=resolution,
        navigation_footprint=snapshot.get("navigation_footprint"),
        odom_origin_world=snapshot.get("odom_origin_world"),
        shape_yx=shape.tolist(),
        floor_z=floor_z,
        robot_top_z=top_z,
        snapshot_step=snapshot.get("step_count"),
        scene=snapshot["scene"],
        geometry_file=str(cache / "geometry.json"),
        map_file=str(path),
        cache_hit=False,
        build_seconds=time.monotonic() - started,
    )
    token = uuid4().hex
    geometry_tmp = cache / (token + ".json")
    geometry_tmp.write_text(json.dumps(snapshot, separators=(",", ":")))
    os.replace(geometry_tmp, cache / "geometry.json")
    temp = cache / (token + ".npz")
    np.savez_compressed(temp, occupancy=occupancy, origin=origin, metadata=json.dumps(meta))
    os.replace(temp, path)
    return NavigationMap(occupancy, origin, resolution, meta)
