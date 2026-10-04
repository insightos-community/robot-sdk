"""二维占据栅格与矩形障碍上的 A*、路径平滑和目标解析。

矩形障碍模型移植自 plugin-mujoco 的 ``AStarGridPlanner``：障碍以 XY 矩形
声明，按膨胀半径一次性栅格化为禁行格，避免旧实现逐格圆形膨胀的
O(占用格×邻域²) 纯 Python 成本。带 ``collision_boxes`` 的栅格走矩形快路径；
只有占据格的栅格（测试地图、Static 地图）保留原有逐格膨胀语义。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from semantic_robot_sdk_core.errors import PlanningError

try:  # numpy 只用于加速矩形栅格化；缺失时退回纯 Python 循环。
    import numpy as _np
except ImportError:  # pragma: no cover - 生产环境随 ruckig 提供 numpy
    _np = None


@dataclass(frozen=True)
class OccupancyGrid:
    width: int
    height: int
    resolution_m: float
    origin_xy: tuple[float, float]
    frame_id: str
    occupied: frozenset[tuple[int, int]]
    collision_boxes: tuple[tuple[float, float, float, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("占据栅格尺寸必须大于零")
        if self.resolution_m <= 0:
            raise ValueError("占据栅格分辨率必须大于零")

    def inflated(self, radius_m: float) -> OccupancyGrid:
        """按 Robot 平面外形膨胀障碍（逐格圆形膨胀，用于只有占据格的地图）。

        规划网格表示的是 Robot 中心可以到达的位置。除 Robot 外接圆外，再加入
        半个栅格对角线，避免离散化后路径中心虽然落在空闲单元，Robot 边缘仍穿过
        相邻障碍。边界外不生成单元，起终点检查仍会给出明确错误。
        """

        if radius_m <= 0:
            return self
        cell_radius = radius_m / self.resolution_m + math.sqrt(2.0) / 2.0
        reach = math.ceil(cell_radius)
        occupied = set(self.occupied)
        for x, y in self.occupied:
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    if math.hypot(dx, dy) > cell_radius:
                        continue
                    cell = (x + dx, y + dy)
                    if 0 <= cell[0] < self.width and 0 <= cell[1] < self.height:
                        occupied.add(cell)
        return OccupancyGrid(
            width=self.width,
            height=self.height,
            resolution_m=self.resolution_m,
            origin_xy=self.origin_xy,
            frame_id=self.frame_id,
            occupied=frozenset(occupied),
            collision_boxes=self.collision_boxes,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor((x - self.origin_xy[0]) / self.resolution_m),
            math.floor((y - self.origin_xy[1]) / self.resolution_m),
        )

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.origin_xy[0] + (cell[0] + 0.5) * self.resolution_m,
            self.origin_xy[1] + (cell[1] + 0.5) * self.resolution_m,
        )


def _center_cell_range(
    grid: OccupancyGrid, minimum: float, maximum: float, *, axis: int
) -> tuple[int, int]:
    """返回格心落在闭区间 [minimum, maximum] 内的格索引范围（含边界）。

    与旧 ``with_carried_aabb`` 完全同一套 ceil/floor 算术：AABB 上边界与栅格
    边界重合时，边界外的下一格没有真实重叠，不能多占。
    """

    origin = grid.origin_xy[axis]
    low = math.ceil((minimum - origin) / grid.resolution_m - 0.5)
    high = math.floor((maximum - origin) / grid.resolution_m - 0.5)
    limit = grid.width if axis == 0 else grid.height
    return max(0, low), min(limit - 1, high)


def rasterize_rects(
    grid: OccupancyGrid,
    rects: tuple[tuple[float, float, float, float], ...],
    *,
    expand: float = 0.0,
) -> frozenset[tuple[int, int]]:
    """把 XY 矩形按方形外扩 ``expand`` 后栅格化为禁行格。

    这是 plugin-mujoco 矩形膨胀的栅格化等价物：每个矩形一次直接光栅化，
    总成本等于膨胀后矩形面积，而不是旧逐格膨胀的面积×邻域²。格心落在
    闭矩形内即禁行，与 A* 的闭区间点判遮挡语义一致。
    """

    if not rects:
        return frozenset()
    cells: set[tuple[int, int]] = set()
    if _np is not None:
        for x_min, y_min, x_max, y_max in rects:
            x_low, x_high = _center_cell_range(grid, x_min - expand, x_max + expand, axis=0)
            y_low, y_high = _center_cell_range(grid, y_min - expand, y_max + expand, axis=1)
            if x_low > x_high or y_low > y_high:
                continue
            xs = _np.arange(x_low, x_high + 1, dtype=_np.int64)
            ys = _np.arange(y_low, y_high + 1, dtype=_np.int64)
            mesh = _np.meshgrid(xs, ys, indexing="ij")
            cells.update(zip(mesh[0].ravel().tolist(), mesh[1].ravel().tolist()))
        return frozenset(cells)
    for x_min, y_min, x_max, y_max in rects:
        x_low, x_high = _center_cell_range(grid, x_min - expand, x_max + expand, axis=0)
        y_low, y_high = _center_cell_range(grid, y_min - expand, y_max + expand, axis=1)
        for cell_x in range(x_low, x_high + 1):
            for cell_y in range(y_low, y_high + 1):
                cells.add((cell_x, cell_y))
    return frozenset(cells)


def carried_aabb_cells(
    grid: OccupancyGrid,
    *,
    offset_xy: tuple[float, float],
    extent_xyz: tuple[float, float, float],
    center_z: float,
) -> frozenset[tuple[int, int]]:
    """把保持固定朝向的实时持物AABB转换为底盘中心禁行格。

    数学与旧 ``OccupancyGrid.with_carried_aabb`` 完全一致，只是改为矩形
    直接光栅化。携物期间底盘保持朝向，箱体相对底盘的平移关系固定；障碍
    矩形与箱体有高度重叠时按 Minkowski 展开生成禁行格，不能只用底盘圆形
    包络，否则箱体会在 Robot 本体尚未碰撞时先扫倒相邻箱。
    """

    if not grid.collision_boxes:
        return frozenset()
    half_x, half_y, half_z = (value / 2.0 for value in extent_xyz)
    carried_min_z = center_z - half_z
    carried_max_z = center_z + half_z
    rects = []
    for min_x, min_y, min_z, max_x, max_y, max_z in grid.collision_boxes:
        if max_z <= carried_min_z or min_z >= carried_max_z:
            continue
        rects.append(
            (
                min_x - half_x - offset_xy[0],
                min_y - half_y - offset_xy[1],
                max_x + half_x - offset_xy[0],
                max_y + half_y - offset_xy[1],
            )
        )
    return rasterize_rects(grid, tuple(rects))


def astar(
    grid: OccupancyGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    blocked: frozenset[tuple[int, int]] | None = None,
    maximum_expanded_nodes: int | None = None,
    clearance: tuple[tuple[tuple[float, float, float, float], ...], float, float, float]
    | None = None,
):
    """A* 搜索并返回世界坐标路径（移植自 plugin-mujoco AStarGridPlanner）。

    与 plugin 实现一致：8 邻域、对角移动不能从两个障碍单元的夹角穿过、
    欧氏启发、可选的膨胀边界软代价和搜索节点上限。``blocked`` 缺省使用
    ``grid.occupied``；节点上限超限时抛出 PlanningError，而不是无限搜索。
    """

    blocked_cells = grid.occupied if blocked is None else blocked
    start = grid.world_to_cell(*start_xy)
    goal = grid.world_to_cell(*goal_xy)
    for label, cell in (("起点", start), ("终点", goal)):
        if not (0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height):
            raise PlanningError(f"{label}不在占据栅格内")
        if cell in blocked_cells:
            raise PlanningError(f"{label}位于障碍物中")

    clearance_rects, clearance_expand, clearance_radius, clearance_weight = (
        clearance if clearance else (),
        0.0,
        0.0,
        0.0,
    )
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    expanded = 0
    while frontier:
        _score, current = heapq.heappop(frontier)
        if current == goal:
            break
        expanded += 1
        if maximum_expanded_nodes is not None and expanded > int(maximum_expanded_nodes):
            raise PlanningError("搜索节点超限，路径规划失败")
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height):
                continue
            # 对角移动不能从两个障碍单元的夹角穿过去。这个检查看似细小，
            # 但对有实际宽度的移动底盘是常见碰撞来源。
            if (
                dx
                and dy
                and (
                    (current[0] + dx, current[1]) in blocked_cells
                    or (current[0], current[1] + dy) in blocked_cells
                )
            ):
                continue
            if nxt in blocked_cells:
                continue
            step = math.sqrt(2) if dx and dy else 1.0
            new_cost = cost[current] + step
            if clearance is not None:
                point = grid.cell_to_world(nxt)
                new_cost += _clearance_penalty(
                    point,
                    clearance_rects,
                    clearance_expand,
                    clearance_radius,
                    clearance_weight,
                )
            if new_cost >= cost.get(nxt, float("inf")):
                continue
            cost[nxt] = new_cost
            parent[nxt] = current
            heuristic = math.dist(nxt, goal)
            heapq.heappush(frontier, (new_cost + heuristic, nxt))
    if goal not in cost:
        raise PlanningError("A* 未找到可行底盘路径")
    cells = [goal]
    while cells[-1] != start:
        cells.append(parent[cells[-1]])
    cells.reverse()
    return [grid.cell_to_world(cell) for cell in cells]


def _clearance_penalty(
    point: tuple[float, float],
    rects: tuple[tuple[float, float, float, float], ...],
    expand: float,
    radius: float,
    weight: float,
) -> float:
    """给靠近障碍膨胀边界的节点增加软代价（plugin clearance_cost 语义）。"""

    if radius <= 0.0 or weight <= 0.0 or not rects:
        return 0.0
    x, y = point
    nearest = math.inf
    for x_min, y_min, x_max, y_max in rects:
        dx = max(x_min - expand - x, 0.0, x - (x_max + expand))
        dy = max(y_min - expand - y, 0.0, y - (y_max + expand))
        nearest = min(nearest, math.hypot(dx, dy))
    if nearest >= radius:
        return 0.0
    normalized = (radius - max(0.0, nearest)) / radius
    return weight * normalized * normalized


def resolve_nearby_free_goal(
    grid: OccupancyGrid,
    blocked: frozenset[tuple[int, int]],
    goal_xy: tuple[float, float],
    *,
    maximum_cell_offset: int = 2,
) -> tuple[float, float]:
    """把离散安全边缘上的目标解析到最近自由格。

    Agent给出的工位是连续空间坐标，而A*使用离散栅格。目标所在原始格确实有
    障碍时必须拒绝；只有原始格空闲、在安全膨胀后落入边缘时，才允许在
    很小的邻域内选择安全格。这样不会用投影掩盖真实碰撞，也不需要模型反复猜
    厘米级坐标。
    """

    raw_cell = grid.world_to_cell(*goal_xy)
    if not (0 <= raw_cell[0] < grid.width and 0 <= raw_cell[1] < grid.height):
        raise PlanningError("终点不在占据栅格内")
    if raw_cell in grid.occupied:
        raise PlanningError("终点位于障碍物中")
    for box in grid.collision_boxes:
        if box[0] <= goal_xy[0] <= box[3] and box[1] <= goal_xy[1] <= box[4]:
            raise PlanningError("终点位于障碍物中")
    if raw_cell not in blocked:
        return goal_xy

    candidates: list[tuple[float, tuple[int, int]]] = []
    for dx in range(-maximum_cell_offset, maximum_cell_offset + 1):
        for dy in range(-maximum_cell_offset, maximum_cell_offset + 1):
            if dx == 0 and dy == 0:
                continue
            if math.hypot(dx, dy) > maximum_cell_offset:
                continue
            cell = (raw_cell[0] + dx, raw_cell[1] + dy)
            if not (0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height):
                continue
            if cell in blocked:
                continue
            point = grid.cell_to_world(cell)
            candidates.append((math.dist(point, goal_xy), cell))

    if not candidates:
        raise PlanningError("终点位于安全膨胀区，附近没有可用工位")
    _distance, cell = min(candidates, key=lambda item: item[0])
    return grid.cell_to_world(cell)


def smooth_path(
    points: list[tuple[float, float]],
    grid: OccupancyGrid,
    blocked: frozenset[tuple[int, int]] | None = None,
) -> list[tuple[float, float]]:
    """删除冗余折点；只连接经过禁行格碰撞检查的直线段。"""

    blocked_cells = grid.occupied if blocked is None else blocked
    if len(points) < 3:
        return points
    result = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        farthest = anchor + 1
        probe = farthest + 1
        while probe < len(points) and _line_is_free(
            grid, points[anchor], points[probe], blocked_cells
        ):
            farthest = probe
            probe += 1
        result.append(points[farthest])
        anchor = farthest
    return result


def _line_is_free(
    grid: OccupancyGrid,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    blocked: frozenset[tuple[int, int]],
) -> bool:
    """检查平滑捷径是否与任一禁行格相交。"""

    # A*经过的是已经膨胀过的安全栅格。这里不能再靠离散采样判断捷径：
    # 一条线即使所有采样点都落在自由格中，也可能恰好擦过垛角对应的占用格
    # 边界。连续线段与闭合格子的相交检查只修正路径平滑方法，不改变目标点、
    # clearance或速度；碰到边/角也保留A*原来的绕行折点。
    min_x = math.floor((min(start_xy[0], end_xy[0]) - grid.origin_xy[0]) / grid.resolution_m) - 1
    max_x = math.floor((max(start_xy[0], end_xy[0]) - grid.origin_xy[0]) / grid.resolution_m)
    min_y = math.floor((min(start_xy[1], end_xy[1]) - grid.origin_xy[1]) / grid.resolution_m) - 1
    max_y = math.floor((max(start_xy[1], end_xy[1]) - grid.origin_xy[1]) / grid.resolution_m)

    for cell_x in range(max(0, min_x), min(grid.width - 1, max_x) + 1):
        for cell_y in range(max(0, min_y), min(grid.height - 1, max_y) + 1):
            if (cell_x, cell_y) not in blocked:
                continue
            cell_min_x = grid.origin_xy[0] + cell_x * grid.resolution_m
            cell_min_y = grid.origin_xy[1] + cell_y * grid.resolution_m
            if _segment_intersects_closed_box(
                start_xy,
                end_xy,
                minimum=(cell_min_x, cell_min_y),
                maximum=(cell_min_x + grid.resolution_m, cell_min_y + grid.resolution_m),
            ):
                return False
    return True


def _segment_intersects_closed_box(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    minimum: tuple[float, float],
    maximum: tuple[float, float],
) -> bool:
    """Liang-Barsky线段裁剪；接触闭合矩形边界也视为相交。"""

    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    lower, upper = 0.0, 1.0
    for coefficient, distance in (
        (-dx, start_xy[0] - minimum[0]),
        (dx, maximum[0] - start_xy[0]),
        (-dy, start_xy[1] - minimum[1]),
        (dy, maximum[1] - start_xy[1]),
    ):
        if math.isclose(coefficient, 0.0, abs_tol=1e-12):
            if distance < 0.0:
                return False
            continue
        ratio = distance / coefficient
        if coefficient < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True
