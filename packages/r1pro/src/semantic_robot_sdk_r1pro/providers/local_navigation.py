"""R1 Pro 本地路径规划和受限底盘轨迹 Provider。

路径规划采用 plugin-mujoco 的矩形障碍 A* 模型：障碍矩形按
``minimum_clearance + 底盘方形半边`` 一次性光栅化为禁行格，起终点判定、
逃逸窗口、目标解析和平滑都在同一份禁行格上进行，最后仍由 Ruckig 生成
时间参数化底盘轨迹。
"""

from __future__ import annotations

import math
import uuid

from semantic_robot_sdk_core import (
    BaseTrajectoryPoint,
    DependencyUnavailable,
    MotionPlan,
    PlanKind,
    PlanningError,
    Pose,
    RobotCapabilities,
    RobotState,
)

from .dependencies import require_ruckig
from .local_motion_algorithms import ruckig_trajectory
from .local_navigation_algorithms import (
    OccupancyGrid,
    astar,
    carried_aabb_cells,
    rasterize_rects,
    resolve_nearby_free_goal,
    smooth_path,
)
from .navigation_timing import base_segment_targets, base_limits, yaw_from_xyzw


class LocalNavigationProvider:
    name = "local"

    def __init__(
        self,
        capabilities: RobotCapabilities,
        *,
        map_source=None,
        allow_debug_fallback=False,
        carrying_motion_scale=1.0,
        base_half_size_m: float | None = None,
        maximum_expanded_nodes: int = 120000,
        clearance_cost_radius: float = 0.0,
        clearance_cost_weight: float = 0.0,
    ):
        self.capabilities = capabilities
        self.map_source = map_source
        self.allow_debug_fallback = allow_debug_fallback
        if not 0 < carrying_motion_scale <= 1:
            raise ValueError("carrying_motion_scale 必须位于 (0, 1] 区间")
        self.carrying_motion_scale = carrying_motion_scale
        # 底盘包络按方形半边建模（plugin-mujoco 语义）。缺省沿用 Profile 的
        # 外接半径：方形包络完全覆盖原圆形，只保守不放松；部署配置可以按
        # 实际底盘尺寸调小。
        footprint = capabilities.base_footprint_radius_m
        if footprint is None:
            raise PlanningError("Robot Profile 缺少底盘外形")
        self.base_half_size_m = float(footprint if base_half_size_m is None else base_half_size_m)
        if self.base_half_size_m <= 0:
            raise ValueError("base_half_size_m 必须大于零")
        if maximum_expanded_nodes <= 0:
            raise ValueError("maximum_expanded_nodes 必须大于零")
        self.maximum_expanded_nodes = int(maximum_expanded_nodes)
        self.clearance_cost_radius = max(0.0, float(clearance_cost_radius))
        self.clearance_cost_weight = max(0.0, float(clearance_cost_weight))

    def plan_route(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goal: Pose,
        occupancy: OccupancyGrid | None = None,
        maximum_speed_mps: float,
        minimum_clearance_m: float | None = None,
        carrying_object_ref: str | None = None,
        carrying_object_pose: Pose | None = None,
        carrying_object_extent_m: tuple[float, float, float] | None = None,
    ) -> MotionPlan:
        if state.base_pose is None:
            raise PlanningError("当前 Robot 没有移动底盘")
        if occupancy is None:
            if self.map_source is None:
                raise PlanningError(
                    "本地导航没有可用地图；请配置 robot.sdk.options.navigation_map_source，"
                    "或在 SDK 装配时注入 NavigationMapSource"
                )
            occupancy = self.map_source.get_occupancy(
                robot_id=robot_id,
                state=state,
                goal=goal,
                excluded_source_refs=frozenset({carrying_object_ref})
                if carrying_object_ref
                else frozenset(),
            )
        if goal.frame_id != state.base_pose.frame_id or occupancy.frame_id != goal.frame_id:
            raise PlanningError("底盘、目标和占据栅格必须位于同一坐标系")
        if minimum_clearance_m is not None and minimum_clearance_m < 0:
            raise PlanningError("minimum_clearance_m 不能为负")
        clearance_m = minimum_clearance_m or 0.0
        motion_scale = self.carrying_motion_scale if carrying_object_ref else 1.0
        # carrying_motion_scale限制的是Robot Profile允许的携物速度上限，不应再
        # 把调用方已经给出的低速请求乘一次比例。否则0.05m/s会被二次缩到
        # 0.0125m/s。加速度和jerk仍在base_limits中同比降低，保留柔和起停。
        speed = min(
            maximum_speed_mps,
            self.capabilities.base_max_velocity_mps * motion_scale,
        )
        if speed <= 0:
            raise PlanningError("maximum_speed_mps 必须大于零")
        start = state.base_pose.position[:2]

        obstacle_rects = tuple(
            (box[0], box[1], box[3], box[4]) for box in occupancy.collision_boxes
        )
        if occupancy.collision_boxes:
            # 矩形快路径：障碍矩形一次光栅化，成本等于膨胀后矩形面积。
            # collision_boxes 与 occupied 来自同一批场景对象，不能重复膨胀。
            # minimum_clearance_m 是总净空要求（与旧 max(footprint, clearance)
            # 圆形膨胀同语义），不是包络之外的附加余量，取 max 而非相加。
            expand = max(self.base_half_size_m, clearance_m)
            footprint_blocked = rasterize_rects(occupancy, obstacle_rects, expand=expand)
        else:
            # 只有占据格的地图（Static/测试地图）保留逐格圆形膨胀语义。
            inflation_radius = max(self.capabilities.base_footprint_radius_m, clearance_m)
            footprint_blocked = occupancy.inflated(inflation_radius).occupied
        blocked = set(footprint_blocked)

        carried_blocked: frozenset[tuple[int, int]] = frozenset()
        if carrying_object_ref and occupancy.collision_boxes:
            if carrying_object_pose is None or carrying_object_extent_m is None:
                raise PlanningError("携物导航缺少实时物体Pose或extent")
            if carrying_object_pose.frame_id != state.base_pose.frame_id:
                raise PlanningError("携物对象与底盘必须位于同一坐标系")
            if len(carrying_object_extent_m) != 3 or any(
                value <= 0 for value in carrying_object_extent_m
            ):
                raise PlanningError("携物对象extent必须包含三个正数")
            offset_xy = (
                carrying_object_pose.position[0] - state.base_pose.position[0],
                carrying_object_pose.position[1] - state.base_pose.position[1],
            )
            # 携物期间底盘保持朝向，因此箱体相对底盘的实时平移关系也是
            # 固定的。这里把箱体AABB与有高度重叠的Scene对象做Minkowski
            # 展开，生成底盘中心禁行格；不能只用底盘圆形包络，否则箱体会
            # 在Robot本体尚未碰撞时先扫倒相邻箱。
            carried_blocked = carried_aabb_cells(
                occupancy,
                offset_xy=offset_xy,
                extent_xyz=carrying_object_extent_m,
                center_z=carrying_object_pose.position[2],
            )
            blocked.update(carried_blocked)

        start_cell = occupancy.world_to_cell(*start)
        if start_cell in occupancy.occupied:
            raise PlanningError("起点位于障碍物中")
        if start_cell in blocked:
            # 当前Robot位姿是Runtime已经接受的事实。栅格取整可能让其中心落入
            # 安全膨胀层，此时只对Robot本体包络应用既有退出窗口；原始障碍格
            # 始终保留。携物禁行区必须在这之后叠加，不能被该窗口整片清除。
            escape_radius = (
                max(self.capabilities.base_footprint_radius_m, clearance_m)
                + occupancy.resolution_m * math.sqrt(2.0) / 2.0
            )
            reach = math.ceil(escape_radius / occupancy.resolution_m) + 1
            for cell_x in range(
                max(0, start_cell[0] - reach), min(occupancy.width, start_cell[0] + reach + 1)
            ):
                for cell_y in range(
                    max(0, start_cell[1] - reach), min(occupancy.height, start_cell[1] + reach + 1)
                ):
                    cell = (cell_x, cell_y)
                    if cell not in blocked or cell in occupancy.occupied:
                        continue
                    if cell in carried_blocked:
                        continue
                    if math.dist(occupancy.cell_to_world(cell), start) > escape_radius:
                        continue
                    blocked.discard(cell)
            if start_cell in blocked:
                # prepare_transport已经用连续碰撞检查确认当前真实姿态安全；
                # 栅格中心取整仍可能把唯一的起点落入禁行格。只释放这一格，
                # 邻格继续受复合包络约束，因此A*只能沿真正增大净空的方向退出。
                blocked.discard(start_cell)
            blocked = frozenset(blocked)

        resolved_goal = resolve_nearby_free_goal(occupancy, frozenset(blocked), goal.position[:2])
        clearance = None
        if self.clearance_cost_radius > 0 and self.clearance_cost_weight > 0:
            clearance = (
                obstacle_rects,
                max(self.base_half_size_m, clearance_m),
                self.clearance_cost_radius,
                self.clearance_cost_weight,
            )
        route = smooth_path(
            astar(
                occupancy,
                start,
                resolved_goal,
                blocked=frozenset(blocked),
                maximum_expanded_nodes=self.maximum_expanded_nodes,
                clearance=clearance,
            ),
            occupancy,
            frozenset(blocked),
        )
        route[0] = start
        route[-1] = resolved_goal
        initial_yaw = yaw_from_xyzw(state.base_pose.quaternion_xyzw)
        final_yaw = yaw_from_xyzw(goal.quaternion_xyzw)
        diagnostics: list[str] = []
        try:
            require_ruckig()
            current = {"base_x": route[0][0], "base_y": route[0][1], "base_yaw": initial_yaw}
            limits = base_limits(self.capabilities, speed, motion_scale=motion_scale)
            points: list[BaseTrajectoryPoint] = []
            elapsed = 0.0
            for target in base_segment_targets(
                route,
                initial_yaw=initial_yaw,
                final_yaw=final_yaw,
                preserve_yaw=bool(carrying_object_ref),
            ):
                segment = ruckig_trajectory(current, target, limits)
                for index, point in enumerate(segment):
                    if points and index == 0:
                        continue
                    points.append(
                        BaseTrajectoryPoint(
                            time_from_start_s=elapsed + point.time_from_start_s,
                            x=point.positions["base_x"],
                            y=point.positions["base_y"],
                            yaw=point.positions["base_yaw"],
                        )
                    )
                elapsed = points[-1].time_from_start_s
                current = target
            planner = "astar-rect+shortcut+ruckig"
        except DependencyUnavailable:
            if not self.allow_debug_fallback:
                raise
            points = [
                BaseTrajectoryPoint(time_from_start_s=0, x=start[0], y=start[1], yaw=initial_yaw)
            ]
            elapsed = 0.0
            previous = start
            for target in base_segment_targets(
                route,
                initial_yaw=initial_yaw,
                final_yaw=final_yaw,
                preserve_yaw=bool(carrying_object_ref),
            ):
                current_xy = (target["base_x"], target["base_y"])
                elapsed += max(math.dist(previous, current_xy) / speed, 1e-3)
                points.append(
                    BaseTrajectoryPoint(
                        time_from_start_s=elapsed,
                        x=current_xy[0],
                        y=current_xy[1],
                        yaw=target["base_yaw"],
                    )
                )
                previous = current_xy
            planner = "astar-rect+shortcut+fake"
            diagnostics.append("Fake 测试使用定速时间戳；不作为真机规划结果")
        return MotionPlan(
            plan_id=str(uuid.uuid4()),
            robot_id=robot_id,
            generation=state.generation,
            kind=PlanKind.BASE,
            resources=["base"],
            frame_id=goal.frame_id,
            start={"x": start[0], "y": start[1], "yaw": initial_yaw},
            goal={"x": resolved_goal[0], "y": resolved_goal[1], "yaw": final_yaw},
            base_trajectory=points,
            collision_checked=True,
            estimated_duration_s=points[-1].time_from_start_s,
            planner=planner,
            diagnostics=diagnostics,
        )
