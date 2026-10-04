"""本地导航使用的地图来源；Ability 不需要构造占据栅格。"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol

from semantic_robot_sdk_core import (
    BackendUnavailable,
    GenerationMismatch,
    PlanningError,
    Pose,
    RobotState,
)

from .local_navigation_algorithms import OccupancyGrid


class NavigationMapSource(Protocol):
    """把部署侧地图转换成当前型号的本地规划输入。"""

    def get_occupancy(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goal: Pose,
        excluded_source_refs: frozenset[str] = frozenset(),
    ) -> OccupancyGrid: ...


class StaticNavigationMapSource:
    """固定测试地图；生产 MuJoCo 使用 RuntimeSceneNavigationMapSource。"""

    def __init__(self, occupancy: OccupancyGrid):
        self.occupancy = occupancy

    @classmethod
    def from_config(cls, value: dict) -> "StaticNavigationMapSource":
        try:
            grid = OccupancyGrid(
                width=int(value["width"]),
                height=int(value["height"]),
                resolution_m=float(value["resolution_m"]),
                origin_xy=tuple(float(item) for item in value["origin_xy"]),
                frame_id=str(value["frame_id"]),
                occupied=frozenset(
                    tuple(int(item) for item in cell) for cell in value.get("occupied", [])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PlanningError(f"navigation_map_source 配置不合法：{error}") from error
        return cls(grid)

    def get_occupancy(self, *, robot_id, state, goal, excluded_source_refs=frozenset()):
        if self.occupancy.frame_id != goal.frame_id:
            raise PlanningError("导航地图与目标坐标系不一致")
        return self.occupancy


class RuntimeSceneNavigationMapSource:
    """把原生 MuJoCo 场景快照转换为 SDK 本地导航占据栅格。

    Scene instance 由部署层明确注入。Robot API 本身没有 scene_instance_id，SDK 不能
    根据“当前唯一场景”猜测，否则同一 Runtime 同时运行多个场景时会给错误 Robot
    规划路线。这里只消费 Runtime 的公共 SceneSnapshot，不读取 MuJoCo geom/body。
    """

    def __init__(
        self,
        endpoint: str,
        scene_instance_id: str,
        *,
        resolution_m: float = 0.05,
        padding_m: float = 1.0,
        timeout_seconds: float = 10.0,
        obstacle_z_min: float = -0.2,
        obstacle_z_max: float = 0.9,
    ):
        if not scene_instance_id:
            raise ValueError("MuJoCo 场景导航必须配置 scene_instance_id")
        if resolution_m <= 0 or padding_m <= 0:
            raise ValueError("导航栅格分辨率和边界留白必须大于零")
        if obstacle_z_min >= obstacle_z_max:
            raise ValueError("obstacle_z_min 必须小于 obstacle_z_max")
        self.endpoint = endpoint.rstrip("/")
        self.scene_instance_id = scene_instance_id
        self.resolution_m = resolution_m
        self.padding_m = padding_m
        self.timeout_seconds = timeout_seconds
        # 只把底盘高度带内的物体当障碍：悬浮或高处的物体不应挡住底盘。
        # 默认区间与 plugin-mujoco 的 obstacle_z_min/z_max 保持一致。
        self.obstacle_z_min = float(obstacle_z_min)
        self.obstacle_z_max = float(obstacle_z_max)

    def _z_overlaps_band(self, item: dict) -> bool:
        """物体 AABB 与底盘障碍高度带 [z_min, z_max] 有重叠才算障碍。"""

        center_z = float(item["pose"]["position"][2])
        half_z = float(item["extent"][2]) / 2.0
        return center_z + half_z >= self.obstacle_z_min and center_z - half_z <= self.obstacle_z_max

    def get_occupancy(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goal: Pose,
        excluded_source_refs: frozenset[str] = frozenset(),
    ) -> OccupancyGrid:
        instance = urllib.parse.quote(self.scene_instance_id, safe="")
        request = urllib.request.Request(
            f"{self.endpoint}/api/v1/scene-instances/{instance}/snapshot",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                snapshot = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise BackendUnavailable(f"无法读取 MuJoCo 场景快照: {error}") from error
        if int(snapshot.get("generation", 0)) != state.generation:
            raise GenerationMismatch("场景快照 generation 与 Robot 当前状态不一致")
        frame_id = str(snapshot.get("coordinate_frame", "world"))
        if frame_id != goal.frame_id:
            raise PlanningError("Runtime 场景快照与导航目标坐标系不一致")

        excluded = {value.rstrip("/").rsplit("/", 1)[-1] for value in excluded_source_refs}
        # 被 Robot 实时持有的物体仍存在于 SceneSnapshot 和碰撞反馈中，只从底盘
        # 中心的占据栅格里排除。否则重规划会把随 Robot 一起运动的箱体误判为
        # 当前路径障碍；这不是 Semantic Map 规则，也不会绕过真实接触安全检查。
        objects = [
            item
            for item in snapshot.get("objects", [])
            if item.get("extent")
            and str(item.get("source_id", "")).rstrip("/").rsplit("/", 1)[-1] not in excluded
            and self._z_overlaps_band(item)
        ]
        points = (
            [
                (state.base_pose.position[0], state.base_pose.position[1]),
                (goal.position[0], goal.position[1]),
            ]
            if state.base_pose
            else [(goal.position[0], goal.position[1])]
        )
        for item in objects:
            pose = item["pose"]["position"]
            extent = item["extent"]
            points.extend(
                (
                    (pose[0] - extent[0] / 2, pose[1] - extent[1] / 2),
                    (pose[0] + extent[0] / 2, pose[1] + extent[1] / 2),
                )
            )
        min_x = min(item[0] for item in points) - self.padding_m
        min_y = min(item[1] for item in points) - self.padding_m
        max_x = max(item[0] for item in points) + self.padding_m
        max_y = max(item[1] for item in points) + self.padding_m
        width = max(1, math.ceil((max_x - min_x) / self.resolution_m))
        height = max(1, math.ceil((max_y - min_y) / self.resolution_m))
        occupied: set[tuple[int, int]] = set()
        collision_boxes = []
        for item in objects:
            x, y, z = (float(value) for value in item["pose"]["position"][:3])
            size_x, size_y, size_z = (float(value) for value in item["extent"][:3])
            collision_boxes.append(
                (
                    x - size_x / 2,
                    # AABB上边界与栅格边界重合时，边界外的下一格没有真实重叠。
                    # 先多占一格、再膨胀底盘包络会让相同工位随浮点原点变化而忽隐忽现。
                    y - size_y / 2,
                    z - size_z / 2,
                    x + size_x / 2,
                    y + size_y / 2,
                    z + size_z / 2,
                )
            )
            x0 = max(0, math.floor((x - size_x / 2 - min_x) / self.resolution_m))
            x1 = min(
                width - 1,
                math.ceil((x + size_x / 2 - min_x) / self.resolution_m) - 1,
            )
            y0 = max(0, math.floor((y - size_y / 2 - min_y) / self.resolution_m))
            y1 = min(
                height - 1,
                math.ceil((y + size_y / 2 - min_y) / self.resolution_m) - 1,
            )
            occupied.update((ix, iy) for ix in range(x0, x1 + 1) for iy in range(y0, y1 + 1))
        return OccupancyGrid(
            width=width,
            height=height,
            resolution_m=self.resolution_m,
            origin_xy=(min_x, min_y),
            frame_id=frame_id,
            occupied=frozenset(occupied),
            collision_boxes=tuple(collision_boxes),
        )


def fake_navigation_map_source() -> StaticNavigationMapSource:
    """Fake 流程专用空地图，不能作为真机或 MuJoCo 的默认地图。"""

    return StaticNavigationMapSource(
        OccupancyGrid(
            width=200,
            height=200,
            resolution_m=0.25,
            origin_xy=(-25.0, -25.0),
            frame_id="world",
            occupied=frozenset(),
        )
    )
