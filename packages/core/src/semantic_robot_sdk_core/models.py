# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robot SDK 公共数据模型。

公共长度单位为米、角度单位为弧度、时间单位为秒，四元数顺序固定为 xyzw。
模型不包含 MuJoCo body/geom、Isaac prim 或其他引擎内部标识。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    # Robot 的位姿、关节和轨迹不能包含 NaN/Inf。物理 Runtime 一旦收到非有限值，
    # 可能污染整个 MjData，后续 stop/hold 也无法保证 Robot 回到可控状态。
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CommandState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class PlanKind(str, Enum):
    JOINT = "joint_trajectory"
    BASE = "base_trajectory"
    GRIPPER = "gripper_command"
    CONTROL = "control_sequence"


class Pose(StrictModel):
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame_id: str

    @model_validator(mode="after")
    def validate_quaternion(self) -> Pose:
        norm = sum(value * value for value in self.quaternion_xyzw) ** 0.5
        if norm < 1e-8:
            raise ValueError("四元数不能为零")
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("四元数必须归一化")
        return self


class EnvironmentCollisionObject(StrictModel):
    """来自 Semantic Map 或场景快照的引擎无关碰撞体。"""

    source_id: str
    shape: Literal["box", "sphere"]
    pose: Pose
    size_xyz: tuple[float, float, float] | None = None
    radius_m: float | None = Field(default=None, gt=0)
    allowed_contact_end_effectors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_shape_parameters(self) -> EnvironmentCollisionObject:
        if self.shape == "box":
            if self.size_xyz is None or any(value <= 0 for value in self.size_xyz):
                raise ValueError("box 碰撞体必须提供正数 size_xyz")
            if self.radius_m is not None:
                raise ValueError("box 碰撞体不能提供 radius_m")
        elif self.radius_m is None:
            raise ValueError("sphere 碰撞体必须提供 radius_m")
        elif self.size_xyz is not None:
            raise ValueError("sphere 碰撞体不能提供 size_xyz")
        return self


class EnvironmentCollisionSet(StrictModel):
    """一次规划使用的静态环境碰撞快照。"""

    frame_id: str
    objects: list[EnvironmentCollisionObject]

    @model_validator(mode="after")
    def validate_frames(self) -> EnvironmentCollisionSet:
        mismatched = [
            item.source_id for item in self.objects if item.pose.frame_id != self.frame_id
        ]
        if mismatched:
            raise ValueError(f"环境碰撞体坐标系不一致: {mismatched}")
        return self


class JointLimit(StrictModel):
    lower: float
    upper: float
    max_velocity: float = Field(gt=0)
    max_acceleration: float = Field(gt=0)
    max_jerk: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> JointLimit:
        if self.lower >= self.upper:
            raise ValueError("关节下限必须小于上限")
        return self


class JointTrajectoryPoint(StrictModel):
    time_from_start_s: float = Field(ge=0)
    positions: dict[str, float]
    velocities: dict[str, float] = Field(default_factory=dict)


class BaseTrajectoryPoint(StrictModel):
    time_from_start_s: float = Field(ge=0)
    x: float
    y: float
    yaw: float


class GripperTarget(StrictModel):
    """Robot SDK 规划后的类型化夹爪目标。"""

    gripper: str = Field(min_length=1)
    opening_m: float = Field(ge=0)
    force_limit_n: float = Field(default=20, ge=0)
    stop_on_contact: bool = False


class ToolDescriptor(StrictModel):
    """跨 Runtime、SDK、Ability 和 Skill 使用的稳定工具描述。"""

    tool_ref: str = Field(min_length=1)
    side: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    frame: str = Field(min_length=1)
    joint: str = Field(min_length=1)
    travel_m: float = Field(gt=0)
    normal_force_n: float = Field(gt=0)
    maximum_force_n: float = Field(gt=0)


class ToolState(StrictModel):
    """末端工具的通用实时传感状态，不包含稳定持物等业务结论。"""

    tool_ref: str = Field(min_length=1)
    side: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    position: float
    velocity: float
    effort: float
    target_position: float | None = None
    reached_target: bool = False
    hook_contact: bool = False
    clamp_contact: bool = False
    sensor_fault: bool = False
    hook_force_n: float = 0.0
    clamp_force_n: float = 0.0
    hook_support_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    hook_tangential_speed_m_s: float = Field(default=0.0, ge=0.0)


class MotionPlan(StrictModel):
    plan_id: str
    robot_id: str
    generation: int = Field(ge=1)
    kind: PlanKind
    resources: list[str] = Field(min_length=1)
    frame_id: str
    start: dict[str, Any]
    goal: dict[str, Any]
    joint_trajectory: list[JointTrajectoryPoint] = Field(default_factory=list)
    base_trajectory: list[BaseTrajectoryPoint] = Field(default_factory=list)
    gripper_command: GripperTarget | None = None
    collision_checked: bool
    estimated_duration_s: float = Field(ge=0)
    planner: str
    diagnostics: list[str] = Field(default_factory=list)
    stop_on_contact: bool = False
    contact_tool_refs: list[str] = Field(default_factory=list)
    max_contact_force_n: float | None = Field(default=None, gt=0)
    position_tolerance_rad: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_payload(self) -> MotionPlan:
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("运动计划不能重复声明同一资源")

        payloads = {
            PlanKind.JOINT: bool(self.joint_trajectory),
            PlanKind.BASE: bool(self.base_trajectory),
            PlanKind.GRIPPER: self.gripper_command is not None,
        }
        if self.kind is PlanKind.CONTROL:
            raise ValueError("定时控制使用 ControlSequence，不属于几何规划 MotionPlan")
        if not payloads[self.kind]:
            raise ValueError(f"{self.kind.value} 计划缺少对应载荷")
        if sum(payloads.values()) != 1:
            raise ValueError("运动计划只能包含与 kind 对应的一种载荷")

        points = self.joint_trajectory if self.kind is PlanKind.JOINT else self.base_trajectory
        if points:
            times = [point.time_from_start_s for point in points]
            if abs(times[0]) > 1e-9:
                raise ValueError("轨迹第一个点必须从 0 秒开始")
            if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
                raise ValueError("轨迹时间必须严格递增")
            if abs(times[-1] - self.estimated_duration_s) > 1e-6:
                raise ValueError("estimated_duration_s 必须等于最后一个轨迹点时间")
        if self.joint_trajectory:
            names = set(self.joint_trajectory[0].positions)
            if not names:
                raise ValueError("关节轨迹点不能为空")
            if any(set(point.positions) != names for point in self.joint_trajectory):
                raise ValueError("同一关节轨迹的每个点必须包含相同关节")
        if self.stop_on_contact and (self.kind is not PlanKind.JOINT or not self.contact_tool_refs):
            raise ValueError("接触完成只适用于声明了目标工具的关节轨迹")
        return self


class RobotCapabilities(StrictModel):
    model: str
    kind: str
    coordinate_frame: str = "world"
    kinematic_root_frame: str = "base_link"
    joint_names: list[str]
    joint_groups: dict[str, list[str]]
    joint_limits: dict[str, JointLimit]
    end_effectors: list[str]
    grippers: list[str]
    tools: list[ToolDescriptor] = Field(default_factory=list)
    sensors: list[str]
    supports_base: bool = False
    base_footprint_radius_m: float | None = Field(default=None, gt=0)
    base_max_velocity_mps: float = Field(default=0.4, gt=0)
    base_max_acceleration_mps2: float = Field(default=0.5, gt=0)
    base_max_jerk_mps3: float = Field(default=1.0, gt=0)
    base_max_angular_velocity_rps: float = Field(default=0.8, gt=0)
    base_max_angular_acceleration_rps2: float = Field(default=1.0, gt=0)
    base_max_angular_jerk_rps3: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def validate_profile(self) -> RobotCapabilities:
        """阻止型号包与 Runtime 使用两套不一致的关节或传感器清单。"""

        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Robot Profile 的 joint_names 不能重复")
        if set(self.joint_limits) != set(self.joint_names):
            raise ValueError("Robot Profile 必须为每个且仅为已声明关节提供限制")
        for group, joints in self.joint_groups.items():
            if not joints:
                raise ValueError(f"关节组 {group} 不能为空")
            unknown = set(joints) - set(self.joint_names)
            if unknown:
                raise ValueError(f"关节组 {group} 包含未声明关节：{sorted(unknown)}")
            if len(set(joints)) != len(joints):
                raise ValueError(f"关节组 {group} 不能包含重复关节")
        for label, values in (
            ("end_effectors", self.end_effectors),
            ("grippers", self.grippers),
            ("sensors", self.sensors),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"Robot Profile 的 {label} 不能重复")
        tool_refs = [tool.tool_ref for tool in self.tools]
        tool_sides = [tool.side for tool in self.tools]
        if len(set(tool_refs)) != len(tool_refs):
            raise ValueError("Robot Profile 的 tool_ref 不能重复")
        if len(set(tool_sides)) != len(tool_sides):
            raise ValueError("一个逻辑侧只能声明一个主动工具")
        unknown_tool_sides = set(tool_sides) - set(self.grippers)
        if unknown_tool_sides:
            raise ValueError(f"工具引用了未声明夹爪：{sorted(unknown_tool_sides)}")
        if self.supports_base and self.base_footprint_radius_m is None:
            raise ValueError("移动 Robot 必须声明底盘平面外形")
        if not self.supports_base and self.base_footprint_radius_m is not None:
            raise ValueError("固定 Robot 不能声明移动底盘外形")
        return self


class SceneObject(StrictModel):
    source_id: str
    category: str
    name: str
    pose: Pose
    extent: tuple[float, float, float] | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class SceneRegion(StrictModel):
    source_id: str
    name: str
    pose: Pose
    extent: tuple[float, float, float] | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class SceneSnapshot(StrictModel):
    scene_key: str
    instance_id: str
    generation: int = Field(ge=1)
    coordinate_frame: str
    objects: list[SceneObject]
    regions: list[SceneRegion]
    observed_at: datetime


class RobotState(StrictModel):
    robot_id: str
    generation: int = Field(ge=1)
    observed_at: datetime = Field(default_factory=utc_now)
    base_pose: Pose | None = None
    joint_positions: dict[str, float]
    end_effectors: dict[str, Pose] = Field(default_factory=dict)
    gripper_openings: dict[str, float] = Field(default_factory=dict)
    tool_states: dict[str, ToolState] = Field(default_factory=dict)
    in_hold: bool = False


class Command(StrictModel):
    command_id: str
    robot_id: str
    generation: int = Field(ge=1)
    kind: PlanKind
    status: CommandState
    progress: float = Field(default=0, ge=0, le=1)
    submitted_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    reason: str | None = None


class Feedback(StrictModel):
    command_id: str
    sequence: int = Field(ge=1)
    progress: float = Field(ge=0, le=1)
    message: str = ""
    observed_at: datetime = Field(default_factory=utc_now)


class Result(StrictModel):
    command: Command
    final_state: RobotState | None = None
    diagnostics: list[str] = Field(default_factory=list)


class SensorFrame(StrictModel):
    sensor_id: str
    sequence: int = Field(ge=1)
    generation: int = Field(ge=1)
    frame_id: str
    encoding: str
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    observed_at: datetime = Field(default_factory=utc_now)
    payload: bytes | dict[str, Any]
