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

"""仿真 SDK 的公共数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence


def utc_now() -> datetime:
    """返回带时区的 UTC 时间，避免各后端产生不可比较的时间。"""
    return datetime.now(timezone.utc)


class BackendKind(StrEnum):
    MUJOCO = "mujoco"
    ISAAC = "isaac"


class CommandStatus(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Pose:
    """右手坐标系中的位姿，位置单位为米，四元数顺序为 xyzw。"""

    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointState:
    names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...] = ()
    efforts: tuple[float, ...] = ()


@dataclass(frozen=True)
class RobotState:
    robot_id: str
    backend: BackendKind
    connected: bool
    mode: str
    base_pose: Pose
    joints: JointState
    held_resources: tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class SensorFrame:
    sensor_id: str
    frame_id: str
    timestamp: datetime
    encoding: str
    data: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityInfo:
    robot_model: str
    backend: BackendKind
    firmware_profile: str
    resources: tuple[str, ...]
    sdk_api_version: str = "1"


@dataclass(frozen=True)
class CommandHandle:
    command_id: str
    robot_id: str
    resource: str
    operation: str
    accepted_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class CommandFeedback:
    command_id: str
    status: CommandStatus
    progress: float
    phase: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    status: CommandStatus
    feedback: CommandFeedback
    output: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


def as_float_tuple(values: Sequence[float], size: int, field_name: str) -> tuple[float, ...]:
    """在进入后端前统一校验维度和数值类型。"""

    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{field_name} 必须包含 {size} 个数值")
    return result
