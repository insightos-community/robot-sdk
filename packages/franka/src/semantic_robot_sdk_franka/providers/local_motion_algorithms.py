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

"""关节路径的时间参数化。"""

from __future__ import annotations

import math

from semantic_robot_sdk_core.errors import PlanningError
from semantic_robot_sdk_core.models import JointLimit, JointTrajectoryPoint
from .dependencies import require_ruckig


def validate_joint_goal(
    current: dict[str, float], goal: dict[str, float], limits: dict[str, JointLimit]
) -> None:
    if not goal:
        raise PlanningError("关节目标不能为空")
    unknown = set(goal) - set(current)
    if unknown:
        raise PlanningError(f"Robot 当前状态不包含关节：{sorted(unknown)}")
    missing_limits = set(goal) - set(limits)
    if missing_limits:
        raise PlanningError(f"Robot Profile 缺少关节限制：{sorted(missing_limits)}")
    for name, value in goal.items():
        limit = limits[name]
        if value < limit.lower or value > limit.upper:
            raise PlanningError(f"关节 {name} 目标 {value} 超出 [{limit.lower}, {limit.upper}]")


def smoothstep_trajectory(
    current: dict[str, float],
    goal: dict[str, float],
    limits: dict[str, JointLimit],
    *,
    sample_period_s: float = 0.05,
) -> list[JointTrajectoryPoint]:
    """显式调试回退。

    五次 smoothstep 保证起止速度和加速度为零；持续时间按速度、加速度和
    加加速度限制保守放大。它不替代 Ruckig，生产调用必须使用后者。
    """

    validate_joint_goal(current, goal, limits)
    duration = sample_period_s
    for name, target in goal.items():
        delta = abs(target - current[name])
        limit = limits[name]
        duration = max(
            duration,
            2.0 * delta / limit.max_velocity,
            math.sqrt(6.0 * delta / limit.max_acceleration) if delta else 0.0,
            (60.0 * delta / limit.max_jerk) ** (1.0 / 3.0) if delta else 0.0,
        )
    count = max(2, math.ceil(duration / sample_period_s) + 1)
    names = list(goal)
    points: list[JointTrajectoryPoint] = []
    for index in range(count):
        ratio = index / (count - 1)
        blend = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
        derivative = 30 * ratio**2 - 60 * ratio**3 + 30 * ratio**4
        positions = {name: current[name] + (goal[name] - current[name]) * blend for name in names}
        velocities = {name: (goal[name] - current[name]) * derivative / duration for name in names}
        points.append(
            JointTrajectoryPoint(
                time_from_start_s=duration * ratio,
                positions=positions,
                velocities=velocities,
            )
        )
    return points


def ruckig_trajectory(
    current: dict[str, float], goal: dict[str, float], limits: dict[str, JointLimit]
) -> list[JointTrajectoryPoint]:
    """使用 Ruckig 生成满足速度、加速度和加加速度限制的轨迹。"""

    validate_joint_goal(current, goal, limits)
    require_ruckig()
    from ruckig import InputParameter, Result, Ruckig, Trajectory  # type: ignore

    names = list(goal)
    dofs = len(names)
    otg = Ruckig(dofs, 0.01)
    inp = InputParameter(dofs)
    inp.current_position = [current[name] for name in names]
    inp.current_velocity = [0.0] * dofs
    inp.current_acceleration = [0.0] * dofs
    inp.target_position = [goal[name] for name in names]
    inp.target_velocity = [0.0] * dofs
    inp.target_acceleration = [0.0] * dofs
    inp.max_velocity = [limits[name].max_velocity for name in names]
    inp.max_acceleration = [limits[name].max_acceleration for name in names]
    inp.max_jerk = [limits[name].max_jerk for name in names]
    trajectory = Trajectory(dofs)
    result = otg.calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise PlanningError(f"Ruckig 轨迹生成失败：{result}")
    count = max(2, math.ceil(trajectory.duration / 0.05) + 1)
    points: list[JointTrajectoryPoint] = []
    for index in range(count):
        at = min(trajectory.duration, index * trajectory.duration / (count - 1))
        positions, velocities, _accelerations = trajectory.at_time(at)
        points.append(
            JointTrajectoryPoint(
                time_from_start_s=at,
                positions=dict(zip(names, positions, strict=True)),
                velocities=dict(zip(names, velocities, strict=True)),
            )
        )
    return points
