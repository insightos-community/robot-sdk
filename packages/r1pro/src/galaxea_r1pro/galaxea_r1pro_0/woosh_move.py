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

"""WooshMove 能力：底盘运动控制（借鉴 Atom WooshMove）。

axis/target_value/speed 与 Atom 对齐（仅保留运动控制相关参数）；
速度带最大/最小限制，speed=None 时使用预设速度；运动前确认底盘控制器就绪。
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .backends.base import RobotBackend
from .utils.types import CommandHandle

logger = logging.getLogger(__name__)


class WooshMove:
    """R1 Pro 底盘运动控制（对齐 Atom WooshMove：预设速度 + 最大/最小速度限制）。"""

    # 预设速度与默认参数（对齐 Atom WooshMove）
    DEFAULT_LINEAR_SPEED_X = 0.1  # X 轴线速度 (m/s)
    DEFAULT_LINEAR_SPEED_Y = 0.08  # Y 轴线速度 (m/s)
    DEFAULT_ANGULAR_SPEED_Z = 15.0  # Z 轴角速度 (°/s)

    # 最大/最小速度限制（对齐 Atom：x/y ±0.3 m/s，z ±30 °/s）
    MAX_LINEAR_SPEED = 0.3
    MAX_ANGULAR_SPEED = 30.0

    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend
        self._emergency_stop = False

    def move_robot(
        self,
        *,
        axis: str,
        target_value: float,
        speed: Optional[float] = None,
    ) -> CommandHandle:
        """执行机器人运动控制（对齐 Atom WooshMove.move_robot）。

        Args:
            axis: 运动轴（x-前后 / y-左右 / z-旋转）
            target_value: 目标值（x/y: 米，z: 度）
            speed: 运动速度（x/y: m/s，z: °/s）；None 时使用预设速度

        Isaac 后端使用 Runtime 定位位移确认 target_value，到达后下发零速并等待静止。
        x/y 方向以提交时的底盘坐标系为准，z 正方向为逆时针。

        Returns:
            CommandHandle: 运动命令句柄

        Raises:
            ValueError: 轴无效
            RuntimeError: 机器人已紧急停止
        """

        # 1. 参数校验（对齐 Atom）
        if axis not in ("x", "y", "z"):
            raise ValueError(f"无效的轴，必须是 x、y 或 z，收到: {axis!r}")
        if self._emergency_stop:
            raise RuntimeError("机器人已紧急停止，请先恢复")

        # 3. 预设速度（对齐 Atom：speed=None 时使用默认值）
        if speed is None:
            speed = {
                "x": self.DEFAULT_LINEAR_SPEED_X,
                "y": self.DEFAULT_LINEAR_SPEED_Y,
                "z": self.DEFAULT_ANGULAR_SPEED_Z,
            }[axis]

        target_value, speed = float(target_value), float(speed)
        if not math.isfinite(target_value) or not math.isfinite(speed):
            raise ValueError("target_value 和 speed 必须为有限数值")
        if target_value and speed == 0:
            raise ValueError("非零 target_value 需要非零 speed")

        # 4. 最大/最小速度限制（对齐 Atom）
        if axis in ("x", "y"):
            speed = max(min(speed, self.MAX_LINEAR_SPEED), -self.MAX_LINEAR_SPEED)
        else:
            speed = max(min(speed, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)
        target_speed = math.copysign(abs(speed), target_value) if target_value else 0.0

        self._unlock_brake()

        # 5. 沿用 dx/dy/dyaw 速度字段；角速度由后端转换为弧度/s。
        if axis == "z":
            dx, dy, dyaw = 0.0, 0.0, target_speed
        elif axis == "y":
            dx, dy, dyaw = 0.0, target_speed, 0.0
        else:  # x
            dx, dy, dyaw = target_speed, 0.0, 0.0

        logger.info("开始%s轴运动：目标 %.3f | 速度 %.3f", axis, target_value, abs(speed))
        return self._backend.start_command(
            "woosh_move",
            "move_robot",
            {"dx": dx, "dy": dy, "dyaw": dyaw, "axis": axis, "target_value": target_value},
        )

    def _unlock_brake(self) -> CommandHandle | None:
        """确认仿真底盘控制器就绪；紧急停止置位时不提交。"""

        if self._emergency_stop:
            return None
        return self._backend.start_command("woosh_move", "unlock_brake", {"brake": False})

    def build_navigation_map(self, refresh: bool = False) -> CommandHandle:
        """异步生成/加载离线地图。完成后 command.get().output['navigation'] 包含文件路径。"""
        if self._emergency_stop:
            raise RuntimeError("机器人已紧急停止")
        if not isinstance(refresh, bool):
            raise ValueError("refresh 必须是 bool")
        return self._backend.start_command(
            "woosh_move", "build_navigation_map", {"refresh": refresh}
        )

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float | None = None,
        *,
        speed: float = 0.20,
        angular_speed: float = 0.50,
        position_tolerance: float = 0.05,
        yaw_tolerance: float = math.radians(5),
        timeout: float | None = None,
    ) -> CommandHandle:
        """离线 A* 导航。x/y 为 odom 坐标米，yaw 为弧度，省略时保持出发朝向。

        speed 单位 m/s，angular_speed 单位 rad/s；timeout 为仿真秒。
        场景假设静态；受阻停车。调用 command.stop(handle.command_id) 可取消。
        """
        if self._emergency_stop:
            raise RuntimeError("机器人已紧急停止")
        target = dict(
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            angular_speed=angular_speed,
            position_tolerance=position_tolerance,
            yaw_tolerance=yaw_tolerance,
            timeout=timeout,
        )
        return self._backend.start_command("woosh_move", "navigate_to_pose", target)

    def stop(self) -> CommandHandle:
        """下发零速度并锁定当前客户端；通过命令结果确认实际静止。"""

        self._emergency_stop = True
        return self._backend.start_command(
            "woosh_move",
            "move_robot",
            {"dx": 0.0, "dy": 0.0, "dyaw": 0.0},
        )
