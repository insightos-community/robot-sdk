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

"""Robot SDK 与仿真/真机之间的最小 Backend 边界。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from .models import (
    Command,
    Feedback,
    MotionPlan,
    RobotCapabilities,
    RobotState,
    SensorFrame,
)
from .sensors import RobotStateSubscription, SensorSubscription


class RobotBackend(Protocol):
    """Backend 只执行已规划轨迹，不负责 IK、导航或失败恢复策略。"""

    def capabilities(self) -> RobotCapabilities: ...

    def state(self, robot_id: str) -> RobotState: ...

    def execute_plan(self, plan: MotionPlan, command_id: str) -> Command: ...

    def command(self, robot_id: str, command_id: str) -> Command: ...

    def feedback(self, robot_id: str, command_id: str) -> Iterator[Feedback]: ...

    def stop(self, robot_id: str, command_id: str) -> Command: ...

    def hold(self, robot_id: str) -> None: ...

    def sensors(self, robot_id: str) -> list[str]: ...

    def latest_sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame: ...

    def subscribe_sensor(
        self,
        robot_id: str,
        sensor_id: str,
        *,
        poll_interval_seconds: float = 0.05,
    ) -> SensorSubscription: ...

    def subscribe_state(
        self,
        robot_id: str,
        *,
        poll_interval_seconds: float = 0.1,
    ) -> RobotStateSubscription: ...
