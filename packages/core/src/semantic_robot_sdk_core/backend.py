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
