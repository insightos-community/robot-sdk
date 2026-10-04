"""各后端必须实现的内部接口。"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..utils.types import CapabilityInfo, CommandHandle, CommandResult, RobotState, SensorFrame


class RobotBackend(Protocol):
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def capabilities(self) -> CapabilityInfo: ...

    def snapshot(self) -> RobotState: ...

    def start_command(
        self,
        resource: str,
        operation: str,
        target: Mapping[str, Any],
    ) -> CommandHandle: ...

    def get_command(
        self,
        command_id: str | None = None,
        *,
        resource: str | None = None,
        operation: str | None = None,
        target: Mapping[str, Any] | None = None,
    ) -> CommandResult: ...

    def stop_command(self, command_id: str, reason: str) -> CommandResult: ...

    def hold(self, resources: tuple[str, ...], reason: str) -> bool: ...

    def capture(self, sensor_id: str, encoding: str) -> SensorFrame: ...

    def capture_rgbd(self, sensor_id: str) -> tuple[SensorFrame, SensorFrame]: ...
