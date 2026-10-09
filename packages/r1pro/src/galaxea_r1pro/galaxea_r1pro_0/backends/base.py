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
