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

"""机器人状态读取。"""

from .backends.base import RobotBackend
from .utils.types import CapabilityInfo, CommandResult, RobotState


class StateClient:
    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend

    def capabilities(self) -> CapabilityInfo:
        return self._backend.capabilities()

    def get_scene_geometry(self) -> CommandResult:
        """按需读取世界坐标下的碰撞几何快照；体积指包围盒体积，勿逐帧调用。"""
        return self._backend.get_command(
            resource="state", operation="get_scene_geometry", target={}
        )

    def snapshot(self) -> RobotState:
        return self._backend.snapshot()

    def get_world_pose(self, name: str) -> CommandResult:
        """按当前场景的唯一 name 读取机器人或物体的实时世界位姿。

        name 必填，机器人同样使用其场景名称（例如 robot_r1）。
        output['name'] 为对象名，output['data'] 沿用位姿反馈格式：
        header.frame_id 为 world，position 为米，orientation 为 xyzw 四元数。
        feedback 包含该仿真步的采样时间、step_count 和 simulation_time；
        查询不会额外推进仿真，同一步内重复查询可得到相同位姿。
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name 必须是非空的场景对象名称")
        return self._backend.get_command(
            resource="state", operation="get_world_pose", target={"name": name}
        )
