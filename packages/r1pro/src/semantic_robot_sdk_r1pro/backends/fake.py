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

"""R1 Pro 确定性 Fake Backend。"""

from collections.abc import Callable

from semantic_robot_sdk_core import Pose, RobotState, ToolState
from semantic_robot_sdk_core.fake_backend import FakeBackend as CoreFakeBackend
from semantic_robot_sdk_core.shared_fake_backend import SharedFakeBackend as CoreSharedFakeBackend

from ..profile import capabilities


def initial_state(robot_id: str) -> RobotState:
    """创建 R1 Pro Fake 初态；内存和共享实现必须从同一初态开始。"""

    profile = capabilities()
    return RobotState(
        robot_id=robot_id,
        generation=1,
        base_pose=Pose(position=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 1), frame_id="world"),
        joint_positions={name: 0.0 for name in profile.joint_names},
        end_effectors={
            name: Pose(
                position=(0, 0, 0),
                quaternion_xyzw=(0, 0, 0, 1),
                frame_id="world",
            )
            for name in profile.end_effectors
        },
        gripper_openings={tool.side: tool.travel_m for tool in profile.tools},
        tool_states={
            tool.tool_ref: ToolState(
                tool_ref=tool.tool_ref,
                side=tool.side,
                kind=tool.kind,
                position=tool.travel_m,
                velocity=0.0,
                effort=0.0,
            )
            for tool in profile.tools
        },
        in_hold=True,
    )


class FakeBackend(CoreFakeBackend):
    def __init__(self, robot_id: str, *, auto_complete=True, stop_confirmed=True):
        super().__init__(
            capabilities(),
            initial_state(robot_id),
            auto_complete=auto_complete,
            stop_confirmed=stop_confirmed,
        )


class SharedFakeBackend(CoreSharedFakeBackend):
    """多个 Ability 进程访问同一台 R1 Pro Fake Robot 的入口。"""

    def __init__(
        self,
        robot_id: str,
        state_path: str,
        *,
        auto_complete=True,
        stop_confirmed=True,
        initializer: Callable[[CoreFakeBackend], None] | None = None,
    ):
        super().__init__(
            capabilities(),
            initial_state(robot_id),
            state_path,
            auto_complete=auto_complete,
            stop_confirmed=stop_confirmed,
            initializer=initializer,
        )
