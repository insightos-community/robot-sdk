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

"""Robot 型号相关的运动学求解接口。"""

from __future__ import annotations

from typing import Protocol

from .models import Pose


class KinematicsSolver(Protocol):
    """实现者通常由 Pinocchio Robot 模型提供。

    `solve_end_effector` 必须检查不可达、奇异位形和关节限位；
    `collision_free` 必须同时检查自碰撞和传入环境碰撞几何。
    """

    def solve_end_effector(
        self,
        end_effector: str,
        target: Pose,
        current_joints: dict[str, float],
    ) -> dict[str, float]: ...

    def collision_free(
        self,
        joint_positions: dict[str, float],
        environment: object | None,
    ) -> bool: ...
