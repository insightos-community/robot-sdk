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

"""不依赖模型资产的 Fake IK，只用于流程与接口测试。"""

from semantic_robot_sdk_core import PlanningError


class FakeKinematicsProvider:
    name = "fake"

    def solve(self, *, end_effector, target, state, **_kwargs):
        prefix = (
            "left_arm"
            if end_effector == "left"
            else "right_arm"
            if end_effector == "right"
            else None
        )
        if prefix is None:
            raise PlanningError(f"未知末端：{end_effector}")
        if max(abs(value) for value in target.position) > 5:
            raise PlanningError("Fake 目标超出可达范围")
        names = [name for name in state.joint_positions if name.startswith(prefix)][:3]
        return {name: target.position[index] for index, name in enumerate(names)}

    def solve_many(self, *, targets, state, **kwargs):
        goal = {}
        for end_effector, target in targets.items():
            goal.update(self.solve(end_effector=end_effector, target=target, state=state, **kwargs))
        return goal

    def reachable(self, **kwargs):
        try:
            self.solve(**kwargs)
            return True
        except PlanningError:
            return False
