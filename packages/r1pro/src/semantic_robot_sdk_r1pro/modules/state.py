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

class StateModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def capabilities(self):
        return self.backend.capabilities()

    def snapshot(self):
        return self.backend.state(self.robot_id)

    def scene_snapshot(self):
        """返回 Backend 提供的场景事实；真机未配置场景来源时明确失败。"""

        operation = getattr(self.backend, "scene_snapshot", None)
        if operation is None:
            raise RuntimeError("当前 Robot Backend 不提供 SceneSnapshot")
        return operation()
