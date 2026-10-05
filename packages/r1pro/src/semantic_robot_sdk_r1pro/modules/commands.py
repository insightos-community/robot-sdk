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

class CommandsModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def get(self, command_id):
        return self.backend.command(self.robot_id, command_id)

    def feedback(self, command_id):
        return self.backend.feedback(self.robot_id, command_id)

    def stop(self, command_id):
        return self.backend.stop(self.robot_id, command_id)
