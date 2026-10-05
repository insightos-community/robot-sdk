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

class SensorsModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def list(self):
        return self.backend.sensors(self.robot_id)

    def latest(self, sensor_id):
        return self.backend.latest_sensor_frame(self.robot_id, sensor_id)

    def subscribe(self, sensor_id, *, poll_interval_seconds=0.05):
        return self.backend.subscribe_sensor(
            self.robot_id, sensor_id, poll_interval_seconds=poll_interval_seconds
        )

    def subscribe_state(self, *, poll_interval_seconds=0.1):
        return self.backend.subscribe_state(
            self.robot_id, poll_interval_seconds=poll_interval_seconds
        )
