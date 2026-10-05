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

"""R1 Pro 真机与仿真统一 Robot SDK。"""

from .backends import FakeBackend, IsaacBackend, RealBackend, SharedFakeBackend
from .profile import LEFT_ARM, RIGHT_ARM, TORSO, capabilities
from .sdk import R1ProSDK, create_mujoco_sdk
from .providers.local_navigation_algorithms import OccupancyGrid

__all__ = [
    "FakeBackend",
    "IsaacBackend",
    "LEFT_ARM",
    "MujocoBackend",
    "OccupancyGrid",
    "R1ProSDK",
    "SharedFakeBackend",
    "create_mujoco_sdk",
    "RIGHT_ARM",
    "RealBackend",
    "TORSO",
    "capabilities",
]


def __getattr__(name: str):
    if name == "MujocoBackend":
        from .backends.mujoco import MujocoBackend

        return MujocoBackend
    raise AttributeError(name)
