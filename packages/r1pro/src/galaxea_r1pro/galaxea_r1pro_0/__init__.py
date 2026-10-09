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

"""R1 Pro Robot SDK 的稳定公开入口。"""

from .utils.errors import (
    BackendUnavailableError,
    CommandNotFoundError,
    ProfileError,
    RobotSDKError,
)
from .utils.profile import RobotProfile
from .sdk import R1ProSDK
from .utils.types import (
    BackendKind,
    CapabilityInfo,
    CommandFeedback,
    CommandHandle,
    CommandResult,
    CommandStatus,
    JointState,
    Pose,
    RobotState,
    SensorFrame,
)

__all__ = [
    "BackendKind",
    "BackendUnavailableError",
    "CapabilityInfo",
    "CommandFeedback",
    "CommandHandle",
    "CommandNotFoundError",
    "CommandResult",
    "CommandStatus",
    "JointState",
    "Pose",
    "ProfileError",
    "R1ProSDK",
    "RobotProfile",
    "RobotSDKError",
    "RobotState",
    "SensorFrame",
]
