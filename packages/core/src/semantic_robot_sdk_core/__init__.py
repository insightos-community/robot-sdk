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

"""Semantic Robot SDK 公共接口；不包含任何具体 Robot 或 Runtime 实现。"""

from .backend import RobotBackend
from .config import CONFIG_ENV, RobotConfig, RobotDeployment, load_robot_deployment
from .fake_backend import FakeBackend, FakeBackendSnapshot, FakeGraspFixture
from .shared_fake_backend import SharedFakeBackend, SQLiteFakeStateStore
from .errors import (
    BackendRequestError,
    BackendUnavailable,
    ConfigurationError,
    DependencyUnavailable,
    GenerationMismatch,
    PlanningError,
    RobotSDKError,
)
from .models import (
    BaseTrajectoryPoint,
    Command,
    CommandState,
    EnvironmentCollisionObject,
    EnvironmentCollisionSet,
    Feedback,
    GripperTarget,
    JointLimit,
    JointTrajectoryPoint,
    MotionPlan,
    PlanKind,
    Pose,
    Result,
    RobotCapabilities,
    RobotState,
    SensorFrame,
    SceneObject,
    SceneRegion,
    SceneSnapshot,
    ToolDescriptor,
    ToolState,
)
from .providers import KinematicsProvider, MotionProvider, NavigationProvider
from .sensors import RobotStateSubscription, SensorSubscription

__all__ = [
    "BackendRequestError",
    "BackendUnavailable",
    "BaseTrajectoryPoint",
    "Command",
    "CommandState",
    "CONFIG_ENV",
    "ConfigurationError",
    "DependencyUnavailable",
    "EnvironmentCollisionObject",
    "FakeBackend",
    "FakeBackendSnapshot",
    "FakeGraspFixture",
    "EnvironmentCollisionSet",
    "Feedback",
    "GenerationMismatch",
    "GripperTarget",
    "JointLimit",
    "JointTrajectoryPoint",
    "KinematicsProvider",
    "MotionPlan",
    "MotionProvider",
    "NavigationProvider",
    "PlanKind",
    "PlanningError",
    "Pose",
    "Result",
    "RobotBackend",
    "RobotCapabilities",
    "RobotConfig",
    "RobotDeployment",
    "RobotSDKError",
    "RobotState",
    "RobotStateSubscription",
    "SensorFrame",
    "SceneObject",
    "SceneRegion",
    "SceneSnapshot",
    "SensorSubscription",
    "ToolDescriptor",
    "ToolState",
    "SharedFakeBackend",
    "SQLiteFakeStateStore",
    "load_robot_deployment",
]
