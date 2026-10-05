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

"""一台 Robot 一份部署配置的公共加载入口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field

from .errors import ConfigurationError
from .models import StrictModel, ToolDescriptor

CONFIG_ENV = "SEMANTIC_ROBOT_CONFIG"
SDK_ENDPOINT_ENV = "SEMANTIC_ROBOT_SDK_ENDPOINT"


class ProviderSelection(StrictModel):
    kinematics: str
    motion: str
    navigation: str | None = None


class SDKConfig(StrictModel):
    package: str
    endpoint: str | None = None
    backend_profile: str | None = None
    firmware_profile: str
    providers: ProviderSelection
    scene_instance_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class KinematicsConfig(StrictModel):
    """由型号包模板写入的本地运动学配置，禁止求解器猜测夹具结构。"""

    urdf_path: str | None = None
    package_directories: list[str] = Field(default_factory=list)
    controlled_joint_groups: dict[str, list[str]] = Field(default_factory=dict)
    disabled_collision_pairs: list[tuple[str, str]] = Field(default_factory=list)
    named_postures: dict[str, dict[str, float]] = Field(default_factory=dict)


class FrameConfig(StrictModel):
    world: str = "world"
    base: str = "base_link"
    end_effectors: dict[str, str]


class SafetyConfig(StrictModel):
    maximum_base_speed: float = Field(default=0.4, gt=0)
    maximum_joint_speed: float = Field(default=0.5, gt=0)
    carrying_motion_scale: float = Field(default=1.0, gt=0, le=1.0)


class RobotConfig(StrictModel):
    id: str = Field(min_length=1)
    display_name: str | None = None
    model: str = Field(min_length=1)
    backend: Literal["fake", "mujoco", "real", "isaac"]
    sdk: SDKConfig
    frames: FrameConfig
    tools: list[ToolDescriptor] = Field(default_factory=list)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class RobotDeployment(StrictModel):
    # RobotDeployment 是启动器、Pilot、Ability 和 SDK 共用的部署文件。SDK 只负责
    # 严格校验自己消费的 robot/abilities 区域；Pilot、Skill 期望状态和进程管理字段
    # 属于部署层，允许它们保留在同一文件中，避免为同一台 Robot 维护多份易漂移配置。
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    api_version: Literal[1]
    robot: RobotConfig
    abilities: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def ability_settings(self, ability_type: str) -> dict[str, Any]:
        """只返回调用方自己的 Ability 配置，避免 Ability 扫描 CR 或彼此配置。"""

        value = self.abilities.get(ability_type, {})
        return dict(value)


def load_robot_deployment(path: str | Path | None = None) -> RobotDeployment:
    """读取统一部署配置，并应用当前进程的 SDK Endpoint 覆盖。

    同一主机可以启动多组 Pilot、AbilityFramework 和 Ability。每组进程读取相同
    配置模板时，可用 ``SEMANTIC_ROBOT_SDK_ENDPOINT`` 指向自己的仿真 Runtime；
    环境变量只覆盖连接地址，不改变 Robot ID、坐标系或安全限制。
    """

    configured = str(path or os.environ.get(CONFIG_ENV, "")).strip()
    if not configured:
        raise ConfigurationError(f"未设置 {CONFIG_ENV}")
    config_path = Path(configured).expanduser()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigurationError(f"无法读取 Robot 部署配置 {config_path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError("Robot 部署配置顶层必须是对象")
    try:
        deployment = RobotDeployment.model_validate(value)
    except Exception as error:
        raise ConfigurationError(f"Robot 部署配置不合法: {error}") from error

    endpoint = os.environ.get(SDK_ENDPOINT_ENV, "").strip()
    if not endpoint:
        return deployment
    sdk = deployment.robot.sdk.model_copy(update={"endpoint": endpoint})
    robot = deployment.robot.model_copy(update={"sdk": sdk})
    return deployment.model_copy(update={"robot": robot})
