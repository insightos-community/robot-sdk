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

from pathlib import Path

import pytest

from semantic_robot_sdk_core import ConfigurationError, load_robot_deployment


def test_deployment_loads_only_from_semantic_robot_config(monkeypatch):
    monkeypatch.delenv("SEMANTIC_ROBOT_CONFIG", raising=False)
    with pytest.raises(ConfigurationError, match="SEMANTIC_ROBOT_CONFIG"):
        load_robot_deployment()


def test_fake_example_is_a_valid_single_robot_deployment(monkeypatch):
    path = Path("examples/robot-deployment.fake.yaml").resolve()
    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(path))
    deployment = load_robot_deployment()
    assert deployment.robot.id == "r1pro-fake-001"
    assert deployment.robot.backend == "fake"
    assert deployment.robot.sdk.providers.navigation == "local"
    assert deployment.ability_settings("navigation") == {}


def test_sdk_endpoint_environment_overrides_deployment_file(tmp_path, monkeypatch):
    source = Path("examples/robot-deployment.mujoco.yaml").read_text(encoding="utf-8")
    path = tmp_path / "robot-deployment.yaml"
    path.write_text(source, encoding="utf-8")
    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(path))
    monkeypatch.setenv("SEMANTIC_ROBOT_SDK_ENDPOINT", "http://127.0.0.1:18091")

    deployment = load_robot_deployment()

    assert deployment.robot.sdk.endpoint == "http://127.0.0.1:18091"
    assert deployment.robot.id == "sim-r1pro-001"
    assert deployment.robot.frames.base == "base_link"


def test_empty_sdk_endpoint_environment_keeps_deployment_value(monkeypatch):
    path = Path("examples/robot-deployment.mujoco.yaml").resolve()
    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(path))
    monkeypatch.setenv("SEMANTIC_ROBOT_SDK_ENDPOINT", "   ")

    deployment = load_robot_deployment()

    assert deployment.robot.sdk.endpoint == "http://127.0.0.1:8090"


def test_sdk_accepts_full_instance_deployment(monkeypatch, tmp_path):
    """部署层字段可以与 SDK 配置共用一份文件，SDK 仍严格校验自身字段。"""

    source = Path("examples/robot-deployment.fake.yaml").read_text(encoding="utf-8")
    source = source.replace(
        "    firmware_profile: fake-v2",
        "    backend_profile: fake-v1\n    firmware_profile: fake-v2",
    )
    source = source.replace("managed_by_pilot: true", "managed_by_instance: true")
    source = source.replace(
        "pilot:\n  robot_skill_directory:",
        "pilot:\n  id: pilot-r1pro-fake-001\n  robot_skill_directory:",
    )
    source = source.replace("worker_timeout: 30", "worker_timeout_seconds: 30")
    source += """
robot_skills:
  - name: grasp-object
    version: 0.1.0
    enabled: true
"""
    path = tmp_path / "robot-deployment.yaml"
    path.write_text(source, encoding="utf-8")
    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(path))

    deployment = load_robot_deployment()

    assert deployment.robot.id == "r1pro-fake-001"
    assert deployment.robot.sdk.backend_profile == "fake-v1"


def test_lazy_provider_initializes_once_on_first_real_use():
    from semantic_robot_sdk_r1pro.sdk import _LazyProvider

    calls: list[int] = []

    class Provider:
        value = "ready"

    provider = _LazyProvider("local", lambda: (calls.append(1), Provider())[1])

    assert provider.name == "local"
    assert calls == []
    assert provider.value == "ready"
    assert provider.value == "ready"
    assert calls == [1]
