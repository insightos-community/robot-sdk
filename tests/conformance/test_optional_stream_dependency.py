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

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_fake_sdk_import_does_not_require_websockets() -> None:
    source_paths = [
        str(ROOT / "packages" / "core" / "src"),
        str(ROOT / "packages" / "r1pro" / "src"),
    ]
    script = f"""
import importlib.abc
import sys

class BlockWebsockets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "websockets" or fullname.startswith("websockets."):
            raise ModuleNotFoundError("blocked for Fake SDK test", name=fullname)
        return None

sys.meta_path.insert(0, BlockWebsockets())
sys.path[:0] = {json.dumps(source_paths)}

import semantic_robot_sdk_core
from semantic_robot_sdk_r1pro import FakeBackend, R1ProSDK

assert FakeBackend is not None
assert R1ProSDK is not None
assert not any(name == "websockets" or name.startswith("websockets.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_stream_transport_reports_missing_optional_dependency() -> None:
    source_paths = [str(ROOT / "packages" / "core" / "src")]
    script = f"""
import importlib.abc
import sys

class BlockWebsockets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "websockets" or fullname.startswith("websockets."):
            raise ModuleNotFoundError("blocked for stream test", name=fullname)
        return None

sys.meta_path.insert(0, BlockWebsockets())
sys.path[:0] = {json.dumps(source_paths)}

from semantic_robot_sdk_core import BackendUnavailable
from semantic_robot_sdk_core.stream_transport import (
    BinarySensorFrameParser,
    BinaryWebSocketFrameTransport,
    SensorStreamDescriptor,
)

parser = BinarySensorFrameParser(
    SensorStreamDescriptor(sensor_id="contact", encoding="json"),
    expected_generation=1,
)
try:
    BinaryWebSocketFrameTransport("ws://runtime/stream", parser)
except BackendUnavailable as error:
    assert "websockets" in str(error)
else:
    raise AssertionError("missing websockets must fail when WebSocket transport is constructed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_websockets_is_declared_only_as_stream_extra() -> None:
    with (ROOT / "packages" / "core" / "pyproject.toml").open("rb") as file:
        core = tomllib.load(file)["project"]
    with (ROOT / "packages" / "r1pro" / "pyproject.toml").open("rb") as file:
        r1pro = tomllib.load(file)["project"]

    assert all(not requirement.startswith("websockets") for requirement in core["dependencies"])
    assert core["optional-dependencies"]["stream"] == ["websockets==17.0.1"]
    assert r1pro["optional-dependencies"]["mujoco"] == ["websockets==17.0.1"]
