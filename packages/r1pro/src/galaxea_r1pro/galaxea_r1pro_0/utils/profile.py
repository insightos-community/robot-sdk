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

"""Robot profile 加载与校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ProfileError
from .types import BackendKind


@dataclass(frozen=True)
class RobotProfile:
    robot_id: str
    backend: BackendKind
    firmware_profile: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RobotProfile":

        robot_id = str(value.get("robot_id", "")).strip()
        firmware_profile = str(value.get("firmware_profile", "")).strip()

        if not robot_id:
            raise ProfileError(
                field="robot_id", reason=ProfileError.INVALID_ROBOT_ID, hint="robot_id 不能为空"
            )

        if not firmware_profile:
            raise ProfileError(
                field="firmware_profile",
                reason=ProfileError.INVALID_FIRMWARE,
                hint="firmware_profile 不能为空",
            )

        try:
            backend = BackendKind(str(value.get("backend", "")))

        except ValueError as exc:
            raise ProfileError(
                field="backend",
                value=value.get("backend"),
                reason=ProfileError.INVALID_BACKEND,
                hint="backend 必须是 isaac 或 mujoco",
            ) from exc

        options = value.get("options") or {}

        if not isinstance(options, Mapping):
            raise ProfileError(
                field="options", reason=ProfileError.INVALID_OPTIONS, hint="options 必须是对象"
            )

        return cls(
            robot_id=robot_id,
            backend=backend,
            firmware_profile=firmware_profile,
            options=dict(options),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RobotProfile":
        profile_path = Path(path)
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(
                field="path",
                value=str(profile_path),
                reason=ProfileError.LOAD_FAILED,
                hint=f"无法读取 Robot profile: {profile_path}",
            ) from exc

        if not isinstance(data, Mapping):
            raise ProfileError(
                reason=ProfileError.INVALID_STRUCTURE, hint="Robot profile 顶层必须是对象"
            )

        return cls.from_mapping(data)
