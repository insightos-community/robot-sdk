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

"""Robot 运动学模型包的公共描述。

SDK 不拥有大型 URDF 与 Mesh，也不能从 Python 环境中猜测模型位置。型号包先把
明确的资产根目录校验成这个不可变对象，随后运动学工厂只消费已经校验的路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RobotModelSource:
    """记录模型来自哪个固定上游版本，供资产发布和许可证审查使用。"""

    url: str
    revision: str
    license_spdx: str


@dataclass(frozen=True, slots=True)
class RobotModelBundle:
    """一次运动学求解所需的完整、已校验模型输入。

    `package_directories` 是 Pinocchio 解析 ``package://`` Mesh URI 使用的根目录，
    不是 SDK 自动搜索路径。坐标系名称来自型号包，调用方不能在创建 SDK 时覆盖。
    """

    model_id: str
    root: Path
    urdf_path: Path
    package_directories: tuple[Path, ...]
    kinematic_root_frame: str
    end_effector_frames: Mapping[str, str]
    disabled_collision_pairs: tuple[tuple[str, str], ...]
    source: RobotModelSource
    manifest_path: Path
    license_path: Path
    notice_path: Path
