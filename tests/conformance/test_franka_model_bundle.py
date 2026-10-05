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

import pytest

from semantic_robot_sdk_core.errors import RobotModelError
from semantic_robot_sdk_franka import load_franka_model_bundle


def _write_bundle(root: Path, **manifest_updates: object) -> Path:
    package = root / "franka_description"
    robots = package / "robots"
    (package / "meshes").mkdir(parents=True)
    robots.mkdir(parents=True)
    (robots / "panda_arm_hand.urdf").write_text(
        '<robot name="panda"><link name="panda_link0"/><link name="panda_hand"/></robot>',
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("Apache License, Version 2.0", encoding="utf-8")
    (root / "NOTICE").write_text("Franka model attribution", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_id": "franka_panda",
        "urdf": "franka_description/robots/panda_arm_hand.urdf",
        "package_roots": ["."],
        "kinematic_root_frame": "panda_link0",
        "end_effectors": {"hand": "panda_hand"},
        "disabled_collision_pairs": [["panda_leftfinger", "panda_rightfinger"]],
        "license": {"spdx": "Apache-2.0", "file": "LICENSE", "notice": "NOTICE"},
        "source": {
            "url": "https://github.com/frankarobotics/franka_ros",
            "revision": "0123456789abcdef0123456789abcdef01234567",
        },
    }
    manifest.update(manifest_updates)
    (root / "robot-model.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return root


def test_franka_bundle_fixes_model_frames_mesh_root_and_provenance(tmp_path: Path) -> None:
    bundle = load_franka_model_bundle(_write_bundle(tmp_path / "model"))

    assert bundle.model_id == "franka_panda"
    assert bundle.urdf_path.name == "panda_arm_hand.urdf"
    assert bundle.package_directories == (bundle.root,)
    assert bundle.kinematic_root_frame == "panda_link0"
    assert dict(bundle.end_effector_frames) == {"hand": "panda_hand"}
    assert bundle.disabled_collision_pairs == (("panda_leftfinger", "panda_rightfinger"),)
    assert bundle.source.license_spdx == "Apache-2.0"
    assert bundle.source.revision == "0123456789abcdef0123456789abcdef01234567"
    assert bundle.notice_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"kinematic_root_frame": "base_link"}, "panda_link0"),
        ({"end_effectors": {"hand": "panda_link8"}}, "panda_hand"),
        (
            {"license": {"spdx": "unknown", "file": "LICENSE", "notice": "NOTICE"}},
            "Apache-2.0",
        ),
        (
            {"source": {"url": "https://example.invalid/model", "revision": "01234567"}},
            "官方仓库",
        ),
        (
            {
                "source": {
                    "url": "https://github.com/frankarobotics/franka_ros",
                    "revision": "main",
                }
            },
            "固定 Tag",
        ),
    ],
)
def test_franka_bundle_rejects_unreviewed_metadata(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    root = _write_bundle(tmp_path / "model", **updates)
    with pytest.raises(RobotModelError, match=message):
        load_franka_model_bundle(root)


def test_franka_bundle_missing_asset_fails_instead_of_searching_environment(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "model")
    (root / "franka_description" / "robots" / "panda_arm_hand.urdf").unlink()

    with pytest.raises(RobotModelError, match="URDF 不存在"):
        load_franka_model_bundle(root)


def test_franka_bundle_rejects_venv_and_path_escape(tmp_path: Path) -> None:
    venv_root = tmp_path / ".venv" / "franka"
    _write_bundle(venv_root)
    with pytest.raises(RobotModelError, match="不能来自 .venv"):
        load_franka_model_bundle(venv_root)

    root = _write_bundle(tmp_path / "model", urdf="../panda_arm_hand.urdf")
    with pytest.raises(RobotModelError, match="不能离开模型根目录"):
        load_franka_model_bundle(root)
