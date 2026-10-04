"""Franka Panda 正式运动学模型包加载器。

生产代码只接受调用方明确传入的资产根目录。这里不会读取 ``FRANKA_MODEL_ROOT``，
更不会遍历 ``site-packages`` 或 ``.venv``；环境变量只属于集成测试入口。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from semantic_robot_sdk_core.errors import RobotModelError
from semantic_robot_sdk_core.model_bundle import RobotModelBundle, RobotModelSource


MANIFEST_NAME = "robot-model.json"
MODEL_ID = "franka_panda"
URDF_NAME = "panda_arm_hand.urdf"
KINEMATIC_ROOT_FRAME = "panda_link0"
END_EFFECTOR_FRAMES = {"hand": "panda_hand"}
LICENSE_SPDX = "Apache-2.0"

# Franka Panda 模型必须来自官方仓库的固定 revision。franka_ros 保留传统 Panda
# 描述，franka_description 是后续官方模型仓；资产 MR 需要记录实际使用的一个。
OFFICIAL_SOURCE_URLS = frozenset(
    {
        "https://github.com/frankarobotics/franka_ros",
        "https://github.com/frankarobotics/franka_description",
    }
)
_MOVING_REVISIONS = frozenset(
    {"main", "master", "develop", "latest", "head", "pinned-tag-or-commit", "replace-me"}
)


def load_franka_model_bundle(model_root: str | Path) -> RobotModelBundle:
    """校验一个可发布的 Franka 模型包并返回不可变路径集合。

    模型根目录必须包含 manifest、Apache-2.0 LICENSE、上游 NOTICE、生成后的
    ``panda_arm_hand.urdf`` 和它引用的 Mesh package root。任何缺失都直接失败，
    不能在无碰撞几何或错误坐标系下继续生成运动计划。
    """

    root = Path(model_root).expanduser().resolve()
    if ".venv" in root.parts:
        raise RobotModelError("Franka 正式模型不能来自 .venv；请使用经审查的资产包")
    if not root.is_dir():
        raise RobotModelError(f"Franka 模型根目录不存在：{root}")

    manifest_path = root / MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    _expect_equal(manifest, "schema_version", 1)
    _expect_equal(manifest, "model_id", MODEL_ID)
    _expect_equal(manifest, "kinematic_root_frame", KINEMATIC_ROOT_FRAME)

    end_effectors = _string_mapping(manifest.get("end_effectors"), "end_effectors")
    if end_effectors != END_EFFECTOR_FRAMES:
        raise RobotModelError(
            "Franka manifest 必须声明 hand -> panda_hand，不能由调用方覆盖末端坐标系"
        )

    urdf_path = _bundle_path(root, _string(manifest, "urdf"), "urdf")
    if urdf_path.name != URDF_NAME:
        raise RobotModelError(f"Franka URDF 必须是 {URDF_NAME}：{urdf_path}")
    if not urdf_path.is_file():
        raise RobotModelError(f"Franka URDF 不存在：{urdf_path}")

    package_values = manifest.get("package_roots")
    if not isinstance(package_values, list) or not package_values:
        raise RobotModelError("Franka manifest 的 package_roots 必须是非空字符串数组")
    package_directories = tuple(
        _bundle_path(root, _plain_string(value, "package_roots"), "package_roots")
        for value in package_values
    )
    missing_packages = [path for path in package_directories if not path.is_dir()]
    if missing_packages:
        raise RobotModelError(f"Franka Mesh package root 不存在：{missing_packages}")

    license_data = manifest.get("license")
    if not isinstance(license_data, dict):
        raise RobotModelError("Franka manifest 缺少 license 对象")
    if license_data.get("spdx") != LICENSE_SPDX:
        raise RobotModelError("Franka 模型许可证必须明确标记为 Apache-2.0")
    license_path = _bundle_path(
        root, _plain_string(license_data.get("file"), "license.file"), "license.file"
    )
    notice_path = _bundle_path(
        root, _plain_string(license_data.get("notice"), "license.notice"), "license.notice"
    )
    _require_nonempty_file(license_path, "Apache-2.0 LICENSE")
    _require_nonempty_file(notice_path, "上游 NOTICE")

    source_data = manifest.get("source")
    if not isinstance(source_data, dict):
        raise RobotModelError("Franka manifest 缺少 source 对象")
    source_url = _plain_string(source_data.get("url"), "source.url").rstrip("/")
    if source_url not in OFFICIAL_SOURCE_URLS:
        raise RobotModelError(f"Franka 模型来源不是已确认的官方仓库：{source_url}")
    revision = _plain_string(source_data.get("revision"), "source.revision")
    if revision.lower() in _MOVING_REVISIONS:
        raise RobotModelError("Franka source.revision 必须是固定 Tag 或完整 Commit 标识")

    disabled_pairs = _collision_pairs(manifest.get("disabled_collision_pairs", []))
    return RobotModelBundle(
        model_id=MODEL_ID,
        root=root,
        urdf_path=urdf_path,
        package_directories=package_directories,
        kinematic_root_frame=KINEMATIC_ROOT_FRAME,
        end_effector_frames=MappingProxyType(dict(END_EFFECTOR_FRAMES)),
        disabled_collision_pairs=disabled_pairs,
        source=RobotModelSource(
            url=source_url,
            revision=revision,
            license_spdx=LICENSE_SPDX,
        ),
        manifest_path=manifest_path,
        license_path=license_path,
        notice_path=notice_path,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RobotModelError(f"Franka 模型缺少 {MANIFEST_NAME}：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobotModelError(f"Franka manifest 无法读取：{path}: {error}") from error
    if not isinstance(value, dict):
        raise RobotModelError("Franka manifest 顶层必须是 JSON 对象")
    return value


def _expect_equal(manifest: dict[str, Any], field: str, expected: object) -> None:
    if manifest.get(field) != expected:
        raise RobotModelError(f"Franka manifest 的 {field} 必须是 {expected!r}")


def _string(manifest: dict[str, Any], field: str) -> str:
    return _plain_string(manifest.get(field), field)


def _plain_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobotModelError(f"Franka manifest 的 {field} 必须是非空字符串")
    return value.strip()


def _string_mapping(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise RobotModelError(f"Franka manifest 的 {field} 必须是非空对象")
    return {
        _plain_string(key, f"{field}.key"): _plain_string(item, f"{field}.{key}")
        for key, item in value.items()
    }


def _bundle_path(root: Path, relative: str, field: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise RobotModelError(f"Franka manifest 的 {field} 不能使用绝对路径")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise RobotModelError(f"Franka manifest 的 {field} 不能离开模型根目录")
    return resolved


def _require_nonempty_file(path: Path, label: str) -> None:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise RobotModelError(f"Franka 模型缺少非空的 {label}：{path}")
    except OSError as error:
        raise RobotModelError(f"无法检查 Franka {label}：{path}: {error}") from error


def _collision_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise RobotModelError("disabled_collision_pairs 必须是数组")
    result: list[tuple[str, str]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise RobotModelError(f"disabled_collision_pairs[{index}] 必须包含两个 link 名")
        first = _plain_string(pair[0], f"disabled_collision_pairs[{index}][0]")
        second = _plain_string(pair[1], f"disabled_collision_pairs[{index}][1]")
        if first == second:
            raise RobotModelError("禁用碰撞对不能引用同一个 link")
        result.append((first, second))
    if len({frozenset(pair) for pair in result}) != len(result):
        raise RobotModelError("disabled_collision_pairs 不能重复")
    return tuple(result)
