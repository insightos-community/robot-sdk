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

"""验证 SDK 消费的 v0.4 公共样例与 Runtime/Framework/Studio 一致。"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_robot_sdk_core.models import JointTrajectoryPoint, MotionPlan, PlanKind


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "v1"


def fixture(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def test_native_profile_and_bundle_use_actual_v1_identifiers() -> None:
    profile = fixture("runtime-profile-native")
    assert set(profile) == {
        "runtime_profile_id",
        "name",
        "engine",
        "loader",
        "api_version",
        "scene_kinds",
        "capabilities",
        "environment",
        "environment_ready",
        "available",
        "unavailable_reason",
    }
    assert profile["runtime_profile_id"] == "native-mujoco"
    assert profile["api_version"] == "v1"
    assert profile["scene_kinds"] == ["scene_document", "asset_scene"]
    assert profile["capabilities"]["robot_models"] == ["r1_pro_chassis"]
    assert profile["environment_ready"] is True
    assert profile["available"] is True

    bundle = fixture("runtime-bundle")
    start = fixture("scene-start-request")
    assert bundle["runtime_profile_id"] == profile["runtime_profile_id"]
    assert bundle["document"]["nodes"][0]["properties"]["model"] == "r1_pro_chassis"
    assert start["runtime_profile_id"] == profile["runtime_profile_id"]
    assert start["runtime_bundle_id"] == bundle["runtime_bundle_id"]


def test_joint_command_fixture_becomes_an_inspectable_motion_plan() -> None:
    command = fixture("robot-command-joint")
    assert set(command) == {
        "command_id",
        "scene_generation",
        "type",
        "timeout_seconds",
        "joint_trajectory",
    }
    trajectory = command["joint_trajectory"]
    points = [
        JointTrajectoryPoint(
            time_from_start_s=point["time_from_start_seconds"],
            positions=point["positions"],
            velocities=point.get("velocities", {}),
        )
        for point in trajectory["points"]
    ]
    plan = MotionPlan(
        plan_id="canonical-joint-plan",
        robot_id="r1-pro-001",
        generation=command["scene_generation"],
        kind=PlanKind(command["type"]),
        resources=trajectory["resources"],
        frame_id=trajectory["frame_id"],
        start=points[0].positions,
        goal=points[-1].positions,
        joint_trajectory=points,
        collision_checked=True,
        estimated_duration_s=points[-1].time_from_start_s,
        planner="canonical-fixture",
    )
    assert plan.kind is PlanKind.JOINT
    assert plan.goal == {"left_arm_joint1": 0.3}
    assert plan.estimated_duration_s == 1


def test_native_evaluator_fixture_remains_profile_evidence() -> None:
    evaluation = fixture("scene-evaluation-profile")
    assert evaluation["runtime_profile_id"] == "libero-robosuite-1.4"
    assert evaluation["scene_key"] == "libero_spatial:0"
    assert evaluation["generation"] == 1
    assert evaluation["metrics"]["suite"] == "libero_spatial"
