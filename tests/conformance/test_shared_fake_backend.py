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

"""共享 Fake Robot 的跨进程行为测试。"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import yaml

from semantic_robot_sdk_core import (
    BaseTrajectoryPoint,
    CommandState,
    MotionPlan,
    PlanKind,
    load_robot_deployment,
)
from semantic_robot_sdk_r1pro import R1ProSDK


def _run_in_child(config_path: str, operation: str, result_queue) -> None:
    """在全新的 Python 进程中执行一次 Robot SDK 操作。"""

    try:
        os.environ["SEMANTIC_ROBOT_CONFIG"] = config_path
        with R1ProSDK.from_environment() as robot:
            if operation == "grasp":
                robot.backend.configure_grasp_fixture("left", "box-17")
                robot.backend.configure_grasp_fixture("right", "box-17")
                robot.end_effector.close_until_contact(
                    "left", command_id="shared-grasp-left", force_limit_n=5
                )
                command = robot.end_effector.close_until_contact(
                    "right", command_id="shared-grasp-right", force_limit_n=5
                )
            elif operation == "navigate":
                # 去重验证必须提交完全相同的计划；若第二次按 Robot 新位置重新规划，
                # 得到的是另一份计划，理应被 SDK 拒绝复用同一个 command_id。
                plan = MotionPlan(
                    plan_id="shared-route-plan",
                    robot_id=robot.robot_id,
                    generation=1,
                    kind=PlanKind.BASE,
                    resources=["base"],
                    frame_id="world",
                    start={"x": 0.0, "y": 0.0, "yaw": 0.0},
                    goal={"x": 1.0, "y": 0.0, "yaw": 0.0},
                    base_trajectory=[
                        BaseTrajectoryPoint(time_from_start_s=0, x=0, y=0, yaw=0),
                        BaseTrajectoryPoint(time_from_start_s=1, x=1, y=0, yaw=0),
                    ],
                    collision_checked=True,
                    estimated_duration_s=1,
                    planner="shared-fake-test",
                )
                command = robot.base.follow_route(plan, command_id="shared-route")
            elif operation == "stop":
                command = robot.safety.stop_and_hold("shared-route")
            else:
                raise ValueError(operation)
            result_queue.put({"status": command.status.value})
    except Exception as error:  # pragma: no cover - 错误正文通过父进程断言展示
        result_queue.put({"error": f"{type(error).__name__}: {error}"})


def _shared_config(tmp_path: Path, *, auto_complete: bool) -> Path:
    source = Path("examples/robot-deployment.fake.yaml")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    options = value["robot"]["sdk"].setdefault("options", {})
    options.update(
        {
            "state_path": str(tmp_path / "fake-robot.sqlite"),
            "auto_complete": auto_complete,
            "stop_confirmed": True,
        }
    )
    target = tmp_path / "robot-deployment.yaml"
    target.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    return target


def _child(config: Path, operation: str) -> dict:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_run_in_child, args=(str(config), operation, queue))
    process.start()
    process.join(timeout=15)
    assert process.exitcode == 0
    result = queue.get(timeout=2)
    assert "error" not in result, result.get("error")
    return result


def test_deployment_initializes_grasp_target_only_for_new_shared_state(tmp_path, monkeypatch):
    config = _shared_config(tmp_path, auto_complete=True)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["robot"]["sdk"]["options"]["initial_grasp_targets"] = {
        side: {
            "object_id": "object://pallet-a/box-17",
            "contact_opening_m": 0.04,
            "contact_force_n": 18.0,
            "minimum_holding_force_n": 2.0,
        }
        for side in ("left", "right")
    }
    config.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")

    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(config))
    with R1ProSDK.from_environment() as robot:
        robot.end_effector.close_until_contact(
            "left", command_id="configured-grasp-left", force_limit_n=5
        )
        robot.end_effector.close_until_contact(
            "right", command_id="configured-grasp-right", force_limit_n=5
        )
        assert all(
            item.hook_contact and item.clamp_contact
            for item in robot.state.snapshot().tool_states.values()
        )
        robot.end_effector.release("left", command_id="configured-release-left")
        robot.end_effector.release("right", command_id="configured-release-right")

    # 模拟另一个 Ability 进程使用修改后的配置重新装配 SDK。已有共享状态不能被
    # 新配置覆盖，否则进程重启会让已经放置的物体重新出现在夹爪接触区。
    value["robot"]["sdk"]["options"]["initial_grasp_targets"]["left"]["object_id"] = (
        "object://unexpected/reset"
    )
    config.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    with R1ProSDK.from_environment() as robot:
        assert all(
            not item.hook_contact and not item.clamp_contact
            for item in robot.state.snapshot().tool_states.values()
        )
        snapshot = robot.backend._store.read(robot.backend._new_backend)
        assert snapshot.grasp_fixtures["left"].object_id == "object://pallet-a/box-17"


def test_independent_processes_share_grasp_and_robot_state(tmp_path, monkeypatch):
    config = _shared_config(tmp_path, auto_complete=True)
    assert _child(config, "grasp") == {"status": CommandState.SUCCEEDED.value}

    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(config))
    with R1ProSDK.from_deployment(load_robot_deployment()) as robot:
        assert all(item.hook_contact for item in robot.state.snapshot().tool_states.values())
        assert robot.sensors.latest("contact").payload == {
            "gripper": "right",
            "contact": True,
            "force_n": 5.0,
            "object_id": "box-17",
        }


def test_command_started_in_one_process_can_be_stopped_from_another(tmp_path, monkeypatch):
    config = _shared_config(tmp_path, auto_complete=False)
    assert _child(config, "navigate") == {"status": CommandState.RUNNING.value}
    assert _child(config, "stop") == {"status": CommandState.STOPPED.value}

    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(config))
    with R1ProSDK.from_deployment(load_robot_deployment()) as robot:
        assert robot.commands.get("shared-route").status is CommandState.STOPPED
        assert robot.state.snapshot().in_hold is True


def test_shared_backend_keeps_command_id_idempotent(tmp_path, monkeypatch):
    config = _shared_config(tmp_path, auto_complete=True)
    assert _child(config, "navigate") == {"status": CommandState.SUCCEEDED.value}
    assert _child(config, "navigate") == {"status": CommandState.SUCCEEDED.value}

    monkeypatch.setenv("SEMANTIC_ROBOT_CONFIG", str(config))
    with R1ProSDK.from_deployment(load_robot_deployment()) as robot:
        assert robot.backend.executions == ["shared-route"]
