"""Franka SDK 与真实 robosuite/LIBERO Profile Runtime 的组合测试。"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from uuid import uuid4

import pytest

from semantic_robot_sdk_core.errors import GenerationMismatch
from semantic_robot_sdk_core.models import (
    CommandState,
    JointTrajectoryPoint,
    MotionPlan,
    PlanKind,
)
from semantic_robot_sdk_franka import MujocoBackend, capabilities


def _request_json(base_url: str, method: str, path: str, body=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _wait_scene(base_url: str, instance_id: str):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        scene = _request_json(base_url, "GET", f"/api/v1/scene-instances/{instance_id}")
        if scene["state"] in {"running", "failed", "stopped"}:
            return scene
        time.sleep(0.05)
    raise TimeoutError("等待真实 Profile 场景启动超时")


def _wait_command(backend: MujocoBackend, command_id: str):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        command = backend.command(backend.robot_id, command_id)
        if command.status in {
            CommandState.SUCCEEDED,
            CommandState.FAILED,
            CommandState.STOPPED,
            CommandState.INTERRUPTED,
        }:
            return command
        time.sleep(0.05)
    raise TimeoutError(f"等待 Franka 命令超时: {command_id}")


def _joint_plan(backend: MujocoBackend, *, command_suffix: str, duration_s: float) -> MotionPlan:
    state = backend.state(backend.robot_id)
    joints = dict(state.joint_positions)
    goal = dict(joints)
    first_joint = backend.capabilities().joint_names[0]
    goal[first_joint] += 0.002
    return MotionPlan(
        plan_id=f"profile-plan-{command_suffix}",
        robot_id=backend.robot_id,
        generation=state.generation,
        kind=PlanKind.JOINT,
        resources=["joints:arm"],
        frame_id=backend.capabilities().kinematic_root_frame,
        start=joints,
        goal=goal,
        joint_trajectory=[
            JointTrajectoryPoint(time_from_start_s=0, positions=joints),
            JointTrajectoryPoint(time_from_start_s=duration_s, positions=goal),
        ],
        collision_checked=True,
        estimated_duration_s=duration_s,
        planner="profile-conformance",
    )


def test_real_franka_profile_command_sensor_stop_and_reset() -> None:
    base_url = os.environ["PLUGIN_MUJOCO_PROFILE_URL"]
    scene_key = os.environ.get("PLUGIN_MUJOCO_PROFILE_SCENE", "Lift")
    started = _request_json(
        base_url,
        "POST",
        f"/api/v1/scenes/{scene_key}/instances",
        {
            "request_id": f"sdk-profile-{uuid4()}",
            "runtime_profile_id": os.environ.get(
                "PLUGIN_MUJOCO_PROFILE_ID",
                "robosuite-1.5",
            ),
            "layout": os.environ.get("PLUGIN_MUJOCO_PROFILE_LAYOUT", "default"),
            "seed": 7,
            "headless": True,
            "render_backend": "egl",
        },
    )
    instance_id = started["instance_id"]
    try:
        scene = _wait_scene(base_url, instance_id)
        assert scene["state"] == "running", scene.get("failure_reason")
        robots = _request_json(
            base_url,
            "GET",
            f"/api/v1/scene-instances/{instance_id}/robots",
        )
        assert len(robots) == 1
        robot_id = robots[0]["robot_id"]
        backend = MujocoBackend(base_url, robot_id, capabilities())

        state = backend.state(robot_id)
        assert state.base_pose is not None
        assert state.base_pose.frame_id == "world"
        # Lift/Stack 与 LIBERO 都按 table/bins offset 放置 Panda；若 Runtime
        # 又退回硬编码 world 原点，这个真实 Profile 门控会直接失败。
        assert abs(state.base_pose.position[0]) > 0.1
        assert sum(value * value for value in state.base_pose.quaternion_xyzw) == pytest.approx(1.0)
        assert set(state.joint_positions) == set(backend.capabilities().joint_names)

        plan = _joint_plan(backend, command_suffix="complete", duration_s=0.2)
        first = backend.execute_plan(plan, "profile-complete")
        duplicate = backend.execute_plan(plan, "profile-complete")
        assert duplicate.command_id == first.command_id
        completed = _wait_command(backend, first.command_id)
        assert completed.status is CommandState.SUCCEEDED, completed.reason
        assert completed.progress == pytest.approx(1)

        moving_plan = _joint_plan(backend, command_suffix="stop", duration_s=2)
        moving = backend.execute_plan(moving_plan, "profile-stop")
        stopped = backend.stop(robot_id, moving.command_id)
        assert stopped.status is CommandState.STOPPED
        backend.hold(robot_id)
        assert backend.state(robot_id).in_hold is True

        sensor_ids = backend.sensors(robot_id)
        rgb_id = next(sensor for sensor in sensor_ids if "rgb" in sensor)
        depth_id = next(sensor for sensor in sensor_ids if "depth" in sensor)
        contact_id = next(sensor for sensor in sensor_ids if "contact" in sensor)
        with (
            backend.subscribe_sensor(robot_id, rgb_id) as rgb_subscription,
            backend.subscribe_sensor(robot_id, depth_id) as depth_subscription,
            backend.subscribe_sensor(robot_id, contact_id) as contact_subscription,
            backend.subscribe_state(robot_id, poll_interval_seconds=0.01) as state_subscription,
        ):
            rgb = next(rgb_subscription)
            assert rgb.encoding == "jpeg" and rgb.payload
            assert rgb.generation == scene["generation"]
            depth = next(depth_subscription)
            assert depth.encoding == "float32-le"
            assert isinstance(depth.payload, bytes)
            assert len(depth.payload) == (depth.width or 0) * (depth.height or 0) * 4
            contact = next(contact_subscription)
            assert contact.encoding == "json" and isinstance(contact.payload, dict)
            assert contact.width is None and contact.height is None
            assert next(state_subscription).generation == scene["generation"]

            reset = _request_json(
                base_url,
                "POST",
                f"/api/v1/scene-instances/{instance_id}/reset",
            )
            assert reset["generation"] == scene["generation"] + 1
            for subscription in (rgb_subscription, depth_subscription, contact_subscription):
                try:
                    next(subscription)
                except GenerationMismatch:
                    continue
                # Profile Runtime 同样允许清掉一帧 reset 前的单槽缓存，之后必须
                # 以 generation 错误终止，不能把 reset 后画面接入旧订阅。
                with pytest.raises(GenerationMismatch):
                    next(subscription)
            # Robot State 与传感器使用同一 latest-frame 单槽语义：reset 返回时最多
            # 清掉一帧旧 generation 缓存，下一次读取必须以 generation 错误终止。
            try:
                next(state_subscription)
            except GenerationMismatch:
                pass
            else:
                with pytest.raises(GenerationMismatch):
                    next(state_subscription)
            with pytest.raises(GenerationMismatch):
                backend.execute_plan(plan, "profile-stale")
    finally:
        _request_json(
            base_url,
            "POST",
            f"/api/v1/scene-instances/{instance_id}/stop",
        )
