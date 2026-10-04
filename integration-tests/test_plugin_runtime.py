"""semantic-robot-sdk 与真实 plugin-mujoco 的组合测试。"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest

from semantic_robot_sdk_core.errors import GenerationMismatch
from semantic_robot_sdk_core.models import CommandState, Pose
from semantic_robot_sdk_r1pro import create_mujoco_sdk


def request_json(base_url: str, method: str, path: str, body=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def wait_scene(base_url: str, instance_id: str):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        scene = request_json(base_url, "GET", f"/api/v1/scene-instances/{instance_id}")
        if scene["state"] in {"running", "failed", "stopped"}:
            return scene
        time.sleep(0.05)
    raise TimeoutError("等待真实 MuJoCo 场景启动超时")


def wait_command(sdk, command_id: str):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        command = sdk.commands.get(command_id)
        if command.status in {
            CommandState.SUCCEEDED,
            CommandState.FAILED,
            CommandState.STOPPED,
            CommandState.INTERRUPTED,
        }:
            return command
        time.sleep(0.05)
    raise TimeoutError(f"等待 Robot 命令超时: {command_id}")


def test_real_plugin_ik_trajectory_stop_sensor_and_reset() -> None:
    base_url = os.environ["PLUGIN_MUJOCO_URL"]
    asset_root = Path(os.environ["R1PRO_ASSET_ROOT"]).resolve()
    started = request_json(
        base_url,
        "POST",
        "/api/v1/scenes/palletizing_depalletizing_tote_v1/instances",
        {
            "request_id": f"sdk-e2e-{uuid4()}",
            "runtime_profile_id": "native-mujoco",
            "layout": "layout_smoke",
            "seed": 7,
            "headless": True,
            "render_backend": "egl",
        },
    )
    instance_id = started["instance_id"]
    try:
        scene = wait_scene(base_url, instance_id)
        assert scene["state"] == "running", scene.get("failure_reason")
        robots = request_json(base_url, "GET", f"/api/v1/scene-instances/{instance_id}/robots")
        assert robots
        robot_id = robots[0]["robot_id"]
        sdk = create_mujoco_sdk(
            endpoint=base_url,
            robot_id=robot_id,
            urdf_path=(
                asset_root / "robot" / "r1_pro_tote_gripper" / "meshes" / "r1_pro_tote_gripper.urdf"
            ),
            scene_instance_id=instance_id,
            package_directories=[str(asset_root / "robot")],
        )

        before = sdk.state.snapshot()
        current_eef = before.end_effectors["left"]
        eef_target = Pose(
            position=(
                current_eef.position[0] + 0.005,
                current_eef.position[1],
                current_eef.position[2],
            ),
            quaternion_xyzw=current_eef.quaternion_xyzw,
            frame_id=current_eef.frame_id,
        )
        eef_plan = sdk.upper_body.plan_end_effector("left", eef_target)
        # 对外记录稳定的 Provider 名称，而不是把内部算法类名写进计划。
        # 当前 local Provider 的实现由 Pinocchio IK 与 Ruckig 组成；以后替换
        # 等价实现时，Ability 和运行证据不需要跟着修改字符串。
        assert eef_plan.planner == "local+ruckig"
        completed = wait_command(sdk, sdk.backend.execute_plan(eef_plan, "sdk-eef").command_id)
        assert completed.status is CommandState.SUCCEEDED, completed.reason

        # 命令进入 succeeded 只能证明关节误差进入容差；还要证明末端确实向
        # 非零笛卡尔目标移动，防止 FK/坐标系或关节映射错误被状态机掩盖。
        after_eef = sdk.state.snapshot().end_effectors["left"]
        requested_distance = math.dist(current_eef.position, eef_target.position)
        actual_distance = math.dist(current_eef.position, after_eef.position)
        remaining_distance = math.dist(after_eef.position, eef_target.position)
        assert actual_distance > 5e-4
        assert remaining_distance < requested_distance

        state = sdk.state.snapshot()
        base_goal = Pose(
            position=(state.base_pose.position[0] + 0.5, state.base_pose.position[1], 0),
            quaternion_xyzw=(0, 0, 0, 1),
            frame_id=state.base_pose.frame_id,
        )
        # 真实 MuJoCo 产品链必须从当前 SceneSnapshot 构造导航数据源。
        base_plan = sdk.base.plan_route(base_goal, maximum_speed_mps=0.1)
        moving = sdk.backend.execute_plan(base_plan, "sdk-stop")
        assert moving.status in {CommandState.ACCEPTED, CommandState.RUNNING}
        stopped = sdk.safety.stop_and_hold(moving.command_id)
        assert stopped.status is CommandState.STOPPED
        assert sdk.backend.state(robot_id).in_hold is True

        sensor_ids = sdk.sensors.list()
        rgb_id = next(sensor for sensor in sensor_ids if sensor.endswith(".rgb"))
        depth_id = next(sensor for sensor in sensor_ids if sensor.endswith(".depth"))
        contact_id = next(sensor for sensor in sensor_ids if sensor == "contact")
        with (
            sdk.sensors.subscribe(rgb_id) as rgb_subscription,
            sdk.sensors.subscribe(depth_id) as depth_subscription,
            sdk.sensors.subscribe(contact_id) as contact_subscription,
            sdk.sensors.subscribe_state(poll_interval_seconds=0.01) as state_subscription,
        ):
            rgb = next(rgb_subscription)
            assert rgb.encoding == "jpeg" and rgb.payload
            assert rgb.generation == scene["generation"]
            depth = next(depth_subscription)
            assert depth.encoding == "png16-mm"
            assert isinstance(depth.payload, bytes) and depth.payload.startswith(b"\x89PNG")
            contact = next(contact_subscription)
            assert contact.encoding == "json" and isinstance(contact.payload, dict)
            assert contact.width is None and contact.height is None
            assert next(state_subscription).generation == scene["generation"]

            reset = request_json(base_url, "POST", f"/api/v1/scene-instances/{instance_id}/reset")
            assert reset["generation"] == scene["generation"] + 1
            for subscription in (rgb_subscription, depth_subscription, contact_subscription):
                try:
                    next(subscription)
                except GenerationMismatch:
                    continue
                # reset 请求返回时，单槽中最多还可能有一帧 reset 前已经产生的画面；
                # 该帧仍带旧 generation，随后订阅必须终止，不能交付新场景画面。
                with pytest.raises(GenerationMismatch):
                    next(subscription)
            try:
                stale_state = next(state_subscription)
            except GenerationMismatch:
                pass
            else:
                # Robot State 与图像共用 latest-frame 单槽。reset 返回时允许读到
                # 一帧已经生成的旧状态，但绝不能跨 generation 继续订阅。
                assert stale_state.generation == scene["generation"]
                with pytest.raises(GenerationMismatch):
                    next(state_subscription)
            with pytest.raises(GenerationMismatch):
                sdk.backend.execute_plan(eef_plan, "stale-plan")
    finally:
        request_json(base_url, "POST", f"/api/v1/scene-instances/{instance_id}/stop")
