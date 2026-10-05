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

"""面向仿真 Runtime 公共 Robot 接口的同步 HTTP Backend。

AbilityFramework 通过本 Backend 使用虚拟 Robot。它只发送 Robot SDK 已经规划好的
低层轨迹，不把末端目标、导航目标或规划算法下沉到 Runtime。网络错误不会被包装成
成功结果；若连接在提交后中断，调用方应查询原 command_id，无法确认时标记 unknown。
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from semantic_robot_sdk_core.errors import (
    BackendRequestError,
    BackendUnavailable,
    GenerationMismatch,
)
from semantic_robot_sdk_core.models import (
    Command,
    CommandState,
    Feedback,
    MotionPlan,
    PlanKind,
    Pose,
    RobotCapabilities,
    RobotState,
    SceneObject,
    SceneRegion,
    SceneSnapshot,
    SensorFrame,
    ToolDescriptor,
    ToolState,
)
from semantic_robot_sdk_core.sensors import (
    RobotStateSubscription,
    SensorSubscription,
    StreamingRobotStateSubscription,
    StreamingSensorSubscription,
)
from semantic_robot_sdk_core.stream_transport import (
    BinarySensorFrameParser,
    BinaryRobotStateFrameParser,
    BinaryWebSocketFrameTransport,
    SensorStreamDescriptor,
)


class MujocoBackend:
    """调用 Plugin Runtime 的类型化 Robot API。

    `capabilities` 来自 Robot 型号包，Runtime profile 只负责验证实际 Robot 与该
    型号一致。这样限位、规划参数和名称映射不会从仿真引擎临时推断。
    """

    def __init__(
        self,
        endpoint: str,
        robot_id: str,
        capabilities: RobotCapabilities,
        *,
        timeout_seconds: float = 10.0,
        stream_connector: Callable[..., Any] | None = None,
        stream_open_timeout_seconds: float = 10.0,
        stream_receive_timeout_seconds: float = 0.5,
        legacy_state_polling: bool = False,
        scene_instance_id: str | None = None,
        joint_position_tolerance_rad: float = 0.002,
    ) -> None:
        if (
            min(
                timeout_seconds,
                stream_open_timeout_seconds,
                stream_receive_timeout_seconds,
            )
            <= 0
        ):
            raise ValueError("Backend 超时时间和轮询间隔必须大于零")
        self.endpoint = endpoint.rstrip("/")
        self.robot_id = robot_id
        self._capabilities = capabilities
        self.timeout_seconds = timeout_seconds
        self._stream_connector = stream_connector
        self._stream_open_timeout_seconds = stream_open_timeout_seconds
        self._stream_receive_timeout_seconds = stream_receive_timeout_seconds
        self._legacy_state_polling = legacy_state_polling
        self._scene_instance_id = scene_instance_id
        if joint_position_tolerance_rad <= 0:
            raise ValueError("关节到位容差必须大于零")
        self._joint_position_tolerance_rad = joint_position_tolerance_rad
        self._verify_profile()

    def capabilities(self) -> RobotCapabilities:
        return self._capabilities.model_copy(deep=True)

    def state(self, robot_id: str) -> RobotState:
        self._check_robot(robot_id)
        return self._decode_robot_state(self._json("GET", self._robot_path("state")))

    def _decode_robot_state(self, raw: dict[str, Any]) -> RobotState:
        """把 Runtime 原始状态转换为公共模型；HTTP 与 WS 必须走同一映射。"""

        if not isinstance(raw, dict):
            raise BackendRequestError("Runtime Robot 状态必须是对象")
        joints = raw.get("joints", {})
        positions = {
            name: float(value.get("position", value)) if isinstance(value, dict) else float(value)
            for name, value in joints.items()
            if name in self._capabilities.joint_names
        }
        missing = set(self._capabilities.joint_names) - set(positions)
        if missing:
            raise BackendRequestError(f"Runtime Robot 状态缺少关节：{sorted(missing)}")

        # 夹具关节不是机械臂的受控规划关节，但它会改变碰撞几何。释放后的撤离规划
        # 必须使用 Runtime 当前开度，不能让 Pinocchio 沿用模型中的闭合默认值；否则
        # 密集堆垛会把已经打开的夹片误判为与相邻箱体碰撞。
        for descriptor in self._capabilities.tools:
            value = joints.get(descriptor.joint)
            if value is not None:
                positions[descriptor.joint] = (
                    float(value.get("position", value)) if isinstance(value, dict) else float(value)
                )

        tools_by_side = {tool.side: tool for tool in self._capabilities.tools}
        tool_states = {}
        for side, value in raw.get("gripper_states", {}).items():
            descriptor = tools_by_side.get(side)
            if descriptor is None:
                raise BackendRequestError(f"Runtime 返回了未声明工具侧：{side}")
            tool_states[descriptor.tool_ref] = ToolState(
                tool_ref=descriptor.tool_ref,
                side=side,
                kind=descriptor.kind,
                **value,
            )

        base = raw.get("base_pose")
        return RobotState(
            robot_id=raw["robot_id"],
            generation=raw["generation"],
            observed_at=raw.get("observed_at", datetime.now(timezone.utc)),
            base_pose=self._pose(base) if base else None,
            joint_positions=positions,
            end_effectors={
                name: self._pose(value) for name, value in raw.get("end_effectors", {}).items()
            },
            gripper_openings={
                name: float(value) for name, value in raw.get("grippers", {}).items()
            },
            tool_states=tool_states,
            in_hold=bool(raw.get("in_hold", False)),
        )

    def scene_snapshot(self) -> SceneSnapshot:
        if not self._scene_instance_id:
            raise BackendRequestError("RobotDeployment 未配置 scene_instance_id")
        instance = urllib.parse.quote(self._scene_instance_id, safe="")
        raw = self._json("GET", f"/api/v1/scene-instances/{instance}/snapshot")
        return SceneSnapshot(
            scene_key=raw["scene_key"],
            instance_id=raw["instance_id"],
            generation=raw["generation"],
            coordinate_frame=raw.get("coordinate_frame", "world"),
            objects=[
                SceneObject(
                    source_id=item["source_id"],
                    category=item["category"],
                    name=item["name"],
                    pose=self._pose(item["pose"]),
                    extent=item.get("extent"),
                    state=item.get("state", {}),
                )
                for item in raw.get("objects", [])
            ],
            regions=[
                SceneRegion(
                    source_id=item["source_id"],
                    name=item["name"],
                    pose=self._pose(item["pose"]),
                    extent=item.get("extent"),
                    properties=dict(item.get("properties") or {}),
                )
                for item in raw.get("regions", [])
            ],
            observed_at=raw["observed_at"],
        )

    def execute_plan(self, plan: MotionPlan, command_id: str) -> Command:
        if plan.robot_id != self.robot_id:
            raise BackendRequestError("MotionPlan 属于其他 Robot")
        timeout_seconds = max(plan.estimated_duration_s * 1.5, 1.0)
        if plan.kind is PlanKind.JOINT:
            # 轨迹时长描述理想位置曲线，真实执行器在最后一个轨迹点后还需要
            # 短时间衰减速度并进入 Runtime 的关节到位容差。这个稳定窗口不
            # 放宽精度，也不影响底盘和夹具命令，只避免轨迹结束即被误判超时。
            timeout_seconds = max(timeout_seconds, plan.estimated_duration_s + 2.0)
        body: dict[str, Any] = {
            "command_id": command_id,
            "scene_generation": plan.generation,
            "type": plan.kind.value,
            "timeout_seconds": timeout_seconds,
        }
        if plan.kind is PlanKind.JOINT:
            body["joint_trajectory"] = {
                "resources": list(plan.resources),
                "frame_id": plan.frame_id,
                "position_tolerance_rad": (
                    plan.position_tolerance_rad
                    if plan.position_tolerance_rad is not None
                    else self._joint_position_tolerance_rad
                ),
                "stop_on_contact": plan.stop_on_contact,
                "contact_tool_refs": list(plan.contact_tool_refs),
                "max_contact_force_n": plan.max_contact_force_n,
                "points": [
                    {
                        "time_from_start_seconds": point.time_from_start_s,
                        "positions": point.positions,
                        "velocities": point.velocities,
                    }
                    for point in plan.joint_trajectory
                ],
            }
        elif plan.kind is PlanKind.BASE:
            body["base_trajectory"] = {
                "frame_id": plan.frame_id,
                "points": [
                    {
                        "time_from_start_seconds": point.time_from_start_s,
                        "positions": {"x": point.x, "y": point.y, "yaw": point.yaw},
                    }
                    for point in plan.base_trajectory
                ],
            }
        else:
            target = plan.gripper_command
            body["gripper_command"] = {
                "gripper_id": target.gripper,
                "position": target.opening_m,
                "max_effort": target.force_limit_n,
                "stop_on_contact": target.stop_on_contact,
            }
        try:
            return self._command(self._json("POST", self._robot_path("commands"), body))
        except BackendUnavailable as submit_error:
            # TCP 连接可能在 Runtime 已接收命令后才中断。此时自动换一个 command_id
            # 会造成重复运动；先按原 ID 查询。若查询同样不可达，只能返回 unknown，
            # 由上层停止/人工确认，绝不能伪造 failed 或重试执行。
            try:
                return self.command(self.robot_id, command_id)
            except BackendRequestError:
                raise submit_error
            except BackendUnavailable:
                return Command(
                    command_id=command_id,
                    robot_id=plan.robot_id,
                    generation=plan.generation,
                    kind=plan.kind,
                    status=CommandState.INTERRUPTED,
                    reason="命令提交时 Runtime 断连，无法确认是否已开始执行",
                )

    def command(self, robot_id: str, command_id: str) -> Command:
        self._check_robot(robot_id)
        return self._command(
            self._json(
                "GET",
                self._robot_path("commands", urllib.parse.quote(command_id, safe="")),
            )
        )

    def feedback(self, robot_id: str, command_id: str) -> Iterator[Feedback]:
        # Backend 的一次 feedback 调用只读取一次当前快照。长期命令由
        # Ability/Pilot 依据 invocation cursor 继续查询；若在这里阻塞轮询到
        # 终态，GetExecution 会被一个 SDK 调用占住数十秒，stop、状态恢复和
        # 其他控制请求都无法及时处理。
        command = self.command(robot_id, command_id)
        yield Feedback(
            command_id=command_id,
            sequence=_feedback_sequence(command),
            progress=command.progress,
            message=command.reason or command.status.value,
        )

    def stop(self, robot_id: str, command_id: str) -> Command:
        self._check_robot(robot_id)
        before = self.command(robot_id, command_id)
        if before.status in {
            CommandState.SUCCEEDED,
            CommandState.FAILED,
            CommandState.STOPPED,
        }:
            return before
        path = self._robot_path(
            "commands",
            urllib.parse.quote(command_id, safe=""),
            "stop",
        )
        try:
            return self._command(self._json("POST", path, {}))
        except BackendUnavailable:
            # stop 响应丢失时同样不能把“请求失败”解释成 Robot 已停止。先查询原命令；
            # 查询也不可用或记录已消失时返回 unknown，要求上层继续尝试 hold/人工确认。
            try:
                return self.command(robot_id, command_id)
            except (BackendUnavailable, BackendRequestError):
                return before.model_copy(
                    update={
                        "status": CommandState.INTERRUPTED,
                        "updated_at": datetime.now(timezone.utc),
                        "reason": "停止请求后 Runtime 断连，无法确认 Robot 是否已停止",
                    }
                )

    def hold(self, robot_id: str) -> None:
        self._check_robot(robot_id)
        generation = self.state(robot_id).generation
        self._json("POST", self._robot_path("hold"), {"scene_generation": generation})

    def sensors(self, robot_id: str) -> list[str]:
        self._check_robot(robot_id)
        return [item.sensor_id for item in self._sensor_descriptors(robot_id)]

    def _sensor_descriptors(self, robot_id: str) -> list[SensorStreamDescriptor]:
        """读取并校验 Runtime 的真实传感器流说明。"""

        self._check_robot(robot_id)
        rows = self._json("GET", self._robot_path("sensors"))
        if not isinstance(rows, list):
            raise BackendRequestError("Runtime sensors 响应必须是数组")
        result: list[SensorStreamDescriptor] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise BackendRequestError("Runtime sensor descriptor 必须是对象")
            if row.get("robot_id", robot_id) != robot_id:
                raise BackendRequestError("Runtime 返回了其他 Robot 的传感器说明")
            try:
                descriptor = SensorStreamDescriptor(
                    sensor_id=row["sensor_id"],
                    encoding=row["encoding"],
                    width=row.get("width"),
                    height=row.get("height"),
                )
            except (KeyError, TypeError, ValidationError) as error:
                raise BackendRequestError(f"Runtime sensor descriptor 不合法: {error}") from error
            if descriptor.sensor_id in seen:
                raise BackendRequestError(f"Runtime 重复声明传感器: {descriptor.sensor_id}")
            seen.add(descriptor.sensor_id)
            result.append(descriptor)
        return result

    def latest_sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame:
        self._check_robot(robot_id)
        path = self._robot_path(
            "sensors",
            urllib.parse.quote(sensor_id, safe=""),
            "frames",
            "latest",
            "content",
        )
        response, payload = self._request("GET", path)
        return SensorFrame(
            sensor_id=sensor_id,
            sequence=int(response.headers.get("X-Semantic-Sequence", "0")),
            generation=int(response.headers["X-Semantic-Generation"]),
            frame_id=response.headers.get("X-Semantic-Frame", sensor_id),
            encoding=response.headers.get("X-Semantic-Encoding", "binary"),
            width=_optional_int(response.headers.get("X-Semantic-Width")),
            height=_optional_int(response.headers.get("X-Semantic-Height")),
            observed_at=response.headers.get("X-Semantic-Observed-At", datetime.now(timezone.utc)),
            payload=payload,
        )

    def subscribe_sensor(
        self,
        robot_id: str,
        sensor_id: str,
        *,
        poll_interval_seconds: float = 0.05,
    ) -> SensorSubscription:
        """通过 Plugin 二进制 WebSocket 订阅经过校验的最新帧。

        poll_interval_seconds 仅为 Fake/旧调用方保留；真实流的 recv 超时由 Backend
        构造参数控制，不会轮询图片，也不会把 WebSocket 暴露给 Ability。
        """

        self._check_robot(robot_id)
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds 不能小于零")
        descriptor = next(
            (item for item in self._sensor_descriptors(robot_id) if item.sensor_id == sensor_id),
            None,
        )
        if descriptor is None:
            raise BackendRequestError(f"Robot 不包含传感器：{sensor_id}")
        generation = self.state(robot_id).generation
        parser = BinarySensorFrameParser(descriptor, expected_generation=generation)
        path = self._robot_path(
            "sensors",
            urllib.parse.quote(sensor_id, safe=""),
            "stream",
        )
        transport = BinaryWebSocketFrameTransport(
            self._stream_url(path),
            parser,
            connector=self._stream_connector,
            open_timeout_seconds=self._stream_open_timeout_seconds,
            receive_timeout_seconds=self._stream_receive_timeout_seconds,
        )
        return StreamingSensorSubscription(transport)

    def subscribe_state(
        self,
        robot_id: str,
        *,
        poll_interval_seconds: float = 0.1,
    ) -> RobotStateSubscription:
        """默认订阅正式 WebSocket 状态流；仅显式 legacy 开关允许 HTTP 轮询。"""

        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds 不能小于零")
        state = self.state(robot_id)
        if self._legacy_state_polling:
            return RobotStateSubscription(
                lambda: self.state(robot_id),
                expected_generation=state.generation,
                poll_interval_seconds=poll_interval_seconds,
            )
        parser = BinaryRobotStateFrameParser(
            robot_id,
            expected_generation=state.generation,
            decode_state=self._decode_robot_state,
        )
        transport = BinaryWebSocketFrameTransport(
            self._stream_url(self._robot_path("state", "stream")),
            parser,
            connector=self._stream_connector,
            open_timeout_seconds=self._stream_open_timeout_seconds,
            receive_timeout_seconds=self._stream_receive_timeout_seconds,
        )
        return StreamingRobotStateSubscription(transport)

    def _verify_profile(self) -> None:
        profile = self._json("GET", self._robot_path("profile"))
        if profile.get("robot_id") != self.robot_id:
            raise BackendRequestError("Runtime 返回了其他 Robot 的 Profile")
        if profile.get("model") != self._capabilities.model:
            raise BackendRequestError(
                f"Robot 型号不匹配：期望 {self._capabilities.model}，实际 {profile.get('model')}"
            )
        actual = set(profile.get("joint_names", []))
        expected = set(self._capabilities.joint_names)
        if actual != expected:
            raise BackendRequestError("Runtime Robot joint_names 与 SDK Profile 不一致")
        if profile.get("coordinate_frame") != self._capabilities.coordinate_frame:
            raise BackendRequestError("Runtime 与 SDK Profile 的公共坐标系不一致")
        if set(profile.get("end_effectors", [])) != set(self._capabilities.end_effectors):
            raise BackendRequestError("Runtime 与 SDK Profile 的末端执行器不一致")
        if set(profile.get("grippers", [])) != set(self._capabilities.grippers):
            raise BackendRequestError("Runtime 与 SDK Profile 的夹爪不一致")
        try:
            actual_tools = [
                ToolDescriptor.model_validate(item) for item in profile.get("tools", [])
            ]
        except ValidationError as error:
            raise BackendRequestError(f"Runtime 工具描述不合法: {error}") from error
        expected_tools = {item.tool_ref: item for item in self._capabilities.tools}
        actual_by_ref = {item.tool_ref: item for item in actual_tools}
        if set(actual_by_ref) != set(expected_tools):
            raise BackendRequestError("Runtime 与 SDK Profile 的工具描述不一致")
        for tool_ref, expected_tool in expected_tools.items():
            actual_tool = actual_by_ref[tool_ref]
            # tool_ref、side、frame 和 joint 是路由与控制身份，必须精确一致；
            # 行程和力是 JSON 浮点测量，不能因 0.039 与
            # 0.03900000000000001 的序列化差异拒绝同一 Profile。
            identity_matches = (
                actual_tool.side == expected_tool.side
                and actual_tool.kind == expected_tool.kind
                and actual_tool.frame == expected_tool.frame
                and actual_tool.joint == expected_tool.joint
            )
            numeric_matches = all(
                math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
                for actual, expected in (
                    (actual_tool.travel_m, expected_tool.travel_m),
                    (actual_tool.normal_force_n, expected_tool.normal_force_n),
                    (actual_tool.maximum_force_n, expected_tool.maximum_force_n),
                )
            )
            if not identity_matches or not numeric_matches:
                raise BackendRequestError(f"Runtime 与 SDK Profile 的工具描述不一致: {tool_ref}")
        capability = profile.get("capabilities", {})
        actual_frames = set(capability.get("frames", []))
        if self._capabilities.kinematic_root_frame not in actual_frames:
            raise BackendRequestError("Runtime Profile 缺少 SDK 运动学根坐标")

        actual_commands = set(capability.get("commands", []))
        required_commands: set[str] = set()
        if self._capabilities.joint_names:
            required_commands.add(PlanKind.JOINT.value)
        if self._capabilities.grippers:
            required_commands.add(PlanKind.GRIPPER.value)
        if self._capabilities.supports_base:
            required_commands.add(PlanKind.BASE.value)
        missing_commands = required_commands - actual_commands
        if missing_commands:
            raise BackendRequestError(f"Runtime 缺少 SDK 所需命令：{sorted(missing_commands)}")

    def _check_robot(self, robot_id: str) -> None:
        if robot_id != self.robot_id:
            raise BackendRequestError(f"Backend 绑定 {self.robot_id}，不能访问 {robot_id}")

    def _robot_path(self, *parts: str) -> str:
        robot = urllib.parse.quote(self.robot_id, safe="")
        suffix = "/".join(parts)
        return f"/api/v1/robots/{robot}/{suffix}"

    def _stream_url(self, path: str) -> str:
        """把 HTTP Runtime 地址转换为同源 WebSocket 地址。"""

        parsed = urllib.parse.urlsplit(self.endpoint)
        schemes = {"http": "ws", "https": "wss"}
        if parsed.scheme not in schemes or not parsed.netloc:
            raise BackendRequestError("Runtime endpoint 必须是完整的 http 或 https 地址")
        if parsed.query or parsed.fragment:
            raise BackendRequestError("Runtime endpoint 不能包含 query 或 fragment")
        base_path = parsed.path.rstrip("/")
        return urllib.parse.urlunsplit(
            (schemes[parsed.scheme], parsed.netloc, base_path + path, "", "")
        )

    def _json(self, method: str, path: str, body: Any | None = None) -> Any:
        _, payload = self._request(method, path, body)
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise BackendRequestError("Runtime 返回的 JSON 无法解析") from error

    def _request(self, method: str, path: str, body: Any | None = None) -> tuple[Any, bytes]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response, response.read()
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            # Runtime 在 reset 后用 409 拒绝旧 generation 命令。这里把传输层
            # 响应还原成 SDK 公共错误，保证 Fake 与真实 Backend 的恢复语义一致；
            # 上层因此会重新规划，而不是把场景失效误当成普通 HTTP 故障。
            try:
                payload = json.loads(details)
                runtime_error = payload.get("error", payload)
            except (json.JSONDecodeError, AttributeError):
                runtime_error = {}
            message = str(runtime_error.get("message", ""))
            code = str(runtime_error.get("code", ""))
            if error.code == 409 and (
                code == "generation_mismatch" or "generation" in message.lower()
            ):
                raise GenerationMismatch(message or "Runtime scene generation 已失效") from error
            raise BackendRequestError(f"Runtime 拒绝请求 ({error.code}): {details}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BackendUnavailable(f"Runtime 不可连接: {error}") from error

    @staticmethod
    def _pose(value: dict[str, Any]) -> Pose:
        return Pose(
            position=tuple(value["position"]),
            quaternion_xyzw=tuple(value["quaternion_xyzw"]),
            frame_id=value.get("frame_id", value.get("frame", "world")),
        )

    @staticmethod
    def _command(value: dict[str, Any]) -> Command:
        return Command(
            command_id=value["command_id"],
            robot_id=value["robot_id"],
            generation=value["scene_generation"],
            kind=PlanKind(value["type"]),
            status=CommandState(
                {"cancelled": "stopped", "canceled": "stopped", "unknown": "interrupted"}.get(
                    value["status"], value["status"]
                )
            ),
            progress=float(value.get("progress", 1.0 if value["status"] == "succeeded" else 0.0)),
            submitted_at=value.get("accepted_at", value.get("updated_at")),
            updated_at=value.get("updated_at"),
            reason=value.get("failure_reason", value.get("message")),
        )


def _feedback_sequence(command: Command) -> int:
    """从单调的命令状态和进度生成稳定 cursor，不在 Backend 保存第二份状态。"""

    if command.status is CommandState.ACCEPTED:
        phase = 0
    elif command.status is CommandState.RUNNING:
        phase = 1
    else:
        phase = 2
    progress = min(1000, max(0, round(command.progress * 1000)))
    return phase * 1001 + progress + 1


def _optional_int(value: str | None) -> int | None:
    if value in (None, "", "0"):
        return None
    return int(value)
