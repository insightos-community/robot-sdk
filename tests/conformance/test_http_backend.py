import json
import io
import struct
import threading
import urllib.error
from datetime import datetime, timezone

import pytest

from semantic_robot_sdk_core.errors import (
    BackendRequestError,
    BackendUnavailable,
    GenerationMismatch,
    StreamProtocolError,
)
from semantic_robot_sdk_r1pro.backends.mujoco import MujocoBackend
from semantic_robot_sdk_core.models import (
    Command,
    CommandState,
    JointLimit,
    JointTrajectoryPoint,
    MotionPlan,
    PlanKind,
    RobotCapabilities,
    ToolDescriptor,
)


def capabilities() -> RobotCapabilities:
    return RobotCapabilities(
        model="test_robot",
        kind="manipulator",
        joint_names=["joint_1"],
        joint_groups={"arm": ["joint_1"]},
        joint_limits={
            "joint_1": JointLimit(lower=-1, upper=1, max_velocity=1, max_acceleration=1, max_jerk=1)
        },
        end_effectors=["tool"],
        grippers=["hand"],
        sensors=["rgb"],
    )


class RecordingBackend(MujocoBackend):
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object]] = []
        super().__init__("http://runtime.test", "robot-1", capabilities())

    def _json(self, method: str, path: str, body=None):
        self.requests.append((method, path, body))
        if path.endswith("/profile"):
            return {
                "robot_id": "robot-1",
                "model": "test_robot",
                "joint_names": ["joint_1"],
                "coordinate_frame": "world",
                "end_effectors": ["tool"],
                "grippers": ["hand"],
                "capabilities": {
                    "commands": ["joint_trajectory", "gripper_command"],
                    "frames": ["world", "base_link", "tool"],
                },
            }
        if path.endswith("/commands"):
            return {
                "command_id": body["command_id"],
                "robot_id": "robot-1",
                "scene_generation": body["scene_generation"],
                "type": body["type"],
                "status": "accepted",
                "accepted_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        raise AssertionError(path)


class MissingRootFrameBackend(RecordingBackend):
    """模拟 Runtime 没有暴露 SDK 规划所使用的运动学根坐标。"""

    def _json(self, method: str, path: str, body=None):
        response = super()._json(method, path, body)
        if path.endswith("/profile"):
            response["capabilities"]["frames"] = ["world"]
        return response


def test_http_backend_rejects_missing_kinematic_root_frame() -> None:
    with pytest.raises(BackendRequestError, match="缺少 SDK 运动学根坐标"):
        MissingRootFrameBackend()


def test_http_backend_only_sends_planned_joint_trajectory() -> None:
    backend = RecordingBackend()
    plan = MotionPlan(
        plan_id="plan-1",
        robot_id="robot-1",
        generation=3,
        kind=PlanKind.JOINT,
        resources=["joints:arm"],
        frame_id="base_link",
        start={"joint_1": 0},
        goal={"joint_1": 0.5},
        joint_trajectory=[
            JointTrajectoryPoint(time_from_start_s=0, positions={"joint_1": 0}),
            JointTrajectoryPoint(time_from_start_s=1, positions={"joint_1": 0.5}),
        ],
        collision_checked=True,
        estimated_duration_s=1,
        planner="test",
        stop_on_contact=True,
        contact_tool_refs=["component://tool/left"],
        max_contact_force_n=25.0,
    )

    command = backend.execute_plan(plan, "command-1")
    assert command.status.value == "accepted"
    body = backend.requests[-1][2]
    assert body["type"] == "joint_trajectory"
    assert body["timeout_seconds"] == pytest.approx(3.0)
    assert body["joint_trajectory"]["resources"] == ["joints:arm"]
    assert body["joint_trajectory"]["position_tolerance_rad"] == pytest.approx(0.002)
    assert body["joint_trajectory"]["stop_on_contact"] is True
    assert body["joint_trajectory"]["contact_tool_refs"] == ["component://tool/left"]
    assert body["joint_trajectory"]["max_contact_force_n"] == pytest.approx(25.0)
    assert body["scene_generation"] == 3
    assert body["joint_trajectory"]["points"][1] == {
        "time_from_start_seconds": 1.0,
        "positions": {"joint_1": 0.5},
        "velocities": {},
    }
    assert "eef_pose" not in body
    assert "target" not in body

    precise = plan.model_copy(update={"position_tolerance_rad": 0.004})
    backend.execute_plan(precise, "command-precise")
    precise_body = backend.requests[-1][2]
    assert precise_body["joint_trajectory"]["position_tolerance_rad"] == pytest.approx(0.004)


def test_http_backend_feedback_returns_one_current_snapshot(monkeypatch) -> None:
    backend = object.__new__(MujocoBackend)
    calls: list[str] = []
    states = [
        Command(
            command_id="command-long",
            robot_id="robot-1",
            generation=3,
            kind=PlanKind.JOINT,
            status=CommandState.RUNNING,
            progress=0.25,
        ),
        Command(
            command_id="command-long",
            robot_id="robot-1",
            generation=3,
            kind=PlanKind.JOINT,
            status=CommandState.SUCCEEDED,
            progress=1.0,
        ),
    ]

    def current(_robot_id: str, command_id: str) -> Command:
        calls.append(command_id)
        return states[len(calls) - 1]

    monkeypatch.setattr(backend, "command", current)
    running = list(backend.feedback("robot-1", "command-long"))
    terminal = list(backend.feedback("robot-1", "command-long"))

    # 单次 feedback 不在 SDK 内等待物理命令终态；调用方可以在两个快照
    # 之间处理 stop、Interaction 或其它 Robot 状态事件。
    assert calls == ["command-long", "command-long"]
    assert len(running) == len(terminal) == 1
    assert running[0].progress == pytest.approx(0.25)
    assert terminal[0].progress == pytest.approx(1.0)
    assert terminal[0].sequence > running[0].sequence


def test_http_backend_maps_runtime_generation_conflict(monkeypatch) -> None:
    backend = object.__new__(MujocoBackend)
    backend.endpoint = "http://runtime.test"
    backend.timeout_seconds = 1
    payload = json.dumps(
        {
            "error": {
                "code": "conflict",
                "message": "命令 generation 已失效",
                "details": {"requested": 1, "current": 2},
            }
        }
    ).encode()

    def reject_generation(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://runtime.test/commands",
            409,
            "Conflict",
            {},
            io.BytesIO(payload),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject_generation)
    with pytest.raises(GenerationMismatch, match="generation 已失效"):
        backend._request("POST", "/commands", {"scene_generation": 1})


def interrupted_plan() -> MotionPlan:
    return MotionPlan(
        plan_id="plan-interrupted",
        robot_id="robot-1",
        generation=3,
        kind=PlanKind.JOINT,
        resources=["joints:arm"],
        frame_id="base_link",
        start={"joint_1": 0},
        goal={"joint_1": 0.2},
        joint_trajectory=[
            JointTrajectoryPoint(time_from_start_s=0, positions={"joint_1": 0}),
            JointTrajectoryPoint(time_from_start_s=1, positions={"joint_1": 0.2}),
        ],
        collision_checked=True,
        estimated_duration_s=1,
        planner="test",
    )


class InterruptedBackend(RecordingBackend):
    """模拟 Runtime 已可能接收 POST，但连接在响应到达前中断。"""

    def __init__(self, *, query_recovers: bool) -> None:
        self.query_recovers = query_recovers
        super().__init__()

    def _json(self, method: str, path: str, body=None):
        self.requests.append((method, path, body))
        if path.endswith("/profile"):
            return {
                "robot_id": "robot-1",
                "model": "test_robot",
                "joint_names": ["joint_1"],
                "coordinate_frame": "world",
                "end_effectors": ["tool"],
                "grippers": ["hand"],
                "capabilities": {
                    "commands": ["joint_trajectory", "gripper_command"],
                    "frames": ["world", "base_link", "tool"],
                },
            }
        if method == "POST" and path.endswith("/commands"):
            raise BackendUnavailable("响应丢失")
        if method == "GET" and "/commands/" in path:
            if not self.query_recovers:
                raise BackendUnavailable("查询时 Runtime 仍离线")
            now = datetime.now(timezone.utc).isoformat()
            return {
                "command_id": path.rsplit("/", 1)[-1],
                "robot_id": "robot-1",
                "scene_generation": 3,
                "type": "joint_trajectory",
                "status": "running",
                "accepted_at": now,
                "updated_at": now,
            }
        raise AssertionError(path)


def test_http_backend_recovers_submit_result_with_same_command_id() -> None:
    backend = InterruptedBackend(query_recovers=True)

    command = backend.execute_plan(interrupted_plan(), "command-recover")

    assert command.status is CommandState.RUNNING
    assert [item[0] for item in backend.requests[-2:]] == ["POST", "GET"]
    assert backend.requests[-1][1].endswith("/commands/command-recover")


def test_http_backend_returns_interrupted_when_submit_cannot_be_confirmed() -> None:
    backend = InterruptedBackend(query_recovers=False)

    command = backend.execute_plan(interrupted_plan(), "command-unknown")

    assert command.status is CommandState.INTERRUPTED
    assert command.command_id == "command-unknown"
    assert command.generation == 3
    assert "无法确认" in (command.reason or "")


class IncompleteStateBackend(RecordingBackend):
    def _json(self, method: str, path: str, body=None):
        if path.endswith("/state"):
            return {
                "robot_id": "robot-1",
                "generation": 1,
                "joints": {},
                "end_effectors": {},
                "grippers": {},
                "in_hold": False,
            }
        return super()._json(method, path, body)


def test_http_backend_rejects_partial_robot_state() -> None:
    backend = IncompleteStateBackend()

    with pytest.raises(BackendRequestError, match="缺少关节"):
        backend.state("robot-1")


def test_http_backend_preserves_passive_tool_joint_for_collision_geometry() -> None:
    backend = object.__new__(MujocoBackend)
    backend._capabilities = capabilities().model_copy(
        update={
            "tools": [
                ToolDescriptor(
                    tool_ref="component://tool/left",
                    side="left",
                    kind="tote_clamp",
                    frame="left_tool",
                    joint="left_tote_clamp_joint",
                    travel_m=0.04,
                    normal_force_n=20.0,
                    maximum_force_n=120.0,
                )
            ]
        }
    )

    state = backend._decode_robot_state(
        {
            "robot_id": "robot-1",
            "generation": 1,
            "joints": {
                "joint_1": {"position": 0.25},
                "left_tote_clamp_joint": {"position": 0.033},
            },
            "end_effectors": {},
            "grippers": {},
            "in_hold": True,
        }
    )

    assert state.joint_positions["joint_1"] == pytest.approx(0.25)
    assert state.joint_positions["left_tote_clamp_joint"] == pytest.approx(0.033)


class InterruptedStopBackend(RecordingBackend):
    def __init__(self) -> None:
        self.command_queries = 0
        super().__init__()

    def _json(self, method: str, path: str, body=None):
        if path.endswith("/profile"):
            return super()._json(method, path, body)
        if method == "GET" and "/commands/" in path:
            self.command_queries += 1
            if self.command_queries > 1:
                raise BackendUnavailable("停止后查询时 Runtime 离线")
            now = datetime.now(timezone.utc).isoformat()
            return {
                "command_id": path.rsplit("/", 1)[-1],
                "robot_id": "robot-1",
                "scene_generation": 3,
                "type": "joint_trajectory",
                "status": "running",
                "progress": 0.4,
                "accepted_at": now,
                "updated_at": now,
            }
        if method == "POST" and path.endswith("/stop"):
            raise BackendUnavailable("停止响应丢失")
        raise AssertionError(path)


def test_http_backend_stop_is_interrupted_when_runtime_cannot_confirm() -> None:
    backend = InterruptedStopBackend()

    command = backend.stop("robot-1", "command-moving")

    assert command.status is CommandState.INTERRUPTED
    assert command.progress == 0.4
    assert "无法确认" in (command.reason or "")


class _OneFrameConnection:
    """只发送一帧，之后等待 close 的同步 WebSocket 测试连接。"""

    def __init__(self, packet: bytes) -> None:
        self.packet = packet
        self.closed = threading.Event()

    def recv(self, timeout: float | None = None) -> bytes:
        if self.packet:
            packet, self.packet = self.packet, b""
            return packet
        if self.closed.wait(timeout):
            raise OSError("测试连接已关闭")
        raise TimeoutError

    def close(self) -> None:
        self.closed.set()


def _jpeg_stream_packet(*, generation: int = 3) -> bytes:
    metadata = {
        "stream": "sensor",
        "sequence": 11,
        "generation": generation,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "media_type": "image/jpeg",
        "encoding": "jpeg",
        "width": 2,
        "height": 1,
        "frame": "camera_front",
        "sensor_id": "front rgb",
        "viewer_session_id": None,
    }
    header = json.dumps(metadata).encode("utf-8")
    return struct.pack(">I", len(header)) + header + b"\xff\xd8frame\xff\xd9"


def _state_stream_packet(*, generation: int = 3, sequence: int = 7) -> bytes:
    observed_at = datetime.now(timezone.utc).isoformat()
    state = {
        "robot_id": "robot-1",
        "generation": generation,
        "observed_at": observed_at,
        "base_pose": {
            "position": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame_id": "world",
        },
        "joints": {"joint_1": {"position": 0.25, "velocity": 0.0}},
        "end_effectors": {},
        "grippers": {"hand": 0.04},
        "in_hold": True,
    }
    metadata = {
        "stream": "robot_state",
        "sequence": sequence,
        "generation": generation,
        "sim_time": 1.25,
        "observed_at": observed_at,
        "frame_id": "world",
        "robot_id": "robot-1",
        "media_type": "application/json",
        "encoding": "json",
    }
    header = json.dumps(metadata).encode("utf-8")
    payload = json.dumps(state).encode("utf-8")
    return struct.pack(">I", len(header)) + header + payload


class StreamRecordingBackend(MujocoBackend):
    def __init__(
        self, connection: _OneFrameConnection, *, legacy_state_polling: bool = False
    ) -> None:
        self.requests: list[tuple[str, str, object]] = []
        self.connection = connection
        self.generation = 3
        self.sensor_encoding = "jpeg"
        self.stream_url = ""
        self.stream_options: dict[str, object] = {}
        super().__init__(
            "http://runtime.test",
            "robot-1",
            capabilities(),
            stream_connector=self._connect_stream,
            stream_open_timeout_seconds=0.2,
            stream_receive_timeout_seconds=0.01,
            legacy_state_polling=legacy_state_polling,
        )

    def _connect_stream(self, url: str, **kwargs):
        self.stream_url = url
        self.stream_options = kwargs
        return self.connection

    def _json(self, method: str, path: str, body=None):
        if path.endswith("/profile"):
            return RecordingBackend._json(self, method, path, body)
        self.requests.append((method, path, body))
        if path.endswith("/sensors"):
            return [
                {
                    "sensor_id": "front rgb",
                    "robot_id": "robot-1",
                    "kind": "rgb",
                    "frame_id": "camera_front",
                    "encoding": self.sensor_encoding,
                    "width": 2,
                    "height": 1,
                    "fps": 20,
                }
            ]
        if path.endswith("/state"):
            return {
                "robot_id": "robot-1",
                "generation": self.generation,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "joints": {"joint_1": {"position": 0.0}},
                "end_effectors": {},
                "grippers": {"hand": 0.04},
                "in_hold": True,
            }
        raise AssertionError(path)


def test_http_backend_hides_binary_websocket_and_quotes_sensor_path() -> None:
    connection = _OneFrameConnection(_jpeg_stream_packet())
    backend = StreamRecordingBackend(connection)

    subscription = backend.subscribe_sensor("robot-1", "front rgb")
    try:
        frame = next(subscription)
        assert frame.sensor_id == "front rgb"
        assert frame.sequence == 11
        assert frame.encoding == "jpeg"
        assert frame.payload == b"\xff\xd8frame\xff\xd9"
    finally:
        subscription.close()

    assert backend.stream_url == (
        "ws://runtime.test/api/v1/robots/robot-1/sensors/front%20rgb/stream"
    )
    assert backend.stream_options["max_size"] == 64 * 1024 * 1024
    assert connection.closed.is_set()


def test_http_backend_state_subscription_uses_websocket_by_default() -> None:
    connection = _OneFrameConnection(_state_stream_packet())
    backend = StreamRecordingBackend(connection)
    subscription = backend.subscribe_state("robot-1")
    try:
        state = next(subscription)
        assert state.generation == 3
        assert state.joint_positions["joint_1"] == pytest.approx(0.25)
    finally:
        subscription.close()

    assert backend.stream_url == "ws://runtime.test/api/v1/robots/robot-1/state/stream"
    assert connection.closed.is_set()


def test_http_backend_state_stream_error_does_not_fallback_to_polling() -> None:
    backend = StreamRecordingBackend(_OneFrameConnection(_jpeg_stream_packet()))
    subscription = backend.subscribe_state("robot-1")
    try:
        with pytest.raises(StreamProtocolError, match="元数据不合法"):
            next(subscription)
    finally:
        subscription.close()

    state_requests = [path for _method, path, _body in backend.requests if path.endswith("/state")]
    assert state_requests == ["/api/v1/robots/robot-1/state"]


def test_http_backend_state_subscription_terminates_after_reset() -> None:
    backend = StreamRecordingBackend(
        _OneFrameConnection(_jpeg_stream_packet()), legacy_state_polling=True
    )
    subscription = backend.subscribe_state("robot-1", poll_interval_seconds=0)

    assert next(subscription).generation == 3
    backend.generation = 4
    with pytest.raises(GenerationMismatch):
        next(subscription)
    with pytest.raises(StopIteration):
        next(subscription)


def test_http_backend_rejects_runtime_sensor_encoding_not_in_public_protocol() -> None:
    backend = StreamRecordingBackend(_OneFrameConnection(_jpeg_stream_packet()))
    backend.sensor_encoding = "base64"

    with pytest.raises(BackendRequestError, match="descriptor 不合法"):
        backend.sensors("robot-1")
