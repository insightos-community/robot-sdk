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

"""Plugin 二进制传感器流协议和 latest-frame 行为测试。"""

from __future__ import annotations

import json
import struct
import threading
from collections import deque
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from semantic_robot_sdk_core.errors import (
    BackendUnavailable,
    GenerationMismatch,
    StreamProtocolError,
)
from semantic_robot_sdk_core.models import Pose, RobotState
from semantic_robot_sdk_core.sensors import StreamingSensorSubscription
from semantic_robot_sdk_core.stream_transport import (
    BinarySensorFrameParser,
    BinaryRobotStateFrameParser,
    BinaryWebSocketFrameTransport,
    SensorStreamDescriptor,
)


_MEDIA_TYPES = {
    "jpeg": "image/jpeg",
    "png16-mm": "image/png",
    "float32-le": "application/octet-stream",
    "json": "application/json",
}


def _packet(
    sequence: int,
    *,
    sensor_id: str = "front.rgb",
    generation: int = 3,
    encoding: str = "jpeg",
    width: int = 2,
    height: int = 1,
    payload: bytes | None = None,
    media_type: str | None = None,
) -> bytes:
    if payload is None:
        payload = {
            "jpeg": b"\xff\xd8frame\xff\xd9",
            "png16-mm": b"\x89PNG\r\n\x1a\npayload",
            "float32-le": b"\x00" * (width * height * 4),
            "json": b'{"contact": true}',
        }[encoding]
    metadata = {
        "stream": "sensor",
        "sequence": sequence,
        "generation": generation,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "media_type": media_type or _MEDIA_TYPES[encoding],
        "encoding": encoding,
        "width": width,
        "height": height,
        "frame": "camera_front",
        "sensor_id": sensor_id,
        "viewer_session_id": None,
    }
    header = json.dumps(metadata).encode("utf-8")
    return struct.pack(">I", len(header)) + header + payload


def _state_packet(
    sequence: int,
    *,
    generation: int = 3,
    metadata_updates: dict | None = None,
    payload_updates: dict | None = None,
) -> bytes:
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
        "joints": {"joint_1": {"position": 0.2}},
        "end_effectors": {},
        "grippers": {"hand": 0.04},
        "in_hold": True,
    }
    metadata = {
        "stream": "robot_state",
        "sequence": sequence,
        "generation": generation,
        "sim_time": 0.5,
        "observed_at": observed_at,
        "frame_id": "world",
        "robot_id": "robot-1",
        "media_type": "application/json",
        "encoding": "json",
    }
    metadata.update(metadata_updates or {})
    state.update(payload_updates or {})
    header = json.dumps(metadata).encode("utf-8")
    payload = json.dumps(state).encode("utf-8")
    return struct.pack(">I", len(header)) + header + payload


def _decode_state(raw: dict) -> RobotState:
    return RobotState(
        robot_id=raw["robot_id"],
        generation=raw["generation"],
        observed_at=raw["observed_at"],
        base_pose=Pose.model_validate(raw["base_pose"]),
        joint_positions={name: float(value["position"]) for name, value in raw["joints"].items()},
        end_effectors={},
        gripper_openings={name: float(value) for name, value in raw["grippers"].items()},
        in_hold=bool(raw["in_hold"]),
    )


def _state_parser(*, generation: int = 3) -> BinaryRobotStateFrameParser:
    return BinaryRobotStateFrameParser(
        "robot-1", expected_generation=generation, decode_state=_decode_state
    )


def test_robot_state_parser_accepts_shared_packet_format() -> None:
    frame = _state_parser().parse(_state_packet(5))

    assert frame.sequence == 5
    assert frame.generation == 3
    assert frame.payload["robot_id"] == "robot-1"
    assert frame.payload["joint_positions"]["joint_1"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("packet", "error", "message"),
    [
        (
            _state_packet(1, metadata_updates={"robot_id": "other"}),
            StreamProtocolError,
            "其他 Robot",
        ),
        (
            _state_packet(1, payload_updates={"robot_id": "other"}),
            StreamProtocolError,
            "robot_id 不一致",
        ),
        (
            _state_packet(1, payload_updates={"generation": 2}),
            StreamProtocolError,
            "generation 不一致",
        ),
        (
            _state_packet(1, metadata_updates={"generation": 4}),
            GenerationMismatch,
            "generation",
        ),
        (
            _state_packet(
                1,
                payload_updates={"observed_at": "2020-01-01T00:00:00+00:00"},
            ),
            StreamProtocolError,
            "observed_at 不一致",
        ),
    ],
)
def test_robot_state_parser_rejects_mismatched_header_and_payload(packet, error, message) -> None:
    with pytest.raises(error, match=message):
        _state_parser().parse(packet)


def _rgb_parser(*, generation: int = 3) -> BinarySensorFrameParser:
    return BinarySensorFrameParser(
        SensorStreamDescriptor(
            sensor_id="front.rgb",
            encoding="jpeg",
            width=2,
            height=1,
        ),
        expected_generation=generation,
    )


def test_parser_accepts_image_depth_and_structured_frames() -> None:
    rgb = _rgb_parser().parse(_packet(7))
    assert rgb.sequence == 7
    assert rgb.payload == b"\xff\xd8frame\xff\xd9"
    assert rgb.generation == 3

    depth = BinarySensorFrameParser(
        SensorStreamDescriptor(
            sensor_id="front.depth",
            encoding="float32-le",
            width=2,
            height=1,
        ),
        expected_generation=3,
    ).parse(
        _packet(
            8,
            sensor_id="front.depth",
            encoding="float32-le",
        )
    )
    assert isinstance(depth.payload, bytes)
    assert len(depth.payload) == 8

    contact = BinarySensorFrameParser(
        SensorStreamDescriptor(sensor_id="contact", encoding="json"),
        expected_generation=3,
    ).parse(
        _packet(
            9,
            sensor_id="contact",
            encoding="json",
            width=640,
            height=480,
        )
    )
    assert contact.payload == {"contact": True}
    assert contact.width is None and contact.height is None


@pytest.mark.parametrize(
    ("packet", "error", "message"),
    [
        ("text", StreamProtocolError, "binary"),
        (b"\x00\x00\x00\x14{}", StreamProtocolError, "不完整"),
        (struct.pack(">I", 2) + b"[]" + b"x", StreamProtocolError, "必须是对象"),
        (_packet(1, sensor_id="other"), StreamProtocolError, "其他 sensor_id"),
        (_packet(1, generation=4), GenerationMismatch, "generation"),
        (
            _packet(1, encoding="json", payload=b'{"ok": true}', width=0, height=0),
            StreamProtocolError,
            "编码",
        ),
        (_packet(1, width=3), StreamProtocolError, "宽度"),
        (
            _packet(1, media_type="application/octet-stream"),
            StreamProtocolError,
            "media_type",
        ),
        (_packet(1, payload=b"not-jpeg"), StreamProtocolError, "JPEG"),
    ],
)
def test_parser_rejects_wrong_metadata_or_payload(packet, error, message) -> None:
    with pytest.raises(error, match=message):
        _rgb_parser().parse(packet)


def test_parser_rejects_invalid_depth_payloads() -> None:
    png_parser = BinarySensorFrameParser(
        SensorStreamDescriptor(
            sensor_id="depth",
            encoding="png16-mm",
            width=2,
            height=1,
        ),
        expected_generation=3,
    )
    with pytest.raises(StreamProtocolError, match="PNG"):
        png_parser.parse(
            _packet(
                1,
                sensor_id="depth",
                encoding="png16-mm",
                payload=b"not-png",
            )
        )

    float_parser = BinarySensorFrameParser(
        SensorStreamDescriptor(
            sensor_id="depth",
            encoding="float32-le",
            width=2,
            height=1,
        ),
        expected_generation=3,
    )
    with pytest.raises(StreamProtocolError, match="载荷长度"):
        float_parser.parse(
            _packet(
                1,
                sensor_id="depth",
                encoding="float32-le",
                payload=b"\x00" * 4,
            )
        )


def test_stream_descriptor_requires_supported_encoding_and_complete_dimensions() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        SensorStreamDescriptor(sensor_id="rgb", encoding="base64", width=2, height=1)
    with pytest.raises(ValidationError, match="同时提供"):
        SensorStreamDescriptor(sensor_id="rgb", encoding="jpeg", width=2)
    with pytest.raises(ValidationError, match="必须提供画面尺寸"):
        SensorStreamDescriptor(sensor_id="rgb", encoding="jpeg")


class _FakeConnection:
    """模拟 websockets.sync 连接；空队列时可阻塞或明确断开。"""

    def __init__(self, packets: list[bytes], *, disconnect_after: bool = False) -> None:
        self._packets = deque(packets)
        self._disconnect_after = disconnect_after
        self._lock = threading.Lock()
        self.closed = threading.Event()
        self.drained = threading.Event()

    def recv(self, timeout: float | None = None) -> bytes:
        with self._lock:
            if self._packets:
                packet = self._packets.popleft()
                if not self._packets:
                    self.drained.set()
                return packet
        if self._disconnect_after:
            raise OSError("测试断连")
        if self.closed.wait(timeout):
            raise OSError("测试关闭")
        raise TimeoutError

    def close(self) -> None:
        self.closed.set()


class _FakeConnector:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.url = ""
        self.kwargs: dict[str, object] = {}

    def __call__(self, url: str, **kwargs):
        self.url = url
        self.kwargs = kwargs
        return self.connection


def _transport(connection: _FakeConnection) -> BinaryWebSocketFrameTransport:
    return BinaryWebSocketFrameTransport(
        "ws://runtime.test/api/v1/robots/r1/sensors/front.rgb/stream",
        _rgb_parser(),
        connector=_FakeConnector(connection),
        open_timeout_seconds=0.2,
        receive_timeout_seconds=0.01,
    )


def test_stream_transport_keeps_only_latest_frame_for_slow_consumer() -> None:
    connection = _FakeConnection([_packet(1), _packet(2), _packet(3)])
    subscription = StreamingSensorSubscription(_transport(connection))
    assert connection.drained.wait(1)

    frame = next(subscription)
    assert frame.sequence == 3
    assert subscription.dropped_frames == 2

    subscription.close()
    with pytest.raises(StopIteration):
        next(subscription)


def test_stream_disconnect_is_terminal_and_not_silent_polling_fallback() -> None:
    connection = _FakeConnection([_packet(1)], disconnect_after=True)
    transport = _transport(connection)
    assert connection.drained.wait(1)

    assert transport.wait_terminated(1)
    with pytest.raises(BackendUnavailable, match="已断开"):
        transport.receive_latest(0)
    transport.close()


def test_stream_duplicate_sequence_is_deduplicated() -> None:
    connection = _FakeConnection([_packet(1), _packet(1), _packet(2)])
    subscription = StreamingSensorSubscription(_transport(connection))
    assert connection.drained.wait(1)

    assert next(subscription).sequence == 2
    assert subscription.dropped_frames == 1
    subscription.close()


@pytest.mark.parametrize(
    ("packets", "error"),
    [
        ([_packet(2), _packet(1)], StreamProtocolError),
        ([_packet(1), _packet(2, generation=4)], GenerationMismatch),
    ],
)
def test_stream_sequence_or_generation_error_terminates_subscription(packets, error) -> None:
    connection = _FakeConnection(packets)
    transport = _transport(connection)
    assert connection.drained.wait(1)

    assert transport.wait_terminated(1)
    with pytest.raises(error):
        transport.receive_latest(0)
    transport.close()


def test_robot_state_stream_reset_generation_is_terminal() -> None:
    connection = _FakeConnection([_state_packet(1), _state_packet(2, generation=4)])
    transport = BinaryWebSocketFrameTransport(
        "ws://runtime.test/api/v1/robots/robot-1/state/stream",
        _state_parser(),
        connector=_FakeConnector(connection),
        open_timeout_seconds=0.2,
        receive_timeout_seconds=0.01,
    )
    assert connection.drained.wait(1)
    assert transport.wait_terminated(1)

    with pytest.raises(GenerationMismatch):
        transport.receive_latest(0)
    transport.close()
