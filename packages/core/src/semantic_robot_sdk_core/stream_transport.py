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

"""Plugin 二进制 WebSocket 帧协议与 latest-frame 传输。

本模块是 Runtime 协议边界。Ability 和 Robot SDK 上层只接触 SensorFrame 与可关闭
订阅，不需要知道 WebSocket、四字节头或 Runtime 路径。
"""

from __future__ import annotations

from collections.abc import Callable
import json
import struct
import threading
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError, model_validator

try:
    from websockets.exceptions import ConnectionClosed, WebSocketException
    from websockets.sync.client import connect
except ModuleNotFoundError as error:
    if error.name is not None and not error.name.startswith("websockets"):
        raise

    class ConnectionClosed(Exception):
        pass

    class WebSocketException(Exception):
        pass

    connect = None
    _WEBSOCKETS_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _WEBSOCKETS_IMPORT_ERROR = None

from .errors import BackendUnavailable, GenerationMismatch, StreamProtocolError
from .models import RobotState, SensorFrame, StrictModel


class _Connection(Protocol):
    def recv(self, timeout: float | None = None) -> bytes | str: ...

    def close(self) -> None: ...


class _Connector(Protocol):
    def __call__(self, uri: str, **kwargs: Any) -> _Connection: ...


class StreamFrameMetadata(StrictModel):
    """Plugin native 与 Profile Runtime 共用的二进制帧头。"""

    stream: Literal["sensor"]
    sequence: int = Field(ge=1)
    generation: int = Field(ge=1)
    observed_at: datetime
    media_type: str = Field(min_length=1)
    encoding: str = Field(min_length=1)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    frame: str = Field(min_length=1)
    sensor_id: str = Field(min_length=1)
    viewer_session_id: None = None


class RobotStateFrameMetadata(StrictModel):
    """Robot State 流头；字段与 Plugin native/Profile 完全一致。"""

    stream: Literal["robot_state"]
    sequence: int = Field(ge=1)
    generation: int = Field(ge=1)
    sim_time: float = Field(ge=0.0)
    observed_at: datetime
    frame_id: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    media_type: Literal["application/json"]
    encoding: Literal["json"]


class SensorStreamDescriptor(StrictModel):
    """建立订阅前从 Robot Profile 读取的传感器说明。"""

    sensor_id: str = Field(min_length=1)
    encoding: Literal["jpeg", "png16-mm", "float32-le", "json"]
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> SensorStreamDescriptor:
        if (self.width is None) != (self.height is None):
            raise ValueError("传感器说明必须同时提供 width 和 height")
        if self.encoding != "json" and self.width is None:
            raise ValueError(f"{self.encoding} 传感器说明必须提供画面尺寸")
        return self


_MEDIA_TYPES = {
    "jpeg": "image/jpeg",
    "png16-mm": "image/png",
    "float32-le": "application/octet-stream",
    "json": "application/json",
}


def _unpack_packet(
    packet: bytes | bytearray | memoryview | str,
    *,
    max_header_bytes: int,
    max_packet_bytes: int,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """解析所有实时流共用的四字节长度头，避免状态流另造封装。"""

    if isinstance(packet, str):
        raise StreamProtocolError(f"{label}必须发送 WebSocket binary message")
    value = bytes(packet)
    if len(value) > max_packet_bytes:
        raise StreamProtocolError(f"{label}超过允许的最大大小")
    if len(value) < 5:
        raise StreamProtocolError(f"{label}缺少四字节头长度或载荷")
    header_size = struct.unpack(">I", value[:4])[0]
    if header_size < 2 or header_size > max_header_bytes:
        raise StreamProtocolError(f"{label} JSON 头长度无效")
    payload_offset = 4 + header_size
    if payload_offset >= len(value):
        raise StreamProtocolError(f"{label}头或载荷不完整")
    try:
        raw_metadata = json.loads(value[4:payload_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StreamProtocolError(f"{label} JSON 头无法解析") from error
    if not isinstance(raw_metadata, dict):
        raise StreamProtocolError(f"{label} JSON 头必须是对象")
    return raw_metadata, value[payload_offset:]


class BinarySensorFrameParser:
    """解析并严格校验 Plugin 的 metadata+payload 二进制包。"""

    def __init__(
        self,
        descriptor: SensorStreamDescriptor,
        *,
        expected_generation: int,
        max_header_bytes: int = 64 * 1024,
        max_packet_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if expected_generation < 1:
            raise ValueError("expected_generation 必须大于零")
        if max_header_bytes < 128 or max_packet_bytes <= max_header_bytes + 4:
            raise ValueError("二进制帧大小限制无效")
        self.descriptor = descriptor
        self.expected_generation = expected_generation
        self.stream_name = descriptor.sensor_id
        self.max_header_bytes = max_header_bytes
        self.max_packet_bytes = max_packet_bytes

    def parse(self, packet: bytes | bytearray | memoryview | str) -> SensorFrame:
        raw_metadata, payload_bytes = _unpack_packet(
            packet,
            max_header_bytes=self.max_header_bytes,
            max_packet_bytes=self.max_packet_bytes,
            label="传感器帧",
        )
        try:
            metadata = StreamFrameMetadata.model_validate(raw_metadata)
        except ValidationError as error:
            raise StreamProtocolError(f"传感器帧元数据不合法: {error}") from error

        if metadata.sensor_id != self.descriptor.sensor_id:
            raise StreamProtocolError(f"传感器流返回其他 sensor_id: {metadata.sensor_id}")
        if metadata.generation != self.expected_generation:
            raise GenerationMismatch(
                "传感器订阅 generation="
                f"{self.expected_generation}，收到 generation={metadata.generation}"
            )
        if metadata.encoding != self.descriptor.encoding:
            raise StreamProtocolError(
                f"传感器编码与订阅说明不一致: {metadata.encoding} != {self.descriptor.encoding}"
            )
        expected_media_type = _MEDIA_TYPES[self.descriptor.encoding]
        if metadata.media_type != expected_media_type:
            raise StreamProtocolError(
                f"传感器 media_type 不匹配: {metadata.media_type} != {expected_media_type}"
            )
        if self.descriptor.width is not None and metadata.width != self.descriptor.width:
            raise StreamProtocolError("传感器帧宽度与订阅说明不一致")
        if self.descriptor.height is not None and metadata.height != self.descriptor.height:
            raise StreamProtocolError("传感器帧高度与订阅说明不一致")

        payload = _validate_payload(metadata, payload_bytes)
        # native Runtime 当前会给 JSON Contact/Holding 帧带上渲染默认尺寸，但
        # 结构化数据没有像素网格。SDK 在协议边界去掉该无意义信息；图像和深度
        # 仍使用上面与 descriptor 严格核对后的尺寸。
        structured = metadata.encoding == "json"
        return SensorFrame(
            sensor_id=metadata.sensor_id,
            sequence=metadata.sequence,
            generation=metadata.generation,
            frame_id=metadata.frame,
            encoding=metadata.encoding,
            width=None if structured else metadata.width,
            height=None if structured else metadata.height,
            observed_at=metadata.observed_at,
            payload=payload,
        )


class BinaryRobotStateFrameParser:
    """解析 Robot State packet，并交叉核对帧头和 JSON payload。"""

    def __init__(
        self,
        robot_id: str,
        *,
        expected_generation: int,
        decode_state: Callable[[dict[str, Any]], RobotState],
        max_header_bytes: int = 64 * 1024,
        max_packet_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not robot_id:
            raise ValueError("robot_id 不能为空")
        if expected_generation < 1:
            raise ValueError("expected_generation 必须大于零")
        if max_header_bytes < 128 or max_packet_bytes <= max_header_bytes + 4:
            raise ValueError("二进制帧大小限制无效")
        self.robot_id = robot_id
        self.expected_generation = expected_generation
        self._decode_state = decode_state
        self.max_header_bytes = max_header_bytes
        self.max_packet_bytes = max_packet_bytes
        self.stream_name = f"state-{robot_id}"

    def parse(self, packet: bytes | bytearray | memoryview | str) -> SensorFrame:
        raw_metadata, payload_bytes = _unpack_packet(
            packet,
            max_header_bytes=self.max_header_bytes,
            max_packet_bytes=self.max_packet_bytes,
            label="Robot State 帧",
        )
        try:
            metadata = RobotStateFrameMetadata.model_validate(raw_metadata)
        except ValidationError as error:
            raise StreamProtocolError(f"Robot State 帧元数据不合法: {error}") from error
        if metadata.robot_id != self.robot_id:
            raise StreamProtocolError(f"状态流返回其他 Robot: {metadata.robot_id}")
        if metadata.generation != self.expected_generation:
            raise GenerationMismatch(
                "Robot State 订阅 generation="
                f"{self.expected_generation}，收到 generation={metadata.generation}"
            )
        try:
            raw_state = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StreamProtocolError("Robot State JSON payload 无法解析") from error
        if not isinstance(raw_state, dict):
            raise StreamProtocolError("Robot State JSON payload 必须是对象")
        try:
            state = self._decode_state(raw_state)
        except Exception as error:
            raise StreamProtocolError(f"Robot State JSON payload 不合法: {error}") from error
        if state.robot_id != metadata.robot_id:
            raise StreamProtocolError("Robot State 帧头与 payload 的 robot_id 不一致")
        if state.generation != metadata.generation:
            raise StreamProtocolError("Robot State 帧头与 payload 的 generation 不一致")
        if state.observed_at != metadata.observed_at:
            raise StreamProtocolError("Robot State 帧头与 payload 的 observed_at 不一致")
        if state.base_pose is not None and state.base_pose.frame_id != metadata.frame_id:
            raise StreamProtocolError("Robot State 帧头与 payload 的 frame_id 不一致")
        return SensorFrame(
            sensor_id="robot_state",
            sequence=metadata.sequence,
            generation=metadata.generation,
            frame_id=metadata.frame_id,
            encoding="json",
            observed_at=metadata.observed_at,
            payload=state.model_dump(mode="json"),
        )


class BinaryWebSocketFrameTransport:
    """后台接收 WebSocket 帧，并只保留尚未消费的最新一帧。

    物理与渲染线程不能被 Ability 的消费速度反压。生产线程覆盖单一帧槽并累计
    dropped_frames；消费方永远不会触发无界内存增长。协议错误、reset generation
    和网络断开都会成为订阅的终止错误，不会退回 HTTP 后悄悄继续。
    """

    def __init__(
        self,
        url: str,
        parser: BinarySensorFrameParser | BinaryRobotStateFrameParser,
        *,
        connector: _Connector | None = None,
        open_timeout_seconds: float = 10,
        receive_timeout_seconds: float = 0.5,
    ) -> None:
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("传感器流 URL 必须使用 ws 或 wss")
        if open_timeout_seconds <= 0 or receive_timeout_seconds <= 0:
            raise ValueError("WebSocket 超时时间必须大于零")
        if connector is None and connect is None:
            raise BackendUnavailable(
                "当前环境未安装 websockets；MuJoCo 实时流需要安装 "
                "semantic-robot-sdk-core[stream] 或使用包含该 Wheel 的 Runtime bundle"
            ) from _WEBSOCKETS_IMPORT_ERROR
        self.url = url
        self.parser = parser
        self._connector = connector or connect
        self._open_timeout_seconds = open_timeout_seconds
        self._receive_timeout_seconds = receive_timeout_seconds
        self._condition = threading.Condition()
        self._closed = threading.Event()
        self._connection: _Connection | None = None
        self._latest: SensorFrame | None = None
        self._last_received_sequence = 0
        self._last_delivered_sequence = 0
        self._dropped_frames = 0
        self._finished = False
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"robot-stream-{parser.stream_name}",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_frames(self) -> int:
        with self._condition:
            return self._dropped_frames

    def wait_terminated(self, timeout_seconds: float | None = None) -> bool:
        """等待接收线程进入终态；主要供生命周期管理和确定性测试使用。"""

        with self._condition:
            return self._condition.wait_for(
                lambda: self._finished,
                timeout=timeout_seconds,
            )

    def receive_latest(self, after_sequence: int) -> SensorFrame:
        """等待并返回比 after_sequence 更新的最新帧。"""

        with self._condition:
            while True:
                if self._closed.is_set():
                    raise StopIteration
                if self._error is not None:
                    raise self._error
                if self._latest is not None and self._latest.sequence > after_sequence:
                    self._last_delivered_sequence = self._latest.sequence
                    return self._latest.model_copy(deep=True)
                if self._finished:
                    raise StopIteration
                self._condition.wait()

    def close(self) -> None:
        self._closed.set()
        with self._condition:
            connection = self._connection
            self._condition.notify_all()
        if connection is not None:
            # close 用于打断正在阻塞的 recv；底层流是只读的，不会停止 Runtime
            # 为其他客户端共享的帧生产器。
            try:
                connection.close()
            except (ConnectionClosed, WebSocketException, OSError):
                # close 本身是幂等清理；连接已断开不应覆盖原始流错误。
                pass
        self._thread.join(timeout=self._open_timeout_seconds + 1)

    def _receive_loop(self) -> None:
        connection: _Connection | None = None
        try:
            connection = self._connector(
                self.url,
                open_timeout=self._open_timeout_seconds,
                close_timeout=1,
                max_size=self.parser.max_packet_bytes,
            )
            with self._condition:
                self._connection = connection
                if self._closed.is_set():
                    try:
                        connection.close()
                    except (ConnectionClosed, WebSocketException, OSError):
                        pass
                    return
            while not self._closed.is_set():
                try:
                    packet = connection.recv(timeout=self._receive_timeout_seconds)
                except TimeoutError:
                    continue
                frame = self.parser.parse(packet)
                with self._condition:
                    if frame.sequence == self._last_received_sequence:
                        # Runtime 在 reset 切换帧生产器时可能重发最后一帧。相同序号
                        # 只去重，不交给 Ability；真正倒退才说明流顺序已损坏。
                        continue
                    if frame.sequence < self._last_received_sequence:
                        raise StreamProtocolError(
                            "传感器帧 sequence 不能倒退: "
                            f"{frame.sequence} < {self._last_received_sequence}"
                        )
                    if (
                        self._latest is not None
                        and self._latest.sequence > self._last_delivered_sequence
                    ):
                        self._dropped_frames += 1
                    self._last_received_sequence = frame.sequence
                    self._latest = frame
                    self._condition.notify_all()
        except (GenerationMismatch, StreamProtocolError) as error:
            self._finish(error)
        except (ConnectionClosed, WebSocketException, OSError) as error:
            if not self._closed.is_set():
                self._finish(BackendUnavailable(f"传感器 WebSocket 已断开: {error}"))
        except Exception as error:
            # 后台线程不能静默退出，否则 Ability 会永久阻塞。未知传输异常转换为
            # 明确的协议错误，同时保留异常类型供日志定位。
            if not self._closed.is_set():
                self._finish(
                    StreamProtocolError(f"传感器流处理失败 ({type(error).__name__}): {error}")
                )
        finally:
            if connection is not None:
                try:
                    connection.close()
                except (ConnectionClosed, WebSocketException, OSError):
                    pass
            with self._condition:
                self._connection = None
                self._finished = True
                self._condition.notify_all()

    def _finish(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()


def _validate_payload(
    metadata: StreamFrameMetadata,
    payload: bytes,
) -> bytes | dict[str, Any]:
    if not payload:
        raise StreamProtocolError("传感器帧载荷不能为空")
    if metadata.encoding == "json":
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StreamProtocolError("结构化传感器 JSON 载荷无法解析") from error
        if not isinstance(value, dict):
            raise StreamProtocolError("结构化传感器 JSON 载荷必须是对象")
        return value
    if metadata.encoding == "jpeg":
        if metadata.width <= 0 or metadata.height <= 0:
            raise StreamProtocolError("JPEG 帧必须提供正数尺寸")
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise StreamProtocolError("JPEG 载荷缺少有效起止标记")
    elif metadata.encoding == "png16-mm":
        if metadata.width <= 0 or metadata.height <= 0:
            raise StreamProtocolError("Depth PNG 帧必须提供正数尺寸")
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise StreamProtocolError("Depth PNG 载荷签名无效")
    elif metadata.encoding == "float32-le":
        if metadata.width <= 0 or metadata.height <= 0:
            raise StreamProtocolError("float32 Depth 帧必须提供正数尺寸")
        expected_size = metadata.width * metadata.height * 4
        if len(payload) != expected_size:
            raise StreamProtocolError(
                f"float32 Depth 载荷长度错误: {len(payload)} != {expected_size}"
            )
    else:
        raise StreamProtocolError(f"不支持的传感器流编码: {metadata.encoding}")
    return payload
