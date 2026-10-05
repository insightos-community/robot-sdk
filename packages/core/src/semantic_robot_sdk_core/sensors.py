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

"""Robot 传感器订阅的公共实现。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from .errors import GenerationMismatch
from .models import RobotState, SensorFrame

if TYPE_CHECKING:
    from .stream_transport import BinaryWebSocketFrameTransport


class SensorSubscription(Iterator[SensorFrame]):
    """可由 Ability 主动关闭的传感器订阅。

    本实现供 FakeBackend 和显式轮询适配器使用；RuntimeHTTPBackend 使用下方的
    StreamingSensorSubscription。轮询同样不建立无界队列，消费方较慢时会自然跳过
    中间帧，只返回比上一帧新的序号。reset 后 generation 改变，
    原订阅立即失败，调用方必须读取新状态并重新订阅，防止旧画面参与新场景决策。
    """

    def __init__(
        self,
        read_latest: Callable[[], SensorFrame],
        *,
        expected_generation: int,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if expected_generation < 1:
            raise ValueError("expected_generation 必须大于零")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds 不能小于零")
        self._read_latest = read_latest
        self._expected_generation = expected_generation
        self._poll_interval_seconds = poll_interval_seconds
        self._closed = threading.Event()
        self._last_sequence = 0

    def __iter__(self) -> SensorSubscription:
        return self

    def __next__(self) -> SensorFrame:
        while not self._closed.is_set():
            frame = self._read_latest()
            if frame.generation != self._expected_generation:
                raise GenerationMismatch(
                    "传感器订阅 generation="
                    f"{self._expected_generation}，最新帧 generation={frame.generation}"
                )
            if frame.sequence > self._last_sequence:
                self._last_sequence = frame.sequence
                return frame
            self._closed.wait(self._poll_interval_seconds)
        raise StopIteration

    def close(self) -> None:
        """解除本地订阅等待；不会停止 Runtime 中由多个客户端共享的帧生产器。"""

        self._closed.set()

    def __enter__(self) -> SensorSubscription:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class StreamingSensorSubscription(SensorSubscription):
    """由二进制 WebSocket transport 驱动的传感器订阅。

    本类保持 Ability 已使用的迭代器/close 接口。WebSocket 路径、协议头以及慢客户端
    丢帧策略全部留在 Backend 层；调用方只能获得已经校验过的 SensorFrame。
    """

    def __init__(self, transport: BinaryWebSocketFrameTransport) -> None:
        self._transport = transport
        self._closed = threading.Event()
        self._last_sequence = 0

    @property
    def dropped_frames(self) -> int:
        """返回消费方来不及读取而被最新帧覆盖的数量。"""

        return self._transport.dropped_frames

    def __next__(self) -> SensorFrame:
        if self._closed.is_set():
            raise StopIteration
        frame = self._transport.receive_latest(self._last_sequence)
        self._last_sequence = frame.sequence
        return frame

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._transport.close()


class RobotStateSubscription(Iterator[RobotState]):
    """可关闭的 Robot State 连续订阅。

    本实现只供 FakeBackend 与显式 legacy_state_polling 使用。网络错误会终止订阅，
    reset 后 generation 改变也会终止旧订阅，不能把两个场景状态拼接起来。
    正常 RuntimeHTTPBackend 使用 StreamingRobotStateSubscription。
    """

    def __init__(
        self,
        read_state: Callable[[], RobotState],
        *,
        expected_generation: int,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        if expected_generation < 1:
            raise ValueError("expected_generation 必须大于零")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds 不能小于零")
        self._read_state = read_state
        self._expected_generation = expected_generation
        self._poll_interval_seconds = poll_interval_seconds
        self._closed = threading.Event()
        self._first = True

    def __iter__(self) -> RobotStateSubscription:
        return self

    def __next__(self) -> RobotState:
        if self._closed.is_set():
            raise StopIteration
        if self._first:
            self._first = False
        elif self._closed.wait(self._poll_interval_seconds):
            raise StopIteration
        try:
            state = self._read_state()
            if state.generation != self._expected_generation:
                raise GenerationMismatch(
                    "Robot State 订阅 generation="
                    f"{self._expected_generation}，当前 generation={state.generation}"
                )
            return state
        except Exception:
            # 订阅错误是终态。调用方需要显式重建订阅，不能在断连或 reset 后
            # 无声恢复，从而把两个场景 generation 的状态拼在一起。
            self._closed.set()
            raise

    def close(self) -> None:
        self._closed.set()

    def __enter__(self) -> RobotStateSubscription:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class StreamingRobotStateSubscription(RobotStateSubscription):
    """通过共享 latest-frame transport 消费 Robot State，不做 HTTP 回退。"""

    def __init__(self, transport: BinaryWebSocketFrameTransport) -> None:
        self._transport = transport
        self._closed = threading.Event()
        self._last_sequence = 0

    @property
    def dropped_frames(self) -> int:
        return self._transport.dropped_frames

    def __next__(self) -> RobotState:
        if self._closed.is_set():
            raise StopIteration
        frame = self._transport.receive_latest(self._last_sequence)
        self._last_sequence = frame.sequence
        # Parser 已交叉核对帧头和 Runtime payload；这里仅恢复公共类型。
        return RobotState.model_validate(frame.payload)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._transport.close()
