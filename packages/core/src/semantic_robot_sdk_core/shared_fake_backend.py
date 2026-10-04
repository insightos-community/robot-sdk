"""供多个 Ability 进程共享的 Fake Robot Backend。"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import RLock
from typing import TypeVar

from .fake_backend import FakeBackend, FakeBackendSnapshot
from .models import Command, Feedback, MotionPlan, RobotCapabilities, RobotState, SensorFrame
from .sensors import RobotStateSubscription, SensorSubscription

T = TypeVar("T")


class SQLiteFakeStateStore:
    """用一个 SQLite 文档保存 Fake Robot 状态。

    每次 SDK 调用只持有一次 ``BEGIN IMMEDIATE`` 事务。事务覆盖读取当前状态、执行
    一个 Backend 操作和写回状态三个步骤，所以资源互斥和 command_id 去重在多进程
    下仍然成立。这里不实现机器人行为，所有行为继续复用 ``FakeBackend``。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local_lock = RLock()
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS fake_robot_state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "payload TEXT NOT NULL)"
            )

    def transact(
        self,
        backend_factory: Callable[[], FakeBackend],
        operation: Callable[[FakeBackend], T],
    ) -> T:
        # 同一进程内先串行，跨进程再由 SQLite 写事务串行。这样不会把数据库锁泄漏给
        # Ability，也不会要求 Ability 知道 Robot 是内存 Fake 还是共享 Fake。
        with self._local_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM fake_robot_state WHERE singleton = 1"
            ).fetchone()
            backend = backend_factory()
            if row is not None:
                backend.restore_snapshot(FakeBackendSnapshot.model_validate_json(row[0]))
            result = operation(backend)
            connection.execute(
                "INSERT INTO fake_robot_state(singleton, payload) VALUES (1, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET payload = excluded.payload",
                (backend.export_snapshot().model_dump_json(),),
            )
            connection.commit()
            return result

    def initialize(
        self,
        backend_factory: Callable[[], FakeBackend],
        operation: Callable[[FakeBackend], None],
    ) -> bool:
        """只在共享状态第一次建立时写入初态。

        七个 Ability 会在不同进程中几乎同时创建 Robot SDK。如果每个进程都再次
        注入 Fake 场景初态，重启某个 Ability 就可能覆盖已经抓取、移动或放置后的
        状态。这里把“是否首次创建”的判断和初态写入放进同一个 SQLite 写事务，
        保证只有第一个进程负责初始化，后续进程只复用现有 Robot 事实。
        """

        with self._local_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM fake_robot_state WHERE singleton = 1"
            ).fetchone()
            if existing is not None:
                connection.commit()
                return False
            backend = backend_factory()
            operation(backend)
            connection.execute(
                "INSERT INTO fake_robot_state(singleton, payload) VALUES (1, ?)",
                (backend.export_snapshot().model_dump_json(),),
            )
            connection.commit()
            return True

    def read(self, backend_factory: Callable[[], FakeBackend]) -> FakeBackendSnapshot:
        return self.transact(backend_factory, lambda backend: backend.export_snapshot())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


class SharedFakeBackend:
    """保持 ``RobotBackend`` 接口不变的共享 Fake 包装器。"""

    def __init__(
        self,
        capabilities: RobotCapabilities,
        initial_state: RobotState,
        state_path: str | Path,
        *,
        auto_complete: bool = True,
        stop_confirmed: bool = True,
        initializer: Callable[[FakeBackend], None] | None = None,
    ) -> None:
        self._capabilities = capabilities.model_copy(deep=True)
        self._initial_state = initial_state.model_copy(deep=True)
        self.auto_complete = auto_complete
        self.stop_confirmed = stop_confirmed
        self._store = SQLiteFakeStateStore(state_path)
        self._store.initialize(self._new_backend, initializer or (lambda _backend: None))

    def _new_backend(self) -> FakeBackend:
        return FakeBackend(
            self._capabilities,
            self._initial_state,
            auto_complete=self.auto_complete,
            stop_confirmed=self.stop_confirmed,
        )

    def _call(self, operation: Callable[[FakeBackend], T]) -> T:
        return self._store.transact(self._new_backend, operation)

    @property
    def executions(self) -> list[str]:
        return list(self._store.read(self._new_backend).executions)

    def capabilities(self) -> RobotCapabilities:
        return self._capabilities.model_copy(deep=True)

    def configure_grasp_fixture(self, gripper: str, object_id: str, **options) -> None:
        self._call(lambda backend: backend.configure_grasp_fixture(gripper, object_id, **options))

    def clear_grasp_fixture(self, gripper: str) -> None:
        self._call(lambda backend: backend.clear_grasp_fixture(gripper))

    def set_tool_contact_state(self, object_id: str | None, **options) -> None:
        self._call(lambda backend: backend.set_tool_contact_state(object_id, **options))

    def state(self, robot_id: str) -> RobotState:
        return self._call(lambda backend: backend.state(robot_id))

    def execute_plan(self, plan: MotionPlan, command_id: str) -> Command:
        return self._call(lambda backend: backend.execute_plan(plan, command_id))

    def complete_command(self, command_id: str) -> Command:
        return self._call(lambda backend: backend.complete_command(command_id))

    def interrupt_command(self, command_id: str, reason: str = "connection lost") -> Command:
        return self._call(lambda backend: backend.interrupt_command(command_id, reason))

    def command(self, robot_id: str, command_id: str) -> Command:
        return self._call(lambda backend: backend.command(robot_id, command_id))

    def feedback(self, robot_id: str, command_id: str) -> Iterator[Feedback]:
        values = self._call(lambda backend: list(backend.feedback(robot_id, command_id)))
        yield from values

    def stop(self, robot_id: str, command_id: str) -> Command:
        return self._call(lambda backend: backend.stop(robot_id, command_id))

    def hold(self, robot_id: str) -> None:
        self._call(lambda backend: backend.hold(robot_id))

    def sensors(self, robot_id: str) -> list[str]:
        return self._call(lambda backend: backend.sensors(robot_id))

    def latest_sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame:
        return self._call(lambda backend: backend.latest_sensor_frame(robot_id, sensor_id))

    def publish_sensor_frame(self, frame: SensorFrame) -> None:
        self._call(lambda backend: backend.publish_sensor_frame(frame))

    def subscribe_sensor(self, robot_id: str, sensor_id: str, *, poll_interval_seconds=0.05):
        state = self.state(robot_id)
        return SensorSubscription(
            lambda: self.latest_sensor_frame(robot_id, sensor_id),
            expected_generation=state.generation,
            poll_interval_seconds=poll_interval_seconds,
        )

    def subscribe_state(self, robot_id: str, *, poll_interval_seconds=0.1):
        state = self.state(robot_id)
        return RobotStateSubscription(
            lambda: self.state(robot_id),
            expected_generation=state.generation,
            poll_interval_seconds=poll_interval_seconds,
        )

    def close(self) -> None:
        return None
