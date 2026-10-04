"""停止、保持与命令状态接口。"""

from ..backends.base import RobotBackend
from .types import CommandResult


class CommandClient:
    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend

    def get(self, command_id: str) -> CommandResult:
        return self._backend.get_command(command_id)

    def stop(self, command_id: str, reason: str) -> CommandResult:
        """请求停止；Isaac 可先返回 RUNNING/stopping，需继续 get 确认终态。"""
        return self._backend.stop_command(command_id, reason)


class SafetyClient:
    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend

    def hold(self, resources: tuple[str, ...], reason: str) -> bool:
        """请求保持资源；Isaac 的 True 表示接受请求，关联命令需查询停止结果。"""
        return self._backend.hold(resources, reason)
