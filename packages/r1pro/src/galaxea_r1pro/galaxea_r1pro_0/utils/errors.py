"""Robot SDK 对 Ability 暴露的错误类型。"""

from __future__ import annotations

from typing import Any

from .types import BackendKind


class RobotSDKError(RuntimeError):
    """Robot SDK 的基础错误。"""


class ProfileError(RobotSDKError):
    """Robot profile 缺少必要配置或包含非法值。

    与 BackendUnavailableError 风格一致，携带结构化字段（field/reason/hint）
    并支持 to_dict()，便于上层记录日志与定位配置问题。
    """

    INVALID_ROBOT_ID = "invalid_robot_id"
    INVALID_FIRMWARE = "invalid_firmware_profile"
    INVALID_BACKEND = "invalid_backend"
    INVALID_OPTIONS = "invalid_options"
    LOAD_FAILED = "profile_load_failed"
    INVALID_STRUCTURE = "invalid_profile_structure"

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str | None = None,
        value: object | None = None,
        reason: str = "invalid_profile",
        hint: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        self.hint = hint
        super().__init__(message if message is not None else self._format_message())

    def _format_message(self) -> str:
        parts = ["Robot profile 配置无效"]
        if self.field:
            parts.append(f"字段: {self.field}")
        parts.append(f"原因: {self.reason}")
        if self.hint:
            parts.append(f"建议: {self.hint}")
        return "；".join(parts)

    def to_dict(self) -> dict:
        return {
            "error": self.__class__.__name__,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "reason": self.reason,
            "hint": self.hint,
            "message": str(self),
        }


class BackendUnavailableError(RobotSDKError):
    """所选后端不可用：未接入、未连接或环境无法提供。

    原因码以类常量形式提供（如 ``BackendUnavailableError.NOT_IMPLEMENTED``），
    供上层（skill-Ability / robot-skill）分类处理；``message`` 是自动拼装的
    可读摘要，``to_dict()`` 供日志、Trace 和前端展示使用。
    """

    UNSUPPORTED_BACKEND = "unsupported_backend"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_CONNECTED = "not_connected"

    def __init__(
        self,
        message: str | None = None,
        *,
        backend: BackendKind | str | None = None,
        robot_id: str | None = None,
        reason: str = "not_implemented",
        operation: str | None = None,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.backend = backend.value if isinstance(backend, BackendKind) else backend
        self.robot_id = robot_id
        self.reason = reason
        self.operation = operation
        self.hint = hint
        self.details = details
        super().__init__(message if message is not None else self._format_message())

    def _format_message(self) -> str:
        target = []
        if self.backend:
            target.append(f"后端 {self.backend}")
        if self.robot_id:
            target.append(f"机器人 {self.robot_id}")
        parts = [(" ".join(target) + " 不可用") if target else "后端不可用"]
        parts.append(f"原因: {self.reason}")
        if self.operation:
            parts.append(f"操作: {self.operation}")
        if self.hint:
            parts.append(f"建议: {self.hint}")
        return "；".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """输出结构化错误，供日志、Trace 和前端展示。"""

        return {
            "error": self.__class__.__name__,
            "backend": self.backend,
            "robot_id": self.robot_id,
            "reason": self.reason,
            "operation": self.operation,
            "hint": self.hint,
            "message": str(self),
            **({"details": self.details} if self.details is not None else {}),
        }


class CommandNotFoundError(RobotSDKError):
    """命令 ID 不属于当前 Robot SDK 实例。

    与 BackendUnavailableError 风格一致，携带结构化字段
    （command_id / reason / hint）并支持 to_dict()。
    """

    COMMAND_NOT_FOUND = "command_not_found"

    def __init__(
        self,
        message: str | None = None,
        *,
        command_id: str | None = None,
        reason: str = "command_not_found",
        hint: str | None = None,
    ) -> None:
        self.command_id = command_id
        self.reason = reason
        self.hint = hint
        super().__init__(message if message is not None else self._format_message())

    def _format_message(self) -> str:
        parts = [f"命令 {self.command_id} 不存在" if self.command_id else "命令不存在"]
        parts.append(f"原因: {self.reason}")
        if self.hint:
            parts.append(f"建议: {self.hint}")
        return "；".join(parts)

    def to_dict(self) -> dict:
        """输出结构化错误，供日志、Trace 和前端展示。"""

        return {
            "error": self.__class__.__name__,
            "command_id": self.command_id,
            "reason": self.reason,
            "hint": self.hint,
            "message": str(self),
        }
