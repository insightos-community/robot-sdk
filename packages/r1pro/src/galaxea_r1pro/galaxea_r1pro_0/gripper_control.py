"""Gripper_Control 能力：夹爪控制。

开度为 0–100；速度为位置目标轨迹速度，力度为驱动力上限，均为模型上限的 0–1 比例。
接触时的实际夹持力由物理仿真决定；慢速轨迹不降低模型的关节硬限速。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .backends.base import RobotBackend
from .utils.types import CommandHandle


class GripperControl:
    """R1 Pro 夹爪控制。"""

    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend

    def control_single_gripper(
        self, *, arm_side: str, position: float, speed: float = 0.5, effort: float = 0.3
    ) -> CommandHandle:
        return self._backend.start_command(
            "gripper_control",
            "control_single_gripper",
            {"side": arm_side, "position": position, "speed": speed, "effort": effort},
        )

    def control_both_gripper(
        self,
        *,
        left_position: float = 50,
        right_position: float = 50,
        left_speed: float = 0.5,
        right_speed: float = 0.5,
        left_effort: float = 0.3,
        right_effort: float = 0.3,
    ) -> list:
        """双夹爪控制：左右夹爪异步多线程并行提交（借鉴 Atom）。"""

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self.control_single_gripper,
                    arm_side=side,
                    position=position,
                    speed=speed,
                    effort=effort,
                )
                for side, position, speed, effort in (
                    ("left", left_position, left_speed, left_effort),
                    ("right", right_position, right_speed, right_effort),
                )
            ]
            return [future.result() for future in futures]
