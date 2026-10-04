"""Robot 型号相关的运动学求解接口。"""

from __future__ import annotations

from typing import Protocol

from .models import Pose


class KinematicsSolver(Protocol):
    """实现者通常由 Pinocchio Robot 模型提供。

    `solve_end_effector` 必须检查不可达、奇异位形和关节限位；
    `collision_free` 必须同时检查自碰撞和传入环境碰撞几何。
    """

    def solve_end_effector(
        self,
        end_effector: str,
        target: Pose,
        current_joints: dict[str, float],
    ) -> dict[str, float]: ...

    def collision_free(
        self,
        joint_positions: dict[str, float],
        environment: object | None,
    ) -> bool: ...
