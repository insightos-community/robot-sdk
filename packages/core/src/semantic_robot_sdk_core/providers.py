"""Robot 型号包需要实现的算法 Provider 接口。"""

from __future__ import annotations

from typing import Protocol

from .models import EnvironmentCollisionSet, MotionPlan, Pose, RobotState


class KinematicsProvider(Protocol):
    name: str

    def solve(
        self,
        *,
        robot_id: str,
        end_effector: str,
        target: Pose,
        state: RobotState,
        environment: EnvironmentCollisionSet | None = None,
    ) -> dict[str, float]: ...

    def reachable(
        self,
        *,
        end_effector: str,
        target: Pose,
        state: RobotState,
        environment: EnvironmentCollisionSet | None = None,
    ) -> bool: ...


class MotionProvider(Protocol):
    name: str

    def plan_joints(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goal: dict[str, float],
        environment: EnvironmentCollisionSet | None = None,
        speed_scale: float = 1.0,
    ) -> MotionPlan: ...


class NavigationProvider(Protocol):
    name: str

    def plan_route(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goal: Pose,
        occupancy: object | None = None,
        maximum_speed_mps: float,
        minimum_clearance_m: float | None = None,
        carrying_object_ref: str | None = None,
        carrying_object_pose: Pose | None = None,
        carrying_object_extent_m: tuple[float, float, float] | None = None,
    ) -> MotionPlan: ...
