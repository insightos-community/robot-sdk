"""R1 Pro 本地关节轨迹 Provider。"""

from __future__ import annotations

import uuid

from semantic_robot_sdk_core import (
    DependencyUnavailable,
    EnvironmentCollisionSet,
    MotionPlan,
    PlanKind,
    PlanningError,
    RobotCapabilities,
    RobotState,
)

from .local_motion_algorithms import ruckig_trajectory, smoothstep_trajectory


class LocalMotionProvider:
    name = "local"

    def __init__(
        self, capabilities: RobotCapabilities, kinematics=None, *, allow_debug_fallback=False
    ):
        self.capabilities = capabilities
        self.kinematics = kinematics
        self.allow_debug_fallback = allow_debug_fallback

    def plan_joints(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goal: dict[str, float],
        environment: EnvironmentCollisionSet | None = None,
        speed_scale: float = 1.0,
    ) -> MotionPlan:
        if not 0 < speed_scale <= 1:
            raise PlanningError("speed_scale 必须在 (0, 1] 范围内")
        limits = {
            name: limit.model_copy(update={"max_velocity": limit.max_velocity * speed_scale})
            for name, limit in self.capabilities.joint_limits.items()
        }
        diagnostics: list[str] = []
        try:
            points = ruckig_trajectory(state.joint_positions, goal, limits)
            planner = "ruckig"
        except DependencyUnavailable:
            if not self.allow_debug_fallback:
                raise
            points = smoothstep_trajectory(state.joint_positions, goal, limits)
            planner = "bounded-smoothstep-fake"
            diagnostics.append("Fake 测试使用受限 smoothstep；不作为真机规划结果")
        collision_checked = False
        if self.kinematics is not None:
            for point in points:
                candidate = dict(state.joint_positions)
                candidate.update(point.positions)
                if not self.kinematics.collision_free(candidate, environment):
                    raise PlanningError("关节路径存在碰撞")
            collision_checked = True
        elif not self.allow_debug_fallback:
            raise PlanningError("未配置碰撞检查器")
        resources = [
            f"joints:{group}"
            for group, names in self.capabilities.joint_groups.items()
            if set(goal).intersection(names)
        ] or ["joints"]
        return MotionPlan(
            plan_id=str(uuid.uuid4()),
            robot_id=robot_id,
            generation=state.generation,
            kind=PlanKind.JOINT,
            resources=resources,
            frame_id=self.capabilities.coordinate_frame,
            start={name: state.joint_positions[name] for name in goal},
            goal=dict(goal),
            joint_trajectory=points,
            collision_checked=collision_checked,
            estimated_duration_s=points[-1].time_from_start_s,
            planner=planner,
            diagnostics=diagnostics,
        )
