"""左右夹爪控制入口。"""

import uuid

from semantic_robot_sdk_core import GripperTarget, MotionPlan, PlanKind, PlanningError


class EndEffectorModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def plan(self, side, opening_m, *, force_limit_n=20, stop_on_contact=False):
        state = self.backend.state(self.robot_id)
        profile = self.backend.capabilities()
        if side not in profile.grippers:
            raise PlanningError(f"Robot 不包含夹爪：{side}")
        if opening_m < 0 or force_limit_n <= 0:
            raise PlanningError("夹爪开度不能为负，力限制必须大于零")
        descriptor = next((tool for tool in profile.tools if tool.side == side), None)
        if descriptor is not None and force_limit_n > descriptor.maximum_force_n:
            raise PlanningError(
                f"夹具 {side} 力限制 {force_limit_n}N 超过Profile峰值 {descriptor.maximum_force_n}N"
            )
        return MotionPlan(
            plan_id=str(uuid.uuid4()),
            robot_id=self.robot_id,
            generation=state.generation,
            kind=PlanKind.GRIPPER,
            resources=[f"gripper:{side}"],
            frame_id=profile.coordinate_frame,
            start={"opening_m": state.gripper_openings.get(side)},
            goal={"opening_m": opening_m, "force_limit_n": force_limit_n},
            gripper_command=GripperTarget(
                gripper=side,
                opening_m=opening_m,
                force_limit_n=force_limit_n,
                stop_on_contact=stop_on_contact,
            ),
            collision_checked=True,
            # tote_clamp 的完整行程是 35 mm；该值只用于生成低层命令的
            # 截止时间，不判断接触是否持续、是否稳定承载或抓取是否成功。
            # 这些带任务语义的判断由 Ability 使用连续实时状态完成。
            estimated_duration_s=8.0,
            planner="r1pro-gripper",
        )

    def set_opening(self, side, opening_m, *, command_id, force_limit_n=20, stop_on_contact=False):
        return self.backend.execute_plan(
            self.plan(
                side,
                opening_m,
                force_limit_n=force_limit_n,
                stop_on_contact=stop_on_contact,
            ),
            command_id,
        )

    def close_until_contact(self, side, target_opening_m=0, *, command_id, force_limit_n):
        return self.set_opening(
            side,
            target_opening_m,
            command_id=command_id,
            force_limit_n=force_limit_n,
            stop_on_contact=True,
        )

    def release(self, side, *, command_id, opening_m=0.08):
        return self.set_opening(side, opening_m, command_id=command_id)
