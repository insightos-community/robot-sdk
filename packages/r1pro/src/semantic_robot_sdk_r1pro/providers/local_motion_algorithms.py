"""关节路径的时间参数化。"""

from __future__ import annotations

import math

from semantic_robot_sdk_core.errors import PlanningError
from semantic_robot_sdk_core.models import JointLimit, JointTrajectoryPoint
from .dependencies import require_ruckig


JOINT_LIMIT_NUMERICAL_TOLERANCE_RAD = 1e-4


def validate_joint_goal(
    current: dict[str, float], goal: dict[str, float], limits: dict[str, JointLimit]
) -> None:
    if not goal:
        raise PlanningError("关节目标不能为空")
    unknown = set(goal) - set(current)
    if unknown:
        raise PlanningError(f"Robot 当前状态不包含关节：{sorted(unknown)}")
    missing_limits = set(goal) - set(limits)
    if missing_limits:
        raise PlanningError(f"Robot Profile 缺少关节限制：{sorted(missing_limits)}")
    for name, value in goal.items():
        limit = limits[name]
        # 数值IK可能返回仅比关节限位多几个微弧度的解。该误差远小于控制器
        # 分辨率，不应让整段动作规划失败；在100µrad内直接夹紧到限位，真正
        # 超界的目标仍保持拒绝。原地修改可同时修正后续Ruckig输入和碰撞采样。
        if limit.lower - JOINT_LIMIT_NUMERICAL_TOLERANCE_RAD <= value < limit.lower:
            goal[name] = limit.lower
        elif limit.upper < value <= limit.upper + JOINT_LIMIT_NUMERICAL_TOLERANCE_RAD:
            goal[name] = limit.upper
        elif value < limit.lower or value > limit.upper:
            raise PlanningError(f"关节 {name} 目标 {value} 超出 [{limit.lower}, {limit.upper}]")


def smoothstep_trajectory(
    current: dict[str, float],
    goal: dict[str, float],
    limits: dict[str, JointLimit],
    *,
    sample_period_s: float = 0.05,
) -> list[JointTrajectoryPoint]:
    """显式调试回退。

    五次 smoothstep 保证起止速度和加速度为零；持续时间按速度、加速度和
    加加速度限制保守放大。它不替代 Ruckig，生产调用必须使用后者。
    """

    validate_joint_goal(current, goal, limits)
    duration = sample_period_s
    for name, target in goal.items():
        delta = abs(target - current[name])
        limit = limits[name]
        duration = max(
            duration,
            2.0 * delta / limit.max_velocity,
            math.sqrt(6.0 * delta / limit.max_acceleration) if delta else 0.0,
            (60.0 * delta / limit.max_jerk) ** (1.0 / 3.0) if delta else 0.0,
        )
    count = max(2, math.ceil(duration / sample_period_s) + 1)
    names = list(goal)
    points: list[JointTrajectoryPoint] = []
    for index in range(count):
        ratio = index / (count - 1)
        blend = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
        derivative = 30 * ratio**2 - 60 * ratio**3 + 30 * ratio**4
        positions = {name: current[name] + (goal[name] - current[name]) * blend for name in names}
        velocities = {name: (goal[name] - current[name]) * derivative / duration for name in names}
        points.append(
            JointTrajectoryPoint(
                time_from_start_s=duration * ratio,
                positions=positions,
                velocities=velocities,
            )
        )
    return points


def ruckig_trajectory(
    current: dict[str, float],
    goal: dict[str, float],
    limits: dict[str, JointLimit],
    *,
    current_velocity: dict[str, float] | None = None,
    target_velocity: dict[str, float] | None = None,
) -> list[JointTrajectoryPoint]:
    """使用 Ruckig 生成满足速度、加速度和加加速度限制的轨迹。"""

    validate_joint_goal(current, goal, limits)
    require_ruckig()
    from ruckig import InputParameter, Result, Ruckig, Trajectory  # type: ignore

    names = list(goal)
    dofs = len(names)
    otg = Ruckig(dofs, 0.01)
    inp = InputParameter(dofs)
    inp.current_position = [current[name] for name in names]
    current_velocity = current_velocity or {}
    target_velocity = target_velocity or {}
    inp.current_velocity = [current_velocity.get(name, 0.0) for name in names]
    inp.current_acceleration = [0.0] * dofs
    inp.target_position = [goal[name] for name in names]
    inp.target_velocity = [target_velocity.get(name, 0.0) for name in names]
    inp.target_acceleration = [0.0] * dofs
    inp.max_velocity = [limits[name].max_velocity for name in names]
    inp.max_acceleration = [limits[name].max_acceleration for name in names]
    inp.max_jerk = [limits[name].max_jerk for name in names]
    trajectory = Trajectory(dofs)
    result = otg.calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise PlanningError(f"Ruckig 轨迹生成失败：{result}")
    if trajectory.duration <= 1e-9:
        # 候选切换或恢复动作可能要求 Robot 保持已经到达的关节位置。Ruckig
        # 会把这种合法的幂等目标计算为零时长；仍按通用 MotionPlan 契约输出
        # 一个控制周期的静止轨迹，避免产生两个 t=0 的非法轨迹点。
        zero_velocity = {name: 0.0 for name in names}
        return [
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={name: current[name] for name in names},
                velocities=zero_velocity,
            ),
            JointTrajectoryPoint(
                time_from_start_s=0.01,
                positions={name: goal[name] for name in names},
                velocities=zero_velocity,
            ),
        ]
    count = max(2, math.ceil(trajectory.duration / 0.05) + 1)
    points: list[JointTrajectoryPoint] = []
    for index in range(count):
        at = min(trajectory.duration, index * trajectory.duration / (count - 1))
        positions, velocities, _accelerations = trajectory.at_time(at)
        points.append(
            JointTrajectoryPoint(
                time_from_start_s=at,
                positions=dict(zip(names, positions, strict=True)),
                velocities=dict(zip(names, velocities, strict=True)),
            )
        )
    return points
