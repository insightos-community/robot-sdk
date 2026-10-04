"""R1 Pro 本地关节轨迹 Provider。"""

from __future__ import annotations

import math
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
from semantic_robot_sdk_core.transforms import relative_collision_set

from .local_motion_algorithms import (
    ruckig_trajectory,
    smoothstep_trajectory,
    validate_joint_goal,
)


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
        limits = self._scaled_limits(speed_scale)
        points, planner, diagnostics = self._plan_segment(state.joint_positions, goal, limits)
        collision_checked = self._check_collision(state, points, environment=environment)
        return MotionPlan(
            plan_id=str(uuid.uuid4()),
            robot_id=robot_id,
            generation=state.generation,
            kind=PlanKind.JOINT,
            resources=self._resources(goal),
            frame_id=self.capabilities.coordinate_frame,
            start={name: state.joint_positions[name] for name in goal},
            goal=dict(goal),
            joint_trajectory=points,
            collision_checked=collision_checked,
            estimated_duration_s=points[-1].time_from_start_s,
            planner=planner,
            diagnostics=diagnostics,
        )

    def plan_joint_waypoints(
        self,
        *,
        robot_id: str,
        state: RobotState,
        goals: list[dict[str, float]],
        environment: EnvironmentCollisionSet | None = None,
        speed_scale: float = 1.0,
    ) -> MotionPlan:
        """把多个已求解关节路点组合为一条有界轨迹。

        多末端搬运不能只对最终位姿求一次 IK 后做关节直插值：双臂在中间
        时刻可能破坏所夹物体的刚性约束。调用方先为同步笛卡尔路点联合求解
        IK，本方法再逐段做 Ruckig 时间参数化。内部路点只在通过速度不会令
        任一关节越过或反向离开当前几何段时连续通过；否则按数值可行性降低
        路点速度，而不是让几十个短段产生无物理意义的往返运动。
        """
        if not goals:
            raise PlanningError("关节路点不能为空")
        names = set(goals[0])
        if not names or any(set(goal) != names for goal in goals):
            raise PlanningError("同一条多段关节路径必须控制相同关节")

        limits = self._scaled_limits(speed_scale)
        base_velocities = self._waypoint_velocities(dict(state.joint_positions), goals, limits)
        selected = None
        last_error: PlanningError | None = None
        # Ruckig 对短段和非零边界速度都满足速度/加速度/jerk限制，但某些
        # 数值可行解会先越过路点再折返。固定末端路径不能接受这种解。
        # 这里仅缩放内部路点速度并重新做时间参数化，不重算IK、不重试物理动作。
        for velocity_scale in (1.0, 0.5, 0.25, 0.0):
            current = dict(state.joint_positions)
            current_velocity = {name: 0.0 for name in names}
            points = []
            diagnostics: list[str] = []
            planners: list[str] = []
            offset = 0.0
            effective_goals: list[dict[str, float]] = []
            follows_geometry = True
            try:
                for goal_index, goal in enumerate(goals):
                    if max(abs(goal[name] - current[name]) for name in names) <= 1e-9:
                        continue
                    target_velocity = {
                        name: value * velocity_scale
                        for name, value in base_velocities[goal_index].items()
                    }
                    segment, planner, segment_diagnostics = self._plan_segment(
                        current,
                        goal,
                        limits,
                        current_velocity=current_velocity,
                        target_velocity=target_velocity,
                    )
                    if not self._segment_follows_waypoint_direction(current, goal, segment):
                        follows_geometry = False
                        break
                    planners.append(planner)
                    diagnostics.extend(segment_diagnostics)
                    for index, point in enumerate(segment):
                        if points and index == 0:
                            continue
                        points.append(
                            point.model_copy(
                                update={"time_from_start_s": (offset + point.time_from_start_s)}
                            )
                        )
                    offset = points[-1].time_from_start_s
                    current.update(goal)
                    current_velocity = target_velocity
                    effective_goals.append(dict(goal))
            except PlanningError as error:
                last_error = error
                follows_geometry = False
            if follows_geometry and points and effective_goals:
                if velocity_scale < 1.0:
                    diagnostics.append(f"内部路点速度缩放为 {velocity_scale:g}，避免短段越过后折返")
                selected = (
                    points,
                    diagnostics,
                    planners,
                    effective_goals,
                )
                break

        if selected is None:
            if last_error is not None:
                raise last_error
            raise PlanningError("关节路点无法生成保持几何方向的受限轨迹")
        points, diagnostics, planners, effective_goals = selected

        # 只有整条多段路径的终点必须停稳。Ruckig终点可能残留约1e-18的
        # 浮点速度，这里按公共轨迹契约明确归零。
        points[-1] = points[-1].model_copy(update={"velocities": {name: 0.0 for name in names}})
        collision_checked = self._check_collision(state, points, environment=environment)
        return MotionPlan(
            plan_id=str(uuid.uuid4()),
            robot_id=robot_id,
            generation=state.generation,
            kind=PlanKind.JOINT,
            resources=self._resources(effective_goals[-1]),
            frame_id=self.capabilities.coordinate_frame,
            start={name: state.joint_positions[name] for name in names},
            goal=dict(effective_goals[-1]),
            joint_trajectory=points,
            collision_checked=collision_checked,
            estimated_duration_s=points[-1].time_from_start_s,
            planner="cartesian-waypoints+" + "+".join(dict.fromkeys(planners)),
            diagnostics=diagnostics,
        )

    def retime_projected_path(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        environment: EnvironmentCollisionSet | None = None,
        speed_scale: float = 1.0,
        waypoint_tolerance_rad: float = 1e-3,
    ) -> MotionPlan:
        """简化固定末端投影路径，并恢复经过物理验证的连续时间轨迹。"""

        points = plan.joint_trajectory
        if len(points) < 2:
            raise PlanningError("投影后的关节路径至少需要两个轨迹点")
        names = set(points[0].positions)
        if any(set(point.positions) != names for point in points):
            raise PlanningError("投影后的关节路径关节集合不一致")
        limits = self._scaled_limits(speed_scale)
        for point in points:
            validate_joint_goal(state.joint_positions, point.positions, limits)

        projected_goals = _simplify_joint_waypoints(
            [dict(point.positions) for point in points],
            tolerance_rad=waypoint_tolerance_rad,
        )
        if (
            max(abs(projected_goals[0][name] - state.joint_positions[name]) for name in names)
            <= 1e-9
        ):
            projected_goals = projected_goals[1:]
        if not projected_goals:
            raise PlanningError("投影路径简化后没有有效运动目标")

        # 固定一侧、移动另一侧的接合路径已经用该方式完成过真实外拉抓取。
        # 它不等同于双侧携物的刚性相对几何：数值投影点可先按容差简化，再由
        # Ruckig形成连续关节轨迹，UpperBody随后仍用真实FK验证固定端误差。
        retimed = self.plan_joint_waypoints(
            robot_id=plan.robot_id,
            state=state,
            goals=projected_goals,
            environment=environment,
            speed_scale=speed_scale,
        )
        return retimed.model_copy(
            update={
                "planner": f"{plan.planner}+retimed+simplified-ruckig",
                "diagnostics": [
                    *plan.diagnostics,
                    *retimed.diagnostics,
                    f"投影路径从 {len(points)} 个数值采样简化为 {len(projected_goals)} 个关节路点",
                ],
            }
        )

    def retime_geometry_preserving_path(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        environment: EnvironmentCollisionSet | None = None,
        speed_scale: float = 1.0,
    ) -> MotionPlan:
        """不改变双侧携物投影几何，只按关节约束统一拉伸时间。"""

        points = plan.joint_trajectory
        if len(points) < 2:
            raise PlanningError("投影后的关节路径至少需要两个轨迹点")
        names = set(points[0].positions)
        if any(set(point.positions) != names for point in points):
            raise PlanningError("投影后的关节路径关节集合不一致")
        limits = self._scaled_limits(speed_scale)
        for point in points:
            validate_joint_goal(state.joint_positions, point.positions, limits)

        time_scale = _projected_path_time_scale(points, limits)
        retimed_points = [
            point.model_copy(
                update={
                    "time_from_start_s": point.time_from_start_s * time_scale,
                    "velocities": {},
                }
            )
            for point in points
        ]
        stopped = {name: 0.0 for name in names}
        retimed_points[0] = retimed_points[0].model_copy(update={"velocities": stopped})
        retimed_points[-1] = retimed_points[-1].model_copy(update={"velocities": stopped})
        collision_checked = self._check_collision(
            state,
            _linear_joint_path_samples(retimed_points),
            environment=environment,
        )
        return plan.model_copy(
            update={
                "joint_trajectory": retimed_points,
                "goal": dict(retimed_points[-1].positions),
                "collision_checked": collision_checked,
                "estimated_duration_s": retimed_points[-1].time_from_start_s,
                "planner": f"{plan.planner}+retimed-uniform",
                "diagnostics": [
                    *plan.diagnostics,
                    f"双侧携物投影路径按关节约束统一放慢 {time_scale:.3f} 倍",
                ],
            }
        )

    def _scaled_limits(self, speed_scale: float):
        if not 0 < speed_scale <= 1:
            raise PlanningError("speed_scale 必须在 (0, 1] 范围内")
        return {
            name: limit.model_copy(
                update={
                    "max_velocity": limit.max_velocity * speed_scale,
                    "max_acceleration": limit.max_acceleration * speed_scale**2,
                    "max_jerk": limit.max_jerk * speed_scale**3,
                }
            )
            for name, limit in self.capabilities.joint_limits.items()
        }

    def _plan_segment(
        self,
        current,
        goal,
        limits,
        *,
        current_velocity=None,
        target_velocity=None,
    ):
        diagnostics: list[str] = []
        try:
            return (
                ruckig_trajectory(
                    current,
                    goal,
                    limits,
                    current_velocity=current_velocity,
                    target_velocity=target_velocity,
                ),
                "ruckig",
                diagnostics,
            )
        except DependencyUnavailable:
            if not self.allow_debug_fallback:
                raise
            diagnostics.append("Fake 测试使用受限 smoothstep；不作为真机规划结果")
            return (
                smoothstep_trajectory(current, goal, limits),
                "bounded-smoothstep-fake",
                diagnostics,
            )

    @staticmethod
    def _waypoint_velocities(current, goals, limits):
        """为内部关节路点生成保守的连续通过速度。

        同一关节在路点前后变向时必须停下；方向一致时只使用其最大速度的
        25%，并按相邻两段较短位移限制速度。该规则只依赖 Robot Profile 的
        物理限位，不包含周转箱或场景特例。最后一个目标始终以零速度结束。
        """
        positions = [current, *goals]
        velocities: list[dict[str, float]] = []
        for index, goal in enumerate(goals, start=1):
            if index == len(positions) - 1:
                velocities.append({name: 0.0 for name in goal})
                continue
            previous = positions[index - 1]
            following = positions[index + 1]
            target: dict[str, float] = {}
            for name in goal:
                incoming = goal[name] - previous[name]
                outgoing = following[name] - goal[name]
                if incoming * outgoing <= 0:
                    target[name] = 0.0
                    continue
                direction = 1.0 if incoming > 0 else -1.0
                distance = min(abs(incoming), abs(outgoing))
                limit = limits[name]
                # 路点通过速度既不能超过关节速度上限，也必须能在相邻短段内
                # 由受限加速度和 jerk 建立起来。这里不能用“位移乘固定常数”
                # 作为速度：固定常数不会随 speed_scale 缩放，会破坏 Ruckig
                # 的时间缩放关系，让低速轨迹反而出现数百秒的异常耗时。
                acceleration_bound = (limit.max_acceleration * distance) ** 0.5
                jerk_bound = (limit.max_jerk * distance**2) ** (1.0 / 3.0)
                target[name] = direction * min(
                    limit.max_velocity * 0.25,
                    acceleration_bound,
                    jerk_bound,
                )
            velocities.append(target)
        return velocities

    @staticmethod
    def _segment_follows_waypoint_direction(current, goal, points) -> bool:
        """拒绝Ruckig为满足过高边界速度而生成的越点折返。"""

        tolerance = 1e-8
        for name, target in goal.items():
            start = current[name]
            direction = target - start
            previous = start
            lower = min(start, target) - tolerance
            upper = max(start, target) + tolerance
            for point in points:
                value = point.positions[name]
                if value < lower or value > upper:
                    return False
                if direction > tolerance and value + tolerance < previous:
                    return False
                if direction < -tolerance and value - tolerance > previous:
                    return False
                if abs(direction) <= tolerance and abs(value - start) > tolerance:
                    return False
                previous = value
        return True

    def _check_collision(self, state, points, *, environment):
        if self.kinematics is None:
            if not self.allow_debug_fallback:
                raise PlanningError("未配置碰撞检查器")
            return False
        collision = environment
        root_frame = self.capabilities.kinematic_root_frame
        if environment is not None and environment.frame_id != root_frame:
            if state.base_pose is None or environment.frame_id != state.base_pose.frame_id:
                raise PlanningError(
                    f"环境碰撞快照无法从 {environment.frame_id} 转换到 {root_frame}"
                )
            # 终点 IK 和轨迹逐点检查必须使用同一坐标边界。场景快照以 world
            # 给出，而 Pinocchio collision model 位于 base_link；若这里只把
            # world 环境直接传下去，安全路径会被错误拒绝，或更糟地绕过检查。
            collision = relative_collision_set(
                state.base_pose,
                environment,
                result_frame_id=root_frame,
            )
        for point in points:
            candidate = dict(state.joint_positions)
            candidate.update(point.positions)
            if not self.kinematics.collision_free(candidate, collision):
                reason = None
                collision_reason = getattr(self.kinematics, "collision_reason", None)
                if callable(collision_reason):
                    reason = collision_reason(candidate, collision)
                detail = f"：{reason}" if reason else ""
                raise PlanningError(f"关节路径在 {point.time_from_start_s:.3f}s 存在碰撞{detail}")
        return True

    def _resources(self, goal):
        return [
            f"joints:{group}"
            for group, names in self.capabilities.joint_groups.items()
            if set(goal).intersection(names)
        ] or ["joints"]


def _projected_path_time_scale(points, limits) -> float:
    """计算满足关节速度、加速度和jerk上限所需的统一时间倍率。"""

    durations = [
        right.time_from_start_s - left.time_from_start_s for left, right in zip(points, points[1:])
    ]
    if any(duration <= 0 for duration in durations):
        raise PlanningError("投影路径时间必须严格递增")

    names = tuple(points[0].positions)
    segment_velocities = [
        {name: (right.positions[name] - left.positions[name]) / duration for name in names}
        for left, right, duration in zip(points, points[1:], durations)
    ]
    velocity_ratio = max(
        abs(velocity[name]) / limits[name].max_velocity
        for velocity in segment_velocities
        for name in names
    )

    accelerations = []
    acceleration_durations = []
    zero = {name: 0.0 for name in names}
    velocity_samples = [zero, *segment_velocities, zero]
    for index, (left, right) in enumerate(zip(velocity_samples, velocity_samples[1:])):
        if index == 0:
            duration = durations[0]
        elif index == len(durations):
            duration = durations[-1]
        else:
            duration = (durations[index - 1] + durations[index]) / 2.0
        acceleration_durations.append(duration)
        accelerations.append({name: (right[name] - left[name]) / duration for name in names})
    acceleration_ratio = max(
        abs(acceleration[name]) / limits[name].max_acceleration
        for acceleration in accelerations
        for name in names
    )

    jerk_ratio = 0.0
    for index, (left, right) in enumerate(zip(accelerations, accelerations[1:])):
        duration = (acceleration_durations[index] + acceleration_durations[index + 1]) / 2.0
        jerk_ratio = max(
            jerk_ratio,
            *(abs(right[name] - left[name]) / duration / limits[name].max_jerk for name in names),
        )
    return max(
        1.0,
        velocity_ratio,
        math.sqrt(acceleration_ratio),
        jerk_ratio ** (1.0 / 3.0),
    )


def _linear_joint_path_samples(points, *, maximum_joint_step_rad: float = 0.02):
    """按Runtime的关节线性插值方式补出碰撞检查采样点。"""

    if not points:
        return []
    samples = [points[0]]
    for left, right in zip(points, points[1:]):
        maximum_delta = max(
            abs(right.positions[name] - left.positions[name]) for name in left.positions
        )
        steps = max(1, math.ceil(maximum_delta / maximum_joint_step_rad))
        for step in range(1, steps + 1):
            ratio = step / steps
            samples.append(
                right.model_copy(
                    update={
                        "time_from_start_s": left.time_from_start_s
                        + (right.time_from_start_s - left.time_from_start_s) * ratio,
                        "positions": {
                            name: left.positions[name]
                            + (right.positions[name] - left.positions[name]) * ratio
                            for name in left.positions
                        },
                        "velocities": {},
                    }
                )
            )
    return samples


def _simplify_joint_waypoints(
    goals: list[dict[str, float]],
    *,
    tolerance_rad: float,
) -> list[dict[str, float]]:
    """去除固定末端IK投影产生的密集数值抖动，同时保留真实路径曲率。"""

    if len(goals) <= 2:
        return goals
    names = tuple(goals[0])
    if any(set(goal) != set(names) for goal in goals):
        raise PlanningError("投影路径的关节集合不一致")

    kept = {0, len(goals) - 1}
    pending = [(0, len(goals) - 1)]
    while pending:
        first, last = pending.pop()
        if last - first <= 1:
            continue
        start = goals[first]
        end = goals[last]
        direction = {name: end[name] - start[name] for name in names}
        direction_norm_squared = sum(value * value for value in direction.values())
        maximum_error = -1.0
        maximum_index = first
        for index in range(first + 1, last):
            point = goals[index]
            if direction_norm_squared <= 1e-18:
                ratio = 0.0
            else:
                # Ruckig按时间采样，点在几何路径上并不等距。按数组下标插值会
                # 把同一直线上的加减速采样误判为弯曲，从而保留数千个路点。
                # 这里按关节空间到首尾线段的投影计算误差；真实转弯仍由递归
                # 分段保留，纯时间采样密度不再影响几何简化结果。
                ratio = (
                    sum((point[name] - start[name]) * direction[name] for name in names)
                    / direction_norm_squared
                )
                ratio = min(1.0, max(0.0, ratio))
            error = max(
                abs(point[name] - (start[name] + direction[name] * ratio)) for name in names
            )
            if error > maximum_error:
                maximum_error = error
                maximum_index = index
        if maximum_error > tolerance_rad:
            kept.add(maximum_index)
            pending.append((first, maximum_index))
            pending.append((maximum_index, last))
    return [goals[index] for index in sorted(kept)]
