# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinocchio 求解器到公共 KinematicsProvider 的适配。"""

import math

from semantic_robot_sdk_core import (
    EnvironmentCollisionSet,
    MotionPlan,
    PlanningError,
    Pose,
    RobotState,
)
from semantic_robot_sdk_core.transforms import (
    quaternion_multiply,
    relative_collision_set,
    relative_pose,
    rotate_vector,
)

from .local_kinematics import PinocchioKinematics


class LocalKinematicsProvider:
    name = "local"

    def __init__(
        self,
        solver: PinocchioKinematics,
        *,
        world_frame="world",
        root_frame="base_link",
        # R1 Pro 周转箱作业的末端软控制和接触顺应本就会产生厘米级
        # 偏差。这里只用于判断重定时轨迹是否保持双工具几何，不是碰撞
        # 边界；碰撞、关节限位和工具接触仍分别由原有检查负责。
        fixed_position_tolerance_m=0.01,
        fixed_orientation_tolerance_rad=0.05,
    ):
        self.solver = solver
        self.world_frame = world_frame
        self.root_frame = root_frame
        self.fixed_position_tolerance_m = float(fixed_position_tolerance_m)
        self.fixed_orientation_tolerance_rad = float(fixed_orientation_tolerance_rad)
        if self.fixed_position_tolerance_m <= 0 or self.fixed_orientation_tolerance_rad <= 0:
            raise ValueError("固定末端位置和姿态容差必须大于零")

    def _local(self, target: Pose, state: RobotState) -> Pose:
        if target.frame_id == self.root_frame:
            return target
        if target.frame_id == self.world_frame and state.base_pose is not None:
            return relative_pose(state.base_pose, target, result_frame_id=self.root_frame)
        raise PlanningError("末端目标无法转换到 Robot 运动学根坐标")

    def fixed_targets_from_state(
        self,
        targets: dict[str, Pose],
        *,
        state: RobotState,
        fixed_end_effectors: frozenset[str],
    ) -> dict[str, Pose]:
        """把固定侧目标锚定到当前关节构型可实现的 FK 位姿。"""

        anchored = dict(targets)
        for name in fixed_end_effectors:
            target = targets[name]
            local = self.solver.forward_kinematics(name, state.joint_positions)
            if target.frame_id == self.root_frame:
                anchored[name] = local
                continue
            if target.frame_id != self.world_frame or state.base_pose is None:
                raise PlanningError("固定末端目标无法从 Robot 运动学根坐标转换")
            base = state.base_pose
            if base.frame_id != self.world_frame:
                raise PlanningError("Robot 基座位姿不在世界坐标系")
            rotated = rotate_vector(base.quaternion_xyzw, local.position)
            anchored[name] = Pose(
                position=tuple(base.position[index] + rotated[index] for index in range(3)),
                quaternion_xyzw=quaternion_multiply(base.quaternion_xyzw, local.quaternion_xyzw),
                frame_id=self.world_frame,
            )
        return anchored

    def solve(
        self,
        *,
        robot_id: str,
        end_effector: str,
        target: Pose,
        state: RobotState,
        environment: EnvironmentCollisionSet | None = None,
    ) -> dict[str, float]:
        del robot_id
        collision = environment
        if environment is not None and environment.frame_id == self.world_frame and state.base_pose:
            collision = relative_collision_set(
                state.base_pose, environment, result_frame_id=self.root_frame
            )
        goal = self.solver.solve_end_effector(
            end_effector,
            self._local(target, state),
            state.joint_positions,
            environment=collision,
        )
        candidate = dict(state.joint_positions)
        candidate.update(goal)
        if not self.solver.collision_free(candidate, collision):
            detail = self.solver.collision_reason(candidate, collision)
            raise PlanningError("IK 结果与环境碰撞" + (f"：{detail}" if detail else ""))
        return goal

    def solve_many(
        self,
        *,
        robot_id: str,
        targets: dict[str, Pose],
        state: RobotState,
        environment: EnvironmentCollisionSet | None = None,
        fixed_end_effectors: frozenset[str] = frozenset(),
        continuous_seed: bool = False,
    ) -> dict[str, float]:
        del robot_id
        local_targets = {name: self._local(target, state) for name, target in targets.items()}
        collision = environment
        if environment is not None and environment.frame_id == self.world_frame and state.base_pose:
            collision = relative_collision_set(
                state.base_pose, environment, result_frame_id=self.root_frame
            )
        solve_options = {"environment": collision}
        if fixed_end_effectors:
            solve_options["fixed_end_effectors"] = fixed_end_effectors
        if continuous_seed:
            solve_options["continuous_seed"] = True
        goal = self.solver.solve_end_effectors(
            local_targets,
            state.joint_positions,
            **solve_options,
        )
        candidate = dict(state.joint_positions)
        candidate.update(goal)
        if not self.solver.collision_free(candidate, collision):
            detail = self.solver.collision_reason(candidate, collision)
            raise PlanningError("多末端 IK 结果与环境碰撞" + (f"：{detail}" if detail else ""))
        return goal

    def path_keeps_fixed_end_effectors(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        targets: dict[str, Pose],
        fixed_end_effectors: frozenset[str],
    ) -> bool:
        """判断一条现有关节轨迹是否自然保持固定末端。

        clearance/transfer优先使用运动臂自己的连续关节轨迹；只有轨迹本身
        已满足固定端容差时才可直接执行。共享关节补偿由同步约束规划处理，
        这里不把密集轨迹逐点投影成另一条昂贵路径。
        """

        local_targets = {name: self._local(targets[name], state) for name in fixed_end_effectors}
        return self._path_keeps_fixed_end_effectors(
            plan,
            state=state,
            local_targets=local_targets,
            position_tolerance_m=self.fixed_position_tolerance_m,
            orientation_tolerance_rad=self.fixed_orientation_tolerance_rad,
        )

    def project_synchronized_end_effector_path(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        targets: dict[str, Pose],
        environment: EnvironmentCollisionSet | None = None,
    ) -> MotionPlan:
        """把双末端关节插值投影回调用方要求的同步笛卡尔路径。

        ``preserve_cartesian_path`` 不能只保证离散 IK 路点正确。双臂共享躯干时，
        Ruckig 在两个路点之间的关节轨迹仍可能改变两只工具的相对几何。这里按
        已有 50 ms 轨迹采样连续求解左右目标，再交回 Motion Provider 重定时；
        它是通用多末端运动约束，不理解箱体、凹槽或仿真物理周期。
        """

        if len(targets) < 2:
            return plan
        duration = float(plan.estimated_duration_s)
        if duration <= 0 or not plan.joint_trajectory:
            raise PlanningError("同步末端路径缺少有效时间轨迹")
        unknown = set(targets) - set(state.end_effectors)
        if unknown:
            raise PlanningError(f"Robot 状态缺少末端位姿：{sorted(unknown)}")

        starts = {name: self._local(state.end_effectors[name], state) for name in targets}
        local_targets = {name: self._local(target, state) for name, target in targets.items()}
        collision = environment
        if environment is not None and environment.frame_id == self.world_frame and state.base_pose:
            collision = relative_collision_set(
                state.base_pose, environment, result_frame_id=self.root_frame
            )

        previous_joints = dict(state.joint_positions)
        projected_points = []
        for index, point in enumerate(plan.joint_trajectory):
            if index == 0 or point.time_from_start_s <= 0:
                positions = dict(point.positions)
            else:
                ratio = min(1.0, max(0.0, point.time_from_start_s / duration))
                point_targets = {
                    name: _interpolate_pose(starts[name], local_targets[name], ratio)
                    for name in targets
                }
                # 优先从上一投影点连续推进，避免Ruckig原始关节插值把冗余
                # 双臂带回另一条IK分支。layout001的真实接合姿态有时恰好位于
                # 局部分支边界：终点和碰撞均可行，但第一个毫米级抬升路点
                # 无法用单一起点收敛。只有这种纯规划失败才做一次有限择优；
                # 后续仍逐点检查碰撞、双工具相对几何和最终目标，不能借此
                # 放宽实际接触或运动安全条件。
                projection_seed = dict(previous_joints)
                try:
                    projected = self.solver.solve_end_effectors(
                        point_targets,
                        projection_seed,
                        environment=collision,
                        continuous_seed=True,
                    )
                except PlanningError:
                    projected = self.solver.solve_end_effectors(
                        point_targets,
                        projection_seed,
                        environment=collision,
                        continuous_seed=False,
                    )
                positions = dict(point.positions)
                positions.update(projected)
            candidate = dict(state.joint_positions)
            candidate.update(positions)
            if not self.solver.collision_free(candidate, collision):
                detail = self.solver.collision_reason(candidate, collision)
                raise PlanningError(
                    "同步末端投影后的关节路径与环境碰撞" + (f"：{detail}" if detail else "")
                )
            previous_joints.update(positions)
            projected_points.append(
                point.model_copy(update={"positions": positions, "velocities": {}})
            )

        return plan.model_copy(
            update={
                "joint_trajectory": projected_points,
                "goal": dict(projected_points[-1].positions),
                "collision_checked": True,
                "planner": f"{plan.planner}+synchronized-cartesian-projection",
            }
        )

    def path_keeps_relative_end_effector_geometry(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        targets: dict[str, Pose],
    ) -> bool:
        """检查整条轨迹是否保持多个末端之间的刚性相对变换。"""

        names = sorted(targets)
        if len(names) < 2:
            return True
        starts = {name: self._local(state.end_effectors[name], state) for name in names}
        local_targets = {name: self._local(targets[name], state) for name in names}
        reference = names[0]
        expected = {
            name: relative_pose(
                starts[reference], starts[name], result_frame_id=f"{reference}:relative"
            )
            for name in names[1:]
        }
        target_relative = {
            name: relative_pose(
                local_targets[reference],
                local_targets[name],
                result_frame_id=f"{reference}:relative",
            )
            for name in names[1:]
        }
        if any(
            not self._pose_within_relative_tolerance(target_relative[name], expected[name])
            for name in names[1:]
        ):
            return False

        for point in plan.joint_trajectory:
            joints = dict(state.joint_positions)
            joints.update(point.positions)
            actual = {name: self.solver.forward_kinematics(name, joints) for name in names}
            for name in names[1:]:
                relative = relative_pose(
                    actual[reference],
                    actual[name],
                    result_frame_id=f"{reference}:relative",
                )
                if not self._pose_within_relative_tolerance(relative, expected[name]):
                    return False
        return True

    def path_reaches_synchronized_end_effector_targets(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        targets: dict[str, Pose],
    ) -> bool:
        """检查重定时轨迹的最终状态是否仍达到两个末端目标。

        投影路径已经逐点满足双工具相对几何，Motion Provider也会按Runtime
        实际采用的关节线性插值复核碰撞。这里不再要求每个中间FK采样严格
        落在理想笛卡尔直线上；真实失稳仍由Ability依据接触、力和滑移停止。
        """

        if not plan.joint_trajectory:
            return False
        joints = dict(state.joint_positions)
        joints.update(plan.joint_trajectory[-1].positions)
        for name, target in targets.items():
            actual = self.solver.forward_kinematics(name, joints)
            if not self._pose_within_relative_tolerance(actual, self._local(target, state)):
                return False
        return True

    def _pose_within_relative_tolerance(self, actual: Pose, target: Pose) -> bool:
        if math.dist(actual.position, target.position) > self.fixed_position_tolerance_m:
            return False
        dot = abs(
            sum(left * right for left, right in zip(actual.quaternion_xyzw, target.quaternion_xyzw))
        )
        angle = 2.0 * math.acos(min(1.0, max(0.0, dot)))
        return angle <= self.fixed_orientation_tolerance_rad

    def project_fixed_end_effector_path(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        targets: dict[str, Pose],
        fixed_end_effectors: frozenset[str],
        environment: EnvironmentCollisionSet | None = None,
    ) -> MotionPlan:
        """把时间参数化后的关节轨迹重新投影到固定末端约束上。

        多末端笛卡尔路点只能约束离散 IK 解。Ruckig 随后在关节空间生成的
        中间采样点仍可能移动已接合的末端，双臂共享躯干时尤其明显。这里
        先只校正每个采样点的位置：运动侧沿原轨迹前进，固定侧始终回到
        调用方给出的笛卡尔 Pose。投影后的旧速度已不再对应新路径，因此在此
        清空并由Motion Provider重新时间参数化。该逻辑只处理Robot运动学约束。
        """

        if not fixed_end_effectors:
            return plan
        if not fixed_end_effectors < set(targets):
            raise PlanningError("固定末端路径至少需要一个运动末端")

        local_fixed_targets = {
            name: self._local(targets[name], state) for name in fixed_end_effectors
        }
        if self._path_keeps_fixed_end_effectors(
            plan,
            state=state,
            local_targets=local_fixed_targets,
            position_tolerance_m=self.fixed_position_tolerance_m,
            orientation_tolerance_rad=self.fixed_orientation_tolerance_rad,
        ):
            # IK路点已经只使用运动侧独立关节时，原Ruckig轨迹会自然保持承载侧。
            # 再逐采样求一次冗余IK不仅没有增加约束，反而可能在等价解之间跳支，
            # 把数秒自由空间动作放大成数百秒。只有真实FK误差超过求解器容差时
            # 才进入下面的投影和重定时。
            return plan
        collision = environment
        if environment is not None and environment.frame_id == self.world_frame and state.base_pose:
            collision = relative_collision_set(
                state.base_pose, environment, result_frame_id=self.root_frame
            )

        previous_joints = dict(state.joint_positions)
        projected_points = []
        moving_end_effectors = set(targets) - fixed_end_effectors
        for point in plan.joint_trajectory:
            original_joints = dict(state.joint_positions)
            original_joints.update(point.positions)
            point_targets = dict(local_fixed_targets)
            point_targets.update(
                {
                    name: self.solver.forward_kinematics(name, original_joints)
                    for name in moving_end_effectors
                }
            )
            projected = self.solver.solve_end_effectors(
                point_targets,
                previous_joints,
                environment=collision,
                fixed_end_effectors=fixed_end_effectors,
                # 投影点按原关节轨迹顺序处理，previous_joints就是上一点
                # 已验证的解。这里必须保持同一冗余分支；若每个密集采样都
                # 重新做多起点搜索，一次transfer会被放大成分钟级离线计算。
                continuous_seed=True,
            )
            # ``solve_end_effectors`` 会优先只使用运动侧独立关节，因此返回值
            # 可能只包含右臂。未返回的躯干和承载臂关节必须沿用上一个已经
            # 满足固定末端约束的投影点；若从原始无约束轨迹回填，它们会把
            # 左侧工具重新带离接触位置，使“投影成功”的路径仍产生厘米级漂移。
            positions = {
                name: projected.get(name, previous_joints[name]) for name in point.positions
            }
            candidate = dict(state.joint_positions)
            candidate.update(positions)
            if not self.solver.collision_free(candidate, collision):
                detail = self.solver.collision_reason(candidate, collision)
                raise PlanningError(
                    "固定末端投影后的关节路径与环境碰撞" + (f"：{detail}" if detail else "")
                )
            previous_joints.update(positions)
            projected_points.append(
                point.model_copy(update={"positions": positions, "velocities": {}})
            )

        if not projected_points:
            raise PlanningError("固定末端路径缺少关节轨迹点")
        return plan.model_copy(
            update={
                "joint_trajectory": projected_points,
                "goal": dict(projected_points[-1].positions),
                "collision_checked": True,
                "planner": f"{plan.planner}+fixed-cartesian-projection",
            }
        )

    def _path_keeps_fixed_end_effectors(
        self,
        plan: MotionPlan,
        *,
        state: RobotState,
        local_targets: dict[str, Pose],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> bool:
        # Runtime在线性插值相邻关节轨迹点；只检查离散端点会遗漏非线性FK在
        # 线段中部产生的固定端漂移。这里按同一执行语义补样后再验证。
        for positions in _linear_joint_path_positions(plan):
            joints = dict(state.joint_positions)
            joints.update(positions)
            for name, target in local_targets.items():
                actual = self.solver.forward_kinematics(name, joints)
                if math.dist(actual.position, target.position) > position_tolerance_m:
                    return False
                dot = abs(
                    sum(
                        left * right
                        for left, right in zip(actual.quaternion_xyzw, target.quaternion_xyzw)
                    )
                )
                angle = 2.0 * math.acos(min(1.0, max(0.0, dot)))
                if angle > orientation_tolerance_rad:
                    return False
        return True

    def reachable(self, **kwargs) -> bool:
        try:
            self.solve(robot_id="reachability", **kwargs)
            return True
        except Exception:
            return False


def _linear_joint_path_positions(plan: MotionPlan, *, maximum_joint_step_rad: float = 0.02):
    points = plan.joint_trajectory
    if not points:
        return
    yield points[0].positions
    for left, right in zip(points, points[1:]):
        maximum_delta = max(
            abs(right.positions[name] - left.positions[name]) for name in left.positions
        )
        steps = max(1, math.ceil(maximum_delta / maximum_joint_step_rad))
        for step in range(1, steps + 1):
            ratio = step / steps
            yield {
                name: left.positions[name] + (right.positions[name] - left.positions[name]) * ratio
                for name in left.positions
            }


def _interpolate_pose(start: Pose, target: Pose, ratio: float) -> Pose:
    position = tuple(
        left + (right - left) * ratio for left, right in zip(start.position, target.position)
    )
    target_quaternion = target.quaternion_xyzw
    if sum(left * right for left, right in zip(start.quaternion_xyzw, target_quaternion)) < 0:
        target_quaternion = tuple(-value for value in target_quaternion)
    quaternion = tuple(
        left + (right - left) * ratio
        for left, right in zip(start.quaternion_xyzw, target_quaternion)
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise PlanningError("末端姿态插值结果无效")
    return Pose(
        position=position,
        quaternion_xyzw=tuple(value / norm for value in quaternion),
        frame_id=start.frame_id,
    )
