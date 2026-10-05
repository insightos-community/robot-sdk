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

"""基于 Pinocchio 的末端 IK 与碰撞检查。

模块导入本身不要求 Pinocchio；只有创建求解器时才检查依赖，因此场景列表、
Robot 状态和直接夹爪操作不会被高级规划依赖阻塞。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from semantic_robot_sdk_core.errors import PlanningError
from semantic_robot_sdk_core.models import EnvironmentCollisionObject, EnvironmentCollisionSet, Pose
from .dependencies import require_pinocchio


class PinocchioKinematics:
    _IK_LOCAL_FALLBACK_ATTEMPTS = 4
    _IK_FALLBACK_ATTEMPTS = 16

    def __init__(
        self,
        urdf_path: str | Path,
        frame_names: dict[str, str],
        *,
        mesh_directories: list[str] | None = None,
        disabled_collision_pairs: list[tuple[str, str]] | None = None,
        model_frame_id: str = "base_link",
        controlled_joints_by_end_effector: dict[str, list[str]] | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        damping: float = 1e-5,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        time_budget_s: float = 3.0,
    ) -> None:
        require_pinocchio()
        import pinocchio as pin  # type: ignore

        self.pin = pin
        self.model, self.geometry_model, _visual = pin.buildModelsFromUrdf(
            str(urdf_path), package_dirs=mesh_directories or []
        )
        self.data = self.model.createData()
        # URDF 只负责声明各 link 的碰撞几何，不会自动建立需要检查的几何对。
        # 若遗漏此步骤，computeCollisions 会稳定返回“无碰撞”，形成危险的假阴性。
        self.geometry_model.addAllCollisionPairs()
        self._remove_adjacent_and_disabled_pairs(disabled_collision_pairs or [])
        self.geometry_data = pin.GeometryData(self.geometry_model)
        self.frames = {
            logical: self.model.getFrameId(actual) for logical, actual in frame_names.items()
        }
        if any(frame_id >= len(self.model.frames) for frame_id in self.frames.values()):
            raise PlanningError("Robot Profile 引用了不存在的末端坐标系")
        self.root_frame = self.model.getFrameId(model_frame_id)
        if self.root_frame >= len(self.model.frames):
            raise PlanningError(f"Robot Profile 引用了不存在的运动学根坐标系：{model_frame_id}")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        # 收敛判据按位置/姿态分组。历史行为是混合单位范数 < tolerance：
        # 姿态被隐式要求到 ~6e-3 度，远超放置类任务的必要精度，也让冗余度
        # 低的构型（固定端+撤钩）被误判为不收敛。默认值与旧判据同规模，
        # 仅在拆分处放宽 √2 倍的组合上界；调用方可按任务逐次覆盖。
        self.position_tolerance = float(
            tolerance if position_tolerance is None else position_tolerance
        )
        self.orientation_tolerance = float(
            tolerance if orientation_tolerance is None else orientation_tolerance
        )
        self.time_budget_s = float(time_budget_s)
        self.model_frame_id = model_frame_id
        mapping = {
            name: frozenset(joints)
            for name, joints in (controlled_joints_by_end_effector or {}).items()
        }
        if set(mapping) - set(self.frames):
            raise PlanningError("受控关节映射包含未知末端")
        known_joints = set(self.model.names)
        for end_effector, joints in mapping.items():
            if not joints or len(joints) != len(
                (controlled_joints_by_end_effector or {})[end_effector]
            ):
                raise PlanningError(f"末端 {end_effector} 的受控关节不能为空或重复")
            unknown = joints - known_joints
            if unknown:
                raise PlanningError(f"末端 {end_effector} 引用了未知关节：{sorted(unknown)}")
        self.controlled_joints_by_end_effector = mapping

    def _remove_adjacent_and_disabled_pairs(
        self, disabled_collision_pairs: list[tuple[str, str]]
    ) -> None:
        """删除不会用于自碰撞判定的几何对。

        直接相连的 link 在关节处通常存在几何重叠，不能当成自碰撞。夹爪手指等
        非相邻但设计上允许接触的组合必须由 Robot Profile 显式列出，不能在算法中
        根据名字猜测。这里的名称使用 URDF link 名；Pinocchio 可能追加 `_0` 后缀。
        """

        disabled = {frozenset(pair) for pair in disabled_collision_pairs}
        kept = []
        for pair in list(self.geometry_model.collisionPairs):
            first = self.geometry_model.geometryObjects[pair.first]
            second = self.geometry_model.geometryObjects[pair.second]
            first_joint, second_joint = first.parentJoint, second.parentJoint
            adjacent = (
                first_joint == second_joint
                or self.model.parents[first_joint] == second_joint
                or self.model.parents[second_joint] == first_joint
            )
            names = frozenset((_link_name(first.name), _link_name(second.name)))
            if not adjacent and names not in disabled:
                kept.append(pair)
        # Pinocchio 的 collisionPairs 是连续容器；逐项删除会令后续索引失效。
        # 因此先清空再重建保留集合，保证每个 Robot Profile 得到确定结果。
        self.geometry_model.removeAllCollisionPairs()
        for pair in kept:
            self.geometry_model.addCollisionPair(pair)

    def forward_kinematics(
        self,
        end_effector: str,
        joint_positions: dict[str, float],
    ) -> Pose:
        """返回末端相对运动学根坐标系的位姿。

        公共四元数始终是 ``xyzw``。这里显式计算 ``root -> end``，不能把
        Pinocchio 的 universe/world 放置误当成 Robot Profile 的根坐标系。
        """

        if end_effector not in self.frames:
            raise PlanningError(f"未知末端：{end_effector}")
        q, _controlled = self._configuration(joint_positions)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        root_to_end = self.data.oMf[self.root_frame].actInv(
            self.data.oMf[self.frames[end_effector]]
        )
        quaternion_xyzw = tuple(
            float(value) for value in self.pin.Quaternion(root_to_end.rotation).coeffs()
        )
        return Pose(
            position=tuple(float(value) for value in root_to_end.translation),
            quaternion_xyzw=quaternion_xyzw,
            frame_id=self.model_frame_id,
        )

    def _configuration(
        self,
        joint_positions: dict[str, float],
        *,
        controlled_names: frozenset[str] | None = None,
    ) -> tuple[object, list[tuple[str, int, int]]]:
        """把整机真实状态填入 q，并单独记录本次 IK 可以改变的自由度。

        未进入 controlled_names 的关节仍保持调用方提供的当前值，只是 Jacobian
        求解不能改变它们；这样另一机械臂不会被错误重置到 neutral。
        """

        q = self.pin.neutral(self.model)
        controlled: list[tuple[str, int, int]] = []
        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            name = self.model.names[joint_id]
            if joint.nq != 1 or name not in joint_positions:
                continue
            q[joint.idx_q] = joint_positions[name]
            if controlled_names is None or name in controlled_names:
                controlled.append((name, joint.idx_q, joint.idx_v))
        return q, controlled

    def solve_end_effector(
        self,
        end_effector: str,
        target: Pose,
        current_joints: dict[str, float],
        environment: EnvironmentCollisionSet | None = None,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        time_budget_s: float | None = None,
    ) -> dict[str, float]:
        return self._solve_targets(
            {end_effector: target},
            current_joints,
            environment=environment,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            time_budget_s=time_budget_s,
        )

    def solve_end_effectors(
        self,
        targets: dict[str, Pose],
        current_joints: dict[str, float],
        environment: EnvironmentCollisionSet | None = None,
        fixed_end_effectors: frozenset[str] = frozenset(),
        continuous_seed: bool = False,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        time_budget_s: float | None = None,
    ) -> dict[str, float]:
        """用同一个关节向量同时满足多个末端目标。

        双臂共享躯干关节，逐侧独立求解会让后一侧覆盖前一侧的躯干结果。因此这里
        堆叠所有末端误差和 Jacobian，在同一次迭代中求解，并在最终状态统一做碰撞检查。

        ``position_tolerance``/``orientation_tolerance`` 只作用于本次调用：
        精密插入阶段保持构造默认，撤钩/转移类动作可显式放宽以扩大收敛域。
        """

        return self._solve_targets(
            targets,
            current_joints,
            environment=environment,
            fixed_end_effectors=fixed_end_effectors,
            continuous_seed=continuous_seed,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            time_budget_s=time_budget_s,
        )

    def _solve_targets(
        self,
        targets: dict[str, Pose],
        current_joints: dict[str, float],
        *,
        environment: EnvironmentCollisionSet | None = None,
        fixed_end_effectors: frozenset[str] = frozenset(),
        continuous_seed: bool = False,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        time_budget_s: float | None = None,
    ) -> dict[str, float]:
        """从当前状态和少量确定性起点寻找一组无碰撞的末端解。

        双臂与躯干组成冗余系统，同一个末端目标存在多组关节解。只从当前
        ``travel`` 姿态做一次局部迭代，会把高位双侧目标误判为不可达，或收敛到
        躯干压住另一侧夹具的碰撞分支。有限多起点只用于寻找同一几何目标的其他
        关节分支。环境碰撞必须在多起点搜索内部参与选择；若先返回一个仅
        无自碰撞的解、再由适配层检查环境，首个分支碰到场景对象时就会错误
        拒绝同一目标的其它可行解。随后生成的整条轨迹仍会再次逐点检查。
        """

        import time

        import numpy as np

        position_tolerance = (
            self.position_tolerance if position_tolerance is None else float(position_tolerance)
        )
        orientation_tolerance = (
            self.orientation_tolerance
            if orientation_tolerance is None
            else float(orientation_tolerance)
        )
        budget_s = self.time_budget_s if time_budget_s is None else float(time_budget_s)
        deadline = None if budget_s <= 0.0 else time.monotonic() + budget_s
        # 停滞早退：总误差在窗口内没有 1% 以上的相对改进就放弃当前起点。
        # 不可达/被卡死的目标不再烧满 max_iterations×起点数 的迭代预算。
        stagnation_window = 30
        stagnation_rtol = 0.01

        if not targets:
            raise PlanningError("末端 IK 至少需要一个目标")
        unknown = set(targets) - set(self.frames)
        if unknown:
            raise PlanningError(f"未知末端：{sorted(unknown)}")
        if any(target.frame_id != self.model_frame_id for target in targets.values()):
            raise PlanningError(f"PinocchioSolver 只接收已转换到 {self.model_frame_id} 的目标")
        if not fixed_end_effectors <= set(targets):
            raise PlanningError("固定末端必须包含在本次多末端目标中")
        active_targets = targets
        configured_by_target = {
            end_effector: frozenset(self.controlled_joints_by_end_effector.get(end_effector, ()))
            for end_effector in active_targets
        }
        configured_active_joints = frozenset().union(*configured_by_target.values())
        controlled_filters: list[frozenset[str] | None] = []
        exclusive_moving_joints = frozenset()
        moving_end_effectors: set[str] = set()
        if len(active_targets) == 1 and not fixed_end_effectors:
            moving_end_effector = next(iter(active_targets))
            other_end_effector_joints = frozenset().union(
                *(
                    joints
                    for name, joints in self.controlled_joints_by_end_effector.items()
                    if name != moving_end_effector
                )
            )
            exclusive_moving_joints = (
                configured_by_target[moving_end_effector] - other_end_effector_joints
            )
            if exclusive_moving_joints:
                # 单臂接近和外拉先保持共享躯干不动，只用对应手臂寻找连续解。
                # 只有手臂自身确实不可达时才允许躯干参与，避免IK为了更容易
                # 收敛而提前大幅扭腰，破坏夹具插入方向和后续另一侧工作空间。
                controlled_filters.append(exclusive_moving_joints)
        if fixed_end_effectors:
            moving_end_effectors = set(active_targets) - fixed_end_effectors
            moving_joints = frozenset().union(
                *(configured_by_target[name] for name in moving_end_effectors)
            )
            fixed_joints = frozenset().union(
                *(configured_by_target[name] for name in fixed_end_effectors)
            )
            exclusive_moving_joints = moving_joints - fixed_joints
            if exclusive_moving_joints:
                # 固定侧已承载时，几何路径的每个连续点都必须先尝试只用
                # 运动侧独立关节。若只在首点使用独立关节，后续点会改用共享
                # 躯干做补偿：命令虽是水平短拉，真实末端却会被向上带出侧面凹槽。
                # 独立关节确实不可达时才允许下面的同步约束求解，且整条路径
                # 仍由固定末端检查和碰撞检查共同约束。
                controlled_filters.append(exclusive_moving_joints)
        # 先使用运动臂独立关节，只有它确实到达工作区边界时才启用共享躯干。
        # 固定侧并不是“禁止所有相关关节运动”，而是末端笛卡尔约束：腰部
        # 小幅参与时，固定臂必须同步补偿并在每个路点保持原Pose。若在这里
        # 彻底禁止同步补偿，第二只手会在独立工作区边界提前失去可达性；
        # 放开无约束单臂IK又会产生大幅扭腰和绕背姿态。两级求解既
        # 优先保持自然局部动作，也保留双臂共享躯干机器人的实际可达空间。
        controlled_filters.append(configured_active_joints or None)
        # 左右目标可能没有独立关节配置，去重后仍保留一次默认全关节求解。
        controlled_filters = list(dict.fromkeys(controlled_filters))

        desired_in_root = {}
        for end_effector, target in active_targets.items():
            x, y, z, w = target.quaternion_xyzw
            desired_in_root[end_effector] = self.pin.SE3(
                self.pin.Quaternion(w, x, y, z).matrix(),
                np.asarray(target.position, dtype=float),
            )

        def _total_error(q_value):
            """当前构型下所有末端的位置+姿态误差总量（无 Jacobian，仅供线搜索）。"""

            self.pin.forwardKinematics(self.model, self.data, q_value)
            self.pin.updateFramePlacements(self.model, self.data)
            total = 0.0
            for end_effector in active_targets:
                desired = self.data.oMf[self.root_frame].act(desired_in_root[end_effector])
                segment = self.pin.log6(
                    self.data.oMf[self.frames[end_effector]].actInv(desired)
                ).vector
                total += float(np.linalg.norm(segment[0:3]))
                total += float(np.linalg.norm(segment[3:6]))
            return total

        converged = False
        had_controlled_joints = False
        collision_reasons: set[str] = set()
        for controlled_filter in controlled_filters:
            initial, controlled = self._configuration(
                current_joints, controlled_names=controlled_filter
            )
            if not controlled:
                continue
            had_controlled_joints = True
            controlled_velocity_indices = [index_v for _name, _index_q, index_v in controlled]
            # 同一末端目标往往存在多组IK解。自由空间动作若返回第一个随机
            # 收敛解，会稳定选择扭腰、绕背或接近关节限位的分支，另一只手
            # 随后就失去工作空间。连续路径只有一个上一点种子，必须立即沿原
            # 分支推进；其余动作则比较本级所有无碰撞解后选择最接近当前整机
            # 状态、且尽量少使用双臂共享关节的解。这是通用运动学选择，不含
            # 箱体、凹槽或某个Skill的经验姿态。
            feasible: list[tuple[float, dict[str, float]]] = []
            shared_joints = (
                frozenset.intersection(*self.controlled_joints_by_end_effector.values())
                if self.controlled_joints_by_end_effector
                else frozenset()
            )
            starts = [initial]
            if not continuous_seed and fixed_end_effectors:
                mirror_seed = self._bilateral_mirror_seed(
                    initial,
                    controlled,
                    moving_end_effectors=moving_end_effectors,
                    fixed_end_effectors=fixed_end_effectors,
                )
                if mirror_seed is not None:
                    starts.append(mirror_seed)
            # 每个自由空间Action的终点或笛卡尔路径首点允许一次有限多起点，
            # 用来选择右臂自身的可达冗余分支；候选随后仍要通过整条关节/
            # 环境碰撞检查。从第二个几何路点起，current_joints已经是上一点
            # 的解，必须只沿连续种子推进，不能每2cm随机跳支并重复昂贵搜索。
            if not continuous_seed:
                starts.extend(self._fallback_configurations(initial, controlled))
            for start in starts:
                q = start.copy()
                stagnation_reference = (float("inf"), -1)
                for _iteration in range(self.max_iterations):
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    self.pin.forwardKinematics(self.model, self.data, q)
                    self.pin.updateFramePlacements(self.model, self.data)
                    self.pin.computeJointJacobians(self.model, self.data, q)
                    errors = []
                    jacobians = []
                    for end_effector in active_targets:
                        desired = self.data.oMf[self.root_frame].act(desired_in_root[end_effector])
                        frame_id = self.frames[end_effector]
                        error_motion = self.pin.log6(self.data.oMf[frame_id].actInv(desired))
                        errors.append(error_motion.vector)
                        # 一次 computeJointJacobians 覆盖全部末端，逐末端只做
                        # getFrameJacobian 取列；双末端场景少算一半 Jacobian。
                        jacobian = self.pin.getFrameJacobian(
                            self.model,
                            self.data,
                            frame_id,
                            self.pin.ReferenceFrame.LOCAL,
                        )
                        # log6 误差对关节的一阶线性化带 Jlog6 修正：直接用
                        # 局部 Jacobian 会在大姿态误差下系统性高估收敛速度，
                        # 修正后迭代方向更准，窄解空间里更少来回震荡。
                        jlog6 = getattr(self.pin, "Jlog6", None)
                        if callable(jlog6):
                            correction = jlog6(self.pin.exp6(error_motion))
                            if correction.shape == (6, 6):
                                jacobian = correction @ jacobian
                        jacobians.append(jacobian)
                    error = np.concatenate(errors)
                    # 收敛按位置/姿态分组判断。Pinocchio log6.vector =
                    # [线速度(m); 角速度(rad)]，混合单位范数会把姿态隐式要求到
                    # 千分之几度，与放置类任务的真实精度需求不匹配。
                    position_ok = True
                    orientation_ok = True
                    total_error = 0.0
                    for segment in errors:
                        position_norm = float(np.linalg.norm(segment[0:3]))
                        orientation_norm = float(np.linalg.norm(segment[3:6]))
                        total_error += position_norm + orientation_norm
                        if position_norm >= position_tolerance:
                            position_ok = False
                        if orientation_norm >= orientation_tolerance:
                            orientation_ok = False
                    if position_ok and orientation_ok:
                        converged = True
                        result = {name: float(q[index_q]) for name, index_q, _index_v in controlled}
                        candidate = dict(current_joints)
                        candidate.update(result)
                        if self.collision_free(candidate, environment):
                            if continuous_seed:
                                return result
                            cost = 0.0
                            for name, index_q, _index_v in controlled:
                                lower = float(self.model.lowerPositionLimit[index_q])
                                upper = float(self.model.upperPositionLimit[index_q])
                                joint_range = upper - lower
                                scale = (
                                    joint_range
                                    if math.isfinite(joint_range) and joint_range > 1e-6
                                    else 1.0
                                )
                                delta = (float(q[index_q]) - float(initial[index_q])) / scale
                                weight = 3.0 if name in shared_joints else 1.0
                                cost += weight * delta * delta
                            feasible.append((cost, result))
                        elif reason := self.collision_reason(candidate, environment):
                            collision_reasons.add(reason)
                        break
                    # 停滞检测放在收敛块之后：有实质改进就刷新参考点，
                    # 连续一个窗口内改进不足 1% 才判定当前起点卡死。
                    if total_error < stagnation_reference[0] * (1.0 - stagnation_rtol):
                        stagnation_reference = (total_error, _iteration)
                    elif (
                        stagnation_reference[0] != float("inf")
                        and _iteration - stagnation_reference[1] >= stagnation_window
                    ):
                        break
                    jacobian = np.vstack(jacobians)
                    # 受控关节必须在求解线性系统前就从 Jacobian 中选出；
                    # 禁止移动的底盘、承载臂或共享关节不能先参与误差分配再清零。
                    controlled_jacobian = jacobian[:, controlled_velocity_indices]
                    system = controlled_jacobian @ controlled_jacobian.T + self.damping * np.eye(
                        len(error)
                    )
                    controlled_inverse = controlled_jacobian.T @ np.linalg.solve(
                        system, np.eye(len(error))
                    )
                    reduced_velocity = controlled_inverse @ error
                    # 双臂同步任务仍有冗余自由度。若只最小化末端误差，连续
                    # 路点会把腕部逐步推到关节限位，最后几厘米即使存在另一
                    # 个可达构型也会数值失效。这里只在任务Jacobian零空间内
                    # 轻微远离关节边界，不改变末端目标，也不写入任何箱型姿态。
                    # 因为参考是通用关节上下限，它同时适用于仿真和真机模型。
                    limit_avoidance = np.zeros(len(controlled))
                    posture_attraction = np.zeros(len(controlled))
                    for index, (_name, index_q, _index_v) in enumerate(controlled):
                        lower = float(self.model.lowerPositionLimit[index_q])
                        upper = float(self.model.upperPositionLimit[index_q])
                        if not (math.isfinite(lower) and math.isfinite(upper)):
                            continue
                        half_range = (upper - lower) / 2.0
                        if half_range <= 1e-6:
                            continue
                        midpoint = (lower + upper) / 2.0
                        normalized = (float(q[index_q]) - midpoint) / half_range
                        limit_avoidance[index] = -0.08 * max(-1.0, min(1.0, normalized))
                        if continuous_seed:
                            # 连续笛卡尔路径的调用方已经提供了期望关节分支。
                            # 初值若只参与第一轮迭代，冗余自由度随后仍会被局部
                            # 最小范数解推向另一分支。只在任务Jacobian零空间内
                            # 轻微吸引到该初值，不改变末端硬目标，也不引入任何
                            # 业务姿态；数值幅度与已有的关节限位回避保持同级。
                            posture_attraction[index] = max(
                                -0.08,
                                min(
                                    0.08,
                                    0.2 * (float(initial[index_q]) - float(q[index_q])),
                                ),
                            )
                    nullspace = np.eye(len(controlled)) - (controlled_inverse @ controlled_jacobian)
                    # 自由目标用关节限位回避选择自然分支；连续路径已经由
                    # 调用方选定可达分支，不能再让两个零空间目标在终点附近
                    # 相互竞争，否则最后几个采样会被错误报告为不收敛。
                    secondary_velocity = posture_attraction if continuous_seed else limit_avoidance
                    reduced_velocity += nullspace @ secondary_velocity
                    velocity = np.zeros(self.model.nv)
                    velocity[controlled_velocity_indices] = reduced_velocity
                    base_step = velocity * 0.1
                    # 回退线搜索：从满步开始逐档折半，只在误差确实下降时接受，
                    # 防止固定步长在窄解空间里来回越过谷底；全档都不降时保持
                    # 原步长，真正的不可达目标交由停滞早退终止。
                    accepted = None
                    for ratio in (1.0, 0.5, 0.25, 0.125, 0.0625):
                        candidate_q = self.pin.integrate(self.model, q, base_step * ratio)
                        candidate_q = np.minimum(
                            np.maximum(candidate_q, self.model.lowerPositionLimit),
                            self.model.upperPositionLimit,
                        )
                        if _total_error(candidate_q) < total_error:
                            accepted = candidate_q
                            break
                    if accepted is not None:
                        q = accepted
                    else:
                        q = self.pin.integrate(self.model, q, base_step)
                        q = np.minimum(
                            np.maximum(q, self.model.lowerPositionLimit),
                            self.model.upperPositionLimit,
                        )
            if feasible:
                return min(feasible, key=lambda item: item[0])[1]
        if not had_controlled_joints:
            raise PlanningError("末端 IK 没有可控制关节")
        if converged:
            detail = "；".join(sorted(collision_reasons)[:3])
            raise PlanningError("Pinocchio IK 未找到无碰撞解" + (f"：{detail}" if detail else ""))
        label = "多末端" if len(active_targets) > 1 else ""
        raise PlanningError(f"Pinocchio {label}IK 未在有限多起点内收敛")

    def _bilateral_mirror_seed(
        self,
        initial,
        controlled,
        *,
        moving_end_effectors: set[str],
        fixed_end_effectors: frozenset[str],
    ):
        """用R1 Pro另一侧有效构型生成确定性的镜像IK起点。"""

        if len(moving_end_effectors) != 1 or len(fixed_end_effectors) != 1:
            return None
        moving = next(iter(moving_end_effectors))
        fixed = next(iter(fixed_end_effectors))
        if {moving, fixed} != {"left", "right"}:
            return None
        controlled_names = {name for name, _index_q, _index_v in controlled}
        # 左右臂URDF关于Robot矢状面对称。绕Y轴的关节保持符号，绕X/Z轴
        # 的关节反号；该关系来自R1 Pro硬件运动学，不包含夹具或箱体语义。
        mirror_signs = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        seed = initial.copy()
        changed = False
        for index, sign in enumerate(mirror_signs, start=1):
            moving_joint = f"{moving}_arm_joint{index}"
            fixed_joint = f"{fixed}_arm_joint{index}"
            if moving_joint not in controlled_names:
                return None
            moving_id = self.model.getJointId(moving_joint)
            fixed_id = self.model.getJointId(fixed_joint)
            if moving_id >= self.model.njoints or fixed_id >= self.model.njoints:
                return None
            moving_index = self.model.joints[moving_id].idx_q
            fixed_index = self.model.joints[fixed_id].idx_q
            value = sign * float(initial[fixed_index])
            value = min(
                float(self.model.upperPositionLimit[moving_index]),
                max(float(self.model.lowerPositionLimit[moving_index]), value),
            )
            if abs(value - float(seed[moving_index])) > 1e-9:
                changed = True
            seed[moving_index] = value
        return seed if changed else None

    def _fallback_configurations(self, initial, controlled):
        """从当前构型向外渐进生成可复现的 IK 起点。"""

        # 局部和全局使用独立随机序列：前置自然姿态搜索不能消耗原有12个
        # 全关节空间样本，否则会让已经验证可达的高层绕障分支随机消失。
        local_rng = random.Random(17)
        global_rng = random.Random(17)
        local_attempts = min(self._IK_LOCAL_FALLBACK_ATTEMPTS, self._IK_FALLBACK_ATTEMPTS)
        local_denominator = max(local_attempts - 1, 1)
        for attempt in range(self._IK_FALLBACK_ATTEMPTS):
            rng = local_rng if attempt < local_attempts else global_rng
            candidate = initial.copy()
            for _name, index_q, _index_v in controlled:
                lower = float(self.model.lowerPositionLimit[index_q])
                upper = float(self.model.upperPositionLimit[index_q])
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    continue
                # 安全区间仍向零位内缩20%，但起点从当前关节位置附近开始，
                # 前四次逐步扩大当前姿态邻域，优先找到小动作分支；其余尝试
                # 恢复完整安全区间采样。不能把全部多起点都消耗在局部邻域，
                # 否则高层或绕障目标只剩一次全局机会，会把原本可达误判为
                # 无解。这里调整搜索顺序，不把“好看姿态”变成可达性硬条件。
                anchor = min(max(0.0, lower), upper)
                safe_lower = anchor + (lower - anchor) * 0.8
                safe_upper = anchor + (upper - anchor) * 0.8
                if attempt < local_attempts:
                    spread = 0.15 + 0.35 * attempt / local_denominator
                    radius = (safe_upper - safe_lower) * spread / 2.0
                    candidate[index_q] = min(
                        safe_upper,
                        max(safe_lower, float(initial[index_q]) + rng.uniform(-radius, radius)),
                    )
                else:
                    # 全局阶段精确保留原有均匀分布及随机序列。
                    candidate[index_q] = rng.uniform(safe_lower, safe_upper)
            yield candidate

    def collision_free(
        self,
        joint_positions: dict[str, float],
        environment: object | None,
    ) -> bool:
        """检查自碰撞以及静态环境碰撞。

        环境输入必须已经转换到运动学根坐标。Robot SDK 在调用本方法前负责从
        世界坐标转换，求解器不猜测 Robot 的世界位姿。
        """

        return self.collision_reason(joint_positions, environment) is None

    def collision_reason(
        self,
        joint_positions: dict[str, float],
        environment: object | None,
    ) -> str | None:
        """返回首个自碰撞或环境碰撞，仅丰富既有规划失败诊断。"""

        q, _controlled = self._configuration(joint_positions)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            q,
        )
        self.pin.computeCollisions(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            q,
            False,
        )
        for pair, result in zip(
            self.geometry_model.collisionPairs,
            self.geometry_data.collisionResults,
            strict=True,
        ):
            if result.isCollision():
                first = self.geometry_model.geometryObjects[pair.first].name
                second = self.geometry_model.geometryObjects[pair.second].name
                return f"自碰撞 {first} 与 {second}"
        if environment is None:
            return None
        if not isinstance(environment, EnvironmentCollisionSet):
            raise PlanningError("环境碰撞输入必须使用 EnvironmentCollisionSet")
        if environment.frame_id != self.model_frame_id:
            raise PlanningError(f"环境碰撞快照必须位于 {self.model_frame_id} 坐标系")
        return self._first_environment_collision(environment)

    def _first_environment_collision(self, environment: EnvironmentCollisionSet) -> str | None:
        """返回首个Robot几何与场景对象碰撞，不改变碰撞判定边界。"""

        import coal
        import numpy as np

        request = coal.CollisionRequest()
        obstacles = []
        for item in environment.objects:
            geometry = _collision_geometry(item)
            x, y, z, w = item.pose.quaternion_xyzw
            rotation = self.pin.Quaternion(w, x, y, z).matrix()
            transform = coal.Transform3s(
                rotation,
                np.asarray(item.pose.position, dtype=float),
            )
            unknown = set(item.allowed_contact_end_effectors) - set(self.frames)
            if unknown:
                raise PlanningError(f"环境碰撞体允许了未知末端接触：{sorted(unknown)}")
            obstacles.append(
                (
                    item.source_id,
                    geometry,
                    transform,
                    frozenset(item.allowed_contact_end_effectors),
                )
            )

        for index, robot_geometry in enumerate(self.geometry_model.geometryObjects):
            placement = self.geometry_data.oMg[index]
            robot_transform = coal.Transform3s(placement.rotation, placement.translation)
            for source_id, obstacle_geometry, obstacle_transform, allowed in obstacles:
                if allowed and self._geometry_belongs_to_any_end_effector(robot_geometry, allowed):
                    continue
                result = coal.CollisionResult()
                collided = coal.collide(
                    robot_geometry.geometry,
                    robot_transform,
                    obstacle_geometry,
                    obstacle_transform,
                    request,
                    result,
                )
                if collided or result.isCollision():
                    return f"{robot_geometry.name} 与场景对象 {source_id}"
        return None

    def _geometry_belongs_to_any_end_effector(
        self, robot_geometry, end_effectors: frozenset[str]
    ) -> bool:
        """判断Robot碰撞几何是否属于一个已允许接触的独立末端分支。"""

        other_joints = {
            joint
            for name, joints in self.controlled_joints_by_end_effector.items()
            if name not in end_effectors
            for joint in joints
        }
        owned_joints = {
            joint
            for name in end_effectors
            for joint in self.controlled_joints_by_end_effector.get(name, ())
        } - other_joints
        joint_id = robot_geometry.parentJoint
        while joint_id:
            if self.model.names[joint_id] in owned_joints:
                return True
            joint_id = self.model.parents[joint_id]
        return False


def _collision_geometry(item: EnvironmentCollisionObject):
    """把公共基本形状转换为 Coal 几何；不接受引擎内部 Mesh 标识。"""

    import coal

    if item.shape == "box":
        assert item.size_xyz is not None
        return coal.Box(*item.size_xyz)
    assert item.radius_m is not None
    return coal.Sphere(item.radius_m)


def _link_name(geometry_name: str) -> str:
    """把 Pinocchio 为同名几何追加的数字后缀还原为 URDF link 名。"""

    stem, separator, suffix = geometry_name.rpartition("_")
    return stem if separator and suffix.isdigit() else geometry_name
