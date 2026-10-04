"""基于 Pinocchio 的末端 IK 与碰撞检查。

模块导入本身不要求 Pinocchio；只有创建求解器时才检查依赖，因此场景列表、
Robot 状态和直接夹爪操作不会被高级规划依赖阻塞。
"""

from __future__ import annotations

from pathlib import Path

from semantic_robot_sdk_core.errors import PlanningError
from semantic_robot_sdk_core.models import EnvironmentCollisionObject, EnvironmentCollisionSet, Pose
from .dependencies import require_pinocchio


class PinocchioKinematics:
    def __init__(
        self,
        urdf_path: str | Path,
        frame_names: dict[str, str],
        *,
        mesh_directories: list[str] | None = None,
        disabled_collision_pairs: list[tuple[str, str]] | None = None,
        model_frame_id: str = "base_link",
        controlled_joints_by_end_effector: dict[str, list[str]] | None = None,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
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
    ) -> dict[str, float]:
        import numpy as np

        if end_effector not in self.frames:
            raise PlanningError(f"未知末端：{end_effector}")
        if target.frame_id != self.model_frame_id:
            raise PlanningError(f"PinocchioSolver 只接收已转换到 {self.model_frame_id} 的目标")
        controlled_names = self.controlled_joints_by_end_effector.get(end_effector)
        q, controlled = self._configuration(
            current_joints,
            controlled_names=controlled_names,
        )
        if not controlled:
            raise PlanningError(f"末端 {end_effector} 没有可控制关节")
        controlled_velocities = {index_v for _name, _index_q, index_v in controlled}

        x, y, z, w = target.quaternion_xyzw
        desired_in_root = self.pin.SE3(
            self.pin.Quaternion(w, x, y, z).matrix(),
            np.asarray(target.position, dtype=float),
        )
        frame_id = self.frames[end_effector]
        for _iteration in range(self.max_iterations):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            desired = self.data.oMf[self.root_frame].act(desired_in_root)
            current = self.data.oMf[frame_id]
            error = self.pin.log6(current.actInv(desired)).vector
            if float(np.linalg.norm(error)) < self.tolerance:
                result = {name: float(q[index_q]) for name, index_q, _index_v in controlled}
                candidate = dict(current_joints)
                candidate.update(result)
                if not self.collision_free(candidate, None):
                    raise PlanningError("IK 结果发生自碰撞")
                return result
            jacobian = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                self.pin.ReferenceFrame.LOCAL,
            )
            system = jacobian @ jacobian.T + self.damping * np.eye(6)
            velocity = jacobian.T @ np.linalg.solve(system, error)
            # 整个 R1 Pro URDF 还包含轮、转向和夹爪自由度。IK 只能改变调用方
            # 提供的关节；否则求解器可能借助未下发的关节得到一个无法执行的结果。
            for index in range(self.model.nv):
                if index not in controlled_velocities:
                    velocity[index] = 0.0
            q = self.pin.integrate(self.model, q, velocity * 0.1)
            q = np.minimum(
                np.maximum(q, self.model.lowerPositionLimit),
                self.model.upperPositionLimit,
            )
        raise PlanningError("Pinocchio IK 未在迭代上限内收敛")

    def collision_free(
        self,
        joint_positions: dict[str, float],
        environment: object | None,
    ) -> bool:
        """检查自碰撞以及静态环境碰撞。

        环境输入必须已经转换到运动学根坐标。Robot SDK 在调用本方法前负责从
        世界坐标转换，求解器不猜测 Robot 的世界位姿。
        """

        q, _controlled = self._configuration(joint_positions)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            q,
        )
        has_collision = self.pin.computeCollisions(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            q,
            False,
        )
        if bool(has_collision):
            return False
        if environment is None:
            return True
        if not isinstance(environment, EnvironmentCollisionSet):
            raise PlanningError("环境碰撞输入必须使用 EnvironmentCollisionSet")
        if environment.frame_id != self.model_frame_id:
            raise PlanningError(f"环境碰撞快照必须位于 {self.model_frame_id} 坐标系")
        return self._environment_collision_free(environment)

    def _environment_collision_free(self, environment: EnvironmentCollisionSet) -> bool:
        """使用 Coal/HPP-FCL 检查每个 Robot collision geom 与环境体。"""

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
            obstacles.append((item.source_id, geometry, transform))

        for index, robot_geometry in enumerate(self.geometry_model.geometryObjects):
            placement = self.geometry_data.oMg[index]
            robot_transform = coal.Transform3s(placement.rotation, placement.translation)
            for _source_id, obstacle_geometry, obstacle_transform in obstacles:
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
                    return False
        return True


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
