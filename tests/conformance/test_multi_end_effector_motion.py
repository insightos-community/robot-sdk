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

"""同步多末端笛卡尔路径回归。"""

from types import SimpleNamespace

import pytest

from semantic_robot_sdk_core import (
    EnvironmentCollisionObject,
    EnvironmentCollisionSet,
    JointLimit,
    JointTrajectoryPoint,
    MotionPlan,
    PlanKind,
    PlanningError,
    Pose,
    RobotCapabilities,
    RobotState,
)
from semantic_robot_sdk_r1pro.modules.upper_body import (
    UpperBodyModule,
    _cartesian_speed_scale,
    _complete_joint_waypoints,
    _pose_steps,
)
from semantic_robot_sdk_r1pro.providers.kinematics import LocalKinematicsProvider
from semantic_robot_sdk_r1pro.providers.local_kinematics import PinocchioKinematics
from semantic_robot_sdk_r1pro.providers.local_motion import (
    LocalMotionProvider,
    _simplify_joint_waypoints,
)
from semantic_robot_sdk_r1pro.providers.local_motion_algorithms import (
    validate_joint_goal,
)


class _Backend:
    def __init__(self, state: RobotState):
        self._state = state

    def state(self, _robot_id: str) -> RobotState:
        return self._state


def test_joint_goal_clamps_only_numerical_limit_overflow() -> None:
    limit = JointLimit(
        lower=-1.5708,
        upper=1.5708,
        max_velocity=1.0,
        max_acceleration=2.0,
        max_jerk=8.0,
    )
    # MuJoCo 放置阶段曾出现 55µrad 的数值溢出；这不是可执行的实际
    # 越界，轨迹参数化前应夹紧到 Profile 声明的硬限位。
    goal = {"joint": 1.5708552314597073}

    validate_joint_goal({"joint": 0.0}, goal, {"joint": limit})

    assert goal["joint"] == limit.upper
    with pytest.raises(PlanningError, match="超出"):
        validate_joint_goal(
            {"joint": 0.0},
            {"joint": limit.lower - 0.002},
            {"joint": limit},
        )


def test_pose_steps_does_not_add_segment_for_floating_point_boundary() -> None:
    start = Pose(
        position=(0.0, 0.0, 1.0712919625649984),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    target = start.model_copy(update={"position": (0.0, 0.0, start.position[2] + 0.08)})

    assert _pose_steps(start, target, maximum_translation_step_m=0.02) == 4


class _WaypointKinematics:
    name = "waypoint-test"

    def __init__(self):
        self.targets: list[dict[str, Pose]] = []
        self.environments: list[EnvironmentCollisionSet | None] = []
        self.fixed_end_effectors: list[frozenset[str]] = []
        self.continuous_seeds: list[bool] = []

    def solve_many(
        self,
        *,
        targets,
        fixed_end_effectors=frozenset(),
        continuous_seed=False,
        **_kwargs,
    ):
        self.targets.append(targets)
        fixed = frozenset(fixed_end_effectors)
        self.fixed_end_effectors.append(fixed)
        self.continuous_seeds.append(bool(continuous_seed))
        moving = next(name for name in targets if name not in fixed)
        return {"shared_joint": targets[moving].position[2]}

    def solve(self, *, end_effector, target, **_kwargs):
        self.targets.append({end_effector: target})
        return {"shared_joint": target.position[2]}

    def collision_free(self, _joints, _environment):
        self.environments.append(_environment)
        return True


class _GoalGuidedWaypointKinematics(_WaypointKinematics):
    """模拟局部连续IK需要最终关节分支引导的冗余系统。"""

    def __init__(self):
        super().__init__()
        self.continuous_seed_positions: list[float] = []

    def solve_many(
        self,
        *,
        targets,
        state,
        fixed_end_effectors=frozenset(),
        continuous_seed=False,
        **_kwargs,
    ):
        fixed = frozenset(fixed_end_effectors)
        moving = next(name for name in targets if name not in fixed)
        height = targets[moving].position[2]
        if not continuous_seed:
            assert height == pytest.approx(0.08)
            return {"shared_joint": 0.4}
        seed = state.joint_positions["shared_joint"]
        progress = height / 0.08
        expected_seed = 0.4 * min(1.0, progress + 0.125)
        self.continuous_seed_positions.append(seed)
        if seed != pytest.approx(expected_seed):
            raise PlanningError("局部IK落入无法到达终点的分支")
        return {"shared_joint": 0.4 * progress}

    def project_synchronized_end_effector_path(self, plan, **_kwargs):
        return plan


class _ProjectionFallbackKinematics(_WaypointKinematics):
    """模拟直达路径在固定末端投影阶段失败。"""

    def __init__(self):
        super().__init__()
        self.projection_attempts = 0

    def project_fixed_end_effector_path(self, plan, **_kwargs):
        self.projection_attempts += 1
        if self.projection_attempts == 1:
            raise PlanningError("直达路径无法保持固定末端")
        return plan


class _JointPathCollisionMotion(LocalMotionProvider):
    """模拟端点可达、但直达关节轨迹扫过障碍。"""

    def __init__(self, capabilities, kinematics):
        super().__init__(capabilities, kinematics)
        self.direct_attempts = 0

    def plan_joints(self, **_kwargs):
        self.direct_attempts += 1
        raise PlanningError("关节路径与环境碰撞")


class _StraightPathCollisionMotion(LocalMotionProvider):
    """模拟直达与直线末端路径碰撞，但升高后转移可用。"""

    def __init__(self, capabilities, kinematics):
        super().__init__(capabilities, kinematics)
        self.waypoint_attempts = 0

    def plan_joints(self, **_kwargs):
        raise PlanningError("关节直达路径与环境碰撞")

    def plan_joint_waypoints(self, **kwargs):
        self.waypoint_attempts += 1
        if self.waypoint_attempts == 1:
            raise PlanningError("直线末端路径与环境碰撞")
        return super().plan_joint_waypoints(**kwargs)


class _VariableWaypointKinematics(_WaypointKinematics):
    """模拟固定侧路径先仅动右臂、随后启用共享关节补偿。"""

    def solve_many(self, *, targets, fixed_end_effectors=frozenset(), **_kwargs):
        self.targets.append(targets)
        self.fixed_end_effectors.append(frozenset(fixed_end_effectors))
        height = targets["right"].position[2]
        if len(self.targets) <= 2:
            return {"right_joint": height}
        return {
            "left_joint": -height / 2.0,
            "right_joint": height,
            "shared_joint": height / 2.0,
        }


class _EnvironmentAwareSolver:
    def __init__(self):
        self.solve_environment = None

    def solve_end_effectors(self, _targets, _joints, *, environment=None):
        self.solve_environment = environment
        return {"shared_joint": 0.1}

    def collision_free(self, _joints, _environment):
        return True


class _FixedPathProjectionSolver:
    """用两个简单关节模拟共享关节插值造成的固定末端漂移。"""

    def __init__(self):
        self.projected_targets: list[dict[str, Pose]] = []
        self.collision_checks = 0

    def forward_kinematics(self, end_effector, joints):
        if end_effector == "left":
            position = (joints["shared"] + joints["left"], 0.4, 0.0)
        else:
            position = (joints["shared"] + joints["right"], -0.4, 0.0)
        return Pose(
            position=position,
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
        )

    def solve_end_effectors(
        self,
        targets,
        joints,
        *,
        environment=None,
        fixed_end_effectors=frozenset(),
        **_options,
    ):
        del environment, fixed_end_effectors
        self.projected_targets.append(targets)
        shared = joints["shared"]
        return {
            "shared": shared,
            "left": targets["left"].position[0] - shared,
            "right": targets["right"].position[0] - shared,
        }

    def collision_free(self, _joints, _environment):
        self.collision_checks += 1
        return True

    def collision_reason(self, _joints, _environment):
        return None


class _NonlinearFixedPathProjectionSolver(_FixedPathProjectionSolver):
    """模拟共享关节下非线性FK，使端点满足约束但关节插值会漂移。"""

    def forward_kinematics(self, end_effector, joints):
        shared = joints["shared"]
        if end_effector == "left":
            position = (shared + joints["left"] + shared**2, 0.4, 0.0)
        else:
            position = (shared + joints["right"], -0.4, 0.0)
        return Pose(
            position=position,
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
        )

    def solve_end_effectors(
        self,
        targets,
        joints,
        *,
        environment=None,
        fixed_end_effectors=frozenset(),
        **_options,
    ):
        del environment, fixed_end_effectors, joints
        self.projected_targets.append(targets)
        shared = targets["right"].position[0] * 0.5
        return {
            "shared": shared,
            "left": targets["left"].position[0] - shared - shared**2,
            "right": targets["right"].position[0] - shared,
        }


class _PartialFixedPathProjectionSolver(_FixedPathProjectionSolver):
    """模拟约束IK在部分采样点只需要返回运动臂关节。"""

    def solve_end_effectors(
        self,
        targets,
        joints,
        *,
        environment=None,
        fixed_end_effectors=frozenset(),
        **_options,
    ):
        del environment, fixed_end_effectors
        self.projected_targets.append(targets)
        # 求解器基于上一投影状态求解；共享和左臂不需要变化时，真实
        # Pinocchio Provider同样只返回右臂，而不是重复返回全部关节。
        return {
            "right": targets["right"].position[0] - joints["shared"],
        }


class _SynchronizedPathProjectionSolver:
    """模拟双臂共享关节插值造成的左右末端相对漂移。"""

    def __init__(self):
        self.collision_checks = 0

    def forward_kinematics(self, end_effector, joints):
        shared = joints["shared"]
        if end_effector == "left":
            height = shared + joints["left"] + shared**2
            lateral = 0.4
        else:
            height = shared + joints["right"] - shared**2
            lateral = -0.4
        return Pose(
            position=(0.0, lateral, height),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
        )

    def solve_end_effectors(self, targets, _joints, *, environment=None, **_options):
        del environment
        shared = (targets["left"].position[2] + targets["right"].position[2]) / 4.0
        return {
            "shared": shared,
            "left": targets["left"].position[2] - shared - shared**2,
            "right": targets["right"].position[2] - shared + shared**2,
        }

    def collision_free(self, _joints, _environment):
        self.collision_checks += 1
        return True

    def collision_reason(self, _joints, _environment):
        return None


class _SynchronizedBoundarySolver:
    """模拟连续分支在毫米级抬升处失效、有限择优仍有近邻解。"""

    def __init__(self):
        self.continuous_seeds: list[bool] = []

    def forward_kinematics(self, end_effector, joints):
        return Pose(
            position=(
                0.0,
                0.4 if end_effector == "left" else -0.4,
                joints["height"],
            ),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
        )

    def solve_end_effectors(self, targets, _joints, *, continuous_seed=False, environment=None):
        del environment
        self.continuous_seeds.append(bool(continuous_seed))
        if continuous_seed:
            raise PlanningError("局部连续分支位于关节边界")
        return {"height": targets["left"].position[2]}

    def collision_free(self, _joints, _environment):
        return True

    def collision_reason(self, _joints, _environment):
        return None


class _CollectiveDriftSolver:
    def forward_kinematics(self, end_effector, joints):
        height = joints["height"]
        return Pose(
            position=(
                joints["drift"],
                0.4 if end_effector == "left" else -0.4,
                height,
            ),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
        )


def test_projected_path_simplification_uses_geometry_not_sample_index() -> None:
    """非等距时间采样的直线路径只能保留几何首尾点。"""

    samples = [
        {
            "joint_a": ratio * 0.4,
            "joint_b": ratio * -0.2,
        }
        for ratio in (0.0, 0.001, 0.01, 0.08, 0.35, 0.72, 0.94, 0.995, 1.0)
    ]

    simplified = _simplify_joint_waypoints(samples, tolerance_rad=1e-6)

    assert simplified == [samples[0], samples[-1]]


def test_variable_ik_joint_sets_keep_previous_values_in_one_path() -> None:
    goals = _complete_joint_waypoints(
        [
            {"right": 0.2},
            {"shared": 0.1, "left": -0.1, "right": 0.3},
        ],
        {"shared": 0.0, "left": 0.0, "right": 0.0},
    )

    assert goals == [
        {"left": 0.0, "right": 0.2, "shared": 0.0},
        {"left": -0.1, "right": 0.3, "shared": 0.1},
    ]


def test_joint_path_transforms_world_collision_snapshot_to_kinematic_root() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(1.0, 2.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared_joint": 0.0},
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=[],
        grippers=[],
        sensors=[],
    )
    environment = EnvironmentCollisionSet(
        frame_id="world",
        objects=[
            EnvironmentCollisionObject(
                source_id="box-1",
                shape="box",
                pose=Pose(
                    position=(2.0, 4.0, 0.5),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id="world",
                ),
                size_xyz=(0.6, 0.4, 0.34),
            )
        ],
    )
    kinematics = _WaypointKinematics()
    motion = LocalMotionProvider(capabilities, kinematics)

    motion.plan_joints(
        robot_id=state.robot_id,
        state=state,
        goal={"shared_joint": 0.1},
        environment=environment,
    )

    assert kinematics.environments
    assert all(item is not None for item in kinematics.environments)
    first = kinematics.environments[0]
    assert first is not None
    assert first.frame_id == "base_link"
    assert first.objects[0].pose.position == pytest.approx((1.0, 2.0, 0.5))


def test_multi_ik_uses_environment_while_selecting_a_solution() -> None:
    """环境碰撞必须参与多起点IK，不能只在首个解返回后检查。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(1.0, 2.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared_joint": 0.0},
    )
    environment = EnvironmentCollisionSet(
        frame_id="world",
        objects=[
            EnvironmentCollisionObject(
                source_id="box-1",
                shape="box",
                pose=Pose(
                    position=(2.0, 4.0, 0.5),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id="world",
                ),
                size_xyz=(0.6, 0.4, 0.34),
            )
        ],
    )
    solver = _EnvironmentAwareSolver()
    provider = LocalKinematicsProvider(solver)

    provider.solve_many(
        robot_id=state.robot_id,
        targets={
            "left": Pose(
                position=(1.5, 2.5, 0.8),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            )
        },
        state=state,
        environment=environment,
    )

    assert solver.solve_environment is not None
    assert solver.solve_environment.frame_id == "base_link"
    assert solver.solve_environment.objects[0].pose.position == pytest.approx((1.0, 2.0, 0.5))


def test_time_parameterized_path_is_projected_back_to_fixed_end_effector() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared": 0.0, "left": 0.0, "right": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    plan = MotionPlan(
        plan_id="plan-fixed-path",
        robot_id=state.robot_id,
        generation=1,
        kind=PlanKind.JOINT,
        resources=["joints:upper_body"],
        frame_id="base_link",
        start={"shared": 0.0, "left": 0.0, "right": 0.0},
        goal={"shared": 0.2, "left": 0.0, "right": 0.2},
        joint_trajectory=[
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={"shared": 0.0, "left": 0.0, "right": 0.0},
                velocities={"shared": 0.0, "left": 0.0, "right": 0.0},
            ),
            JointTrajectoryPoint(
                time_from_start_s=1.0,
                positions={"shared": 0.1, "left": 0.0, "right": 0.1},
                velocities={"shared": 0.1, "left": 0.0, "right": 0.1},
            ),
            JointTrajectoryPoint(
                time_from_start_s=2.0,
                positions={"shared": 0.2, "left": 0.0, "right": 0.2},
                velocities={"shared": 0.0, "left": 0.0, "right": 0.0},
            ),
        ],
        collision_checked=True,
        estimated_duration_s=2.0,
        planner="cartesian-waypoints+ruckig",
    )
    solver = _FixedPathProjectionSolver()
    provider = LocalKinematicsProvider(solver)

    projected = provider.project_fixed_end_effector_path(
        plan,
        state=state,
        targets={
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(update={"position": (0.4, -0.4, 0.0)}),
        },
        fixed_end_effectors=frozenset({"left"}),
    )

    for point in projected.joint_trajectory:
        joints = dict(state.joint_positions)
        joints.update(point.positions)
        assert solver.forward_kinematics("left", joints).position[0] == pytest.approx(0.0)
    final_joints = dict(state.joint_positions)
    final_joints.update(projected.joint_trajectory[-1].positions)
    assert solver.forward_kinematics("right", final_joints).position[0] == pytest.approx(0.4)
    # 投影改变了路径几何，旧速度不能继续沿用；Motion Provider会随后重定时。
    assert projected.joint_trajectory[1].velocities == {}
    assert solver.collision_checks == len(projected.joint_trajectory)
    assert "fixed-cartesian-projection" in projected.planner


def test_partial_fixed_projection_keeps_previous_constrained_joints() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared": 0.0, "left": 0.0, "right": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
        },
    )
    plan = MotionPlan(
        plan_id="plan-partial-fixed-path",
        robot_id=state.robot_id,
        generation=1,
        kind=PlanKind.JOINT,
        resources=["joints:upper_body"],
        frame_id="base_link",
        start=dict(state.joint_positions),
        goal={"shared": 0.2, "left": 0.0, "right": 0.2},
        joint_trajectory=[
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={"shared": 0.0, "left": 0.0, "right": 0.0},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=1.0,
                positions={"shared": 0.1, "left": 0.0, "right": 0.1},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=2.0,
                positions={"shared": 0.2, "left": 0.0, "right": 0.2},
                velocities={},
            ),
        ],
        collision_checked=True,
        estimated_duration_s=2.0,
        planner="test",
    )
    solver = _PartialFixedPathProjectionSolver()
    provider = LocalKinematicsProvider(solver)

    projected = provider.project_fixed_end_effector_path(
        plan,
        state=state,
        targets={
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(update={"position": (0.4, -0.4, 0.0)}),
        },
        fixed_end_effectors=frozenset({"left"}),
    )

    for point in projected.joint_trajectory:
        # 未被部分IK返回的共享和左臂关节必须保持上一约束点，不能从原始
        # 无约束轨迹恢复，否则左末端会随shared产生同等幅度的漂移。
        assert point.positions["shared"] == pytest.approx(0.0)
        assert point.positions["left"] == pytest.approx(0.0)
        joints = dict(state.joint_positions)
        joints.update(point.positions)
        assert solver.forward_kinematics("left", joints).position[0] == pytest.approx(0.0)
    final_joints = dict(state.joint_positions)
    final_joints.update(projected.joint_trajectory[-1].positions)
    assert solver.forward_kinematics("right", final_joints).position[0] == pytest.approx(0.4)


def test_preserved_multi_end_effector_path_projects_between_ik_waypoints() -> None:
    """双末端同步约束必须覆盖Ruckig路点之间，不能只保证IK端点。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared": 0.0, "left": 0.0, "right": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    limits = {
        name: JointLimit(
            lower=-2.0,
            upper=2.0,
            max_velocity=1.0,
            max_acceleration=2.0,
            max_jerk=8.0,
        )
        for name in state.joint_positions
    }
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=list(limits),
        joint_groups={"upper_body": list(limits)},
        joint_limits=limits,
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    solver = _SynchronizedPathProjectionSolver()
    kinematics = LocalKinematicsProvider(solver)
    module = UpperBodyModule(
        state.robot_id,
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, solver),
    )

    plan = module.plan_end_effectors(
        {
            name: pose.model_copy(update={"position": (pose.position[0], pose.position[1], 0.08)})
            for name, pose in state.end_effectors.items()
        },
        # 携物调用只声明相对几何；Local Provider自行选择连续笛卡尔路径。
        preserve_cartesian_path=False,
        preserve_relative_geometry=True,
    )

    assert "synchronized-cartesian-projection+retimed" in plan.planner
    assert "retimed+simplified-ruckig" in plan.planner
    assert plan.estimated_duration_s < 5.0
    assert solver.collision_checks > len(plan.joint_trajectory)
    for point in plan.joint_trajectory:
        joints = dict(state.joint_positions)
        joints.update(point.positions)
        left = solver.forward_kinematics("left", joints)
        right = solver.forward_kinematics("right", joints)
        assert abs(left.position[2] - right.position[2]) <= (kinematics.fixed_position_tolerance_m)


def test_safe_collective_curve_is_not_rejected_after_retiming() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"height": 0.0, "drift": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
        },
    )
    targets = {
        name: pose.model_copy(update={"position": (0.0, pose.position[1], 0.08)})
        for name, pose in state.end_effectors.items()
    }
    plan = MotionPlan(
        plan_id="plan-collective-drift",
        robot_id=state.robot_id,
        generation=1,
        kind=PlanKind.JOINT,
        resources=["joints:upper_body"],
        frame_id="base_link",
        start=state.joint_positions,
        goal={"height": 0.08, "drift": 0.0},
        joint_trajectory=[
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={"height": 0.0, "drift": 0.0},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=0.5,
                positions={"height": 0.04, "drift": 0.01},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=1.0,
                positions={"height": 0.08, "drift": 0.0},
                velocities={},
            ),
        ],
        collision_checked=True,
        estimated_duration_s=1.0,
        planner="test",
    )
    provider = LocalKinematicsProvider(_CollectiveDriftSolver())

    assert provider.path_keeps_relative_end_effector_geometry(plan, state=state, targets=targets)
    # 两端保持相对几何并最终到达目标时，中间共同形成小幅曲线是合法的。
    # 碰撞以及运行中的接触/滑移仍由Motion和Ability分别判断。
    assert provider.path_reaches_synchronized_end_effector_targets(
        plan, state=state, targets=targets
    )


def test_synchronized_projection_recovers_from_local_branch_boundary() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"height": 0.0},
        end_effectors={
            side: Pose(
                position=(0.0, 0.4 if side == "left" else -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            )
            for side in ("left", "right")
        },
    )
    targets = {
        side: pose.model_copy(update={"position": (0.0, pose.position[1], 0.02)})
        for side, pose in state.end_effectors.items()
    }
    plan = MotionPlan(
        plan_id="plan-boundary",
        robot_id=state.robot_id,
        generation=1,
        kind=PlanKind.JOINT,
        resources=["joints:upper_body"],
        frame_id="base_link",
        start={"height": 0.0},
        goal={"height": 0.02},
        joint_trajectory=[
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={"height": 0.0},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=0.5,
                positions={"height": 0.01},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=1.0,
                positions={"height": 0.02},
                velocities={},
            ),
        ],
        collision_checked=True,
        estimated_duration_s=1.0,
        planner="test",
    )
    solver = _SynchronizedBoundarySolver()
    projected = LocalKinematicsProvider(solver).project_synchronized_end_effector_path(
        plan, state=state, targets=targets
    )

    assert projected.joint_trajectory[-1].positions["height"] == pytest.approx(0.02)
    assert solver.continuous_seeds == [True, False, True, False]


def test_synchronized_path_must_reach_both_end_effector_targets() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"height": 0.0, "drift": 0.0},
        end_effectors={
            side: Pose(
                position=(0.0, offset, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            )
            for side, offset in (("left", 0.4), ("right", -0.4))
        },
    )
    targets = {
        name: pose.model_copy(update={"position": (0.0, pose.position[1], 0.08)})
        for name, pose in state.end_effectors.items()
    }
    plan = MotionPlan(
        plan_id="plan-incomplete-lift",
        robot_id=state.robot_id,
        generation=1,
        kind=PlanKind.JOINT,
        resources=["joints:upper_body"],
        frame_id="base_link",
        start=state.joint_positions,
        goal={"height": 0.06, "drift": 0.0},
        joint_trajectory=[
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={"height": 0.0, "drift": 0.0},
                velocities={},
            ),
            JointTrajectoryPoint(
                time_from_start_s=1.0,
                positions={"height": 0.06, "drift": 0.0},
                velocities={},
            ),
        ],
        collision_checked=True,
        estimated_duration_s=1.0,
        planner="test",
    )
    provider = LocalKinematicsProvider(_CollectiveDriftSolver())

    assert not provider.path_reaches_synchronized_end_effector_targets(
        plan, state=state, targets=targets
    )


def test_path_already_within_fixed_tolerance_is_not_projected_again() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared": 0.0, "left": 0.0, "right": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    plan = MotionPlan(
        plan_id="plan-independent-arm",
        robot_id=state.robot_id,
        generation=1,
        kind=PlanKind.JOINT,
        resources=["joints:right_arm"],
        frame_id="base_link",
        start={"right": 0.0},
        goal={"right": 0.4},
        joint_trajectory=[
            JointTrajectoryPoint(
                time_from_start_s=0.0,
                positions={"right": 0.0},
                velocities={"right": 0.0},
            ),
            JointTrajectoryPoint(
                time_from_start_s=1.0,
                positions={"right": 0.4},
                velocities={"right": 0.0},
            ),
        ],
        collision_checked=True,
        estimated_duration_s=1.0,
        planner="ruckig",
    )
    solver = _FixedPathProjectionSolver()
    solver.tolerance = 1e-4
    provider = LocalKinematicsProvider(solver)

    projected = provider.project_fixed_end_effector_path(
        plan,
        state=state,
        targets={
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(update={"position": (0.4, -0.4, 0.0)}),
        },
        fixed_end_effectors=frozenset({"left"}),
    )

    # 右臂独立轨迹已经保持左侧末端时，不应再求一次冗余IK并引入分支跳变。
    assert projected is plan
    assert solver.projected_targets == []

    # IK收敛精度不是动作期间的物理保持容差。1 mm的共享关节插值漂移仍在
    # RobotDeployment声明的2 mm保持范围内，不应触发逐采样重投影。
    slightly_drifting = plan.model_copy(
        update={
            "joint_trajectory": [
                plan.joint_trajectory[0].model_copy(
                    update={"positions": {"shared": 0.0, "right": 0.0}}
                ),
                plan.joint_trajectory[1].model_copy(
                    update={"positions": {"shared": 0.001, "right": 0.4}}
                ),
            ]
        }
    )
    provider = LocalKinematicsProvider(
        solver,
        fixed_position_tolerance_m=0.002,
        fixed_orientation_tolerance_rad=0.02,
    )
    within_physical_tolerance = provider.project_fixed_end_effector_path(
        slightly_drifting,
        state=state,
        targets={
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(update={"position": (0.4, -0.4, 0.0)}),
        },
        fixed_end_effectors=frozenset({"left"}),
    )
    assert within_physical_tolerance is slightly_drifting


def test_continuous_constrained_waypoints_keep_fixed_end_effector() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared": 0.0, "left": 0.0, "right": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    limits = {
        name: JointLimit(
            lower=-2.0,
            upper=2.0,
            max_velocity=1.0,
            max_acceleration=2.0,
            max_jerk=8.0,
        )
        for name in state.joint_positions
    }
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=list(limits),
        joint_groups={"upper_body": list(limits)},
        joint_limits=limits,
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    solver = _NonlinearFixedPathProjectionSolver()
    kinematics = LocalKinematicsProvider(
        solver,
        fixed_position_tolerance_m=0.002,
        fixed_orientation_tolerance_rad=0.02,
    )
    motion = LocalMotionProvider(capabilities, solver)
    module = UpperBodyModule(state.robot_id, _Backend(state), kinematics, motion)

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(update={"position": (0.4, -0.4, 0.0)}),
        },
        preserve_cartesian_path=False,
        fixed_end_effectors={"left"},
    )

    assert "fixed-cartesian-projection+retimed" in plan.planner
    assert "retimed+simplified-ruckig" in plan.planner
    assert "fixed-constraint" in plan.planner
    assert plan.joint_trajectory[0].velocities == {
        "shared": 0.0,
        "left": 0.0,
        "right": 0.0,
    }
    assert plan.joint_trajectory[-1].velocities == {
        "shared": 0.0,
        "left": 0.0,
        "right": 0.0,
    }
    for point in plan.joint_trajectory:
        joints = dict(state.joint_positions)
        joints.update(point.positions)
        assert abs(solver.forward_kinematics("left", joints).position[0]) <= (
            kinematics.fixed_position_tolerance_m
        )
    # 投影后的几何点和Ruckig重新生成的时间轨迹都经过碰撞复查；两次采样
    # 数量无需相同。非线性固定端路径可以比无约束直达更慢，但不能因数值
    # 投影噪声膨胀到分钟级动作。
    assert solver.collision_checks > len(plan.joint_trajectory)
    assert plan.estimated_duration_s < 30.0


def test_fixed_path_anchors_contact_deformation_to_current_joint_fk() -> None:
    """受力后的Runtime末端偏差不能被规划成一次新的承载侧运动。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        base_pose=Pose(
            position=(1.0, 2.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        joint_positions={"shared": 0.0, "left": 0.0, "right": 0.0},
        end_effectors={
            # 模拟承载接触和柔性造成的1 cm Runtime观测偏差；刚性FK仍为x=0。
            "left": Pose(
                position=(1.01, 2.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(1.0, 1.6, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    limits = {
        name: JointLimit(
            lower=-2.0,
            upper=2.0,
            max_velocity=1.0,
            max_acceleration=2.0,
            max_jerk=8.0,
        )
        for name in state.joint_positions
    }
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=list(limits),
        joint_groups={"upper_body": list(limits)},
        joint_limits=limits,
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    solver = _FixedPathProjectionSolver()
    solver.tolerance = 1e-4
    kinematics = LocalKinematicsProvider(solver)
    motion = LocalMotionProvider(capabilities, solver)
    module = UpperBodyModule(state.robot_id, _Backend(state), kinematics, motion)

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": Pose(
                position=(1.4, 1.6, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
        preserve_cartesian_path=False,
        fixed_end_effectors={"left"},
    )

    assert plan.estimated_duration_s < 10.0
    assert "fixed-cartesian-projection" not in plan.planner
    for point in plan.joint_trajectory:
        joints = dict(state.joint_positions)
        joints.update(point.positions)
        assert solver.forward_kinematics("left", joints).position[0] == pytest.approx(0.0)
    # 对外结果仍记录调用方请求的真实Runtime位姿，而不是内部FK锚点。
    assert plan.goal["targets"]["left"]["position"] == [1.01, 2.4, 0.0]


def test_multi_end_effector_path_solves_synchronized_cartesian_waypoints() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    module = UpperBodyModule(
        "r1pro-test",
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    plan = module.plan_end_effectors(
        {
            name: pose.model_copy(update={"position": (pose.position[0], pose.position[1], 0.08)})
            for name, pose in state.end_effectors.items()
        },
        speed_scale=0.5,
        preserve_cartesian_path=True,
    )

    # 第一次调用求终点引导；随后8 cm同步抬升拆成四个2 cm路点，
    # 每个路点同时包含左右末端。
    assert len(kinematics.targets) == 5
    assert [targets["left"].position[2] for targets in kinematics.targets[1:]] == pytest.approx(
        [0.02, 0.04, 0.06, 0.08]
    )
    assert all(set(targets) == {"left", "right"} for targets in kinematics.targets)
    # 内部 2 cm 路点精确经过但不停车；只有整段末端速度归零。这样能保持
    # 双末端同步几何，又不会把 8 cm 抬升拆成四次独立启停。
    internal = [
        point
        for point in plan.joint_trajectory
        if point.positions["shared_joint"] == pytest.approx(0.02, abs=1e-6)
    ]
    assert internal
    assert internal[-1].velocities["shared_joint"] > 0
    assert plan.joint_trajectory[-1].velocities["shared_joint"] == pytest.approx(0.0)
    assert plan.goal["targets"]["left"]["position"] == pytest.approx((0.0, 0.4, 0.08))
    assert "cartesian-waypoints+ruckig" in plan.planner
    assert plan.joint_trajectory[-1].positions["shared_joint"] == pytest.approx(0.08)


@pytest.mark.parametrize(
    "path_options",
    [
        {"preserve_cartesian_path": True},
        {"preserve_relative_geometry": True},
    ],
)
def test_synchronized_path_uses_reachable_goal_branch_to_guide_waypoints(
    path_options,
) -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _GoalGuidedWaypointKinematics()
    module = UpperBodyModule(
        state.robot_id,
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    plan = module.plan_end_effectors(
        {
            name: pose.model_copy(update={"position": (pose.position[0], pose.position[1], 0.08)})
            for name, pose in state.end_effectors.items()
        },
        **path_options,
    )

    # 先求出的可达终点分支按几何进度引导每个路点；若仍只以上一局部
    # 解作为种子，本测试模拟的冗余IK会在中途进入无法到达终点的分支。
    assert kinematics.continuous_seed_positions == pytest.approx([0.15, 0.25, 0.35, 0.4])
    assert plan.joint_trajectory[-1].positions["shared_joint"] == pytest.approx(0.4)


def test_fixed_cartesian_path_uses_reachable_goal_branch_to_guide_waypoints() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _GoalGuidedWaypointKinematics()
    module = UpperBodyModule(
        state.robot_id,
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(
                update={"position": (0.0, -0.4, 0.08)}
            ),
        },
        preserve_cartesian_path=True,
        fixed_end_effectors={"left"},
    )

    assert kinematics.continuous_seed_positions == pytest.approx([0.15, 0.25, 0.35, 0.4])
    assert plan.joint_trajectory[-1].positions["shared_joint"] == pytest.approx(0.4)


def test_fixed_side_falls_back_to_continuous_constrained_waypoint_plan() -> None:
    """关节路径碰撞时应回退同步约束路点，并保持承载侧。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    motion = _JointPathCollisionMotion(capabilities, kinematics)
    module = UpperBodyModule(state.robot_id, _Backend(state), kinematics, motion)

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(
                update={"position": (0.0, -0.4, 0.08)}
            ),
        },
        preserve_cartesian_path=False,
        fixed_end_effectors={"left"},
    )

    assert motion.direct_attempts == 1
    # 一次终点求解失败后，四个同步路点从起点就包含固定端约束。
    assert len(kinematics.targets) == 5
    assert "cartesian-waypoints+ruckig+fixed-constraint" in plan.planner
    assert plan.joint_trajectory[-1].positions["shared_joint"] == pytest.approx(0.08)


def test_fixed_projection_failure_uses_cartesian_fallback() -> None:
    """固定侧投影失败属于当前路径失败，不能阻断后续通用回退。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            side: Pose(
                position=(0.0, offset, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            )
            for side, offset in (("left", 0.4), ("right", -0.4))
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _ProjectionFallbackKinematics()
    module = UpperBodyModule(
        state.robot_id,
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(
                update={"position": (0.0, -0.4, 0.08)}
            ),
        },
        fixed_end_effectors={"left"},
    )

    assert kinematics.projection_attempts == 2
    assert "collision-fallback" in plan.planner


def test_straight_path_collision_uses_elevated_transfer_detour() -> None:
    """直线路径也受阻时，SDK先在当前XY升高，再连续转移到目标。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    motion = _StraightPathCollisionMotion(capabilities, kinematics)
    module = UpperBodyModule(state.robot_id, _Backend(state), kinematics, motion)

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(
                update={
                    "position": (0.3, -0.1, 0.08),
                    "quaternion_xyzw": (0.0, 0.0, 0.7071068, 0.7071068),
                }
            ),
        },
        preserve_cartesian_path=False,
        fixed_end_effectors={"left"},
    )

    assert motion.waypoint_attempts == 2
    elevated = next(
        targets["right"]
        for targets in kinematics.targets
        if targets["right"].position == pytest.approx((0.0, -0.4, 0.08))
    )
    assert elevated.quaternion_xyzw == state.end_effectors["right"].quaternion_xyzw
    aligned = next(
        targets["right"]
        for targets in kinematics.targets
        if targets["right"].position == pytest.approx((0.3, -0.1, 0.08))
        and targets["right"].quaternion_xyzw == state.end_effectors["right"].quaternion_xyzw
    )
    assert aligned.position == pytest.approx((0.3, -0.1, 0.08))
    assert "+elevated-detour" in plan.planner


def test_multi_end_effector_preflight_can_start_from_predicted_state() -> None:
    """连续候选预检的下一段必须从上一段预测终态开始。"""

    backend_state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            side: Pose(
                position=(0.0, offset, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            )
            for side, offset in (("left", 0.4), ("right", -0.4))
        },
    )
    predicted_state = backend_state.model_copy(
        update={
            "joint_positions": {"shared_joint": 0.2},
            "end_effectors": {
                side: pose.model_copy(update={"position": (0.0, pose.position[1], 0.2)})
                for side, pose in backend_state.end_effectors.items()
            },
        }
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    module = UpperBodyModule(
        "r1pro-test",
        _Backend(backend_state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    module.plan_end_effectors(
        {
            side: pose.model_copy(update={"position": (0.0, pose.position[1], 0.24)})
            for side, pose in predicted_state.end_effectors.items()
        },
        state=predicted_state,
        preserve_cartesian_path=True,
    )

    assert len(kinematics.targets) == 3
    assert kinematics.targets[1]["left"].position[2] == pytest.approx(0.22)


def test_multi_end_effector_path_can_keep_one_end_effector_fixed() -> None:
    """短接触动作保持固定端，并保留多个Cartesian几何段。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    module = UpperBodyModule(
        state.robot_id,
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(
                update={"position": (0.0, -0.4, 0.012)}
            ),
        },
        preserve_cartesian_path=True,
        fixed_end_effectors={"left"},
    )
    assert len(kinematics.targets) == 4
    assert kinematics.fixed_end_effectors
    assert all(value == frozenset({"left"}) for value in kinematics.fixed_end_effectors)
    assert kinematics.continuous_seeds == [False, True, True, True]


def test_fixed_side_path_accepts_variable_ik_joint_sets() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"left_joint": 0.0, "right_joint": 0.0, "shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
            "right": Pose(
                position=(0.0, -0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            ),
        },
    )
    limits = {
        name: JointLimit(
            lower=-1.0, upper=1.0, max_velocity=1.0, max_acceleration=2.0, max_jerk=8.0
        )
        for name in state.joint_positions
    }
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=list(state.joint_positions),
        joint_groups={"upper_body": list(state.joint_positions)},
        joint_limits=limits,
        end_effectors=["left", "right"],
        grippers=[],
        sensors=[],
    )
    kinematics = _VariableWaypointKinematics()
    module = UpperBodyModule(
        state.robot_id,
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    plan = module.plan_end_effectors(
        {
            "left": state.end_effectors["left"],
            "right": state.end_effectors["right"].model_copy(
                update={"position": (0.0, -0.4, 0.08)}
            ),
        },
        preserve_cartesian_path=True,
        fixed_end_effectors={"left"},
    )

    assert all(
        set(point.positions) == {"left_joint", "right_joint", "shared_joint"}
        for point in plan.joint_trajectory
    )


def test_single_end_effector_path_preserves_cartesian_waypoints() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            )
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    module = UpperBodyModule(
        "r1pro-test",
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    plan = module.plan_end_effector(
        "left",
        state.end_effectors["left"].model_copy(update={"position": (0.0, 0.4, 0.08)}),
        preserve_cartesian_path=True,
    )

    assert len(kinematics.targets) == 4
    assert [targets["left"].position[2] for targets in kinematics.targets] == pytest.approx(
        [0.02, 0.04, 0.06, 0.08]
    )
    assert "cartesian-waypoints+ruckig" in plan.planner


def test_long_cartesian_path_uses_bounded_adaptive_waypoints() -> None:
    """长距离安全抬升不应因固定2 cm路点生成不可用的分钟级轨迹。"""

    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
        end_effectors={
            "left": Pose(
                position=(0.0, 0.4, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            )
        },
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=["left"],
        grippers=[],
        sensors=[],
    )
    kinematics = _WaypointKinematics()
    module = UpperBodyModule(
        "r1pro-test",
        _Backend(state),
        kinematics,
        LocalMotionProvider(capabilities, kinematics),
    )

    module.plan_end_effector(
        "left",
        state.end_effectors["left"].model_copy(update={"position": (0.0, 0.4, 0.64)}),
        preserve_cartesian_path=True,
    )

    assert len(kinematics.targets) == 8
    assert kinematics.targets[-1]["left"].position[2] == pytest.approx(0.64)


def test_joint_waypoint_duration_obeys_time_scaling() -> None:
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"shared_joint": 0.0},
    )
    capabilities = RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["shared_joint"],
        joint_groups={"upper_body": ["shared_joint"]},
        joint_limits={
            "shared_joint": JointLimit(
                lower=-1.0,
                upper=1.0,
                max_velocity=1.0,
                max_acceleration=2.0,
                max_jerk=8.0,
            )
        },
        end_effectors=[],
        grippers=[],
        sensors=[],
    )
    motion = LocalMotionProvider(capabilities, _WaypointKinematics())
    goals = [{"shared_joint": value} for value in (0.02, 0.04, 0.06, 0.08)]

    full = motion.plan_joint_waypoints(
        robot_id="r1pro-test",
        state=state,
        goals=goals,
        speed_scale=1.0,
    )
    slow = motion.plan_joint_waypoints(
        robot_id="r1pro-test",
        state=state,
        goals=goals,
        speed_scale=0.2,
    )

    # velocity/acceleration/jerk 分别按 s/s²/s³ 缩放时，同一几何路径
    # 应只在时间轴上拉伸 1/s；若路点速度混入未缩放常数，此断言会出现
    # 数量级偏差，正是原生 MuJoCo 抬升曾生成数百秒轨迹的根因。
    assert slow.estimated_duration_s == pytest.approx(
        full.estimated_duration_s / 0.2,
        rel=0.08,
    )


def test_cartesian_speed_limit_only_slows_a_plan_that_is_actually_too_fast() -> None:
    starts = {
        "left": Pose(
            position=(0.0, 0.4, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        "right": Pose(
            position=(0.0, -0.4, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
    }
    targets = {
        name: start.model_copy(update={"position": (start.position[0], start.position[1], 0.08)})
        for name, start in starts.items()
    }

    # 8 cm / 0.05 m/s 的线速度上限要求至少 1.6 秒。关节约束已经让轨迹
    # 长达 12 秒时不能再减速；只有原轨迹快于 1.6 秒时才拉长时间轴。
    assert _cartesian_speed_scale(
        starts,
        targets,
        planned_duration_s=12.0,
        maximum_speed_mps=0.05,
        current_scale=1.0,
    ) == pytest.approx(1.0)
    assert _cartesian_speed_scale(
        starts,
        targets,
        planned_duration_s=0.8,
        maximum_speed_mps=0.05,
        current_scale=1.0,
    ) == pytest.approx(0.5)


def test_ik_fallback_starts_near_current_configuration_before_expanding() -> None:
    """冗余IK先探索当前姿态附近，不能从任意大幅翻腕分支开始。"""

    solver = PinocchioKinematics.__new__(PinocchioKinematics)
    solver.model = SimpleNamespace(
        lowerPositionLimit=[-1.0],
        upperPositionLimit=[1.0],
    )
    initial = [0.2]
    controlled = [("joint", 0, 0)]

    candidates = list(solver._fallback_configurations(initial, controlled))

    assert len(candidates) == solver._IK_FALLBACK_ATTEMPTS
    assert abs(candidates[0][0] - initial[0]) <= 0.12
    assert max(abs(item[0] - initial[0]) for item in candidates[4:]) > 0.3
    assert all(-0.8 <= item[0] <= 0.8 for item in candidates)
