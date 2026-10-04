"""R1 Pro 本地运动规划的物理时间缩放回归。"""

import pytest

from semantic_robot_sdk_core import (
    JointLimit,
    JointTrajectoryPoint,
    PlanningError,
    RobotCapabilities,
    RobotState,
)
from semantic_robot_sdk_r1pro.providers.local_motion import LocalMotionProvider


class _CollisionFreeKinematics:
    def collision_free(self, _joints, _environment):
        return True


def _capabilities() -> RobotCapabilities:
    limit = JointLimit(
        lower=-2.0,
        upper=2.0,
        max_velocity=1.0,
        max_acceleration=2.0,
        max_jerk=8.0,
    )
    return RobotCapabilities(
        model="r1pro",
        kind="mobile_manipulator",
        joint_names=["joint_1"],
        joint_groups={"arm": ["joint_1"]},
        joint_limits={"joint_1": limit},
        end_effectors=[],
        grippers=[],
        sensors=[],
    )


def test_speed_scale_applies_consistent_time_scaling() -> None:
    provider = LocalMotionProvider(
        _capabilities(),
        _CollisionFreeKinematics(),
    )
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"joint_1": 0.0},
    )

    normal = provider.plan_joints(
        robot_id="r1pro-test",
        state=state,
        goal={"joint_1": 0.2},
        speed_scale=1.0,
    )
    slow = provider.plan_joints(
        robot_id="r1pro-test",
        state=state,
        goal={"joint_1": 0.2},
        speed_scale=0.1,
    )

    # 速度比例描述同一路径的时间伸缩，因此速度、加速度和 jerk 必须分别
    # 按 s、s²、s³ 缩放。只缩最大速度会让短距离动作仍受未缩放的加速度
    # 主导，携物抬升即使请求低速也会以近乎原速度执行。
    assert slow.estimated_duration_s >= normal.estimated_duration_s * 9.5


def test_current_joint_goal_produces_valid_stationary_plan() -> None:
    provider = LocalMotionProvider(
        _capabilities(),
        _CollisionFreeKinematics(),
    )
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"joint_1": 0.25},
    )

    plan = provider.plan_joints(
        robot_id="r1pro-test",
        state=state,
        goal={"joint_1": 0.25},
    )

    assert [point.time_from_start_s for point in plan.joint_trajectory] == [0.0, 0.01]
    assert all(point.positions == {"joint_1": 0.25} for point in plan.joint_trajectory)


class _OvershootWaypointMotion(LocalMotionProvider):
    """模拟过高内部边界速度令短段先越点再折返。"""

    def _plan_segment(
        self,
        current,
        goal,
        limits,
        *,
        current_velocity=None,
        target_velocity=None,
    ):
        target_velocity = target_velocity or {}
        if any(abs(value) > 0.2 for value in target_velocity.values()):
            name = next(iter(goal))
            return (
                [
                    JointTrajectoryPoint(
                        time_from_start_s=0.0,
                        positions={name: current[name]},
                        velocities={name: 0.0},
                    ),
                    JointTrajectoryPoint(
                        time_from_start_s=0.1,
                        positions={name: goal[name] + 0.01},
                        velocities={name: target_velocity[name]},
                    ),
                    JointTrajectoryPoint(
                        time_from_start_s=0.2,
                        positions={name: goal[name]},
                        velocities={name: target_velocity[name]},
                    ),
                ],
                "overshoot-test",
                [],
            )
        return super()._plan_segment(
            current,
            goal,
            limits,
            current_velocity=current_velocity,
            target_velocity=target_velocity,
        )


def test_waypoint_velocity_is_reduced_when_segment_would_reverse() -> None:
    provider = _OvershootWaypointMotion(
        _capabilities(),
        _CollisionFreeKinematics(),
    )
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"joint_1": 0.0},
    )

    plan = provider.plan_joint_waypoints(
        robot_id=state.robot_id,
        state=state,
        goals=[{"joint_1": 0.1}, {"joint_1": 0.2}],
    )

    values = [point.positions["joint_1"] for point in plan.joint_trajectory]
    assert values == sorted(values)
    assert any("内部路点速度缩放为 0.5" in item for item in plan.diagnostics)


def test_machine_precision_at_joint_limit_is_not_a_real_limit_violation() -> None:
    provider = LocalMotionProvider(
        _capabilities(),
        _CollisionFreeKinematics(),
    )
    state = RobotState(
        robot_id="r1pro-test",
        generation=1,
        joint_positions={"joint_1": 0.0},
    )

    plan = provider.plan_joints(
        robot_id=state.robot_id,
        state=state,
        goal={"joint_1": 2.0000000000000004},
    )
    assert plan.joint_trajectory[-1].positions["joint_1"] == pytest.approx(2.0)

    with pytest.raises(PlanningError, match="超出"):
        provider.plan_joints(
            robot_id=state.robot_id,
            state=state,
            # 1e-4 rad 内属于数值求解/浮点余量；这里使用明确超过该余量的
            # 偏差，验证真正越界仍会被拒绝。
            goal={"joint_1": 2.0002},
        )
