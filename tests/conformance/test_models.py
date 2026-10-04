import pytest
from pydantic import ValidationError

from semantic_robot_sdk_core.models import (
    BaseTrajectoryPoint,
    EnvironmentCollisionObject,
    EnvironmentCollisionSet,
    JointLimit,
    JointTrajectoryPoint,
    MotionPlan,
    PlanKind,
    Pose,
    RobotCapabilities,
    SensorFrame,
)
from semantic_robot_sdk_core.transforms import relative_collision_set, relative_pose


def test_public_pose_requires_xyzw_unit_quaternion() -> None:
    pose = Pose(position=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 1), frame_id="world")
    assert pose.quaternion_xyzw == (0, 0, 0, 1)
    with pytest.raises(ValidationError):
        Pose(position=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 2), frame_id="world")


def test_sensor_frame_keeps_generation_and_sequence() -> None:
    frame = SensorFrame(
        sensor_id="front_rgb",
        sequence=9,
        generation=3,
        frame_id="camera_front",
        encoding="jpeg",
        width=640,
        height=480,
        payload=b"frame",
    )
    assert frame.sequence == 9
    assert frame.generation == 3


def test_world_target_is_transformed_into_moving_base_frame() -> None:
    # 底盘在 world 中平移 (1, 2) 并旋转 90°；world 的 +y 对应底盘的 +x。
    root = Pose(
        position=(1, 2, 0),
        quaternion_xyzw=(0, 0, 2**-0.5, 2**-0.5),
        frame_id="world",
    )
    target = Pose(position=(1, 3, 0), quaternion_xyzw=(0, 0, 0, 1), frame_id="world")
    local = relative_pose(root, target, result_frame_id="base_link")
    assert local.position == pytest.approx((1, 0, 0))
    assert local.frame_id == "base_link"


def test_world_collision_snapshot_is_transformed_into_robot_frame() -> None:
    # Robot 位于 world=(1, 2)，朝向 +y；world 中 Robot 前方一米的障碍应变为 base +x。
    root = Pose(
        position=(1, 2, 0),
        quaternion_xyzw=(0, 0, 2**-0.5, 2**-0.5),
        frame_id="world",
    )
    environment = EnvironmentCollisionSet(
        frame_id="world",
        objects=[
            EnvironmentCollisionObject(
                source_id="box-1",
                shape="box",
                pose=Pose(
                    position=(1, 3, 0),
                    quaternion_xyzw=(0, 0, 0, 1),
                    frame_id="world",
                ),
                size_xyz=(0.2, 0.4, 0.6),
                allowed_contact_end_effectors=("left",),
            )
        ],
    )
    local = relative_collision_set(root, environment, result_frame_id="base_link")

    assert local.frame_id == "base_link"
    assert local.objects[0].source_id == "box-1"
    assert local.objects[0].pose.position == pytest.approx((1, 0, 0))
    assert local.objects[0].pose.frame_id == "base_link"
    assert local.objects[0].allowed_contact_end_effectors == ("left",)


def test_environment_collision_model_rejects_incomplete_or_mixed_frames() -> None:
    with pytest.raises(ValidationError, match="size_xyz"):
        EnvironmentCollisionObject(
            source_id="invalid-box",
            shape="box",
            pose=Pose(
                position=(0, 0, 0),
                quaternion_xyzw=(0, 0, 0, 1),
                frame_id="world",
            ),
        )

    with pytest.raises(ValidationError, match="坐标系不一致"):
        EnvironmentCollisionSet(
            frame_id="world",
            objects=[
                EnvironmentCollisionObject(
                    source_id="sphere-1",
                    shape="sphere",
                    pose=Pose(
                        position=(0, 0, 0),
                        quaternion_xyzw=(0, 0, 0, 1),
                        frame_id="map",
                    ),
                    radius_m=0.1,
                )
            ],
        )


def _valid_joint_plan_payload() -> dict:
    return {
        "plan_id": "plan-1",
        "robot_id": "robot-1",
        "generation": 1,
        "kind": PlanKind.JOINT,
        "resources": ["joints:arm"],
        "frame_id": "base_link",
        "start": {"joint_1": 0},
        "goal": {"joint_1": 0.5},
        "joint_trajectory": [
            JointTrajectoryPoint(time_from_start_s=0, positions={"joint_1": 0}),
            JointTrajectoryPoint(time_from_start_s=1, positions={"joint_1": 0.5}),
        ],
        "collision_checked": True,
        "estimated_duration_s": 1,
        "planner": "test",
    }


def test_motion_plan_rejects_mixed_payload_and_invalid_time_axis() -> None:
    mixed = _valid_joint_plan_payload()
    mixed["base_trajectory"] = [
        BaseTrajectoryPoint(time_from_start_s=0, x=0, y=0, yaw=0),
        BaseTrajectoryPoint(time_from_start_s=1, x=1, y=0, yaw=0),
    ]
    with pytest.raises(ValidationError, match="一种载荷"):
        MotionPlan.model_validate(mixed)

    repeated_time = _valid_joint_plan_payload()
    repeated_time["joint_trajectory"] = [
        JointTrajectoryPoint(time_from_start_s=0, positions={"joint_1": 0}),
        JointTrajectoryPoint(time_from_start_s=0, positions={"joint_1": 0.5}),
    ]
    with pytest.raises(ValidationError, match="严格递增"):
        MotionPlan.model_validate(repeated_time)

    wrong_duration = _valid_joint_plan_payload()
    wrong_duration["estimated_duration_s"] = 2
    with pytest.raises(ValidationError, match="最后一个轨迹点"):
        MotionPlan.model_validate(wrong_duration)
    contact_plan = _valid_joint_plan_payload()
    contact_plan["stop_on_contact"] = True
    with pytest.raises(ValidationError, match="目标工具"):
        MotionPlan.model_validate(contact_plan)


def test_public_models_reject_non_finite_physics_values() -> None:
    with pytest.raises(ValidationError):
        Pose(
            position=(float("nan"), 0, 0),
            quaternion_xyzw=(0, 0, 0, 1),
            frame_id="world",
        )


def test_robot_profile_rejects_incomplete_joint_mapping() -> None:
    limit = JointLimit(
        lower=-1,
        upper=1,
        max_velocity=1,
        max_acceleration=1,
        max_jerk=1,
    )
    with pytest.raises(ValidationError, match="每个且仅为"):
        RobotCapabilities(
            model="invalid",
            kind="manipulator",
            joint_names=["joint_1", "joint_2"],
            joint_groups={"arm": ["joint_1", "joint_2"]},
            joint_limits={"joint_1": limit},
            end_effectors=["tool"],
            grippers=[],
            sensors=["rgb"],
        )
