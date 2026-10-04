"""Franka 型号包；v0.5 用于公共接口和模型包验证。"""

from pathlib import Path

from .backends import MujocoBackend
from semantic_robot_sdk_core import JointLimit, RobotCapabilities
from semantic_robot_sdk_core.model_bundle import RobotModelBundle

from .model_bundle import load_franka_model_bundle
from .providers import PinocchioKinematics

JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
_POSITION_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]
_VELOCITY_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]


def capabilities() -> RobotCapabilities:
    return RobotCapabilities(
        model="franka_panda",
        kind="manipulator",
        kinematic_root_frame="panda_link0",
        joint_names=JOINTS,
        joint_groups={"arm": JOINTS},
        joint_limits={
            name: JointLimit(
                lower=_POSITION_LIMITS[index][0],
                upper=_POSITION_LIMITS[index][1],
                max_velocity=_VELOCITY_LIMITS[index],
                max_acceleration=2.5,
                max_jerk=10,
            )
            for index, name in enumerate(JOINTS)
        },
        end_effectors=["hand"],
        grippers=["hand"],
        sensors=["rgb", "depth", "contact", "robot_state"],
        supports_base=False,
    )


def create_kinematics(model_root: str | Path) -> PinocchioKinematics:
    return create_kinematics_from_bundle(load_franka_model_bundle(model_root))


def create_kinematics_from_bundle(bundle: RobotModelBundle) -> PinocchioKinematics:
    if bundle.model_id != "franka_panda":
        raise ValueError(f"Franka 工厂不能加载其他型号：{bundle.model_id}")
    return PinocchioKinematics(
        bundle.urdf_path,
        dict(bundle.end_effector_frames),
        mesh_directories=[str(path) for path in bundle.package_directories],
        disabled_collision_pairs=list(bundle.disabled_collision_pairs),
        model_frame_id=bundle.kinematic_root_frame,
    )


__all__ = [
    "MujocoBackend",
    "JOINTS",
    "capabilities",
    "create_kinematics",
    "create_kinematics_from_bundle",
    "load_franka_model_bundle",
]
