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

"""按统一部署配置装配 R1 Pro 的模块、Backend 和 Provider。"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Callable

from semantic_robot_sdk_core import BackendUnavailable, RobotDeployment, load_robot_deployment

from .backends import FakeBackend, RealBackend, SharedFakeBackend
from .modules import (
    BaseModule,
    CommandsModule,
    EndEffectorModule,
    SafetyModule,
    SensorsModule,
    StateModule,
    UpperBodyModule,
)
from .profile import capabilities
from .providers import (
    FakeKinematicsProvider,
    LocalKinematicsProvider,
    LocalMotionProvider,
    LocalNavigationProvider,
    RuntimeSceneNavigationMapSource,
    StaticNavigationMapSource,
    VendorKinematicsProvider,
    VendorMotionProvider,
    VendorNavigationProvider,
)
from .providers.navigation_map import fake_navigation_map_source
from .providers.local_kinematics import PinocchioKinematics


class _LazyProvider:
    """首次使用时才创建重型 Provider；对外仍保持原 Provider 接口。"""

    def __init__(self, name: str, factory: Callable[[], object]) -> None:
        self.name = name
        self._factory = factory
        self._value: object | None = None
        self._lock = RLock()

    def _load(self) -> object:
        if self._value is not None:
            return self._value
        with self._lock:
            if self._value is None:
                self._value = self._factory()
            return self._value

    def __getattr__(self, attribute: str):
        return getattr(self._load(), attribute)


def _initialize_fake_backend(backend, options: dict) -> None:
    """把 RobotDeployment 中声明的 Fake 环境事实写入新 Backend。

    抓取目标属于 Fake 测试环境，而不是 Ability 或 Robot Skill 的业务捷径。正式
    MuJoCo/真机从传感器和场景获得这些事实；Fake 没有场景 Runtime，因此允许类型包
    在统一 RobotDeployment 中声明初始可接触物体。共享 Backend 只会在状态库首次
    创建时调用本函数，Ability 重启不会重置已经发生的物理状态。
    """

    targets = options.get("initial_grasp_targets", {})
    if targets is None:
        return
    if not isinstance(targets, dict):
        raise ValueError("robot.sdk.options.initial_grasp_targets 必须是按夹爪命名的对象")
    for gripper, target in targets.items():
        if not isinstance(target, dict):
            raise ValueError(f"Fake 抓取目标 {gripper} 必须是对象")
        backend.configure_grasp_fixture(str(gripper), **target)


class R1ProSDK:
    """Ability 使用的稳定门面；底层差异在装配时一次性确定。"""

    def __init__(self, deployment: RobotDeployment, backend, kinematics, motion, navigation):
        self.deployment = deployment
        self.robot_id = deployment.robot.id
        self.backend = backend
        self.kinematics_provider = kinematics
        self.motion_provider = motion
        self.navigation_provider = navigation
        self.state = StateModule(self.robot_id, backend)
        self.base = BaseModule(self.robot_id, backend, navigation)
        self.upper_body = UpperBodyModule(self.robot_id, backend, kinematics, motion)
        self.end_effector = EndEffectorModule(self.robot_id, backend)
        self.sensors = SensorsModule(self.robot_id, backend)
        self.commands = CommandsModule(self.robot_id, backend)
        self.safety = SafetyModule(self.robot_id, backend)

    @classmethod
    def from_environment(cls, *, driver=None, navigation_map_source=None):
        return cls.from_deployment(
            load_robot_deployment(),
            driver=driver,
            navigation_map_source=navigation_map_source,
        )

    @classmethod
    def from_deployment(
        cls, deployment: RobotDeployment, *, driver=None, navigation_map_source=None
    ):
        robot = deployment.robot
        if robot.model != "r1_pro_chassis":
            raise ValueError(f"R1 Pro SDK 不能加载其他型号：{robot.model}")
        if robot.sdk.package not in {"semantic-robot-sdk-r1pro", "semantic_robot_sdk_r1pro"}:
            raise ValueError(f"部署配置引用了其他 SDK 包：{robot.sdk.package}")
        options = robot.sdk.options
        profile = capabilities(
            robot.tools or None,
            base_footprint_radius_m=float(options.get("base_footprint_radius_m", 0.42)),
        )
        if robot.backend == "fake":
            fake_options = {
                "auto_complete": bool(options.get("auto_complete", True)),
                "stop_confirmed": bool(options.get("stop_confirmed", True)),
            }
            state_path = options.get("state_path")
            if state_path:
                backend = SharedFakeBackend(
                    robot.id,
                    str(state_path),
                    initializer=lambda value: _initialize_fake_backend(value, options),
                    **fake_options,
                )
            else:
                backend = FakeBackend(robot.id, **fake_options)
                _initialize_fake_backend(backend, options)
        elif robot.backend == "mujoco":
            from .backends.mujoco import MujocoBackend

            if not robot.sdk.endpoint:
                raise ValueError("MuJoCo Backend 必须配置 endpoint")
            backend = MujocoBackend(
                robot.sdk.endpoint,
                robot.id,
                profile,
                timeout_seconds=float(options.get("timeout_seconds", 10)),
                legacy_state_polling=bool(options.get("legacy_state_polling", False)),
                scene_instance_id=robot.sdk.scene_instance_id,
                # Runtime 只在每个受控关节进入严格容差后报告轨迹完成，末端语义仍由
                # 后续 Ability Observation 独立复核；此处不能用过宽关节容差掩盖
                # 尚未真正完成的抬升，否则箱体会在 Robot 继续运动时被误报成功。
                joint_position_tolerance_rad=float(
                    options.get("joint_position_tolerance_rad", 0.002)
                ),
            )
        elif robot.backend == "real":
            backend = RealBackend(driver=driver)
        else:
            raise BackendUnavailable(
                "拆码垛型号不支持 Isaac；BEHAVIOR 使用 r1pro 普通夹爪的 IsaacBackend"
            )

        selection = robot.sdk.providers
        if robot.backend == "fake" and selection.kinematics in {"fake", "local"}:
            kinematics = FakeKinematicsProvider()
        elif selection.kinematics == "local":
            kinematics_config = robot.kinematics
            if not kinematics_config.urdf_path:
                raise BackendUnavailable("本地 IK Provider 必须配置 robot.kinematics.urdf_path")
            controlled_joints = {}
            for end_effector, groups in kinematics_config.controlled_joint_groups.items():
                unknown = set(groups) - set(profile.joint_groups)
                if unknown:
                    raise ValueError(f"末端 {end_effector} 引用了未知关节组：{sorted(unknown)}")
                controlled_joints[end_effector] = [
                    joint for group in groups for joint in profile.joint_groups[group]
                ]

            def build_local_kinematics():
                solver = PinocchioKinematics(
                    Path(kinematics_config.urdf_path),
                    dict(robot.frames.end_effectors),
                    mesh_directories=list(kinematics_config.package_directories),
                    disabled_collision_pairs=list(kinematics_config.disabled_collision_pairs),
                    controlled_joints_by_end_effector=controlled_joints,
                    model_frame_id=robot.frames.base,
                )
                return LocalKinematicsProvider(
                    solver,
                    world_frame=robot.frames.world,
                    root_frame=robot.frames.base,
                    fixed_position_tolerance_m=float(
                        options.get("fixed_end_effector_position_tolerance_m", 0.002)
                    ),
                    fixed_orientation_tolerance_rad=float(
                        options.get("fixed_end_effector_orientation_tolerance_rad", 0.02)
                    ),
                )

            # 七类Ability共享一份SDK装配代码，但只有抓取规划和机械臂运动
            # 会使用Pinocchio。延迟到首次真实运动请求再加载URDF/Coal，避免
            # RobotState、SensorCapture等只读进程各自常驻数百MB运动学模型。
            kinematics = _LazyProvider("local", build_local_kinematics)
        elif selection.kinematics == "vendor":
            kinematics = VendorKinematicsProvider(driver)
        else:
            raise ValueError(f"不支持的 Kinematics Provider：{selection.kinematics}")

        if selection.motion == "local":
            motion = _LazyProvider(
                "local",
                lambda: LocalMotionProvider(
                    profile,
                    getattr(kinematics, "solver", None),
                    allow_debug_fallback=robot.backend == "fake",
                ),
            )
        elif selection.motion == "vendor":
            motion = VendorMotionProvider(driver)
        else:
            raise ValueError(f"不支持的 Motion Provider：{selection.motion}")

        if selection.navigation == "local":
            configured_map_source = options.get("navigation_map_source")
            if navigation_map_source is None and configured_map_source is not None:
                if not isinstance(configured_map_source, dict):
                    raise ValueError("robot.sdk.options.navigation_map_source 必须是对象")
                navigation_map_source = StaticNavigationMapSource.from_config(configured_map_source)
            if navigation_map_source is None and robot.backend == "mujoco":
                if not robot.sdk.endpoint or not robot.sdk.scene_instance_id:
                    raise ValueError(
                        "MuJoCo 本地导航必须配置 sdk.endpoint 和 sdk.scene_instance_id"
                    )
                navigation_map_source = RuntimeSceneNavigationMapSource(
                    robot.sdk.endpoint,
                    robot.sdk.scene_instance_id,
                    resolution_m=float(options.get("navigation_resolution_m", 0.05)),
                    padding_m=float(options.get("navigation_padding_m", 1.0)),
                    timeout_seconds=float(options.get("timeout_seconds", 10)),
                    obstacle_z_min=float(options.get("navigation_obstacle_z_min", -0.2)),
                    obstacle_z_max=float(options.get("navigation_obstacle_z_max", 0.9)),
                )
            if navigation_map_source is None and robot.backend == "fake":
                navigation_map_source = fake_navigation_map_source()
            navigation = LocalNavigationProvider(
                profile,
                map_source=navigation_map_source,
                allow_debug_fallback=robot.backend == "fake",
                carrying_motion_scale=robot.safety.carrying_motion_scale,
                base_half_size_m=options.get("navigation_base_half_size_m"),
                maximum_expanded_nodes=int(options.get("navigation_max_expanded_nodes", 120000)),
                clearance_cost_radius=float(options.get("navigation_clearance_cost_radius", 0.0)),
                clearance_cost_weight=float(options.get("navigation_clearance_cost_weight", 0.0)),
            )
        elif selection.navigation == "vendor":
            navigation = VendorNavigationProvider(driver)
        else:
            raise ValueError("R1 Pro 必须配置 local 或 vendor Navigation Provider")
        return cls(deployment, backend, kinematics, motion, navigation)

    def close(self):
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()


def create_mujoco_sdk(
    endpoint,
    robot_id,
    urdf_path,
    *,
    scene_instance_id,
    package_directories=None,
    timeout_seconds=10,
):
    """显式测试/集成工厂；正式 Ability 使用 ``from_environment``。"""
    deployment = RobotDeployment.model_validate(
        {
            "api_version": 1,
            "robot": {
                "id": robot_id,
                "model": "r1_pro_chassis",
                "backend": "mujoco",
                "sdk": {
                    "package": "semantic-robot-sdk-r1pro",
                    "endpoint": endpoint,
                    "backend_profile": "r1pro-tote-mujoco-v1",
                    "firmware_profile": "r1pro-tote-mujoco-v1",
                    # 场景快照属于具体场景实例，不能使用会跨实例读错状态的占位值。
                    "scene_instance_id": scene_instance_id,
                    "providers": {"kinematics": "local", "motion": "local", "navigation": "local"},
                    "options": {
                        "timeout_seconds": timeout_seconds,
                        "joint_position_tolerance_rad": 0.002,
                    },
                },
                "frames": {
                    "world": "world",
                    "base": "base_link",
                    "end_effectors": {
                        "left": "left_tote_load_frame",
                        "right": "right_tote_load_frame",
                    },
                },
                "kinematics": {
                    "urdf_path": str(urdf_path),
                    "package_directories": package_directories or [],
                    "controlled_joint_groups": {
                        "left": ["torso", "left_arm"],
                        "right": ["torso", "right_arm"],
                    },
                    "disabled_collision_pairs": [
                        ["left_tote_hook", "left_tote_clamp"],
                        ["right_tote_hook", "right_tote_clamp"],
                    ],
                },
            },
            "ability_framework": {"endpoint": "http://127.0.0.1:8080"},
            "pilot": {"robot_skill_directory": "/opt/semantic/robot-skills"},
        }
    )
    return R1ProSDK.from_deployment(deployment)
