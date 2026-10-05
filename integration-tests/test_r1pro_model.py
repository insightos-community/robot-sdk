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

"""R1 Pro 真实运动学资产集成测试。

该测试用于阻止三类问题再次出现：URDF package 名与资产目录不一致、碰撞对未
建立导致假阴性、IK 借用未下发的轮或夹爪自由度得到不可执行结果。
"""

from __future__ import annotations

import os
from pathlib import Path

import pinocchio as pin
import pytest

from semantic_robot_sdk_core import PlanningError
from semantic_robot_sdk_core.models import EnvironmentCollisionObject, EnvironmentCollisionSet, Pose
from semantic_robot_sdk_r1pro.providers.local_kinematics import PinocchioKinematics


def test_r1pro_nonzero_ik_and_collision_model() -> None:
    asset_root = Path(os.environ["R1PRO_ASSET_ROOT"]).resolve()
    robot_root = asset_root / "robot" / "r1_pro_chassis"
    urdf = robot_root / "meshes" / "r1_pro_with_gripper.urdf"
    assert urdf.is_file(), f"缺少 R1 Pro URDF: {urdf}"
    assert "package://r1_pro_with_gripper/" not in urdf.read_text(encoding="utf-8")

    disabled_pairs = [
        ("left_gripper_finger_link1", "left_gripper_finger_link2"),
        ("right_gripper_finger_link1", "right_gripper_finger_link2"),
    ]
    torso = [f"torso_joint{index}" for index in range(1, 5)]
    left_arm = [f"left_arm_joint{index}" for index in range(1, 8)]
    right_arm = [f"right_arm_joint{index}" for index in range(1, 8)]

    solver = PinocchioKinematics(
        urdf,
        {"left": "left_gripper_link", "right": "right_gripper_link"},
        mesh_directories=[str(asset_root / "robot")],
        disabled_collision_pairs=disabled_pairs,
        model_frame_id="base_link",
        controlled_joints_by_end_effector={
            "left": [*torso, *left_arm],
            "right": [*torso, *right_arm],
        },
    )
    joints = {name: 0.0 for name in [*torso, *left_arm, *right_arm]}
    assert solver.geometry_model.collisionPairs
    assert solver.collision_free(joints, None)
    blocking_environment = EnvironmentCollisionSet(
        frame_id="base_link",
        objects=[
            EnvironmentCollisionObject(
                source_id="blocking-box",
                shape="box",
                pose=Pose(
                    position=(0.0, 0.0, 0.5),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id="base_link",
                ),
                size_xyz=(4.0, 4.0, 4.0),
            )
        ],
    )
    assert not solver.collision_free(joints, blocking_environment)

    clear_environment = EnvironmentCollisionSet(
        frame_id="base_link",
        objects=[
            EnvironmentCollisionObject(
                source_id="far-sphere",
                shape="sphere",
                pose=Pose(
                    position=(100.0, 100.0, 100.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id="base_link",
                ),
                radius_m=0.5,
            )
        ],
    )
    assert solver.collision_free(joints, clear_environment)

    # R1 Pro 左右臂是镜像机构。固定一侧已经形成有效工作姿态后，另一侧
    # 不能只依赖随机多起点去碰运气；先从固定侧构型生成确定性镜像起点，
    # 再由双末端 IK 在保持承载端 Pose 的约束下收敛。这里验证的是 Robot
    # 硬件运动学关系，不包含周转箱、凹槽或具体 Robot Skill 的业务参数。
    mirrored_source = {
        **joints,
        "right_arm_joint1": 0.20,
        "right_arm_joint2": -0.35,
        "right_arm_joint3": 0.18,
        "right_arm_joint4": 0.42,
        "right_arm_joint5": -0.16,
        "right_arm_joint6": 0.25,
        "right_arm_joint7": -0.12,
    }
    initial, controlled = solver._configuration(
        mirrored_source,
        controlled_names=frozenset([*torso, *left_arm, *right_arm]),
    )
    mirror_seed = solver._bilateral_mirror_seed(
        initial,
        controlled,
        moving_end_effectors={"left"},
        fixed_end_effectors=frozenset({"right"}),
    )
    assert mirror_seed is not None
    mirror_signs = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    for index, sign in enumerate(mirror_signs, start=1):
        left_id = solver.model.getJointId(f"left_arm_joint{index}")
        right_id = solver.model.getJointId(f"right_arm_joint{index}")
        left_q = solver.model.joints[left_id].idx_q
        right_q = solver.model.joints[right_id].idx_q
        expected = sign * float(initial[right_q])
        expected = min(
            float(solver.model.upperPositionLimit[left_q]),
            max(float(solver.model.lowerPositionLimit[left_q]), expected),
        )
        assert float(mirror_seed[left_q]) == expected

    q = pin.neutral(solver.model)
    pin.forwardKinematics(solver.model, solver.data, q)
    pin.updateFramePlacements(solver.model, solver.data)
    current = solver.data.oMf[solver.frames["left"]]
    quaternion_xyzw = tuple(float(value) for value in pin.Quaternion(current.rotation).coeffs())
    target = Pose(
        position=(
            float(current.translation[0]),
            float(current.translation[1]),
            float(current.translation[2] + 0.01),
        ),
        quaternion_xyzw=quaternion_xyzw,
        frame_id="base_link",
    )
    result = solver.solve_end_effector("left", target, joints)

    # 单臂小动作应先只使用对应手臂；共享躯干只有在手臂自身不可达时才参与。
    assert set(result) == set(left_arm)
    assert not set(torso) & set(result)
    assert any(abs(value) > 1e-5 for value in result.values())
    assert solver.collision_free({**joints, **result}, None)

    # 两侧目标都能由各自手臂满足时，先组合互不重叠的局部解。同步轨迹
    # 随后仍会统一时间参数化并逐点检查碰撞；这里阻止大规模联合IK把
    # 显然可达的纯双臂抬升误判为不收敛，或为了收敛而无故扭动躯干。
    left_start = solver.forward_kinematics("left", joints)
    right_start = solver.forward_kinematics("right", joints)
    lift_targets = {
        name: start.model_copy(
            update={
                "position": (
                    start.position[0],
                    start.position[1],
                    start.position[2] + 0.01,
                )
            }
        )
        for name, start in {"left": left_start, "right": right_start}.items()
    }
    bilateral_lift = solver.solve_end_effectors(lift_targets, joints)
    assert set(bilateral_lift) == set([*left_arm, *right_arm])
    assert not set(torso) & set(bilateral_lift)
    lifted_joints = {**joints, **bilateral_lift}
    for name, target in lift_targets.items():
        actual = solver.forward_kinematics(name, lifted_joints)
        assert actual.position == pytest.approx(target.position, abs=1e-4)
    assert solver.collision_free(lifted_joints, None)

    # 固定右端的短左移会被拆成连续几何路点。后续路点即使使用上一点作为
    # 连续种子，也必须继续优先只动左臂；切换到共享躯干会把已钩入的左端抬高。
    left_start = solver.forward_kinematics("left", joints)
    right_anchor = solver.forward_kinematics("right", joints)
    first_target = left_start.model_copy(
        update={
            "position": (
                left_start.position[0] + 0.004,
                left_start.position[1],
                left_start.position[2],
            )
        }
    )
    first = solver.solve_end_effectors(
        {"left": first_target, "right": right_anchor},
        joints,
        fixed_end_effectors=frozenset({"right"}),
    )
    assert set(first) == set(left_arm)

    after_first = {**joints, **first}
    left_after_first = solver.forward_kinematics("left", after_first)
    right_after_first = solver.forward_kinematics("right", after_first)
    second_target = left_after_first.model_copy(
        update={
            "position": (
                left_after_first.position[0] + 0.004,
                left_after_first.position[1],
                left_after_first.position[2],
            )
        }
    )
    second = solver.solve_end_effectors(
        {"left": second_target, "right": right_after_first},
        after_first,
        fixed_end_effectors=frozenset({"right"}),
        continuous_seed=True,
    )
    assert set(second) == set(left_arm)
    assert not set(torso) & set(second)

    shared_only_target = left_start.model_copy(
        update={
            "position": (
                left_start.position[0] + 0.02,
                left_start.position[1],
                left_start.position[2],
            )
        }
    )
    try:
        larger_move = solver.solve_end_effectors(
            {"left": shared_only_target, "right": right_anchor},
            joints,
            fixed_end_effectors=frozenset({"right"}),
            continuous_seed=True,
        )
    except PlanningError:
        pass
    else:
        compensated = {**joints, **larger_move}
        actual_left = solver.forward_kinematics("left", compensated)
        actual_right = solver.forward_kinematics("right", compensated)
        endpoint_tolerance = solver.tolerance * 2.0 + 1e-9
        assert actual_left.position == pytest.approx(
            shared_only_target.position, abs=endpoint_tolerance
        )
        assert actual_right.position == pytest.approx(right_anchor.position, abs=endpoint_tolerance)
        assert actual_right.quaternion_xyzw == pytest.approx(
            right_anchor.quaternion_xyzw, abs=endpoint_tolerance
        )
