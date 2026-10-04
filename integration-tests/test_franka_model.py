"""Franka 正式模型、Pinocchio 与 Ruckig 的不可跳过门控。"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from semantic_robot_sdk_core.models import Pose, RobotState
from semantic_robot_sdk_franka import capabilities, create_kinematics
from semantic_robot_sdk_franka.providers import LocalMotionProvider


def test_franka_nonzero_fk_ik_and_ruckig_trajectory() -> None:
    raw_root = os.environ.get("FRANKA_MODEL_ROOT")
    assert raw_root, "必须显式设置 FRANKA_MODEL_ROOT，不能从 .venv 猜测或跳过正式模型门控"

    model_root = Path(raw_root).resolve()
    solver = create_kinematics(model_root)
    profile = capabilities()
    assert solver.model_frame_id == profile.kinematic_root_frame == "panda_link0"
    assert solver.model.frames[solver.frames["hand"]].name == "panda_hand"

    # 该姿态来自 Franka Panda 常用的非奇异 ready configuration。使用非零姿态能
    # 发现关节顺序、root frame 或 Mesh package root 配错后被中性位掩盖的问题。
    joints = dict(
        zip(
            profile.joint_names,
            (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398),
            strict=True,
        )
    )
    assert solver.collision_free(joints, None)
    current = solver.forward_kinematics("hand", joints)
    assert current.frame_id == "panda_link0"

    target = Pose(
        position=(current.position[0] + 0.01, current.position[1], current.position[2]),
        quaternion_xyzw=current.quaternion_xyzw,
        frame_id="panda_link0",
    )
    goal = solver.solve_end_effector("hand", target, joints)
    plan = LocalMotionProvider(profile, solver).plan_joints(
        robot_id="franka-model-gate",
        state=RobotState(robot_id="franka-model-gate", generation=1, joint_positions=joints),
        goal=goal,
    )

    assert plan.planner == "ruckig"
    assert plan.collision_checked is True
    assert plan.estimated_duration_s > 0
    assert len(plan.joint_trajectory) >= 2
    final_joints = plan.joint_trajectory[-1].positions
    assert any(abs(final_joints[name] - joints[name]) > 1e-5 for name in profile.joint_names)

    reached = solver.forward_kinematics("hand", final_joints)
    position_error = math.dist(reached.position, target.position)
    assert position_error < 1e-3
    assert plan.joint_trajectory[-1].time_from_start_s == pytest.approx(plan.estimated_duration_s)
