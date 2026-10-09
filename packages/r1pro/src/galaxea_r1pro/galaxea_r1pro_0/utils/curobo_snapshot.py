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

"""Build the native numerical planner from an immutable scene snapshot."""

from types import SimpleNamespace
import numpy as np


def build(snapshot):
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.types.state import JointState
    from curobo.types.math import Pose
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from .curobo_motion import MotionPlanningError

    device = TensorDeviceType(device=torch.device("cuda:0"))
    meshes = [
        Mesh(name=m["name"], pose=[0, 0, 0, 1, 0, 0, 0], vertices=m["vertices"], faces=m["faces"])
        for m in snapshot["mesh_snapshot"]
    ]
    cfg = MotionGenConfig.load_from_robot_config(
        RobotConfig.from_dict(snapshot["configuration"], device),
        WorldConfig(mesh=meshes),
        device,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=False,
        num_ik_seeds=128,
        num_batch_ik_seeds=128,
        num_batch_trajopt_seeds=1,
        num_trajopt_noisy_seeds=1,
        ik_opt_iters=100,
        optimize_dt=True,
        num_trajopt_seeds=4,
        num_graph_seeds=4,
        interpolation_dt=snapshot["dt"],
        collision_activation_distance=0.005,
        self_collision_check=True,
        maximum_trajectory_dt=None,
        fixed_iters_trajopt=True,
        finetune_trajopt_iters=100,
        finetune_dt_scale=1.05,
        position_threshold=0.001,
        rotation_threshold=0.01,
        collision_cache={"mesh": len(meshes) + 10},
    )
    mg = MotionGen(cfg)
    indices = np.array([snapshot["full_names"].index(n) for n in mg.kinematics.joint_names])
    if set(indices) != snapshot["expected_indices"]:
        raise MotionPlanningError("joint_mapping_invalid", "快照与规划关节映射不一致")
    start = JointState.from_position(
        device.to_device(snapshot["q"][indices][None]), joint_names=mg.kinematics.joint_names
    )

    def pose(value):
        return Pose(
            position=device.to_device([value["position"]]),
            quaternion=device.to_device([np.asarray(value["orientation"])[[3, 0, 1, 2]].tolist()]),
        )

    attached = []
    for row in snapshot["attachments"]:
        if not mg.attach_objects_to_robot(
            start,
            row["meshes"],
            ee_pose=pose(row["pose"]),
            link_name=row["link"],
            merge_meshes=True,
            voxelize_method="subdivide",
            scale=1.0,
        ):
            raise MotionPlanningError("attachment_failed", "附着物碰撞模型生成失败")
        attached.append(row)

    def detach(rows, _):
        for row in rows:
            mg.detach_object_from_robot(object_names=row["meshes"], link_name=row["link"])
            for name in row["meshes"]:
                mg.world_coll_checker.enable_obstacle(name=name, enable=True)

    def plan_batch(state, goal, config):
        result = mg.plan_single(state, goal, config)
        return (
            result,
            result.success,
            [result.get_interpolated_plan()] if bool(result.success.any()) else [],
        )

    generator = SimpleNamespace(
        mg={"default": mg},
        tensor_args=device,
        plan_batch=plan_batch,
        _detach_objects_from_robot=detach,
    )
    return dict(
        snapshot,
        generator=generator,
        start=start,
        goal=pose(snapshot["target"]),
        attached=attached,
        indices=indices,
        max_velocity=snapshot["full_max_velocity"][indices],
        lower=snapshot["full_lower"][indices],
        upper=snapshot["full_upper"][indices],
        limited=snapshot["full_limited"][indices],
        motion_velocity=np.minimum(
            snapshot["full_max_velocity"][indices],
            [0.4 if i in snapshot["torso_indices"] else 0.5 for i in indices],
        ),
        motion_acceleration=np.full(len(indices), 1.2),
    )
