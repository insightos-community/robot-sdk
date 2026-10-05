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

"""Robot SDK 公共坐标转换。

公共四元数固定为 xyzw。转换集中放在本模块，禁止 Backend、Ability 或规划器
各自隐藏地交换顺序；测试同时覆盖平移和旋转，防止移动底盘后 IK 目标漂移。
"""

from __future__ import annotations

import math

from .errors import PlanningError
from .models import EnvironmentCollisionSet, Pose


def relative_pose(parent_in_world: Pose, child_in_world: Pose, *, result_frame_id: str) -> Pose:
    """计算 `parent^-1 * child`，返回 child 在 parent 坐标中的位姿。"""

    if parent_in_world.frame_id != child_in_world.frame_id:
        raise PlanningError("两个世界位姿的 frame_id 不一致")
    inverse_parent = quaternion_conjugate(parent_in_world.quaternion_xyzw)
    delta = tuple(
        child_in_world.position[index] - parent_in_world.position[index] for index in range(3)
    )
    return Pose(
        position=rotate_vector(inverse_parent, delta),
        quaternion_xyzw=quaternion_multiply(inverse_parent, child_in_world.quaternion_xyzw),
        frame_id=result_frame_id,
    )


def relative_collision_set(
    parent_in_world: Pose,
    environment: EnvironmentCollisionSet,
    *,
    result_frame_id: str,
) -> EnvironmentCollisionSet:
    """把世界坐标环境快照转换到 Robot 运动学根坐标。

    环境几何在一次规划中保持不变。转换后的对象仍保留 source_id，规划失败时
    可以把具体障碍反馈给上层，而不暴露仿真引擎 body/geom 标识。
    """

    if parent_in_world.frame_id != environment.frame_id:
        raise PlanningError("Robot 与环境碰撞快照的世界坐标系不一致")
    return EnvironmentCollisionSet(
        frame_id=result_frame_id,
        objects=[
            item.model_copy(
                update={
                    "pose": relative_pose(
                        parent_in_world,
                        item.pose,
                        result_frame_id=result_frame_id,
                    )
                }
            )
            for item in environment.objects
        ],
    )


def quaternion_conjugate(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """返回单位四元数的共轭，公共输入和输出都保持 xyzw。"""

    x, y, z, w = value
    return (-x, -y, -z, w)


def quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    norm = math.sqrt(sum(item * item for item in result))
    if norm < 1e-12:
        raise PlanningError("坐标转换产生了零四元数")
    return tuple(item / norm for item in result)


def rotate_vector(
    quaternion_xyzw: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """用单位四元数旋转三维向量，不引入 NumPy 作为 core 的强制依赖。"""

    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
