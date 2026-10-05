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

"""移动底盘路径的连续朝向和 Ruckig 限制辅助。"""

import math

from semantic_robot_sdk_core import JointLimit, PlanningError


def yaw_from_xyzw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _nearest_angle(reference, target):
    return reference + (target - reference + math.pi) % (2 * math.pi) - math.pi


def base_segment_targets(route, *, initial_yaw, final_yaw, preserve_yaw=False):
    if not route:
        raise PlanningError("底盘路径不能为空")
    if len(route) == 1:
        route = [route[0], route[0]]
    targets = []
    yaw = initial_yaw
    for index in range(1, len(route)):
        current = route[index]
        if preserve_yaw:
            target_yaw = initial_yaw
        elif index == len(route) - 1:
            target_yaw = _nearest_angle(yaw, final_yaw)
        else:
            following = route[index + 1]
            target_yaw = _nearest_angle(
                yaw, math.atan2(following[1] - current[1], following[0] - current[0])
            )
        targets.append({"base_x": current[0], "base_y": current[1], "base_yaw": target_yaw})
        yaw = target_yaw
    if preserve_yaw:
        # R1 Pro 是全向底盘。携物平移时保持当前朝向，可以避免双臂夹持的大箱体
        # 随 A* 折线反复扫转；若业务目标要求新朝向，最后再原地完成一次受限旋转。
        target_yaw = _nearest_angle(initial_yaw, final_yaw)
        if not math.isclose(target_yaw, initial_yaw, abs_tol=1e-6):
            current = route[-1]
            targets.append({"base_x": current[0], "base_y": current[1], "base_yaw": target_yaw})
    return targets


def base_limits(capabilities, maximum_speed_mps, *, motion_scale=1.0):
    """生成底盘连续轨迹限制。

    ``motion_scale`` 同比缩放速度、加速度和 jerk。只降低速度而保留空载加速度，
    仍会在起停阶段向双臂和载荷施加过大的惯性冲击，因此三阶约束必须一起缩放。
    """

    return {
        "base_x": JointLimit(
            lower=-1e6,
            upper=1e6,
            max_velocity=maximum_speed_mps,
            max_acceleration=capabilities.base_max_acceleration_mps2 * motion_scale,
            max_jerk=capabilities.base_max_jerk_mps3 * motion_scale,
        ),
        "base_y": JointLimit(
            lower=-1e6,
            upper=1e6,
            max_velocity=maximum_speed_mps,
            max_acceleration=capabilities.base_max_acceleration_mps2 * motion_scale,
            max_jerk=capabilities.base_max_jerk_mps3 * motion_scale,
        ),
        "base_yaw": JointLimit(
            lower=-1e6,
            upper=1e6,
            max_velocity=capabilities.base_max_angular_velocity_rps * motion_scale,
            max_acceleration=capabilities.base_max_angular_acceleration_rps2 * motion_scale,
            max_jerk=capabilities.base_max_angular_jerk_rps3 * motion_scale,
        ),
    }
