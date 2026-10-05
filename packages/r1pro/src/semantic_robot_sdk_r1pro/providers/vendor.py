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

"""厂商 Provider 的稳定边界；真实驱动未注入时明确失败。"""

from semantic_robot_sdk_core import BackendUnavailable


class VendorKinematicsProvider:
    name = "vendor"

    def __init__(self, driver=None):
        self.driver = driver

    def solve(self, **kwargs):
        if self.driver is None:
            raise BackendUnavailable("R1 Pro 厂商 IK Provider 尚未配置真实驱动")
        return self.driver.solve_ik(**kwargs)

    def reachable(self, **kwargs):
        if self.driver is None:
            raise BackendUnavailable("R1 Pro 厂商 IK Provider 尚未配置真实驱动")
        return self.driver.reachable(**kwargs)


class VendorMotionProvider:
    name = "vendor"

    def __init__(self, driver=None):
        self.driver = driver

    def plan_joints(self, **kwargs):
        if self.driver is None:
            raise BackendUnavailable("R1 Pro 厂商 Motion Provider 尚未配置真实驱动")
        return self.driver.plan_joints(**kwargs)


class VendorNavigationProvider:
    name = "vendor"

    def __init__(self, driver=None):
        self.driver = driver

    def plan_route(
        self,
        *,
        robot_id,
        state,
        goal,
        occupancy=None,
        maximum_speed_mps,
        minimum_clearance_m=None,
        carrying_object_ref=None,
        carrying_object_pose=None,
        carrying_object_extent_m=None,
    ):
        if self.driver is None:
            raise BackendUnavailable("R1 Pro 厂商 Navigation Provider 尚未配置真实驱动")
        arguments = {
            "robot_id": robot_id,
            "state": state,
            "goal": goal,
            "occupancy": occupancy,
            "maximum_speed_mps": maximum_speed_mps,
        }
        # 未指定净空时不向旧版厂商驱动增加关键字，保持固件 SDK 升级前后的兼容性。
        if minimum_clearance_m is not None:
            arguments["minimum_clearance_m"] = minimum_clearance_m
        # 持物对象引用是本地 Scene 导航避免“把自己携带的物体当障碍物”的提示。
        # 厂商导航驱动有自己的载荷模型，不能把这个框架内部关键字强塞给旧驱动。
        _ = (
            carrying_object_ref,
            carrying_object_pose,
            carrying_object_extent_m,
        )
        return self.driver.plan_route(**arguments)
