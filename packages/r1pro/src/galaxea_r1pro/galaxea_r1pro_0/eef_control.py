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

"""Joint_EEFPose_Control 能力：沿用现有位置、欧拉角与坐标系参数。"""

from __future__ import annotations

import math

from .backends.base import RobotBackend
from .utils.types import CommandHandle, Pose
from .utils.utils_common import euler_to_quaternionx


class EEFControl:
    """R1 Pro 末端执行器位姿控制。"""

    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend

    def control_single_arm(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        frame_id: str = "torso_link4",
        axis: str = "left",
        motion_mode: str = "ik",
        position_tolerance_m: float | None = None,
        plan_only: bool = False,
        use_torso: bool = True,
        pregrasp_planner: str = "curobo",
        approach_offset: list[float] | None = None,
        attached_object_ref: str | None = None,
        contact_object_ref: str | None = None,
        allow_support_contact: bool = False,
    ) -> CommandHandle:
        """位置单位为米；角度单位为度，沿用 Atom 的内禀 YXZ 顺序。

        仿真后端在提交时将 frame_id 内的目标转换到世界坐标系并固定。
        返回句柄后通过 command.get / command.stop 查询或停止。
        """
        if motion_mode not in ("ik", "curobo"):
            raise ValueError("motion_mode must be ik or curobo")
        if motion_mode != "curobo" and (
            plan_only or approach_offset is not None or attached_object_ref or contact_object_ref
        ):
            raise ValueError("planning options require motion_mode=curobo")
        if pregrasp_planner not in ("curobo", "ik_filter"):
            raise ValueError("invalid pregrasp_planner")
        if pregrasp_planner != "curobo" and (motion_mode != "curobo" or approach_offset is None):
            raise ValueError("ik_filter requires curobo mode and approach_offset")
        if not isinstance(allow_support_contact, bool):
            raise ValueError("allow_support_contact must be boolean")
        if allow_support_contact and (
            motion_mode != "curobo" or not attached_object_ref or approach_offset is not None
        ):
            raise ValueError(
                "support contact requires an attached curobo carry without approach_offset"
            )
        planning = (
            {}
            if motion_mode == "ik"
            else dict(
                motion_mode=motion_mode,
                plan_only=plan_only,
                use_torso=use_torso,
                approach_offset=approach_offset,
                pregrasp_planner=pregrasp_planner,
                allow_support_contact=allow_support_contact,
                attached_object_ref=attached_object_ref,
                contact_object_ref=contact_object_ref,
            )
        )
        precision = {}
        if position_tolerance_m is not None:
            if (
                isinstance(position_tolerance_m, bool)
                or not math.isfinite(position_tolerance_m)
                or position_tolerance_m <= 0
            ):
                raise ValueError("position_tolerance_m must be positive and finite")
            precision["position_tolerance_m"] = float(position_tolerance_m)
        quat = tuple(float(value) for value in euler_to_quaternionx([roll, pitch, yaw], seq="yxz"))
        return self._backend.start_command(
            "eef_control",
            "control_single_arm",
            {
                "axis": axis,
                **planning,
                **precision,
                "target": Pose(
                    frame_id=frame_id,
                    position=(x, y, z),
                    orientation_xyzw=quat,
                ),
            },
        )

    def control_both_arm(
        self,
        *,
        left_x: float,
        left_y: float,
        left_z: float,
        left_roll: float,
        left_pitch: float,
        left_yaw: float,
        right_x: float,
        right_y: float,
        right_z: float,
        right_roll: float,
        right_pitch: float,
        right_yaw: float,
        left_frame_id: str = "1",
        right_frame_id: str = "1",
    ) -> list:
        """双臂末端控制：左右臂异步多线程并行提交（借鉴 Atom）。"""

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self.control_single_arm,
                    axis=side,
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    frame_id=frame_id,
                )
                for side, x, y, z, roll, pitch, yaw, frame_id in (
                    (
                        "left",
                        left_x,
                        left_y,
                        left_z,
                        left_roll,
                        left_pitch,
                        left_yaw,
                        left_frame_id,
                    ),
                    (
                        "right",
                        right_x,
                        right_y,
                        right_z,
                        right_roll,
                        right_pitch,
                        right_yaw,
                        right_frame_id,
                    ),
                )
            ]
            return [future.result() for future in futures]
