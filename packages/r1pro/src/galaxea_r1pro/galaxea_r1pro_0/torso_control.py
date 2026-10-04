"""Torso_Control 能力：沿用现有躯干位置与速度参数。"""

from __future__ import annotations

from typing import List, Optional

from .backends.base import RobotBackend
from .sensors import SensorsClient
from .utils.types import CommandHandle, CommandResult


class TorsoControl:
    """R1 Pro 躯干关节控制。"""

    def __init__(self, backend: RobotBackend, sensors: Optional[SensorsClient] = None) -> None:
        self._backend = backend
        self._sensors = sensors or SensorsClient(backend)

    def compute_observation_pose(
        self,
        *,
        object_position,
        standing_pose,
        frame_id="odom",
        object_bounds=None,
    ) -> CommandResult:
        """Compute head-camera viewing angles without moving the robot.

        object_position: XYZ in meters; standing_pose: body [x,y,z,yaw], yaw in rad.
        Both use frame_id (odom by default, or world). Optional object_bounds is
        [min_xyz,max_xyz] in that same frame; its center replaces object_position.
        Successful output['data'] contains torso_joint1..4 angles in rad.
        Only field of view is checked, not occlusion or motion collisions.
        """
        return self._backend.get_command(
            resource="torso_control",
            operation="compute_observation_pose",
            target={
                "object_position": list(object_position),
                "standing_pose": list(standing_pose),
                "frame_id": frame_id,
                "object_bounds": None
                if object_bounds is None
                else [list(row) for row in object_bounds],
            },
        )

    def control_torso_joints(
        self,
        *,
        positions: List[float],
        speeds: Optional[List[float]] = None,
        effort: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
    ) -> CommandHandle:
        """关节位置为 rad；speeds 为 rad/s 上限，留空使用模型上限。

        仿真默认按 torso_joint1 至 torso_joint4 排列；effort 暂需留空。
        """
        return self._backend.start_command(
            "torso_control",
            "control_torso_joints",
            {
                "positions": positions,
                "velocities": speeds or [],
                "efforts": effort or [],
                "names": names or [],
            },
        )

    def control_torso_trajectory(
        self, *, positions: List[float], duration_seconds: float
    ) -> CommandHandle:
        """Execute a finite quintic trajectory, including a final 0.5 s hold.

        Duration uses simulation time. Queue completion confirms all samples ran;
        callers must verify measured joint positions before advancing the task.
        Model drive limits are restored by Runtime when it takes sequence control.
        """
        return self._backend.start_command(
            "torso_control",
            "control_torso_trajectory",
            {"positions": positions, "duration_seconds": duration_seconds},
        )

    def control_single_torso_joint(
        self,
        *,
        joint_id: int,
        target_pos: float,
        speed: float = 1.5,
        effort: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
    ) -> CommandHandle:
        """单独控制躯干某个关节，joint_id 为 0 至 3；其他关节保持当前位置。"""

        if isinstance(joint_id, bool) or not isinstance(joint_id, int) or not 0 <= joint_id < 4:
            raise ValueError("joint_id 必须为 0 至 3 的整数")

        current_result = self._sensors.get_torso_positions()
        current = (current_result.output or {}).get("data")
        positions = list(current)
        positions[joint_id] = target_pos
        speeds = [speed if i == joint_id else 0.05 for i in range(4)]
        return self.control_torso_joints(
            positions=positions, speeds=speeds, effort=effort, names=names
        )
