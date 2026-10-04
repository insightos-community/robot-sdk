"""图像、关节、末端及定位读取接口。"""

from .backends.base import RobotBackend
from .utils.types import CommandResult, SensorFrame


class SensorsClient:
    def __init__(self, backend: RobotBackend) -> None:
        self._backend = backend

    def capture_rgb(self, sensor_id: str) -> SensorFrame:
        return self._backend.capture(sensor_id, "rgb8")

    def capture_depth(self, sensor_id: str) -> SensorFrame:
        return self._backend.capture(sensor_id, "32FC1")

    def capture_rgbd(self, sensor_id: str) -> tuple[SensorFrame, SensorFrame]:
        """读取同一仿真步的 RGB 和米制 optical_z 深度及标定。"""
        return self._backend.capture_rgbd(sensor_id)

    def get_camera_calibration(
        self, sensor_id: str = "head_camera", calibration_id: str | None = None
    ) -> CommandResult:
        """读取内参及 optical(+X右/+Y下/+Z前) 到 body 的米制 4x4 变换。

        使用 capture_rgbd 帧中的 calibration_id 可读取同帧快照；省略时读取当前标定。
        """
        return self._backend.get_command(
            resource="sensors",
            operation="get_camera_calibration",
            target={"sensor_id": sensor_id, "calibration_id": calibration_id},
        )

    def get_ee_pose(self, axis: str = "left") -> CommandResult:
        """读取单臂末端位姿（axis: left/right）。

        output['data'] 沿用 {'header': {'frame_id': ...}, 'pose': ...}。
        仿真返回 torso_link4 坐标系，位置为米，orientation 为 xyzw 四元数。
        """

        if axis not in ("left", "right"):
            raise ValueError(f"无效的轴: {axis!r}")
        return self._backend.get_command(
            resource="eef_control", operation=f"get_{axis}_ee_pose", target={}
        )

    def get_gripper_state(self, axis: str = "left") -> CommandResult:
        """读取单侧夹爪实际开度，output['data'] 为 0–100 的数值（axis: left/right）。"""

        if axis not in ("left", "right"):
            raise ValueError(f"无效的轴: {axis!r}")
        return self._backend.get_command(
            resource="gripper_control", operation=f"get_{axis}_gripper_state", target={}
        )

    def get_joint_state(self, axis: str = "left") -> CommandResult:
        """读取单臂关节位置：output['data'] 为按该臂模型顺序排列的弧度列表。"""

        if axis not in ("left", "right"):
            raise ValueError(f"无效的轴: {axis!r}")
        return self._backend.get_command(
            resource="joint_control", operation=f"get_{axis}_joint_state", target={}
        )

    def get_torso_positions(self) -> CommandResult:
        """读取躯干四关节 data（rad）、joint_names、velocities_rad_s，按模型顺序排列。"""

        return self._backend.get_command(
            resource="torso_control", operation="get_torso_positions", target={}
        )

    def get_localization(self) -> CommandResult:
        """读取速度定位：output["data"] 为 odom 位姿，base_qvel 为底盘局部 vx/vy/wz。"""

        return self._backend.get_command(
            resource="woosh_move", operation="get_localization", target={}
        )
