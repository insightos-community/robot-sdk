"""完整开发观测 HTTP 客户端。仿真生命周期、相机和物理步进由 Runtime 管理。"""

from urllib.parse import urlencode
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from semantic_robot_sdk_core.errors import BackendRequestError
from .isaac import decode_policy_observation


@dataclass(frozen=True)
class BehaviorSensorFrame:
    """现用感知 Ability 的 RGBD 数据结构，包含同帧像素与标定。"""

    sensor_id: str
    frame_id: str
    timestamp: datetime
    encoding: str
    data: Any
    metadata: dict


def legacy_pose(value):
    """现用 Ability 的米制 xyzw PoseStamped 输出。"""
    return {
        "header": {"frame_id": value["frame_id"]},
        "pose": {
            "position": dict(zip(("x", "y", "z"), value["position"])),
            "orientation": dict(zip(("x", "y", "z", "w"), value["quaternion_xyzw"])),
        },
    }


class DevelopmentObservationClient:
    def __init__(self, backend, scene_instance_id, generation):
        if not isinstance(scene_instance_id, str) or not scene_instance_id:
            raise ValueError("scene_instance_id 必须来自 Semantic 场景部署")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation 必须为正整数")
        self.backend, self.scene_instance_id, self.generation = (
            backend,
            scene_instance_id,
            generation,
        )

    def _path(self, *parts, **query):
        return (
            self.backend._path(*parts)
            + "?"
            + urlencode(
                {
                    "scene_instance_id": self.scene_instance_id,
                    "generation": self.generation,
                    **query,
                }
            )
        )

    def _check(self, value):
        if (value.get("generation"), value.get("scene_instance_id")) != (
            self.generation,
            self.scene_instance_id,
        ):
            raise BackendRequestError("观测属于其他场景或旧代次，请重新绑定 Semantic 部署")
        return value

    def _get(self, *parts, **query):
        return self._check(self.backend._json("GET", self._path(*parts, **query)))

    def state(self):
        return self._get("development", "state")

    def get_scene_geometry(self):
        """包含碰撞网格的 world 快照，字段保持现有 build_map 契约。"""
        return self._get("development", "scene-geometry")

    def get_world_pose(self, name):
        value = self._get("development", "world-pose", name=name)
        return {**value, "data": legacy_pose(value["pose"])}

    def get_localization(self):
        state = self.state()
        value = state["localization"]
        return {
            **value,
            "data": legacy_pose(value["pose"]),
            "generation": state["generation"],
            "scene_instance_id": state["scene_instance_id"],
            "step_count": state["step_count"],
        }

    def get_gripper_state(self, side):
        self._side(side)
        state = self.state()
        return {
            **state["gripper_feedback"][side],
            "generation": state["generation"],
            "scene_instance_id": state["scene_instance_id"],
            "step_count": state["step_count"],
        }

    def get_ee_pose(self, side, frame_id="body"):
        self._side(side)
        field = {
            "body": "end_effectors_body",
            "base_link": "end_effectors_body",
            "torso_link4": "end_effectors_torso",
            "world": "end_effectors",
        }[frame_id]
        state = self.state()
        value = {**state[field][side], "frame_id": frame_id}
        return {
            "data": legacy_pose(value),
            "generation": state["generation"],
            "scene_instance_id": state["scene_instance_id"],
            "step_count": state["step_count"],
        }

    @staticmethod
    def _side(side):
        if side not in {"left", "right"}:
            raise ValueError("side 必须为 left 或 right")

    def navigation_contacts(self):
        return self._get("development", "navigation-contacts")

    def capture_rgbd(self, sensor_id="head"):
        """返回 (metadata, {rgb: uint8 HWC, depth: float32 HW})，深度单位米。"""
        import numpy as np

        header, values = decode_policy_observation(
            self.backend._request("GET", self._path("sensors", sensor_id, "rgbd"))
        )
        self._check(header)
        rgb, depth = values["rgb"], values["depth"]
        calibration = header["calibration"]
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or depth.dtype != np.float32
            or depth.shape != rgb.shape[:2]
            or calibration["generation"] != self.generation
            or calibration["sequence"] != header["sequence"]
            or (calibration["height"], calibration["width"]) != depth.shape
        ):
            raise BackendRequestError("RGBD 图像与标定的形状、代次或采样帧不一致")
        return header, values

    def get_camera_calibration(self, sensor_id="head", calibration_id=None):
        query = {} if calibration_id is None else {"calibration_id": calibration_id}
        return self._get("sensors", sensor_id, "calibration", **query)

    def capture_rgbd_frames(self, sensor_id="head"):
        """转换为现有 ObjectPerception.validate_pair 接收的两个 SensorFrame。"""
        import numpy as np
        from scipy.spatial.transform import Rotation

        header, arrays = self.capture_rgbd(sensor_id)
        calibration = header["calibration"]
        optical_from_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        world_from_camera = np.asarray(calibration["world_from_camera_optical"]) @ optical_from_usd
        body_from_camera = np.asarray(calibration["body_from_camera_optical"]) @ optical_from_usd
        timestamp = datetime.fromisoformat(header["observed_at"])
        common = {
            "scene_instance_id": self.scene_instance_id,
            "generation": self.generation,
            "sequence": header["sequence"],
            "step_count": header["step_count"],
            "simulation_time": header["sim_time"],
            "calibration_id": calibration["calibration_id"],
            "sensor_name": calibration["frame_id"],
            "intrinsic_matrix": calibration["intrinsic_matrix"],
            "camera_position_world": world_from_camera[:3, 3].tolist(),
            "camera_orientation_world_xyzw": Rotation.from_matrix(world_from_camera[:3, :3])
            .as_quat()
            .tolist(),
            "camera_pose_body": body_from_camera.tolist(),
        }
        return tuple(
            BehaviorSensorFrame(
                sensor_id,
                calibration["frame_id"],
                timestamp,
                encoding,
                arrays[key],
                {**common, "shape": arrays[key].shape, "units": units, "depth_type": depth_type},
            )
            for key, encoding, units, depth_type in (
                ("rgb", "rgb8", "uint8", None),
                ("depth", "32FC1", "m", "optical_z"),
            )
        )
