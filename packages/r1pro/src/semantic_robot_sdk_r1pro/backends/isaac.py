"""R1Pro 普通夹爪的 Isaac 定时控制后端。

这个出口只依赖公共 SDK 和 HTTP；OmniGibson、Torch 与模型依赖留在各自进程。
MuJoCo 拆码垛仍使用原有 Backend、规划器和工具配置。
"""

import json
import struct
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from semantic_robot_sdk_core.errors import BackendRequestError, BackendUnavailable
from semantic_robot_sdk_core.models import Command, PlanKind, Pose, RobotState, SensorFrame


class IsaacBackend:
    def __init__(self, endpoint: str, robot_id: str, *, timeout_seconds: float = 10):
        self.endpoint = endpoint.rstrip("/")
        self.robot_id = robot_id
        self.timeout_seconds = timeout_seconds
        self.profile = self._json("GET", self._path("profile"))
        if (self.profile.get("model"), self.profile.get("backend_profile")) != (
            "r1pro",
            "behavior-omnigibson",
        ):
            raise BackendRequestError("部署不是 R1Pro 普通夹爪 BEHAVIOR Profile")

    def _path(self, *parts):
        return "/api/v1/robots/" + "/".join(
            quote(item, safe="") for item in (self.robot_id, *parts)
        )

    def _request(self, method, path, body=None, *, with_headers=False):
        request = Request(
            self.endpoint + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = response.read()
                return (response.headers, data) if with_headers else data
        except HTTPError as error:
            raise BackendRequestError(
                f"Isaac Runtime HTTP {error.code}: {error.read().decode()}"
            ) from error
        except (URLError, TimeoutError) as error:
            # 提交后断线不能自动重发新的 command_id；调用者查询原命令或停止。
            raise BackendUnavailable(f"Isaac Runtime 不可达: {error}") from error

    def _json(self, method, path, body=None):
        return json.loads(self._request(method, path, body))

    def _check_robot(self, robot_id):
        if robot_id != self.robot_id:
            raise BackendRequestError("请求 Robot 与部署不一致")

    def timed_control_configuration(self):
        return dict(self.profile["capabilities"])

    def development_observations(self, scene_instance_id: str, generation: int):
        """绑定 Semantic 场景 UUID 与代次；读取 RGBD、反馈和真值导航数据。"""
        from .isaac_observations import DevelopmentObservationClient

        return DevelopmentObservationClient(self, scene_instance_id, generation)

    def atomic_controls(self, scene_instance_id: str, generation: int):
        """Return the existing Ability facade, bound to this Runtime deployment."""
        from galaxea_r1pro import BackendKind, RobotProfile, R1ProSDK

        return R1ProSDK.open(
            RobotProfile(
                self.robot_id,
                BackendKind.ISAAC,
                "behavior",
                {
                    "runtime_endpoint": self.endpoint,
                    "scene_instance_id": scene_instance_id,
                    "generation": generation,
                    "request_timeout": self.timeout_seconds,
                },
            )
        )

    def state(self, robot_id):
        self._check_robot(robot_id)
        value = self._json("GET", self._path("state"))
        return RobotState(
            robot_id=robot_id,
            generation=value["generation"],
            observed_at=value["observed_at"],
            base_pose=Pose(**value["base_pose"]),
            joint_positions={key: item["position"] for key, item in value["joints"].items()},
            end_effectors={key: Pose(**item) for key, item in value["end_effectors"].items()},
            gripper_openings=value["grippers"],
            in_hold=value["in_hold"],
        )

    def execute_sequence(self, sequence, command_id):
        self._check_robot(sequence.robot_id)
        value = self._json(
            "POST",
            self._path("commands"),
            {
                "command_id": command_id,
                "execution_id": sequence.execution_id,
                "scene_generation": sequence.generation,
                "type": "control_sequence",
                "control_sequence": {
                    "control_period_s": sequence.control_period_s,
                    "samples": [sample.model_dump() for sample in sequence.samples],
                },
            },
        )
        return self._command(value)

    def _command(self, value):
        return Command(
            command_id=value["command_id"],
            robot_id=self.robot_id,
            generation=value["scene_generation"],
            kind=PlanKind.CONTROL,
            status=value["status"],
            progress=value.get("progress", 1.0 if value["status"] == "succeeded" else 0.0),
        )

    def command(self, robot_id, command_id):
        self._check_robot(robot_id)
        return self._command(self._json("GET", self._path("commands", command_id)))

    def cancel_execution(self, robot_id, execution_id, generation):
        self._check_robot(robot_id)
        result = self._json(
            "POST",
            self._path("executions", execution_id, "cancel"),
            {"scene_generation": generation},
        )
        if result != {
            "execution_id": execution_id,
            "generation": generation,
            "status": "cancelled",
        }:
            raise BackendRequestError("Runtime 未确认本次执行取消")

    def scene_snapshot(self, instance_id, *, include_objects=True):
        path = f"/api/v1/scene-instances/{quote(instance_id, safe='')}/snapshot"
        if not include_objects:
            path += "?include_objects=false"
        value = self._json("GET", path)
        if self.robot_id not in {item["robot_id"] for item in value["robots"]}:
            raise BackendRequestError("快照不属于当前 Robot")
        return value

    def native_policy_observation(self):
        """型号 Ability 的原生策略观测出口，Framework 不解析此内容。

        精确保留上游相机字段、shape 和数值精度，模型绑定负责选择需要的字段。
        NumPy 只在此处延迟导入，普通拆码垛 SDK 导入不增加 Isaac/VLA 依赖。
        """
        return decode_policy_observation(self._request("GET", self._path("policy-observation")))

    def latest_sensor_frame(self, robot_id, sensor_id):
        self._check_robot(robot_id)
        headers, payload = self._request(
            "GET",
            self._path("sensors", sensor_id, "frames", "latest", "content"),
            with_headers=True,
        )
        return SensorFrame(
            sensor_id=sensor_id,
            sequence=int(headers["X-Semantic-Sequence"]),
            generation=int(headers["X-Semantic-Generation"]),
            frame_id=headers["X-Semantic-Frame"],
            encoding=headers["X-Semantic-Encoding"],
            observed_at=headers["X-Semantic-Observed-At"],
            width=int(headers["X-Semantic-Width"]),
            height=int(headers["X-Semantic-Height"]),
            payload=payload,
        )


def decode_policy_observation(packet):
    import numpy as np

    if len(packet) < 4:
        raise BackendRequestError("原生观测头不完整")
    size = struct.unpack(">I", packet[:4])[0]
    if size > len(packet) - 4:
        raise BackendRequestError("原生观测头被截断")
    header = json.loads(packet[4 : 4 + size])
    payload = memoryview(packet)[4 + size :]
    values = {}
    for key, spec in header["arrays"].items():
        dtype, shape = np.dtype(spec["dtype"]), tuple(spec["shape"])
        offset, length = spec["offset"], spec["length"]
        if (
            dtype.hasobject
            or any(dim < 0 for dim in shape)
            or offset < 0
            or length < 0
            or offset + length > len(payload)
            or int(np.prod(shape)) * dtype.itemsize != length
        ):
            raise BackendRequestError(f"原生观测字段长度无效: {key}")
        values[key] = (
            np.frombuffer(payload[offset : offset + length], dtype=dtype).reshape(shape).copy()
        )
    return header, values
