"""HTTP adapter for the existing Ability SDK contract. Runtime owns all simulation."""

from dataclasses import asdict, is_dataclass
from datetime import datetime
from . import torso_trajectory, joint_trajectory
from uuid import uuid4

from semantic_robot_sdk_r1pro.backends.isaac import IsaacBackend as RuntimeHTTP
from ..utils.types import (
    BackendKind,
    CapabilityInfo,
    CommandHandle,
    CommandFeedback,
    CommandResult,
    CommandStatus,
    Pose,
    RobotState,
    JointState,
    SensorFrame,
)
from ..utils.errors import BackendUnavailableError


def encode(value):
    if is_dataclass(value):
        return encode(asdict(value))
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def result(value):
    feedback = value["feedback"]
    return CommandResult(
        value["command_id"],
        CommandStatus(value["status"]),
        CommandFeedback(
            feedback["command_id"],
            CommandStatus(feedback["status"]),
            feedback["progress"],
            feedback["phase"],
            feedback["metrics"],
            datetime.fromisoformat(feedback["timestamp"]),
        ),
        value.get("output", {}),
        value.get("error_code"),
        value.get("error_message"),
    )


class IsaacBackend:
    def __init__(self, profile):
        self.profile = profile
        self.http = self.observations = None
        self._connected = False

    def connect(self):
        options = self.profile.options
        self.http = RuntimeHTTP(
            options["runtime_endpoint"],
            self.profile.robot_id,
            timeout_seconds=options.get("request_timeout", 30),
        )
        self.observations = self.http.development_observations(
            options["scene_instance_id"], options["generation"]
        )
        self.observations.state()
        self._connected = True

    def close(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def _require_connected(self):
        if not self._connected:
            raise BackendUnavailableError(reason="not_connected", hint="请先连接 Semantic Runtime")

    def _call(self, method, *parts, body=None):
        self._require_connected()
        packet = self.http._json(method, self.observations._path("atomic", *parts), encode(body))
        self.observations._check(packet)
        return packet["value"]

    def start_command(self, resource, operation, target, *, command_id=None, execution_id=None):
        if (resource, operation) == ("joint_control", "control_joint_trajectory"):
            self._require_connected()
            command_id = command_id or joint_trajectory.PREFIX + str(uuid4())
            if not command_id.startswith(joint_trajectory.PREFIX):
                raise ValueError("关节轨迹 command_id 需要专属前缀")
            joint_trajectory.start(self, target, command_id, execution_id or command_id)
            return CommandHandle(command_id, self.profile.robot_id, resource, operation)
        if (resource, operation) == ("torso_control", "control_torso_trajectory"):
            self._require_connected()
            command_id = command_id or torso_trajectory.PREFIX + str(uuid4())
            if not command_id.startswith(torso_trajectory.PREFIX):
                raise ValueError("躯干轨迹 command_id 需要专属前缀以支持重连查询")
            torso_trajectory.start(
                self,
                target["positions"],
                target["duration_seconds"],
                command_id,
                execution_id or command_id,
            )
            return CommandHandle(command_id, self.profile.robot_id, resource, operation)
        command_id = command_id or str(uuid4())
        value = self._call(
            "POST",
            "commands",
            body=dict(
                command_id=command_id,
                execution_id=execution_id or command_id,
                resource=resource,
                operation=operation,
                target=target,
            ),
        )
        return CommandHandle(
            value["command_id"],
            value["robot_id"],
            value["resource"],
            value["operation"],
            datetime.fromisoformat(value["accepted_at"]),
        )

    def get_command(self, command_id=None, *, resource=None, operation=None, target=None):
        if command_id is not None and command_id.startswith(
            (torso_trajectory.PREFIX, joint_trajectory.PREFIX)
        ):
            return self._trajectory_result(command_id)
        if command_id is not None:
            return result(self._call("GET", "commands", command_id))
        if (resource, operation) == ("sensors", "get_camera_calibration"):
            self._require_connected()
            data = self.observations.get_camera_calibration(**(target or {}))
            return CommandResult(
                "",
                CommandStatus.SUCCEEDED,
                CommandFeedback("", CommandStatus.SUCCEEDED, 1.0, "sampled"),
                {"data": data},
            )
        return result(
            self._call(
                "POST",
                "query",
                body=dict(resource=resource, operation=operation, target=target or {}),
            )
        )

    def stop_command(self, command_id, reason):
        if command_id.startswith((torso_trajectory.PREFIX, joint_trajectory.PREFIX)):
            return self._trajectory_result(command_id, stop=True)
        return result(self._call("POST", "commands", command_id, "stop", body={"reason": reason}))

    def _trajectory_result(self, command_id, stop=False):
        self._require_connected()
        self.observations.state()  # Ensure cancellation/query cannot cross reset.
        path = self.http._path("commands", command_id, *(["stop"] if stop else []))
        value = self.http._json("POST" if stop else "GET", path, {} if stop else None)
        if value["scene_generation"] != self.observations.generation:
            raise ValueError("躯干轨迹属于其他 generation")
        status = CommandStatus(value["status"])
        details = value["details"]
        period = float(self.http.profile["capabilities"]["control_period_s"])
        metrics = {
            "executed_samples": details["executed_samples"],
            "total_samples": details["total_samples"],
            "elapsed_sim_s": details["executed_samples"] * period,
            "trajectory_duration_sim_s": details["total_samples"] * period,
        }
        return CommandResult(
            command_id,
            status,
            CommandFeedback(
                command_id, status, value["progress"], "trajectory_" + status.value, metrics
            ),
            {"verification_level": "trajectory_samples_executed"},
        )

    def hold(self, resources, reason):
        return self._call("POST", "hold", body=dict(resources=resources, reason=reason))

    def capabilities(self):
        return CapabilityInfo(
            "R1Pro",
            BackendKind.ISAAC,
            self.profile.firmware_profile,
            (
                "state",
                "sensors",
                "safety",
                "joint_control",
                "eef_control",
                "torso_control",
                "woosh_move",
                "gripper_control",
            ),
        )

    def snapshot(self):
        self._require_connected()
        state = self.observations.state()
        pose = state["localization"]["pose"]
        names = [
            name
            for group, names in state["control_groups"].items()
            if group in {"arm_left", "arm_right", "trunk"}
            for name in names
        ]
        joints = state["joints"]
        return RobotState(
            self.profile.robot_id,
            BackendKind.ISAAC,
            True,
            "hold" if state["in_hold"] else "ready",
            Pose("odom", tuple(pose["position"]), tuple(pose["quaternion_xyzw"])),
            JointState(
                tuple(names),
                tuple(joints[n]["position"] for n in names),
                tuple(joints[n]["velocity"] for n in names),
                tuple(joints[n]["effort"] for n in names),
            ),
            timestamp=datetime.fromisoformat(state["observed_at"]),
        )

    def capture_rgbd(self, sensor_id):
        self._require_connected()
        return tuple(
            SensorFrame(f.sensor_id, f.frame_id, f.timestamp, f.encoding, f.data, f.metadata)
            for f in self.observations.capture_rgbd_frames(sensor_id)
        )

    def capture(self, sensor_id, encoding):
        if encoding not in {"rgb8", "32FC1"}:
            raise ValueError("encoding must be rgb8 or 32FC1")
        return self.capture_rgbd(sensor_id)[0 if encoding == "rgb8" else 1]
