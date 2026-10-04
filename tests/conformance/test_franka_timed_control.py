import json
import struct

import pytest

from semantic_robot_sdk_core.control import ControlSequence
from semantic_robot_sdk_core.errors import BackendRequestError
from semantic_robot_sdk_core.models import PlanKind
from semantic_robot_sdk_franka import capabilities
from semantic_robot_sdk_franka.backends.mujoco import MujocoBackend


class Backend(MujocoBackend):
    def __init__(self):
        self.calls = []
        self.packet = b""
        super().__init__("http://runtime.test", "franka-0", capabilities())

    def _json(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.endswith("/profile"):
            return {
                "robot_id": "franka-0",
                "model": "franka_panda",
                "coordinate_frame": "world",
                "joint_names": capabilities().joint_names,
                "end_effectors": ["hand"],
                "grippers": ["hand"],
                "capabilities": {
                    "frames": ["world", "panda_link0"],
                    "commands": ["control_sequence"],
                    "control_mode": "OSC_POSE",
                    "control_period_s": 0.05,
                },
            }
        if path.endswith("/cancel"):
            return {"execution_id": "skill-1", "status": "cancelled"}
        if path.endswith("/snapshot"):
            return {"instance_id": "scene-test", "robots": [{"robot_id": "franka-0"}]}
        return {
            **body,
            "robot_id": "franka-0",
            "status": "accepted",
            "updated_at": "2026-09-09T00:00:00Z",
        }

    def _request(self, method, path, body=None):
        assert path.endswith("/observation")
        return None, self.packet


def test_franka_sequence_has_physical_units_and_uses_existing_command_status():
    backend = Backend()
    assert backend.timed_control_configuration() == {
        "control_mode": "OSC_POSE",
        "control_period_s": 0.05,
    }
    sequence = ControlSequence(
        robot_id="franka-0",
        execution_id="skill-1",
        generation=3,
        control_period_s=0.05,
        samples=[
            {
                "controls": [
                    {
                        "kind": "end_effector_delta",
                        "group": "arm",
                        "frame_id": "world",
                        "translation_m": [0.025, 0, 0],
                        "rotation_axis_angle_rad": [0, 0, 0],
                    },
                    {"kind": "gripper_direction", "group": "hand", "closing_direction": 1},
                ]
            }
        ],
    )
    result = backend.execute_sequence(sequence, "chunk-1")
    body = backend.calls[-1][2]
    assert body["scene_generation"] == 3
    assert body["execution_id"] == "skill-1"
    assert body["control_sequence"]["samples"][0]["controls"][0]["translation_m"][0] == 0.025
    assert result.kind is PlanKind.CONTROL
    backend.cancel_execution("franka-0", "skill-1", 3)
    assert backend.calls[-1][1].endswith("/executions/skill-1/cancel")


def test_franka_scene_snapshot_checks_actual_robot_and_instance():
    backend = Backend()
    assert backend.scene_snapshot("scene-test")["robots"][0]["robot_id"] == "franka-0"
    with pytest.raises(BackendRequestError, match="快照"):
        backend.scene_snapshot("another-scene")


def test_franka_decodes_raw_rgb_and_finger_state_without_pixel_transforms():
    backend = Backend()
    header = {
        "generation": 2,
        "sequence": 10,
        "sim_time": 0.45,
        "observed_at": "2026-09-09T00:00:00Z",
        "robot_state": {
            "robot_id": "franka-0",
            "generation": 2,
            "joints": dict.fromkeys(capabilities().joint_names, 0.0),
        },
        "gripper_joint_positions": {"panda_finger_joint1": 0.03, "panda_finger_joint2": -0.02},
        "images": [
            {
                "sensor_id": "agentview_rgb",
                "frame_id": "agentview",
                "encoding": "rgb8",
                "width": 2,
                "height": 1,
                "offset": 0,
                "length": 6,
            }
        ],
    }
    raw_header = json.dumps(header).encode()
    pixels = bytes([255, 0, 0, 0, 255, 0])
    backend.packet = struct.pack(">I", len(raw_header)) + raw_header + pixels
    observation = backend.synchronized_observation("franka-0")
    assert observation.images[0].payload == pixels
    assert observation.images[0].generation == observation.state.generation == 2
    assert observation.gripper_joint_positions["panda_finger_joint2"] == -0.02
    backend.packet = backend.packet[:-1]
    with pytest.raises(BackendRequestError, match="RGB 数据长度"):
        backend.synchronized_observation("franka-0")
