"""Isaac 独立后端的契约测试；不替代真实物理验证。"""

import json
import struct
from unittest.mock import patch

import numpy as np
import pytest

from semantic_robot_sdk_core.control import ControlSequence
from semantic_robot_sdk_core.errors import BackendRequestError
from semantic_robot_sdk_r1pro.backends.isaac import IsaacBackend, decode_policy_observation


def test_native_arrays_keep_pixel_channels_and_precision():
    value = np.array([0.123456789, -0.999999999], dtype=np.float64)
    header = json.dumps(
        {
            "generation": 3,
            "arrays": {
                "robot_r1::proprio": {
                    "dtype": value.dtype.str,
                    "shape": [2],
                    "offset": 0,
                    "length": value.nbytes,
                }
            },
        }
    ).encode()
    metadata, arrays = decode_policy_observation(
        struct.pack(">I", len(header)) + header + value.tobytes()
    )
    assert metadata["generation"] == 3
    np.testing.assert_array_equal(arrays["robot_r1::proprio"], value)
    with pytest.raises(BackendRequestError):
        decode_policy_observation(struct.pack(">I", len(header)) + header + value.tobytes()[:-1])


def test_sequence_preserves_identity_and_physical_units():
    sent = []

    def request(self, method, path, body=None):
        sent.append((method, path, body))
        if method == "GET":
            return {"model": "r1pro", "backend_profile": "behavior-omnigibson"}
        return {"command_id": "cmd1", "scene_generation": 4, "status": "accepted"}

    with patch.object(IsaacBackend, "_json", request):
        backend = IsaacBackend("http://localhost:18100", "robot_r1")
        sequence = ControlSequence(
            robot_id="robot_r1",
            execution_id="run1",
            generation=4,
            control_period_s=1 / 30,
            samples=[
                {
                    "controls": [
                        {
                            "kind": "base_velocity",
                            "group": "base",
                            "frame_id": "base_link",
                            "linear_mps": [0.1, -0.2],
                            "angular_rps": 0.3,
                        }
                    ]
                }
            ],
        )
        command = backend.execute_sequence(sequence, "cmd1")
    assert command.robot_id == "robot_r1"
    body = sent[-1][2]
    assert body["scene_generation"] == 4 and body["execution_id"] == "run1"
    assert body["control_sequence"]["samples"][0]["controls"][0]["linear_mps"] == (0.1, -0.2)


def test_rejects_depalletizing_profile():
    with patch.object(
        IsaacBackend,
        "_json",
        return_value={"model": "r1_pro_chassis", "backend_profile": "native-mujoco"},
    ):
        with pytest.raises(BackendRequestError, match="普通夹爪"):
            IsaacBackend("http://localhost:18100", "robot_r1")


def test_evaluation_snapshot_can_skip_map_objects():
    with patch.object(
        IsaacBackend,
        "_json",
        return_value={
            "model": "r1pro",
            "backend_profile": "behavior-omnigibson",
            "robots": [{"robot_id": "robot_r1"}],
        },
    ) as request:
        backend = IsaacBackend("http://localhost:18100", "robot_r1")
        backend.scene_snapshot("scene")
        assert request.call_args.args[1].endswith("/snapshot")
        backend.scene_snapshot("scene", include_objects=False)
        assert request.call_args.args[1].endswith("/snapshot?include_objects=false")
