import json
import struct
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from semantic_robot_sdk_core.errors import BackendRequestError
from semantic_robot_sdk_r1pro.backends.isaac_observations import DevelopmentObservationClient


def client_with_state():
    calls = []
    value = {
        "generation": 7,
        "scene_instance_id": "scene/a",
        "step_count": 12,
        "end_effectors_body": {
            "left": {
                "position": [0.3, 0.2, 0.8],
                "quaternion_xyzw": [0, 0, 0, 1],
                "frame_id": "body",
            }
        },
        "gripper_feedback": {"left": {"positions_m": [0.01, 0.02], "efforts_n": [10.0, 12.0]}},
        "localization": {
            "pose": {"position": [1, 2, 0], "quaternion_xyzw": [0, 0, 0, 1], "frame_id": "odom"},
            "source": "simulation_ground_truth",
            "valid": True,
        },
    }
    backend = SimpleNamespace(_path=lambda *parts: "/" + "/".join(parts))

    def get(method, path):
        calls.append(path)
        return value

    backend._json = get
    return DevelopmentObservationClient(backend, "scene/a", 7), calls, value


def test_existing_pose_and_gripper_feedback_keep_units_and_identity():
    client, calls, value = client_with_state()
    assert client.get_ee_pose("left")["data"]["pose"]["position"] == {"x": 0.3, "y": 0.2, "z": 0.8}
    assert client.get_gripper_state("left")["efforts_n"] == [10, 12]
    assert client.get_localization()["data"]["header"]["frame_id"] == "odom"
    assert parse_qs(urlsplit(calls[0]).query)["scene_instance_id"] == ["scene/a"]
    value["generation"] += 1
    with pytest.raises(BackendRequestError, match="旧代次"):
        client.get_gripper_state("left")


def packet(calibration_sequence=13):
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.array([[0.5, 1, 2], [3, np.inf, 4]], dtype=np.float32)
    header = {
        "generation": 7,
        "scene_instance_id": "scene/a",
        "sequence": 13,
        "step_count": 12,
        "sim_time": 0.4,
        "observed_at": "2026-09-20T07:00:00+00:00",
        "calibration": {
            "generation": 7,
            "sequence": calibration_sequence,
            "height": 2,
            "width": 3,
            "frame_id": "head",
            "calibration_id": "7:12:head",
            "intrinsic_matrix": np.eye(3).tolist(),
            "world_from_camera_optical": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
            "body_from_camera_optical": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
        },
        "arrays": {
            "rgb": {"dtype": "uint8", "shape": [2, 3, 3], "offset": 0, "length": rgb.nbytes},
            "depth": {
                "dtype": "float32",
                "shape": [2, 3],
                "offset": rgb.nbytes,
                "length": depth.nbytes,
            },
        },
    }
    data = json.dumps(header).encode()
    return struct.pack(">I", len(data)) + data + rgb.tobytes() + depth.tobytes()


def test_rgbd_preserves_meter_depth_and_rejects_mismatched_calibration_frame():
    client, _, _ = client_with_state()
    client.backend._request = lambda *args: packet()
    metadata, arrays = client.capture_rgbd()
    np.testing.assert_array_equal(arrays["rgb"], np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    assert arrays["depth"].dtype == np.float32
    assert arrays["depth"][0, 0] == 0.5
    assert np.isinf(arrays["depth"][1, 1])
    client.backend._request = lambda *args: packet(calibration_sequence=12)
    with pytest.raises(BackendRequestError, match="采样帧不一致"):
        client.capture_rgbd()


def test_existing_perception_frames_share_timestamp_and_usd_calibration():
    client, _, _ = client_with_state()
    client.backend._request = lambda *args: packet()
    rgb, depth = client.capture_rgbd_frames()
    assert rgb.timestamp == depth.timestamp
    assert rgb.metadata["step_count"] == depth.metadata["step_count"]
    assert depth.encoding == "32FC1" and depth.metadata["units"] == "m"
    np.testing.assert_allclose(depth.metadata["camera_pose_body"], np.eye(4))
    np.testing.assert_allclose(rgb.metadata["camera_orientation_world_xyzw"], [0, 0, 0, 1])
