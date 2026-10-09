# semantic-robot-sdk-r1pro

[English](README.md) | [简体中文](README.zh-CN.md)

Stable modules for the R1 Pro, Fake/MuJoCo/real/Isaac Backends, and local/vendor Provider assembly.

## Full BEHAVIOR development observations

Use the endpoint, robot ID, scene UUID, and generation provided by Semantic scene deployment:

```python
from semantic_robot_sdk_r1pro.backends.isaac import IsaacBackend

backend = IsaacBackend(endpoint, robot_id)
observations = backend.development_observations(scene_instance_id, generation)
state = observations.state()
header, arrays = observations.capture_rgbd("head")
rgb, depth_m = arrays["rgb"], arrays["depth"]
calibration = header["calibration"]
geometry = observations.get_scene_geometry()
localization = observations.get_localization()
left_eef = observations.get_ee_pose("left", frame_id="body")
gripper = observations.get_gripper_state("left")
```

RGBD and binary policy observations require the `isaac` extra (NumPy / SciPy). Calibration is
returned with the image, and the original frame can also be read via
`get_camera_calibration("head", calibration_id)`. The end-effector interface supports
`body`, `base_link`, `torso_link4`, and `world`; the output `data` keeps the PoseStamped
field format of the current Ability, and the gripper retains the meter, meter/second, and
newton readback fields.

`capture_rgbd_frames()` returns the RGB/depth SensorFrame required by the existing
perception Ability, with same-frame timestamp, step_count, intrinsics, and USD camera
extrinsics, ready to hand to `validate_pair()`. This client only reads Runtime data; the
overall SDK facade of the current Ability and the migration of atomic motion control still
need to be integrated. The client does not create simulations, streams, or resets.
After a reset, create the observation client with the new generation. Old clients receive
an explicit error. Localization and collision meshes are part of the full development
observations, with localization marked `simulation_ground_truth`.


## BEHAVIOR atomic control facade

With the Isaac extra dependencies installed, use `from galaxea_r1pro import R1ProSDK, RobotProfile, BackendKind`
to connect to a Semantic-managed scene. Profile.options must include `runtime_endpoint`,
`scene_instance_id` (UUID), and `generation`, with optional `request_timeout`.
`R1ProSDK.open(profile)` returns the joint / eef / torso / gripper /
woosh_move / state / sensors / command / safety interfaces used by the existing Ability.

Control requests are sent to the Runtime over HTTP; measured position, velocity, and
native episode results decide the action status. The SDK's close only disconnects the
client. Simulation start/stop/reset and camera streaming are managed by the Semantic
Runtime. Re-bind to the new generation after a reset. The chassis feedback source is the
scene ground truth in development_full mode. The existing timed policy backend can also
obtain the same facade via `backend.atomic_controls(scene_instance_id, generation)`.
The two control modes are mutually exclusively scheduled by the Runtime.

See `docs-atomic-sdk.md` in the adjacent Isaac Runtime for the atomic protocol, fields,
and native action validation scripts.
