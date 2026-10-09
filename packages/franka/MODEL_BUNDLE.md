# Franka Panda Official Model Bundle

[English](MODEL_BUNDLE.md) | [简体中文](MODEL_BUNDLE.zh-CN.md)

The SDK repository does not distribute large Robot assets, nor does it search for
models in `.venv`, robosuite, Isaac Sim, or ROS installation directories. The release
pipeline must first produce a standalone, reviewable model bundle in the asset
repository, then hand the bundle root directory to the dedicated tests via
`FRANKA_MODEL_ROOT`; production code passes that path directly.

## Sources and Licensing

Models are accepted only from the official Franka Robotics repositories:

- `https://github.com/frankarobotics/franka_ros`: source of the legacy Panda description.
- `https://github.com/frankarobotics/franka_description`: the current official model repository.

Both are released under Apache-2.0 and provide `LICENSE` and `NOTICE` in the
repository. An asset MR must:

1. Pin a Tag or Commit; it must not record `main`, `master`, or `latest`.
2. Keep the complete `LICENSE` and `NOTICE` of the same revision; links in the MR description alone are not enough.
3. Record the command and inputs that generate `panda_arm_hand.urdf`; the root Link of the generated result must be
   `panda_link0`, and the hand end-effector Frame must be `panda_hand`.
4. Keep the collision Meshes referenced by the URDF. A bundle with only visual Meshes and no collision Meshes cannot pass
   the Pinocchio self-collision gate.
5. Have the asset owner verify file provenance and distributable scope. Programmatic validation can only prevent missing files or wrong pinned versions;
   it cannot replace a license review.

## Directory and Manifest

The recommended layout is as follows; `package_roots` also allows another
self-contained hierarchy, but it must not leave the model bundle root directory.

```text
franka-model-bundle/
├── robot-model.json
├── LICENSE
├── NOTICE
└── franka_description/
    ├── robots/panda_arm_hand.urdf
    └── meshes/...
```

`robot-model.json` uses the in-repo `model-manifest.example.json` as its template. The
validator checks the model, URDF filename, package root, root frame, end-effector
frame, the Apache-2.0 marker, LICENSE, NOTICE, the official source, and the pinned
revision. Paths must be relative to the model root directory.

The Franka factory only accepts the model root directory:

```python
from semantic_robot_sdk_franka import create_mujoco_sdk

sdk = create_mujoco_sdk(runtime_url, robot_id, "/opt/semantic-assets/franka-panda")
```

Callers can no longer override the URDF, Mesh root, or `panda_hand`, preventing the
Runtime Profile and IK from using two different model definitions.

## Official Algorithm Gate

```bash
FRANKA_MODEL_ROOT=/opt/semantic-assets/franka-panda make test-franka
```

This test actually loads the URDF and collision Meshes, computes non-zero FK/IK with
Pinocchio, and then uses Ruckig to generate trajectories bounded by joint velocity,
acceleration, and jerk limits. A missing environment variable, missing model,
incomplete license materials, or unavailable algorithm dependencies all fail the test;
it will not disguise acceptance with a `skip`.
