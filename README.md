# Semantic Robot SDK

[English](README.md) | [简体中文](README.zh-CN.md)

This repository maintains the public specification of the Robot SDK and the
implementations for each robot model. The root directory is only a development
workspace; it does not publish a mixed Wheel containing all models:

```text
packages/core    semantic-robot-sdk-core
packages/r1pro   semantic-robot-sdk-r1pro
packages/franka  semantic-robot-sdk-franka
```

`core` provides only common types, errors, the Backend/Provider interfaces, command
feedback and stopping, unified deployment configuration, the subscription protocol,
and shared Fake tests. Robot models, joint names, URDF, Pinocchio, Ruckig, navigation
algorithms, vendor interfaces, and MuJoCo HTTP paths all stay in the corresponding
model package.

## Project Structure

| Path | Responsibility |
|---|---|
| `packages/core/` | Common models, resource contracts, backends, and motion primitives |
| `packages/r1pro/` | R1 Pro adaptation |
| `packages/franka/` | Franka adaptation |
| `tests/` | Contract and unit tests |
| `integration-tests/` | Integration tests that depend on assets / Runtime |

## Installation and Usage

The Ability environment installs the common Core and the Wheel matching the current
Robot model, and uses the SDK through normal Python imports; production deployments
do not add this repository's source directory to `PYTHONPATH`.

```bash
python -m pip install \
  dist/semantic_robot_sdk_core-*.whl \
  dist/semantic_robot_sdk_r1pro-*.whl
```

```python
from semantic_robot_sdk_r1pro import R1ProSDK

with R1ProSDK.from_environment() as robot:
    state = robot.state.snapshot()
    frame = robot.sensors.latest("camera.rgb")
```

Robots of the same model share the same set of Wheels; each Robot isolates its
identity, connections, commands, and state through its own `RobotDeployment` and
runtime data directory.

## R1 Pro Structure

For native cuRobo single-link and fixed multi-link object-carrying support, see the
[fixed multi-link attachment guide](packages/r1pro/FIXED_ATTACHMENTS.md).

```text
semantic_robot_sdk_r1pro
├── modules       Stable resource interfaces used by Abilities
├── backends      fake / mujoco / real / isaac
├── providers     local/vendor kinematics, motion, navigation
├── profile.py    R1 Pro fixed model capabilities
└── sdk.py        Assembles the above components from the deployment configuration
```

- The Backend only accesses state, low-level trajectories, the gripper, sensors, stop, and hold.
- Providers handle IK, trajectory generation, and path planning; the MuJoCo Runtime does not implement these algorithms.
- `real` must be injected with the actual vendor driver; `isaac` explicitly returns unimplemented in v0.5.
- Fake provides command deduplication, feedback ordering, resource mutual exclusion, state transitions, and stop evidence.

The Fake grasp environment is declared via `robot.sdk.options.initial_grasp_targets`
in the RobotDeployment. When the shared state file is first created, the Robot SDK
automatically writes these environment facts; the seven Abilities only read the same
state on subsequent starts or restarts and never re-initialize it. Low-level tests can
still call `configure_grasp_fixture(...)` or `set_tool_contact_state(...)` directly to
set up contact test state. Afterwards, closing and releasing only update tool contact,
force, and the `contact` SensorFrame; stable carrying and grasp success are still
determined by the Ability from continuous state. Without a fixture, closing the
gripper does not fake a contact success.

## One Robot, One Configuration

Ability and Pilot both read:

```bash
export SEMANTIC_ROBOT_CONFIG=/etc/semantic/robots/r1pro-001/robot-deployment.yaml
```

Configuration examples live in `examples/robot-deployment.fake.yaml` and
`examples/robot-deployment.mujoco.yaml`. Endpoint, firmware, and Provider are written
only in this deployment configuration, not repeated in every Ability CR.

When running multiple simulation Robots on the same host, each group of Pilot,
AbilityFramework, and Ability can override its own Runtime address. Environment
variables take precedence over `robot.sdk.endpoint` in the YAML:

```bash
SEMANTIC_ROBOT_CONFIG=/etc/semantic/robots/r1pro-sim.yaml \
SEMANTIC_ROBOT_SDK_ENDPOINT=http://127.0.0.1:8091 \
semantic-pilot ...
```

If multiple virtual Robots live in the same Runtime, they can use the same Endpoint
and are routed precisely by their respective `robot.id`:

```yaml
# r1pro-001/robot-deployment.yaml
robot:
  id: r1pro-001
  model: r1_pro_chassis
  backend: mujoco
  sdk:
    package: semantic-robot-sdk-r1pro
    endpoint: http://127.0.0.1:8090
```

```yaml
# r1pro-002/robot-deployment.yaml
robot:
  id: r1pro-002
  model: r1_pro_chassis
  backend: mujoco
  sdk:
    package: semantic-robot-sdk-r1pro
    endpoint: http://127.0.0.1:8090
```

If the Robots live in different Runtimes, each group of processes uses a different
`SEMANTIC_ROBOT_SDK_ENDPOINT` to override the template address. Robot ID, coordinate
frames, Provider, and safety limits are still kept in the deployment configuration;
the environment variable only overrides the Endpoint and does not change the Robot
identity.

Multiple Robot instances of the same model do not duplicate Wheels. Production
deployments share the SDK artifacts through the robot type package and only generate
`robot-deployment.yaml` in each Robot's instance directory.

The v0.5 stable entry points used by Abilities are:

- `state.capabilities()`, `state.snapshot()`.
- `base.plan_route(goal, occupancy=None, minimum_clearance_m=None)`,
  `base.follow_route(plan, command_id=...)`,
  `base.navigate(goal, command_id=..., occupancy=None, minimum_clearance_m=None)`.
- `upper_body.plan_joints(...)`, `move_joints(...)`, `plan_end_effector(...)`, and
  `move_end_effector(...)`.
- `end_effector.set_opening(...)`, `close_until_contact(...)`, `release(...)`.
- `sensors.list()`, `latest(...)`, `subscribe(...)`, and `subscribe_state(...)`.
- `commands.get(...)`, `feedback(...)`, `stop(...)`.
- `safety.stop_and_hold(...)`, `hold()`.

The `occupancy` parameter of local navigation is an optional override and does not
require the Navigation Ability to construct an SDK-internal grid. The SDK prefers the
`NavigationMapSource` injected at assembly time, and can also read the fixed scene map
in `robot.sdk.options.navigation_map_source`. The Fake backend uses an explicitly
marked empty test map when nothing is configured; MuJoCo and physical robots fail
explicitly when local planning is used without a map source. A vendor Navigation
Provider can fetch maps from the vendor navigation service itself.

The motion interfaces always use the caller-provided stable `command_id`. A repeated
ID with identical content returns the original command; different content is rejected.
Stopping goes through `robot.safety.stop_and_hold(command_id)`, enters the low level,
and verifies hold; when the physical state cannot be confirmed it returns
`interrupted`.

## MuJoCo Boundary

The R1 Pro MuJoCo Backend only consumes the existing Robot Profile, state, command,
stop/hold, sensor, and WebSocket streaming interfaces of
`plugin-mujoco-v040-platform`. This repository does not modify or embed the Runtime.
For real combined tests, the caller starts the pinned-version Runtime first:

```bash
PLUGIN_MUJOCO_URL=http://127.0.0.1:8090 \
R1PRO_ASSET_ROOT=/path/to/mujoco_asset make test-plugin
```

## Development and Build

```bash
uv sync --all-packages --group test
make lint
make test
make build
```

`make build` produces the three Wheels separately. Model and Runtime-specific tests do
not fake a pass when external assets are missing; you must explicitly provide the
corresponding environment variables before running `make test-r1pro`,
`make test-franka`, or `make test-plugin`.

## FAQ

- Fake contract tests do not validate real hardware or rendering.
- Integration targets require explicitly provided assets and a compatible running Runtime; some tests create and stop simulation scenes.
- When a native dependency fails to load, check the Wheel ABI, Python, and shared library versions instead of modifying import paths.
- Deployments use the installed Wheels; source `PYTHONPATH` overrides are for development only.

## Related Documents

[Detailed technical reference](README.reference.md) · [Build and test targets](Makefile)

## License

Copyright 2026 InsightOS. First-party code is licensed under [Apache-2.0](LICENSE); for
third-party components and assets, see [NOTICE](NOTICE) and the
[license scope](LICENSE_SCOPE.md).

## Reproducible Builds on Three Platforms

See the [glibc, musl, and macOS build instructions](README.build.md): pinned source
versions, actual script entry points, tool requirements, local and CI commands,
artifact locations, and platform validation scope.
