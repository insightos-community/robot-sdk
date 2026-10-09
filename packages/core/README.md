# semantic-robot-sdk-core

[English](README.md) | [简体中文](README.zh-CN.md)

Common Robot SDK types, Backend/Provider interfaces, unified deployment configuration, and Fake testing foundations. Does not include specific robot models or planning algorithms.

## Optional timed control

`semantic_robot_sdk_core.control` provides `ControlSequence` and `TimedControlBackend`.
This interface is independent of the existing `RobotBackend.execute_plan` and does not
require older models to emulate policy control.

A control sequence expresses, per group, absolute joint angles, end-effector
translation/axis-angle increments, chassis velocity, and gripper open/close direction.
Each segment carries the Robot, execution identity, generation, and control period; raw
model arrays must be converted in the model adaptation layer before submission.
`cancel_execution` revokes the entire execution identity, including inference results
that have not yet returned, not just the commands of a single chunk. Sequence completion
only means the control has been consumed; task acceptance is still performed by the
Robot Skill.

The Franka MuJoCo Backend adds this interface plus `synchronized_observation`, which
preserves the raw RGB, EEF, and two-finger joint state without Web image processing.
Actual capabilities are as declared by the Runtime Profile; LIBERO's OSC configuration
is deployed separately from the original JOINT_POSITION configuration.
