"""Forward a finite planned joint sequence without IK or target reconstruction."""

import math
from semantic_robot_sdk_core.control import ControlSample, ControlSequence, JointPositionControl

PREFIX = "joint-sequence-"


def start(backend, trajectory, command_id, execution_id):
    backend.observations.state()
    profile = backend.http.profile["capabilities"]
    period = float(trajectory["control_period_s"])
    if not math.isclose(period, float(profile["control_period_s"]), rel_tol=0, abs_tol=1e-9):
        raise ValueError("轨迹控制周期与 Runtime 不一致")
    names = list(trajectory["joint_names"])
    rows = trajectory["positions_rad"]
    if not names or len(set(names)) != len(names) or not rows:
        raise ValueError("轨迹需要唯一关节名和非空采样序列")
    groups = {}
    for group in ("arm_left", "arm_right", "trunk"):
        actual = list(profile["control_groups"][group]["joint_names"])
        selected = set(actual) & set(names)
        if selected:
            if selected != set(actual):
                raise ValueError("轨迹必须包含所选控制组的全部关节")
            groups[group] = actual
    if set(names) != {n for values in groups.values() for n in values}:
        raise ValueError("轨迹仅支持躯干和手臂位置控制")
    samples = []
    for row in rows:
        if len(row) != len(names) or not all(math.isfinite(v) for v in row):
            raise ValueError("轨迹采样需要与关节名等长的有限角度")
        positions = dict(zip(names, row))
        samples.append(
            ControlSample(
                controls=[
                    JointPositionControl(
                        group=group, positions_rad={n: positions[n] for n in joints}
                    )
                    for group, joints in groups.items()
                ]
            )
        )
    sequence = ControlSequence(
        robot_id=backend.profile.robot_id,
        execution_id=execution_id,
        generation=backend.observations.generation,
        control_period_s=period,
        samples=samples,
    )
    return backend.http.execute_sequence(sequence, command_id)
