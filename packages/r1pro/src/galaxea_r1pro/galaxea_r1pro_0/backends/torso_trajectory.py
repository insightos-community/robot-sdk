"""Finite R1Pro torso trajectories for the existing Runtime sequence endpoint."""

import math
from semantic_robot_sdk_core.control import ControlSample, ControlSequence, JointPositionControl

NAMES = tuple(f"torso_joint{i}" for i in range(1, 5))
# R1Pro model contract, radians and rad/s.
LOWER = (-1.1345, -2.7925, -1.8326, -3.0543)
UPPER = (1.8326, 2.5307, 1.5708, 3.0543)
SPEED_LIMIT = 2.5
PREFIX = "torso-sequence-"


def plan(start, goal, duration, period):
    start, goal = list(start), list(goal)
    if len(start) != 4 or len(goal) != 4 or not all(math.isfinite(v) for v in start + goal):
        raise ValueError("躯干轨迹需要四个有限起始角和目标角")
    if not math.isfinite(duration) or not 0.5 < duration <= 30:
        raise ValueError("躯干轨迹总时长须大于 0.5 且不超过 30 秒仿真时间")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Runtime 控制周期无效")
    if any(not lo <= value <= hi for value, lo, hi in zip(goal, LOWER, UPPER)):
        raise ValueError("躯干轨迹目标超出模型关节限位")
    total = math.floor(duration / period + 1e-9)
    hold = math.ceil(0.5 / period - 1e-9)
    moving = total - hold
    if moving < 1:
        raise ValueError("轨迹时长不足以容纳运动和到位复核窗口")
    travel = moving * period
    delta = [b - a for a, b in zip(start, goal)]
    peak_speed = [1.875 * abs(d) / travel for d in delta]
    if max(peak_speed) > SPEED_LIMIT + 1e-9:
        raise ValueError("指定时长内的平滑轨迹超过模型速度上限")
    rows = []
    for i in range(1, total + 1):
        t = min(i / moving, 1.0)
        blend = t * t * t * (10 + t * (-15 + 6 * t))
        positions = goal[:] if i >= moving else [a + blend * d for a, d in zip(start, delta)]
        rows.append(
            ControlSample(
                controls=[
                    JointPositionControl(group="trunk", positions_rad=dict(zip(NAMES, positions)))
                ]
            )
        )
    return rows, {
        "trajectory_duration_sim_s": total * period,
        "motion_duration_sim_s": travel,
        "hold_duration_sim_s": hold * period,
        "peak_target_speeds_rad_s": peak_speed,
        "peak_target_accelerations_rad_s2": [
            10 / math.sqrt(3) * abs(d) / (travel * travel) for d in delta
        ],
    }


def start(backend, positions, duration, command_id, execution_id):
    state = backend.observations.state()  # Also rejects stale scene/generation bindings.
    actual_names = backend.http.profile["capabilities"]["control_groups"]["trunk"]["joint_names"]
    if set(actual_names) != set(NAMES):
        raise ValueError("部署的躯干关节与 R1Pro 轨迹模型不匹配")
    period = float(backend.http.profile["capabilities"]["control_period_s"])
    samples, _ = plan([state["joints"][n]["position"] for n in NAMES], positions, duration, period)
    sequence = ControlSequence(
        robot_id=backend.profile.robot_id,
        execution_id=execution_id,
        generation=backend.observations.generation,
        control_period_s=period,
        samples=samples,
    )
    return backend.http.execute_sequence(sequence, command_id)
