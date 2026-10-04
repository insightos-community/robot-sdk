import pytest
from pydantic import ValidationError

from semantic_robot_sdk_core.control import ControlSample, ControlSequence


def sample():
    return {
        "controls": [
            {
                "kind": "end_effector_delta",
                "group": "arm",
                "frame_id": "world",
                "translation_m": [0.01, 0, 0],
                "rotation_axis_angle_rad": [0, 0, 0],
            },
            {"kind": "gripper_direction", "group": "hand", "closing_direction": 0},
        ],
    }


def test_sequence_preserves_physical_units_and_execution_identity():
    sequence = ControlSequence(
        robot_id="franka-0",
        execution_id="skill-1",
        generation=2,
        control_period_s=0.05,
        samples=[sample(), sample()],
    )
    assert sequence.samples[0].controls[0].translation_m == (0.01, 0, 0)
    assert sequence.execution_id == "skill-1"
    assert sequence.generation == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2.0])
def test_sequence_rejects_invalid_gripper_direction(value):
    payload = sample()
    payload["controls"][1]["closing_direction"] = value
    with pytest.raises(ValidationError):
        ControlSample.model_validate(payload)


def test_sequence_rejects_conflicting_groups():
    payload = sample()
    payload["controls"][1]["group"] = "arm"
    with pytest.raises(ValidationError, match="同一控制组"):
        ControlSample.model_validate(payload)


def test_sequence_does_not_silently_drop_a_group():
    second = sample()
    second["controls"].pop()
    with pytest.raises(ValidationError, match="语义必须保持一致"):
        ControlSequence(
            robot_id="franka-0",
            execution_id="skill-1",
            generation=2,
            control_period_s=0.05,
            samples=[sample(), second],
        )
