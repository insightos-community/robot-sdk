from types import SimpleNamespace
import pytest
from galaxea_r1pro.galaxea_r1pro_0.backends.torso_trajectory import plan, NAMES, PREFIX
from galaxea_r1pro.galaxea_r1pro_0.backends.isaac import IsaacBackend


@pytest.mark.parametrize("period", [1 / 30, 1 / 60])
@pytest.mark.parametrize("duration", [1.5, 3.0, 5.0])
def test_finite_smooth_trajectory_preserves_held_joints_and_respects_budget(period, duration):
    start = [0.29, 0.31, -0.53, 0.17]
    goal = [0.29, 0.31, 0.6, 0.17]
    rows, metrics = plan(start, goal, duration, period)
    values = [row.controls[0].positions_rad for row in rows]
    assert len(rows) * period <= duration + 1e-12
    hold = round(metrics["hold_duration_sim_s"] / period)
    assert metrics["hold_duration_sim_s"] >= 0.5 - 1e-12
    assert all(v == dict(zip(NAMES, goal)) for v in values[-hold:])
    assert all([v[NAMES[i]] for i in (0, 1, 3)] == [start[i] for i in (0, 1, 3)] for v in values)
    q = [start[2]] + [v[NAMES[2]] for v in values]
    speed = [(b - a) / period for a, b in zip(q, q[1:])]
    assert min(speed) >= -1e-12
    assert max(speed) <= metrics["peak_target_speeds_rad_s"][2] + 1e-12
    acceleration = [(b - a) / period for a, b in zip([0.0] + speed, speed + [0.0])]
    assert max(map(abs, acceleration)) <= metrics["peak_target_accelerations_rad_s2"][2] + 1e-8


@pytest.mark.parametrize(
    "goal,duration",
    [([0.29, 0.31, 0.6, 0.17], 0.6), ([0.29, 0.31, 2.0, 0.17], 3), ([0, 0, float("nan"), 0], 3)],
)
def test_invalid_or_too_fast_plan_is_rejected(goal, duration):
    with pytest.raises(ValueError):
        plan([0.29, 0.31, -0.53, 0.17], goal, duration, 1 / 30)


def test_timed_status_and_stop_route_survive_client_recreation():
    calls = []
    b = IsaacBackend(SimpleNamespace(robot_id="r1"))
    b._connected = True
    b.observations = SimpleNamespace(generation=7, state=lambda: calls.append("generation_checked"))

    def http(method, path, body):
        calls.append((method, path, body))
        return {
            "scene_generation": 7,
            "status": "cancelled" if method == "POST" else "succeeded",
            "progress": 1.0,
            "details": {"executed_samples": 90, "total_samples": 90},
        }

    b.http = SimpleNamespace(
        _path=lambda *p: "/".join(p),
        _json=http,
        profile={"capabilities": {"control_period_s": 1 / 30}},
    )
    cid = PREFIX + "persisted-id"
    result = b.get_command(cid)
    assert (
        result.status == "succeeded"
        and result.output["verification_level"] == "trajectory_samples_executed"
    )
    assert result.feedback.metrics["elapsed_sim_s"] == pytest.approx(3.0)
    assert b.stop_command(cid, "test").status == "cancelled"
    assert calls == [
        "generation_checked",
        ("GET", f"commands/{cid}", None),
        "generation_checked",
        ("POST", f"commands/{cid}/stop", {}),
    ]


def test_stale_binding_cannot_cancel_new_generation():
    b = IsaacBackend(SimpleNamespace(robot_id="r1"))
    b._connected = True

    def stale():
        raise ValueError("generation changed")

    b.observations = SimpleNamespace(state=stale)
    b.http = SimpleNamespace(_json=lambda *args: pytest.fail("must reject before HTTP mutation"))
    with pytest.raises(ValueError, match="generation changed"):
        b.stop_command(PREFIX + "old", "test")
