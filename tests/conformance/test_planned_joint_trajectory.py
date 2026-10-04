from types import SimpleNamespace
import pytest
from galaxea_r1pro.galaxea_r1pro_0.backends.joint_trajectory import start, PREFIX
from galaxea_r1pro.galaxea_r1pro_0.backends.isaac import IsaacBackend


def fixture():
    groups = {
        g: {"joint_names": [f"{prefix}{i}" for i in range(1, count + 1)]}
        for g, prefix, count in [
            ("arm_left", "left_arm_joint", 7),
            ("arm_right", "right_arm_joint", 7),
            ("trunk", "torso_joint", 4),
        ]
    }
    names = [n for group in groups.values() for n in group["joint_names"]]
    wire = {
        "joint_names": names,
        "positions_rad": [[0.0] * 18, [0.05] * 18, [0.1] * 18],
        "control_period_s": 1 / 30,
    }
    calls = []
    b = SimpleNamespace(
        profile=SimpleNamespace(robot_id="r1"),
        observations=SimpleNamespace(state=lambda: None, generation=3),
        http=SimpleNamespace(
            profile={"capabilities": {"control_period_s": 1 / 30, "control_groups": groups}},
            execute_sequence=lambda seq, cid: calls.append((seq, cid)),
        ),
    )
    return b, wire, calls


def test_all_groups_are_forwarded_in_one_sequence_without_replanning():
    b, w, calls = fixture()
    start(b, w, "command", "execution")
    assert len(calls) == 1
    seq, cid = calls[0]
    assert cid == "command" and seq.generation == 3
    assert len(seq.samples) == len(w["positions_rad"])
    for row, sample in zip(w["positions_rad"], seq.samples):
        actual = {k: v for control in sample.controls for k, v in control.positions_rad.items()}
        assert actual == dict(zip(w["joint_names"], row))


@pytest.mark.parametrize("corruption", ["partial_group", "nonfinite", "period"])
def test_invalid_sequences_never_submit(corruption):
    b, w, calls = fixture()
    if corruption == "partial_group":
        w["joint_names"] = w["joint_names"][:-1]
        w["positions_rad"] = [r[:-1] for r in w["positions_rad"]]
    if corruption == "nonfinite":
        w["positions_rad"][1][0] = float("nan")
    if corruption == "period":
        w["control_period_s"] = 1 / 60
    with pytest.raises(ValueError):
        start(b, w, "command", "execution")
    assert not calls


def test_sequence_status_and_stop_use_same_runtime_command():
    b = IsaacBackend(SimpleNamespace(robot_id="r1"))
    b._connected = True
    calls = []
    b.observations = SimpleNamespace(generation=3, state=lambda: None)

    def http(method, path, body):
        calls.append((method, path))
        return {
            "scene_generation": 3,
            "status": "cancelled" if method == "POST" else "succeeded",
            "progress": 1.0,
            "details": {"executed_samples": 3, "total_samples": 3},
        }

    b.http = SimpleNamespace(
        _path=lambda *p: "/".join(p),
        _json=http,
        profile={"capabilities": {"control_period_s": 1 / 30}},
    )
    cid = PREFIX + "example"
    assert b.get_command(cid).status == "succeeded"
    assert b.stop_command(cid, "user").status == "cancelled"
    assert calls == [("GET", f"commands/{cid}"), ("POST", f"commands/{cid}/stop")]
