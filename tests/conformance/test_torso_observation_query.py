from galaxea_r1pro.galaxea_r1pro_0.torso_control import TorsoControl
from galaxea_r1pro.galaxea_r1pro_0.backends.isaac import IsaacBackend
from galaxea_r1pro import RobotProfile, BackendKind


def test_observation_request_uses_read_query_and_keeps_solver_feedback():
    backend = IsaacBackend(RobotProfile("robot_r1", BackendKind.ISAAC, "behavior"))
    calls = []

    def request(method, *parts, body=None):
        calls.append((method, parts, body))
        return dict(
            command_id="",
            status="failed",
            output={},
            error_code="observation_pose_not_found",
            error_message="no feasible view",
            feedback=dict(
                command_id="",
                status="failed",
                progress=1.0,
                phase="computed",
                metrics={},
                timestamp="2026-09-20T00:00:00+00:00",
            ),
        )

    backend._call = request
    out = TorsoControl(backend).compute_observation_pose(
        object_position=(1, 2, 3),
        standing_pose=(0, 0, 0, 0.5),
        object_bounds=((0, 1, 2), (2, 3, 4)),
    )
    method, parts, body = calls[0]
    assert method == "POST" and parts == ("query",)
    assert body["target"]["object_position"] == [1, 2, 3]
    assert body["target"]["standing_pose"] == [0, 0, 0, 0.5]
    assert body["target"]["object_bounds"] == [[0, 1, 2], [2, 3, 4]]
    assert out.status == "failed" and out.error_code == "observation_pose_not_found"
    assert len(calls) == 1
