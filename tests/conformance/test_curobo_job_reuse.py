import copy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_job import CuroboJobs
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_motion import MotionPlanningError


def snapshot():
    return dict(
        full_names=["arm", "other_arm", "finger"],
        q=np.zeros(3),
        expected_indices={0},
        configuration={
            "kinematics": {
                "cspace": {"retract_config": [0.0]},
                "lock_joints": {"other_arm": 0.0, "finger": 0.0},
                "collision_spheres": {"finger": [{"radius": 0.01, "center": [0.0, 0.0, 0.0]}]},
            }
        },
        mesh_snapshot=[{"name": "item", "vertices": [[0.2, 0.3, 0.4]], "faces": [[0, 0, 0]]}],
        target={"position": [0.2, 0.3, 0.4], "orientation": [0.0, 0.0, 0.0, 1.0]},
        side="left",
        torso=False,
        attachments=[],
        contact_meshes=["item"],
        approach=np.array([0.0, 0.0, -0.1]),
        dt=1 / 30,
    )


def setup():
    rt = NS(q=np.zeros(3), dt=1 / 30)
    snap = snapshot()

    class Planner:
        calls = 0
        prepares = 0

        def prepare(self, *a, **kw):
            self.prepares += 1
            out = copy.deepcopy(snap)
            out["q"] = rt.q.copy()
            return out

        def solve(self, request):
            self.calls += 1
            return dict(
                indices=np.array([0]),
                positions=np.array([[0.0], [0.005], [0.01]]),
                dt=1 / 30,
                torso_used=False,
                planning_wall_s=100.0,
            )

    planner = Planner()
    jobs = CuroboJobs(rt, lambda _: planner)
    return rt, snap, planner, jobs


def command(plan_only):
    return dict(
        side="left",
        eef_goal=np.eye(4),
        use_torso=False,
        plan_only=plan_only,
        planning_intent=dict(
            side="left",
            frame_id="body",
            position=(0.2, 0.3, 0.4),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            use_torso=False,
            approach_offset=(0.0, 0.0, -0.1),
            attached_object_ref=None,
            contact_object_ref="item",
        ),
    )


def planned(jobs):
    execution = command(True)
    targets = np.zeros(3)
    assert jobs.tick(execution, targets) == "planning"
    execution["planning_future"].result(timeout=3)
    assert jobs.tick(execution, targets) == "planned"
    np.testing.assert_array_equal(targets, np.zeros(3))
    return execution


def test_preplan_is_consumed_once_and_preserves_every_waypoint():
    rt, snap, planner, jobs = setup()
    try:
        pre = planned(jobs)
        ex = command(False)
        targets = np.zeros(3)
        seen = []
        for i in range(3):
            phase = jobs.tick(ex, targets)
            seen.append(targets[0])
        assert phase == "settling" and planner.calls == 1
        np.testing.assert_array_equal(seen, pre["trajectory"][:, 0])
        assert jobs.summary(ex)["plan_reused"] and jobs.summary(ex)["planning_wall_s"] == 0
        assert jobs.summary(ex)["source_planning_wall_s"] == 100
        assert jobs.prepared_plan is None
        another = command(False)
        assert jobs.tick(another, targets) == "planning"
        another["planning_future"].result(timeout=3)
        assert planner.calls == 2
    finally:
        jobs.close()


@pytest.mark.parametrize(
    "change", ["goal", "side", "period", "approach", "attachment", "contact", "torso", "frame"]
)
def test_different_command_requires_new_plan(change):
    rt, snap, planner, jobs = setup()
    try:
        planned(jobs)
        ex = command(False)
        intent = ex["planning_intent"]
        if change == "goal":
            intent["position"] = (0.21, 0.3, 0.4)
        elif change == "side":
            intent["side"] = "right"
            ex["side"] = "right"
        elif change == "period":
            rt.dt = 1 / 60
        elif change == "approach":
            intent["approach_offset"] = None
        elif change == "attachment":
            intent["attached_object_ref"] = "held"
        elif change == "contact":
            intent["contact_object_ref"] = "different-item"
        elif change == "torso":
            intent["use_torso"] = True
        elif change == "frame":
            intent["frame_id"] = "world"
        assert jobs.tick(ex, np.zeros(3)) == "planning"
        ex["planning_future"].result(timeout=3)
        assert planner.calls == 2 and not ex.get("plan_reused", False)
        assert ex["plan_reuse_reason"] == "different_command"
    finally:
        jobs.close()


def test_scene_and_locked_joint_changes_do_not_resample_or_replan():
    rt, snap, planner, jobs = setup()
    try:
        pre = planned(jobs)
        # The actual reported static jitter exceeded the former 1e-5 threshold.
        rt.q[1] = 0.000016897916793823242
        snap["mesh_snapshot"][0]["vertices"][0][0] += 0.001
        snap["configuration"]["kinematics"]["lock_joints"]["other_arm"] = rt.q[1]
        snap["configuration"]["kinematics"]["cspace"]["retract_config"] = [0.001]

        def forbidden(*args, **kwargs):
            raise AssertionError("reuse must not prepare a new scene snapshot")

        planner.prepare = forbidden
        ex = command(False)
        ex["eef_goal"][0, 3] += 0.00003
        target = np.zeros(3)
        actual = []
        for _ in range(3):
            jobs.tick(ex, target)
            actual.append(target[0])
        np.testing.assert_array_equal(actual, pre["trajectory"][:, 0])
        assert planner.calls == 1 and planner.prepares == 1 and ex["plan_reused"]
        assert ex["plan_reuse_reason"] == "prepared_plan"
        assert "planning_request" not in pre
    finally:
        jobs.close()


def test_unaccepted_or_failed_result_is_never_cached():
    rt, snap, planner, jobs = setup()
    try:
        ex = command(True)
        jobs.tick(ex, np.zeros(3))
        ex["planning_future"].result(timeout=3)
        assert jobs.prepared_plan is None
        rt.q[0] = 0.1
        with pytest.raises(MotionPlanningError, match="起点"):
            jobs.tick(ex, np.zeros(3))
        assert jobs.prepared_plan is None
    finally:
        jobs.close()


def test_failed_new_candidate_discards_older_preplan():
    rt, snap, planner, jobs = setup()
    try:
        planned(jobs)

        def failed(_):
            raise MotionPlanningError("collision_plan_failed", "blocked")

        planner.solve = failed
        ex = command(True)
        assert jobs.tick(ex, np.zeros(3)) == "planning"
        with pytest.raises(MotionPlanningError):
            ex["planning_future"].result(timeout=3)
        with pytest.raises(MotionPlanningError):
            jobs.tick(ex, np.zeros(3))
        assert jobs.prepared_plan is None
    finally:
        jobs.close()


def test_reuse_checks_actual_first_waypoint_not_only_snapshot():
    rt, snap, planner, jobs = setup()
    try:
        planned(jobs)
        jobs.prepared_plan[1]["positions"][0, 0] = -0.019
        rt.q[0] = 0.019
        ex = command(False)
        with pytest.raises(MotionPlanningError, match="起点"):
            jobs.tick(ex, np.zeros(3))
        assert planner.calls == 1 and ex["plan_reuse_reason"] == "start_mismatch"
        assert jobs.prepared_plan is None
    finally:
        jobs.close()


def test_solver_normalization_does_not_mutate_cached_request():
    rt, snap, planner, jobs = setup()
    original = planner.solve

    def normalize(request):
        result = original(request)
        request["configuration"]["kinematics"]["collision_spheres"] = "normalized"
        return result

    planner.solve = normalize
    try:
        planned(jobs)
        ex = command(False)
        assert jobs.tick(ex, np.zeros(3)) == "executing"
        assert planner.calls == 1 and ex["plan_reused"]
    finally:
        jobs.close()
