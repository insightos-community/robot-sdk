import sys
from types import SimpleNamespace as NS
import numpy as np
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_motion import CuroboPlanner


def test_owner_snapshots_without_constructing_motion_generator(monkeypatch):
    monkeypatch.setitem(sys.modules, "omnigibson", NS(sim=NS(floor_plane=None)))
    mesh = {
        "mesh": "target_mesh",
        "vertices_world": [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
        "faces": [[0, 1, 2]],
    }
    scene = {"objects": [{"name": "item", "colliders": [mesh]}]}
    base = np.eye(4)
    base[0, 3] = 1.0
    rt = NS(
        q=np.array([0.1, 0.2]),
        names=("arm", "torso"),
        arm_indices={"left": [0]},
        torso_indices=[1],
        max_velocity=np.array([2.0, 2.0]),
        lower=np.array([-3.0, -3.0]),
        upper=np.array([3.0, 3.0]),
        has_limits=np.array([True, True]),
        dt=1 / 30,
        robot=NS(name="robot", _ag_obj_in_hand={}),
        frame_pose=lambda _: base,
        scene_geometry=lambda: scene,
    )
    planner = CuroboPlanner(rt)
    monkeypatch.setattr(planner, "_configuration", lambda *a: {"kinematics": {"base_link": "root"}})
    goal = np.eye(4)
    goal[0, 3] = 2.0
    request = planner.prepare(
        "left", goal, contact_object_ref="item", approach_world=np.array([0.0, 0.0, -0.1])
    )
    rt.q[:] = 9.0
    mesh["vertices_world"][0][0] = 99.0
    np.testing.assert_allclose(request["q"], [0.1, 0.2])
    np.testing.assert_allclose(request["mesh_snapshot"][0]["vertices"][0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(request["target"]["position"], [1.0, 0.0, 0.0])
    assert request["contact_meshes"] == ["target_mesh"] and "generator" not in request
