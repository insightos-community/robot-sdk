"""Fixed assemblies, planning-frame snapshots and attachment lifecycle."""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_attachments import require_rigid_attachment
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_motion import CuroboPlanner, MotionPlanningError


@pytest.fixture
def usd():
    return pytest.importorskip("pxr.Usd"), pytest.importorskip("pxr.UsdPhysics")


def assembly(usd, count=2):
    Usd, Physics = usd
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/held", "Xform")
    links = {}
    for i in range(count):
        path = f"/held/link{i}"
        prim = stage.DefinePrim(path, "Xform")
        Physics.RigidBodyAPI.Apply(prim)
        links[str(i)] = NS(prim_path=path, prim=prim)
    return stage, NS(name="held", prim=root, links=links)


def joint(usd, stage, a, b, *, kind="FixedJoint", enabled=True, name="joint"):
    value = getattr(usd[1], kind).Define(stage, "/held/" + name)
    value.CreateBody0Rel().SetTargets([a] if a else [])
    value.CreateBody1Rel().SetTargets([b] if b else [])
    value.CreateJointEnabledAttr(enabled)
    return value


def test_single_link_keeps_old_path_without_usd_access():
    require_rigid_attachment(NS(links={"base": object()}))


def test_fixed_chain_including_geometryless_link_is_supported(usd):
    stage, obj = assembly(usd, 3)
    joint(usd, stage, "/held/link1", "/held/link0", name="first")
    joint(usd, stage, "/held/link1", "/held/link2", name="second")
    require_rigid_attachment(obj)


@pytest.mark.parametrize("kind", ["RevoluteJoint", "PrismaticJoint", "SphericalJoint", "Joint"])
def test_movable_joints_are_rejected_even_when_stationary(usd, kind):
    stage, obj = assembly(usd)
    joint(usd, stage, "/held/link0", "/held/link1", kind=kind)
    with pytest.raises(MotionPlanningError, match="仅支持固定关节") as error:
        require_rigid_attachment(obj)
    assert error.value.code == "unsupported_attached_object"


@pytest.mark.parametrize("mode", ["missing", "disabled", "outside", "world", "self"])
def test_unconnected_or_external_assemblies_fail_closed(usd, mode):
    stage, obj = assembly(usd)
    if mode != "missing":
        b = {"outside": "/external", "world": None, "self": "/held/link0"}.get(mode, "/held/link1")
        joint(usd, stage, "/held/link0", b, enabled=mode != "disabled")
    with pytest.raises(MotionPlanningError):
        require_rigid_attachment(obj)


def planner(monkeypatch, obj):
    monkeypatch.setitem(sys.modules, "omnigibson", NS(sim=NS(floor_plane=None)))
    meshes = [
        dict(
            mesh=f"/held/link{i}/mesh",
            link=f"/held/link{i}",
            vertices_world=[[2.0, float(i), 0.0], [3.0, float(i), 0.0], [2.0, float(i), 1.0]],
            faces=[[0, 1, 2]],
        )
        for i in range(len(obj.links))
    ]
    scene = {"objects": [dict(name="held", colliders=meshes)]}
    # Nonidentity rotation and translation catches wrong per-link frame handling.
    base = np.array(
        [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    eef = np.eye(4)
    eef[:3, 3] = [2.0, 3.0, 1.0]
    robot = NS(
        name="robot",
        grasping_mode="assisted",
        _ag_obj_in_hand={"left": obj},
        eef_link_names={"left": "left_eef"},
        curobo_attached_object_link_names={"left_eef": "held_link"},
    )
    rt = NS(
        robot=robot,
        names=["arm"],
        q=np.array([0.0]),
        arm_indices={"left": [0]},
        torso_indices=[],
        max_velocity=np.array([1.0]),
        lower=np.array([-3.0]),
        upper=np.array([3.0]),
        has_limits=np.array([True]),
        dt=1 / 30,
        scene=NS(object_registry=lambda *a: obj),
        scene_geometry=lambda: scene,
        frame_pose=lambda name: base if name == "base" else eef,
    )
    p = CuroboPlanner(rt)
    monkeypatch.setattr(p, "_configuration", lambda *a: {"kinematics": {"base_link": "base"}})
    return p, rt, meshes, base, eef


@pytest.mark.parametrize("count", [1, 2])
def test_all_colliders_share_frame_attach_together_and_release_returns_to_world(
    monkeypatch, usd, count
):
    stage, obj = assembly(usd, count)
    if count == 2:
        joint(usd, stage, "/held/link0", "/held/link1")
    p, rt, meshes, base, eef = planner(monkeypatch, obj)
    snapshot = p.prepare("left", np.eye(4), attached_object_ref="held")
    assert snapshot["attachments"][0]["meshes"] == [m["mesh"] for m in meshes]
    inv = np.linalg.inv(base)
    for original, copied in zip(meshes, snapshot["mesh_snapshot"]):
        expected = np.asarray(original["vertices_world"]) @ inv[:3, :3].T + inv[:3, 3]
        np.testing.assert_allclose(copied["vertices"], expected)
        assert copied["faces"] == original["faces"]
    np.testing.assert_allclose(snapshot["attachments"][0]["pose"]["position"], (inv @ eef)[:3, 3])
    # Detaching in the scene removes the carried representation on next prepare;
    # all meshes stay in the world snapshot and are not silently filtered out.
    rt.robot._ag_obj_in_hand = {}
    released = p.prepare("left", np.eye(4))
    assert released["attachments"] == []
    assert released["mesh_snapshot"] == snapshot["mesh_snapshot"]
