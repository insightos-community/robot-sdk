from types import SimpleNamespace as NS
import numpy as np
import pytest
from galaxea_r1pro.galaxea_r1pro_0.utils import curobo_support as support
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_motion import MotionPlanningError

torch = pytest.importorskip("torch")


def scene(z=0):
    return [
        dict(
            name="can",
            colliders=[
                dict(vertices_world=[[-0.02, -0.02, z], [0.02, -0.02, z], [0, 0.02, z + 0.1]])
            ],
        ),
        dict(
            name="table",
            colliders=[
                dict(
                    link="/table",
                    mesh="top",
                    vertices_world=[[-1, -1, z], [1, -1, z], [0, 1, z]],
                    faces=[[0, 1, 2]],
                )
            ],
        ),
    ]


@pytest.mark.parametrize("z", [0, 0.75])
def test_support_requires_contact_and_transforms_geometry_in_same_frame(z):
    attachment = dict(link="held", pose=dict(position=[0, 0, z], orientation=[0, 0, 0, 1]))
    result = support.support_snapshot(scene(z), "can", {"/table"}, np.eye(4), attachment)
    assert result["meshes"] == ["top"] and result["surface_z"] == z
    assert result["vertices_eef"][:, 2].min() == 0
    with pytest.raises(MotionPlanningError):
        support.support_snapshot(scene(z), "can", set(), np.eye(4), attachment)


@pytest.mark.parametrize(
    "feasible,heights",
    [
        ([False, False], [0, 0.1]),
        ([False, True, False], [0, 0.1, 0]),
        ([False, True, True], [0, -0.001, 0.1]),
        ([False, True], [-0.01, 0.1]),
    ],
)
def test_rejects_failure_to_depart_recontact_and_downward_penetration(feasible, heights):
    with pytest.raises(MotionPlanningError):
        support.departure_metrics(feasible, heights, 0)


def test_accepts_departure_with_contact_only_at_start():
    result = support.departure_metrics([False, True, True], [-0.000006, 0.01, 0.1], 0)
    assert result == dict(first_full_clear_sample=1, extra_penetration_m=0)


@pytest.mark.parametrize("fail_stage", [None, 1, 2])
def test_pair_masks_restore_dynamic_attachment_and_support_even_on_failure(monkeypatch, fail_stage):
    original = torch.tensor([[1.0, 2.0, 3.0, 0.02]])

    class Kin:
        spheres = original.clone()

        def get_link_spheres(self, link):
            return self.spheres

        def update_link_spheres(self, link, value):
            self.spheres = value.clone()

        def disable_link_spheres(self, link):
            self.spheres[:, 3] = -100

    kin = Kin()
    world = {"enabled": True}
    calls = []

    def obstacle(**kw):
        world["enabled"] = kw["enable"]

    def check(*args):
        calls.append((bool((kin.spheres[:, 3] > 0).all()), world["enabled"]))
        if len(calls) == fail_stage:
            raise MotionPlanningError("collision_plan_failed", "collision")

    monkeypatch.setattr(support, "validate_contact_path", check)
    mg = NS(kinematics=NS(kinematics_config=kin), world_coll_checker=NS(enable_obstacle=obstacle))
    if fail_stage:
        with pytest.raises(MotionPlanningError):
            support.validate_support_pair(mg, None, [[0]], dict(link="held", meshes=["table"]))
    else:
        support.validate_support_pair(mg, None, [[0]], dict(link="held", meshes=["table"]))
        assert calls == [(False, True), (True, False)]
    assert torch.equal(kin.spheres, original) and world["enabled"]


def test_coincident_ground_geometry_is_included_but_wall_bottom_is_not():
    from copy import deepcopy

    rows = scene()
    floor = deepcopy(rows[1])
    floor["name"] = "native_floor"
    floor["colliders"][0].update(link="/ground", mesh="ground")
    wall = deepcopy(rows[1])
    wall["name"] = "wall"
    wall["colliders"][0].update(link="/wall", mesh="wall")
    wall["colliders"][0]["vertices_world"].append([0, 0, 2])
    rows.extend([floor, wall])
    result = support.support_snapshot(
        rows,
        "can",
        {"/table"},
        np.eye(4),
        dict(link="held", pose=dict(position=[0, 0, 0], orientation=[0, 0, 0, 1])),
    )
    assert result["meshes"] == ["ground", "top"]
