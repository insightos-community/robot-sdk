from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from galaxea_r1pro.galaxea_r1pro_0.utils.fixed_spheres import collision_spheres
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_motion import CuroboPlanner


def test_planner_config_mutation_cannot_change_saved_geometry():
    expected = collision_spheres()
    first = collision_spheres()
    link = next(iter(first))
    first[link][0]["center"][0] += 10
    first[link].append({"radius": 3.0, "center": [0.0, 0.0, 0.0]})
    assert collision_spheres() == expected


@pytest.mark.parametrize("side,torso", [("left", True), ("right", True), ("right", False)])
def test_live_pose_changes_update_locks_without_reading_or_refitting_meshes(tmp_path, side, torso):
    fixed = collision_spheres()
    names = ["left_arm_joint1", "right_arm_joint1", "torso_joint1"]
    attachments = {f"attached_object_{s}_eef_link": [] for s in ("left", "right")}
    config = {
        "robot_cfg": {
            "kinematics": dict(
                collision_link_names=list(fixed) + list(attachments),
                collision_spheres={},  # Native YAML does not preallocate attached-object slots.
                extra_collision_spheres={n: 32 for n in attachments},
                cspace={"joint_names": names},
                lock_joints={},
                self_collision_ignore={},
            )
        }
    }
    path = tmp_path / "robot.yml"
    path.write_text(yaml.safe_dump(config))

    class Robot:
        curobo_path = {"arm": path}
        eef_link_names = {"left": "left_eef_link", "right": "right_eef_link"}
        usd_path = "fixed-model.usd"

        @property
        def links(self):
            raise AssertionError("Planning must not read live collision meshes")

    rt = SimpleNamespace(
        robot=Robot(),
        names=names,
        q=np.array([0.1, 0.2, 0.3]),
        arm_indices={"left": [0], "right": [1]},
        torso_indices=[2],
    )
    planner = CuroboPlanner(rt)
    a = planner._configuration(side, torso)["kinematics"]
    rt.q += 0.7
    b = planner._configuration(side, torso)["kinematics"]
    assert not set(attachments).intersection(a["collision_spheres"])
    assert set(attachments).issubset(a["extra_collision_spheres"])
    assert a["collision_spheres"] == b["collision_spheres"]
    assert {k: v for k, v in b["collision_spheres"].items() if k in fixed} == fixed
    other = "right_arm_joint1" if side == "left" else "left_arm_joint1"
    assert b["lock_joints"][other] - a["lock_joints"][other] == pytest.approx(0.7)
    # cuRobo normalization of one request cannot leak into the next request.
    b["collision_spheres"][next(iter(fixed))][0]["radius"] = 100
    assert (
        planner._configuration(side, torso)["kinematics"]["collision_spheres"]
        == a["collision_spheres"]
    )
