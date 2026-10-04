import numpy as np
import pytest
from types import SimpleNamespace as NS
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_path import retime, validate_contact_path
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_motion import MotionPlanningError


def test_retiming_limits_actual_samples_and_preserves_endpoints_and_bounds():
    u = np.linspace(0, 1, 31)
    p = np.column_stack([1.5 * (10 * u**3 - 15 * u**4 + 6 * u**5), -0.8 * u * u])
    timed = retime(p, 1 / 30, [0.4, 0.5], [1.2, 1.2])
    np.testing.assert_allclose(timed[[0, -1]], p[[0, -1]])
    assert np.all(np.abs(np.diff(timed, axis=0)) * 30 <= np.array([0.4, 0.5]) * 1.01)
    assert np.all(np.abs(np.diff(timed, axis=0, n=2)) * 900 <= 1.2 * 1.01)
    assert np.all(timed >= p.min(0) - 1e-9) and np.all(timed <= p.max(0) + 1e-9)
    assert len(timed) > len(p)


def test_contact_masks_cover_self_and_other_objects_and_restore_after_failure():
    fingers_on = True
    target_on = True
    checks = []

    def toggle(links, enabled):
        nonlocal fingers_on
        fingers_on = enabled

    def obstacle(*, name, enable):
        nonlocal target_on
        assert name == "target"
        target_on = enable

    def constraint(*args, **kwargs):
        checks.append((fingers_on, target_on))
        return NS(feasible=np.array([not fingers_on]))

    # numpy booleans expose .all().item() like the native tensor result.
    cost = NS(enable_cost=lambda: None)
    mg = NS(
        toggle_link_collision=toggle,
        world_coll_checker=NS(enable_obstacle=obstacle),
        rollout_fn=NS(
            primitive_collision_constraint=cost,
            robot_self_collision_constraint=cost,
            rollout_constraint=constraint,
        ),
    )
    tensor = NS(to_device=lambda x: NS(unsqueeze=lambda _: x))
    with pytest.raises(MotionPlanningError):
        validate_contact_path(mg, tensor, np.array([[0], [0.01]]), ["finger"], ["target"])
    assert checks == [(False, True), (True, False)]
    assert fingers_on and target_on
