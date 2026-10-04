"""No simulator: proposals must pass the entire path and restore checker state."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from galaxea_r1pro.galaxea_r1pro_0.utils.curobo_ik_filter import (
    filter_ready_path,
    fresh_collision_results,
)

torch = pytest.importorskip("torch")


class Cost:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def enable_cost(self):
        self.enabled = True

    def disable_cost(self):
        self.enabled = False


def fixture(solutions, *, fail=False):
    costs = [Cost(), Cost(), Cost(), Cost(False)]
    rollout = NS(
        **dict(
            zip(
                (
                    "primitive_collision_constraint",
                    "robot_self_collision_constraint",
                    "primitive_collision_cost",
                    "robot_self_collision_cost",
                ),
                costs,
            )
        )
    )

    def solve(goal, **kwargs):
        assert kwargs["return_seeds"] == 8
        assert not any(c.enabled for c in costs)
        if fail:
            raise RuntimeError("solver failed")
        return NS(
            solution=torch.tensor(solutions, dtype=torch.float64),
            success=torch.ones(len(solutions), dtype=torch.bool),
        )

    samples = []

    def check(q, **kwargs):
        assert all(c.enabled for c in costs[:3])
        samples.extend(q[:, 0].tolist())
        # Positive path has a collision in its middle, but a clear endpoint.
        bad = (q[..., 0] > 0.03) & (q[..., 0] < 0.07)
        return NS(feasible=~bad)

    mg = NS(
        kinematics=NS(joint_names=["joint"]),
        ik_solver=NS(get_all_rollout_instances=lambda: [rollout, rollout], solve_single=solve),
        rollout_fn=NS(
            primitive_collision_constraint=Cost(),
            robot_self_collision_constraint=Cost(),
            rollout_constraint=check,
        ),
    )
    args = NS(to_device=lambda a: torch.tensor(a, dtype=torch.float64))
    start = NS(position=torch.tensor([[0.0]], dtype=torch.float64))
    request = dict(dt=1 / 30, motion_velocity=np.array([0.5]), motion_acceleration=np.array([1.2]))
    return mg, args, start, request, costs, samples


def test_rejects_collision_between_clear_endpoints_and_uses_next_ik():
    mg, args, start, request, costs, samples = fixture([[0.1], [-0.2]])
    positions, info = filter_ready_path(mg, args, start, object(), request)
    assert info["selected_candidate"] == 2
    assert any(0.03 < row[0] < 0.07 for row in samples)
    assert positions[0, 0] == 0 and positions[-1, 0] == -0.2
    assert np.max(np.abs(np.diff(positions, axis=0))) / request["dt"] <= 0.5 * 1.01
    assert [c.enabled for c in costs] == [True, True, True, False]


@pytest.mark.parametrize("solutions", [[[0.1]], []])
def test_no_accepted_path_requests_original_planner(solutions):
    mg, args, start, request, _, _ = fixture(solutions)
    positions, metrics = filter_ready_path(mg, args, start, object(), request)
    assert positions is None
    assert metrics["used"] == "curobo"
    assert metrics["fallback_reason"] == "no_collision_free_path"


def test_solver_error_restores_all_previously_enabled_costs():
    mg, args, start, request, costs, _ = fixture([[0.1]], fail=True)
    with pytest.raises(RuntimeError, match="solver failed"):
        filter_ready_path(mg, args, start, object(), request)
    assert [c.enabled for c in costs] == [True, True, True, False]


def test_self_collision_result_does_not_leak_between_equal_sized_batches():
    class Checker:
        def update_batch_size(self, spheres):
            if not hasattr(self, "_out_distance"):
                self._out_distance = torch.zeros(1)

        def forward(self, spheres):
            self.update_batch_size(spheres)
            if spheres:
                self._out_distance.fill_(1)
            return self._out_distance.clone()

    checker = Checker()
    original = checker.forward
    mg = NS(rollout_fn=NS(robot_self_collision_constraint=checker))
    with pytest.raises(RuntimeError):
        with fresh_collision_results(mg):
            assert checker.forward(True).item() == 1
            assert checker.forward(False).item() == 0
            raise RuntimeError("exit worker")
    assert checker.forward == original
