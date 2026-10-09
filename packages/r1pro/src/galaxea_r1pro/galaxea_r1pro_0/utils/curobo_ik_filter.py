# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate collision-blind IK proposals; accept only fully checked paths."""

from contextlib import contextmanager
import time

import numpy as np

from .curobo_path import retime, validate_contact_path


@contextmanager
def fresh_collision_results(mg):
    """The slow cuRobo self-collision kernel only writes positive distances.

    Its reusable output must be cleared before each independent constraint
    evaluation. This changes neither collision geometry nor enabled checks.
    Only the worker-owned validation rollout is wrapped, not optimizer costs.
    """
    checker = mg.rollout_fn.robot_self_collision_constraint
    original = checker.forward

    def forward(spheres):
        checker.update_batch_size(spheres)
        checker._out_distance.zero_()
        return original(spheres)

    checker.forward = forward
    try:
        yield
    finally:
        checker.forward = original


def propose_ik(mg, start, goal, velocity):
    """Generate online IK proposals and restore all original collision cost states."""

    started = time.monotonic()
    disabled = []
    try:
        for rollout in mg.ik_solver.get_all_rollout_instances():
            for name in (
                "primitive_collision_constraint",
                "robot_self_collision_constraint",
                "primitive_collision_cost",
                "robot_self_collision_cost",
            ):
                cost = getattr(rollout, name, None)
                if cost is not None and cost.enabled:
                    disabled.append(cost)
                    cost.disable_cost()
        result = mg.ik_solver.solve_single(
            goal, retract_config=start.position.clone(), return_seeds=8, num_seeds=128
        )
        solutions = result.solution.reshape(-1, len(mg.kinematics.joint_names))
        solutions = solutions[result.success.reshape(-1)].detach().cpu().numpy()
    finally:
        for cost in disabled:
            cost.enable_cost()

    metrics = dict(
        requested="ik_filter",
        used="ik_filter",
        ik_s=time.monotonic() - started,
        ik_solutions=len(solutions),
        checked_candidates=0,
        selected_candidate=None,
    )
    q0 = start.position[0].detach().cpu().numpy()
    candidates = []
    for solution in sorted(
        (s for s in solutions if np.isfinite(s).all()),
        key=lambda s: float(np.max(np.abs(s - q0) / velocity)),
    ):
        if all(np.max(np.abs(solution - other)) > np.deg2rad(1) for other in candidates):
            candidates.append(solution)
    metrics["unique_candidates"] = len(candidates)
    return candidates, metrics


def filter_ready_path(mg, tensor_args, start, goal, request):
    from .curobo_motion import MotionPlanningError

    candidates, metrics = propose_ik(mg, start, goal, request["motion_velocity"])
    q0 = start.position[0].detach().cpu().numpy()
    velocity = request["motion_velocity"]
    started = time.monotonic()
    for index, solution in enumerate(candidates):
        metrics["checked_candidates"] += 1
        positions = retime(
            np.linspace(q0, solution, 32), request["dt"], velocity, request["motion_acceleration"]
        )
        try:
            # Full world and self collision, including both hands and attachments.
            # No target-contact exceptions apply to the pregrasp segment.
            validate_contact_path(mg, tensor_args, positions, [], [])
        except MotionPlanningError as error:
            if error.code != "collision_plan_failed":
                raise
            continue
        metrics.update(
            used="ik_filter", selected_candidate=index + 1, check_s=time.monotonic() - started
        )
        return positions, metrics
    metrics.update(
        used="curobo", check_s=time.monotonic() - started, fallback_reason="no_collision_free_path"
    )
    return None, metrics
