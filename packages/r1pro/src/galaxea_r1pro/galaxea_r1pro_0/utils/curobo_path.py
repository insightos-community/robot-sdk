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

"""Numerical path validation and timing; no simulator access."""

import numpy as np
from scipy.interpolate import PchipInterpolator


def retime(positions, dt, velocity, acceleration):
    p = np.asarray(positions, dtype=float)
    v = np.broadcast_to(np.asarray(velocity, dtype=float), (p.shape[1],))
    a = np.broadcast_to(np.asarray(acceleration, dtype=float), (p.shape[1],))
    if len(p) < 2 or dt <= 0 or not np.isfinite(p).all() or np.any(v <= 0) or np.any(a <= 0):
        raise ValueError("Invalid trajectory or timing limits")
    # Endpoint duplicates impose zero endpoint derivatives without joint overshoot.
    p = np.vstack([p[0], p, p[-1]])
    knots = np.arange(len(p)) * dt
    curve = PchipInterpolator(knots, p, axis=0)
    probes = np.unique(
        np.r_[knots, np.nextafter(knots[1:], -np.inf), np.linspace(0, knots[-1], len(p) * 8)]
    )
    scale = max(
        1.0,
        float(np.max(np.abs(curve(probes, 1)) / v)),
        float(np.sqrt(np.max(np.abs(curve(probes, 2)) / a))),
    )
    count = int(np.ceil(knots[-1] * scale / dt)) + 1
    timed = curve(np.linspace(0, knots[-1], count))
    timed[0], timed[-1] = p[0], p[-1]
    return timed


def densify(positions, max_joint_step=np.deg2rad(0.5)):
    result = [positions[0]]
    for start, end in zip(positions[:-1], positions[1:]):
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_joint_step)))
        result.extend(start + (end - start) * t for t in np.linspace(0, 1, steps + 1)[1:])
    return np.asarray(result)


def validate_contact_path(mg, tensor_args, positions, finger_links, target_meshes):
    """Only selected fingers may touch the named target: validate complementary masks."""
    from .curobo_motion import MotionPlanningError

    samples = densify(positions)

    def check():
        mg.rollout_fn.primitive_collision_constraint.enable_cost()
        mg.rollout_fn.robot_self_collision_constraint.enable_cost()
        for first in range(0, len(samples), 32):
            q = tensor_args.to_device(samples[first : first + 32]).unsqueeze(1)
            if not bool(
                mg.rollout_fn.rollout_constraint(q, use_batch_env=False).feasible.all().item()
            ):
                raise MotionPlanningError(
                    "collision_plan_failed", "接触阶段路径未通过选择性碰撞复核"
                )

    if not finger_links:
        check()
        return
    try:
        # All objects, including the target, remain obstacles to the palm and arm.
        mg.toggle_link_collision(finger_links, False)
        check()
        mg.toggle_link_collision(finger_links, True)
        # Restore fingers and check self-collision and every non-target obstacle.
        for name in target_meshes:
            mg.world_coll_checker.enable_obstacle(enable=False, name=name)
        check()
    finally:
        mg.toggle_link_collision(finger_links, True)
        for name in target_meshes:
            mg.world_coll_checker.enable_obstacle(enable=True, name=name)
        mg.rollout_fn.primitive_collision_constraint.enable_cost()
        mg.rollout_fn.robot_self_collision_constraint.enable_cost()
