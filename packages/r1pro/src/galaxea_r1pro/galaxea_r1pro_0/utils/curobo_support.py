"""Depart a measured horizontal support without a separate lift command."""

import time
import numpy as np
from scipy.spatial.transform import Rotation

from .curobo_motion import MotionPlanningError
from .curobo_path import retime, densify, validate_contact_path


def support_snapshot(objects, object_ref, contact_paths, base, attachment):
    """Use actual support contacts plus horizontal surface geometry, not labels."""
    held = next(o for o in objects if o["name"] == object_ref)
    vertices = np.concatenate([m["vertices_world"] for m in held["colliders"]])
    bottom = float(vertices[:, 2].min())
    low, high = vertices.min(0), vertices.max(0)
    probes = np.vstack([vertices[vertices[:, 2] <= bottom + 0.002, :2], (low[:2] + high[:2]) / 2])
    candidates = []
    for obj in objects:
        if obj["name"] == object_ref:
            continue
        for mesh in obj["colliders"]:
            contacted = any(
                path == mesh["link"] or path.startswith(mesh["link"] + "/")
                for path in contact_paths
            )
            points = np.asarray(mesh["vertices_world"])
            lo, hi = points.min(0), points.max(0)
            if np.any(hi[:2] < low[:2]) or np.any(lo[:2] > high[:2]):
                continue
            for face in mesh["faces"]:
                triangle = points[face]
                z = float(triangle[:, 2].mean())
                if (
                    np.ptp(triangle[:, 2]) > 0.001
                    or abs(z - bottom) > 0.015
                    or abs(hi[2] - z) > 0.002
                ):
                    continue
                a, b, c = triangle[:, :2]
                matrix = np.column_stack([b - a, c - a])
                if abs(np.linalg.det(matrix)) < 1e-10:
                    continue
                uv = np.linalg.solve(matrix, (probes - a).T).T
                if np.any((uv[:, 0] >= -1e-6) & (uv[:, 1] >= -1e-6) & (uv.sum(1) <= 1 + 1e-6)):
                    candidates.append((mesh["mesh"], z, contacted))
                    break
    contact_levels = [z for _, z, contacted in candidates if contacted]
    if not contact_levels or np.ptp(contact_levels) > 0.002:
        raise MotionPlanningError("support_departure_unavailable", "未确定同一水平支撑面")
    # A native ground plane and scene floor can represent the same surface.
    # Include only coincident top surfaces under the object; never wall bases.
    supports = [name for name, z, _ in candidates if abs(z - max(contact_levels)) <= 0.002]
    levels = [z for _, z, _ in candidates if abs(z - max(contact_levels)) <= 0.002]
    inverse = np.linalg.inv(base)
    body_vertices = vertices @ inverse[:3, :3].T + inverse[:3, 3]
    pose = attachment["pose"]
    eef_vertices = (
        Rotation.from_quat(pose["orientation"]).inv().apply(body_vertices - pose["position"])
    )
    return dict(
        meshes=sorted(set(supports)),
        link=attachment["link"],
        vertices_eef=eef_vertices,
        world_from_base=base.copy(),
        surface_z=max(levels),
    )


def validate_support_pair(mg, tensor_args, positions, support):
    """Complementary checks exempt exactly attachment/support, restoring live spheres."""
    kin = mg.kinematics.kinematics_config
    link = support["link"]
    saved = kin.get_link_spheres(link).clone()
    if not bool((saved[:, 3] > 0).any().item()):
        raise MotionPlanningError("attachment_failed", "持物碰撞模型为空")
    try:
        kin.disable_link_spheres(link)
        validate_contact_path(mg, tensor_args, positions, [], [])
        # enable_link_spheres would restore empty pre-attachment slots.
        kin.update_link_spheres(link, saved)
        for name in support["meshes"]:
            mg.world_coll_checker.enable_obstacle(enable=False, name=name)
        validate_contact_path(mg, tensor_args, positions, [], [])
    finally:
        kin.update_link_spheres(link, saved)
        for name in support["meshes"]:
            mg.world_coll_checker.enable_obstacle(enable=True, name=name)


def departure_metrics(feasible, heights, surface_z):
    """Reject inward motion or renewed contact after first full clearance."""
    clear = next((i for i, value in enumerate(feasible) if value), len(feasible))
    penetration = max(0.0, float(heights[0] - min(heights)))
    if (
        clear == len(feasible)
        or not all(feasible[clear:])
        or penetration > 0.0001
        or heights[0] < surface_z - 0.005
        or heights[-1] <= surface_z
    ):
        raise MotionPlanningError("collision_plan_failed", "携物路径未持续离开支撑面或向内压入")
    return dict(first_full_clear_sample=clear, extra_penetration_m=penetration)


def filter_carry_path(mg, tensor_args, start, goal, request):
    support = request["support_departure"]
    from .curobo_ik_filter import propose_ik

    candidates, metrics = propose_ik(mg, start, goal, request["motion_velocity"])
    metrics.update(
        strategy="support_departure", support_meshes=support["meshes"], checked_candidates=0
    )
    began = time.monotonic()
    q0 = start.position[0].detach().cpu().numpy()
    base = np.asarray(support["world_from_base"])
    vertices = np.asarray(support["vertices_eef"])
    for index, solution in enumerate(candidates):
        metrics["checked_candidates"] += 1
        path = retime(
            np.linspace(q0, solution, 32),
            request["dt"],
            request["motion_velocity"],
            request["motion_acceleration"],
        )
        try:
            validate_support_pair(mg, tensor_args, path, support)
            samples = densify(path)
            feasible, heights = [], []
            for first in range(0, len(samples), 32):
                q = tensor_args.to_device(samples[first : first + 32])
                feasible.extend(
                    mg.rollout_fn.rollout_constraint(q.unsqueeze(1), use_batch_env=False)
                    .feasible.flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                state = mg.kinematics.get_state(q)
                for position, quaternion in zip(
                    state.ee_position.detach().cpu().numpy(),
                    state.ee_quaternion.detach().cpu().numpy(),
                ):
                    body = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).apply(vertices) + position
                    heights.append(float((body @ base[:3, :3].T + base[:3, 3])[:, 2].min()))
            checks = departure_metrics(feasible, heights, support["surface_z"])
        except MotionPlanningError as error:
            if error.code != "collision_plan_failed":
                raise
            continue
        times = [0.0]
        for frame, (a, b) in enumerate(zip(path[:-1], path[1:])):
            steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(0.5))))
            times.extend((frame + t) * request["dt"] for t in np.linspace(0, 1, steps + 1)[1:])
        metrics.update(
            checks,
            selected_candidate=index + 1,
            check_s=time.monotonic() - began,
            full_collision_from_sim_s=times[checks["first_full_clear_sample"]],
        )
        return path, metrics
    raise MotionPlanningError("support_departure_unavailable", "未找到可直接离开支撑面的携物路径")
