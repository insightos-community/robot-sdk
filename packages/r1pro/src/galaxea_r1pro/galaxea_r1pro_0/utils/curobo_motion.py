"""cuRobo adapter: prepare on the simulation thread, solve snapshots in one worker.

No simulation access is permitted in solve(). The native motion generator owns
robot geometry, collision checking and trajectory optimization; Semantic Runtime owns stepping.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .utils_common import matrix_transform


def initialize_planning_runtime():
    """Call before importing Isaac, which adds its bundled Warp to module paths."""
    import importlib
    import warp as wp

    for name in ("warp.context", "warp.torch", "warp.types", "warp.codegen"):
        importlib.import_module(name)
    root = Path(wp.__file__).resolve().parent
    for name in ("warp.context", "warp.torch", "warp.types", "warp.codegen"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise MotionPlanningError(
                "planning_runtime_mismatch",
                "Warp 模块来源冲突；使用 SEMANTIC_CUROBO=1 在加载 Isaac 前初始化 SDK 规划环境",
            )
    wp.init()


class MotionPlanningError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class CuroboPlanner:
    def __init__(self, runtime):
        self.runtime = runtime

    def _configuration(self, side, torso, joint_names=None):
        """Read fixed link-local geometry and update live joint state."""
        import yaml

        rt = self.runtime
        config = yaml.safe_load(Path(rt.robot.curobo_path["arm"]).read_text())["robot_cfg"]
        kin = config["kinematics"]
        active = (
            {rt.names.index(n) for n in joint_names}
            if joint_names is not None
            else set(rt.arm_indices[side]) | (set(rt.torso_indices) if torso else set())
        )
        for name in kin["cspace"]["joint_names"]:
            index = rt.names.index(name)
            if index in active:
                kin["lock_joints"].pop(name, None)
            else:
                kin["lock_joints"][name] = float(rt.q[index])
        kin.update(
            ee_link=rt.robot.eef_link_names[side],
            link_names=[],
            use_usd_kinematics=True,
            usd_path=str(rt.robot.usd_path),
        )
        kin["extra_collision_spheres"] = {
            name: (128 if side in name else 32) for name in kin["extra_collision_spheres"]
        }
        kin["cspace"]["retract_config"] = [
            float(rt.q[rt.names.index(n)]) for n in kin["cspace"]["joint_names"]
        ]
        from .fixed_spheres import collision_spheres

        fixed = collision_spheres()
        # cuRobo allocates attached-object slots from extra_collision_spheres.
        # Native YAML has no corresponding entries in collision_spheres.
        kin["collision_spheres"] = {
            name: fixed[name]
            for name in kin["collision_link_names"]
            if name not in kin["extra_collision_spheres"]
        }
        for hand in ("left", "right"):
            for finger in (1, 2):
                ignored = kin["self_collision_ignore"].setdefault(
                    f"{hand}_gripper_finger_link{finger}", []
                )
                if f"{hand}_realsense_link" not in ignored:
                    ignored.append(f"{hand}_realsense_link")
        return config

    def prepare(
        self,
        side,
        target_world,
        *,
        torso=True,
        approach_world=None,
        attached_object_ref=None,
        contact_object_ref=None,
        joint_goal=None,
        allow_support_contact=False,
    ):
        """Snapshot only; no MotionGen construction or warmup on the owner thread."""
        rt = self.runtime
        configuration = self._configuration(
            side, torso, list(joint_goal) if joint_goal is not None else None
        )
        base = rt.frame_pose(configuration["kinematics"]["base_link"])
        inverse = np.linalg.inv(base)
        scene = rt.scene_geometry()
        meshes = []
        by_object = {}
        for obj in scene["objects"]:
            if obj["name"] == rt.robot.name:
                continue
            names = []
            for mesh in obj["colliders"]:
                vertices = np.asarray(mesh["vertices_world"]) @ inverse[:3, :3].T + inverse[:3, 3]
                meshes.append(
                    dict(name=mesh["mesh"], vertices=vertices.tolist(), faces=mesh["faces"])
                )
                names.append(mesh["mesh"])
            by_object[obj["name"]] = names
        # Include the native global floor plane when present, in the same frame.
        import omnigibson as og

        if og.sim.floor_plane is not None:
            from curobo.util.usd_helper import get_mesh_attrs

            plane = get_mesh_attrs(og.sim.floor_plane.prim.GetChildren()[0], transform=inverse)
            # Normalize its pose into vertex coordinates while still on the owner.
            from scipy.spatial.transform import Rotation

            pose = plane.pose
            rotation = Rotation.from_quat(np.asarray(pose[3:])[[1, 2, 3, 0]]).as_matrix()
            vertices = np.asarray(plane.vertices) * np.asarray(
                plane.scale if plane.scale is not None else [1, 1, 1]
            )
            meshes.append(
                dict(
                    name=plane.name,
                    vertices=(vertices @ rotation.T + pose[:3]).tolist(),
                    faces=plane.faces,
                )
            )
        contact_meshes = by_object.get(contact_object_ref, []) if contact_object_ref else []
        if contact_object_ref and not contact_meshes:
            raise MotionPlanningError("contact_object_missing", contact_object_ref)
        objects = {
            hand: obj
            for hand, obj in getattr(rt.robot, "_ag_obj_in_hand", {}).items()
            if obj is not None
        }
        if attached_object_ref:
            obj = rt.scene.object_registry("name", attached_object_ref)
            if obj is None:
                raise MotionPlanningError("attached_object_missing", attached_object_ref)
            if rt.robot.grasping_mode == "assisted" and objects.get(side) is not obj:
                raise MotionPlanningError(
                    "attached_object_not_held", "原生持物身份与携物目标不一致"
                )
            objects[side] = obj
        attachments = []
        for hand, obj in objects.items():
            from .curobo_attachments import require_rigid_attachment

            require_rigid_attachment(obj)
            eef = rt.robot.eef_link_names[hand]
            # Every enabled collider from every link is already expressed in
            # the same planning frame above. The numerical worker merges the
            # complete list into EEF-local attachment spheres and disables all
            # corresponding world obstacles until detach.
            attachments.append(
                dict(
                    meshes=by_object[obj.name],
                    link=rt.robot.curobo_attached_object_link_names[eef],
                    pose=matrix_transform(inverse @ rt.frame_pose(eef)),
                )
            )
        support = None
        if allow_support_contact:
            from omnigibson.utils.usd_utils import RigidContactAPI
            from .curobo_support import support_snapshot

            obj = objects[side]
            paths = {link.prim_path for link in obj.links.values()}
            pairs = RigidContactAPI.get_contact_pairs(rt.scene.idx, paths, None, True)
            contacts = {str(b) for a, b in pairs if str(a) in paths}
            # This is support geometry, not a new gripper confirmation action.
            excluded = {rt.robot.name, *(o.name for h, o in objects.items() if h != side)}
            rows = [row for row in scene["objects"] if row["name"] not in excluded]
            if og.sim.floor_plane is not None:
                vertices_world = np.asarray(meshes[-1]["vertices"]) @ base[:3, :3].T + base[:3, 3]
                rows.append(
                    dict(
                        name="native_floor",
                        colliders=[
                            dict(
                                link=og.sim.floor_plane.prim_path,
                                mesh=plane.name,
                                vertices_world=vertices_world.tolist(),
                                faces=plane.faces,
                            )
                        ],
                    )
                )
            attachment = next(
                a
                for a in attachments
                if a["link"]
                == rt.robot.curobo_attached_object_link_names[rt.robot.eef_link_names[side]]
            )
            support = support_snapshot(rows, attached_object_ref, contacts, base, attachment)
        approach, goal_frame = approach_constraint(base, target_world, approach_world)
        expected = (
            {rt.names.index(n) for n in joint_goal}
            if joint_goal is not None
            else set(rt.arm_indices[side]) | (set(rt.torso_indices) if torso else set())
        )
        return dict(
            configuration=configuration,
            mesh_snapshot=meshes,
            attachments=attachments,
            support_departure=support,
            full_names=list(rt.names),
            q=rt.q.copy(),
            expected_indices=expected,
            target=matrix_transform(inverse @ target_world),
            joint_goal=joint_goal,
            side=side,
            torso=torso,
            dt=rt.dt,
            contact_meshes=contact_meshes,
            approach=approach,
            approach_in_goal_frame=goal_frame,
            full_max_velocity=rt.max_velocity.copy(),
            full_lower=rt.lower.copy(),
            full_upper=rt.upper.copy(),
            full_limited=rt.has_limits.copy(),
            torso_indices=set(rt.torso_indices),
        )

    @staticmethod
    def solve(request):
        """Numeric worker only: never read robot, USD, sensors or advance physics."""
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        if "configuration" in request:
            from .curobo_snapshot import build

            request = build(request)
        generator, start, goal = (request[k] for k in ("generator", "start", "goal"))
        mg = generator.mg["default"]
        begin = time.monotonic()
        pregrasp_metrics = {}
        support_metrics = {}
        cfg = MotionGenPlanConfig(
            max_attempts=5,
            timeout=5.0,
            enable_graph=False,
            enable_graph_attempt=None,
            ik_fail_return=5,
        )
        try:
            from .curobo_ik_filter import fresh_collision_results

            with fresh_collision_results(mg):
                if request.get("support_departure") is not None:
                    from .curobo_support import filter_carry_path

                    positions, support_metrics = filter_carry_path(
                        mg, generator.tensor_args, start, goal, request
                    )
                elif request.get("joint_goal") is not None:
                    from curobo.types.state import JointState
                    from .curobo_path import retime, validate_contact_path

                    names = mg.kinematics.joint_names
                    goal = JointState.from_position(
                        generator.tensor_args.to_device(
                            [[request["joint_goal"][n] for n in names]]
                        ),
                        joint_names=names,
                    )
                    result = mg.plan_single_js(start, goal, cfg)
                    if not bool(result.success.any().item()):
                        raise MotionPlanningError("collision_plan_failed", str(result.status))
                    path = result.get_interpolated_plan().get_ordered_joint_state(names)
                    positions = retime(
                        path.position.detach().cpu().numpy().astype(float),
                        request["dt"],
                        request["motion_velocity"],
                        request["motion_acceleration"],
                    )
                    validate_contact_path(mg, generator.tensor_args, positions, [], [])
                    if (
                        np.max(np.abs(positions[-1] - [request["joint_goal"][n] for n in names]))
                        > 0.01
                    ):
                        raise MotionPlanningError(
                            "joint_goal_not_reached", "规划终点未达到关节目标"
                        )
                elif request.get("approach") is not None:
                    from curobo.types.math import Pose

                    offset = request["approach"]
                    weights = [1.0] * 6
                    weights[3 + int(np.argmax(np.abs(offset)))] = 0.0
                    from curobo.rollout.cost.pose_cost import PoseCostMetric

                    single_goal = Pose(position=goal.position[:1], quaternion=goal.quaternion[:1])
                    delta = Pose.from_list(
                        [*offset.tolist(), 1.0, 0.0, 0.0, 0.0], tensor_args=generator.tensor_args
                    )
                    ready_goal = (
                        single_goal.multiply(delta)
                        if request["approach_in_goal_frame"]
                        else delta.multiply(single_goal)
                    )
                    ready_p = None
                    if request.get("pregrasp_planner", "curobo") == "ik_filter":
                        from .curobo_ik_filter import filter_ready_path
                        from curobo.types.state import JointState

                        ready_p, pregrasp_metrics = filter_ready_path(
                            mg, generator.tensor_args, start[:1], ready_goal, request
                        )
                        if ready_p is not None:
                            ready_end = JointState.from_position(
                                generator.tensor_args.to_device(ready_p[-1:]),
                                joint_names=mg.kinematics.joint_names,
                            )
                    if ready_p is None:
                        ready = mg.plan_single(start[:1], ready_goal, cfg.clone())
                        if not bool(ready.success.all().item()):
                            raise MotionPlanningError("collision_plan_failed", str(ready.status))
                        ready_end = ready.optimized_plan[-1].unsqueeze(0)
                    fingers = (
                        [f"{request['side']}_gripper_finger_link{i}" for i in (1, 2)]
                        if request["contact_meshes"]
                        else []
                    )
                    final_cfg = cfg.clone()
                    final_cfg.pose_cost_metric = PoseCostMetric(
                        hold_partial_pose=True,
                        hold_vec_weight=generator.tensor_args.to_device(weights),
                        project_to_goal_frame=request["approach_in_goal_frame"],
                    )
                    try:
                        mg.toggle_link_collision(fingers, False)
                        final = mg.plan_single(ready_end, single_goal, final_cfg)
                    finally:
                        mg.toggle_link_collision(fingers, True)
                    if not bool(final.success.all().item()):
                        raise MotionPlanningError("collision_plan_failed", str(final.status))
                    from .curobo_path import retime, validate_contact_path

                    if ready_p is None:
                        ready_p = (
                            ready.get_interpolated_plan()
                            .get_ordered_joint_state(mg.kinematics.joint_names)
                            .position.detach()
                            .cpu()
                            .numpy()
                        )
                        ready_p = retime(
                            ready_p,
                            request["dt"],
                            request["motion_velocity"],
                            request["motion_acceleration"],
                        )
                    final_p = (
                        final.get_interpolated_plan()
                        .get_ordered_joint_state(mg.kinematics.joint_names)
                        .position.detach()
                        .cpu()
                        .numpy()
                    )
                    final_p = retime(
                        final_p,
                        request["dt"],
                        request["motion_velocity"],
                        request["motion_acceleration"],
                    )
                    validate_contact_path(mg, generator.tensor_args, ready_p, [], [])
                    validate_contact_path(
                        mg, generator.tensor_args, final_p, fingers, request["contact_meshes"]
                    )
                    validate_contact_path(
                        mg,
                        generator.tensor_args,
                        np.vstack([ready_p[-1:], final_p[:2]]),
                        fingers,
                        request["contact_meshes"],
                    )
                    positions = np.vstack([ready_p, final_p[1:]])
                else:
                    result, success, paths = generator.plan_batch(start, goal, cfg)
                    if not bool(success.any().item()):
                        status = str(result.status)
                        code = (
                            "invalid_start_state"
                            if "INVALID_START" in status
                            else "planning_timeout"
                            if "TIMEOUT" in status
                            else "collision_plan_failed"
                        )
                        raise MotionPlanningError(code, status)
                    index = int(success.nonzero()[0].item())
                    path = paths[index].get_ordered_joint_state(mg.kinematics.joint_names)
                    from .curobo_path import retime, validate_contact_path

                    positions = retime(
                        path.position.detach().cpu().numpy().astype(float),
                        request["dt"],
                        request["motion_velocity"],
                        request["motion_acceleration"],
                    )
                    validate_contact_path(mg, generator.tensor_args, positions, [], [])
                positions = CuroboPlanner.validate_trajectory(positions, request)
                return dict(
                    positions=positions,
                    indices=request["indices"],
                    torso_used=request["torso"],
                    planning_wall_s=time.monotonic() - begin,
                    dt=request["dt"],
                    pregrasp_planning=pregrasp_metrics,
                    support_departure=support_metrics,
                )
        finally:
            # Pure cuRobo cleanup; no scene reads in native detach helper.
            generator._detach_objects_from_robot(request["attached"], "default")

    @staticmethod
    def validate_trajectory(positions, request):
        idx, limited = request["indices"], request["limited"]
        if (
            positions.ndim != 2
            or positions.shape[1] != len(idx)
            or len(positions) < 2
            or not np.isfinite(positions).all()
        ):
            raise MotionPlanningError("invalid_trajectory", "非有限或无效的 cuRobo 轨迹")
        if np.any(positions[:, limited] < request["lower"][limited] - 1e-6) or np.any(
            positions[:, limited] > request["upper"][limited] + 1e-6
        ):
            raise MotionPlanningError("joint_limit_violation", "cuRobo 轨迹超出模型关节限位")
        if np.max(np.abs(positions[0] - request["q"][idx])) > 0.02:
            raise MotionPlanningError("invalid_trajectory_start", "轨迹起点偏离采样关节位置")
        # Do not clamp trajectory coordinates: that would change the collision-checked path.
        # Reject incompatible time sampling rather than secretly speeding up execution.
        speed = np.abs(np.diff(positions, axis=0)) / request["dt"]
        if np.any(speed > request["max_velocity"][None, :] * 1.01 + 1e-6):
            raise MotionPlanningError("trajectory_velocity_limit", "轨迹超过当前关节速度上限")
        return positions


def approach_constraint(base, target_world, approach_world):
    """Use native single-axis constraints in either base or target coordinates."""
    if approach_world is None:
        return None, False
    value = np.asarray(approach_world, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-8:
        raise MotionPlanningError("invalid_approach", "接近偏移必须为有限非零三维向量")
    for rotation, goal_frame in ((base[:3, :3], False), (target_world[:3, :3], True)):
        offset = rotation.T @ value
        if np.linalg.norm(np.delete(offset, np.argmax(np.abs(offset)))) <= 1e-5:
            return offset, goal_frame
    raise MotionPlanningError(
        "unsupported_approach_direction", "接近方向须沿规划基坐标轴或末端局部轴"
    )
