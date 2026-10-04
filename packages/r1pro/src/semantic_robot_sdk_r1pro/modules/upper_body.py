"""躯干和双臂的类型化运动入口。"""

import math

from semantic_robot_sdk_core import PlanningError, Pose


class UpperBodyModule:
    def __init__(self, robot_id, backend, kinematics, motion):
        self.robot_id = robot_id
        self.backend = backend
        self.kinematics = kinematics
        self.motion = motion

    def plan_joints(self, goal, *, environment=None, speed_scale=1.0):
        return self.motion.plan_joints(
            robot_id=self.robot_id,
            state=self.backend.state(self.robot_id),
            goal=goal,
            environment=environment,
            speed_scale=speed_scale,
        )

    def move_joints(self, goal, *, command_id, environment=None, speed_scale=1.0):
        plan = self.plan_joints(goal, environment=environment, speed_scale=speed_scale)
        return self.backend.execute_plan(plan, command_id)

    def plan_end_effector(
        self,
        end_effector,
        target,
        *,
        state=None,
        environment=None,
        speed_scale=1.0,
        maximum_speed_mps=None,
        preserve_cartesian_path=False,
    ):
        state = state or self.backend.state(self.robot_id)
        start = state.end_effectors.get(end_effector)
        if start is None:
            raise PlanningError(f"Robot 状态缺少末端位姿：{end_effector}")
        if not preserve_cartesian_path:
            goal = self.kinematics.solve(
                robot_id=self.robot_id,
                end_effector=end_effector,
                target=target,
                state=state,
                environment=environment,
            )
            plan = self.motion.plan_joints(
                robot_id=self.robot_id,
                state=state,
                goal=goal,
                environment=environment,
                speed_scale=speed_scale,
            )
        else:
            if start.frame_id != target.frame_id:
                raise PlanningError("末端路径的起点和目标必须使用同一坐标系")
            # 狭窄接口中的单侧插入也必须保持工具姿态。只求终点IK再做关节
            # 插值会让开放压片在中途扫过箱体，低层接触停止虽然能避免继续
            # 推撞，却永远到不了真实侧面凹槽。这里复用通用笛卡尔路点与
            # 关节时间参数化，不引入任何周转箱、场景或仿真专用判断。
            working_state = state
            goals = []
            steps = _pose_steps(start, target)
            for index in range(1, steps + 1):
                waypoint = _interpolate_pose(start, target, index / steps)
                goal = self.kinematics.solve(
                    robot_id=self.robot_id,
                    end_effector=end_effector,
                    target=waypoint,
                    state=working_state,
                    environment=environment,
                )
                goals.append(goal)
                joints = dict(working_state.joint_positions)
                joints.update(goal)
                end_effectors = dict(working_state.end_effectors)
                end_effectors[end_effector] = waypoint
                working_state = working_state.model_copy(
                    update={"joint_positions": joints, "end_effectors": end_effectors}
                )
            # 分段路径可能先只用运动臂，接近工作区边界后再启用共享
            # 躯干和承载臂补偿。Motion Provider要求一条路径的关节集合一致，
            # 因此在时间参数化前补齐各段；未参与该段IK的关节沿用上一段值，
            # 不能回零，也不能伪造一次额外运动。
            goals = _complete_joint_waypoints(goals, state.joint_positions)
            plan = self.motion.plan_joint_waypoints(
                robot_id=self.robot_id,
                state=state,
                goals=goals,
                environment=environment,
                speed_scale=speed_scale,
            )
        limited_scale = _cartesian_speed_scale(
            {end_effector: start},
            {end_effector: target},
            plan.estimated_duration_s,
            maximum_speed_mps,
            speed_scale,
        )
        if limited_scale < speed_scale:
            plan = self.motion.plan_joints(
                robot_id=self.robot_id,
                state=state,
                goal=goal,
                environment=environment,
                speed_scale=limited_scale,
            )
        return plan.model_copy(
            update={
                "goal": {
                    "end_effector": end_effector,
                    **target.model_dump(mode="json"),
                },
                "planner": f"{self.kinematics.name}+{plan.planner}",
            }
        )

    def plan_end_effectors(
        self,
        targets,
        *,
        state=None,
        environment=None,
        speed_scale=1.0,
        maximum_speed_mps=None,
        preserve_cartesian_path=False,
        preserve_relative_geometry=False,
        fixed_end_effectors=(),
    ):
        state = state or self.backend.state(self.robot_id)
        if len(targets) < 2:
            raise ValueError("多末端规划至少需要两个目标")
        fixed_end_effectors = frozenset(fixed_end_effectors)
        requested_targets = dict(targets)
        anchor_fixed_targets = getattr(self.kinematics, "fixed_targets_from_state", None)
        if fixed_end_effectors and callable(anchor_fixed_targets):
            # Runtime返回的是受力后的真实末端位姿，IK使用的是刚性Robot模型；
            # 两者的微小形变偏差不能被规划器解释成一次新的承载侧校正动作。
            # 固定约束锚定到动作开始关节构型的FK位姿，真实接触是否仍在则由
            # Ability持续监控。它表达的是“承载侧不额外移动”，不是让刚性
            # 模型追赶接触形变后的不可实现精确Pose。
            targets = anchor_fixed_targets(
                targets,
                state=state,
                fixed_end_effectors=fixed_end_effectors,
            )
        current = {}
        for name in targets:
            current_pose = state.end_effectors.get(name)
            if current_pose is None:
                raise PlanningError(f"Robot 状态缺少末端位姿：{name}")
            if current_pose.frame_id != targets[name].frame_id:
                raise PlanningError("多末端路径的起点和目标必须使用同一坐标系")
            current[name] = targets[name] if name in fixed_end_effectors else current_pose

        def finalize_path(
            plan,
            *,
            joint_goal=None,
            waypoint_goals=None,
            allow_fixed_projection=True,
        ):
            effective_speed_scale = speed_scale
            limited_scale = _cartesian_speed_scale(
                current,
                targets,
                plan.estimated_duration_s,
                maximum_speed_mps,
                speed_scale,
            )
            if limited_scale < speed_scale:
                effective_speed_scale = limited_scale
                if waypoint_goals is not None:
                    plan = self.motion.plan_joint_waypoints(
                        robot_id=self.robot_id,
                        state=state,
                        goals=waypoint_goals,
                        environment=environment,
                        speed_scale=limited_scale,
                    )
                elif joint_goal is not None:
                    plan = self.motion.plan_joints(
                        robot_id=self.robot_id,
                        state=state,
                        goal=joint_goal,
                        environment=environment,
                        speed_scale=limited_scale,
                    )

            synchronized_projector = getattr(
                self.kinematics, "project_synchronized_end_effector_path", None
            )
            if (
                preserve_relative_geometry
                and not fixed_end_effectors
                and callable(synchronized_projector)
            ):
                projected = synchronized_projector(
                    plan,
                    state=state,
                    targets=targets,
                    environment=environment,
                )
                if projected is not plan:
                    # 双侧携物必须保留投影后的相对几何；固定一侧的接合路径
                    # 则继续使用下方已通过真实抓取验证的简化+Ruckig重定时。
                    # 两种路径不能共用同一策略，否则修复放置会反向劣化抓取。
                    retime = getattr(self.motion, "retime_geometry_preserving_path", None)
                    reaches_targets = getattr(
                        self.kinematics,
                        "path_reaches_synchronized_end_effector_targets",
                        None,
                    )
                    if not callable(retime) or not callable(reaches_targets):
                        raise PlanningError("同步末端轨迹缺少重定时或完成校验能力")
                    candidate = None
                    simplified_retime = getattr(self.motion, "retime_projected_path", None)
                    if callable(simplified_retime):
                        try:
                            simplified = simplified_retime(
                                projected,
                                state=state,
                                environment=environment,
                                speed_scale=effective_speed_scale,
                                waypoint_tolerance_rad=1e-3,
                            )
                        except PlanningError:
                            simplified = None
                        if simplified is not None and reaches_targets(
                            simplified, state=state, targets=targets
                        ):
                            # 同步IK会把一条平滑动作展开成大量数值采样点；直接按
                            # 最差采样统一拉长时间会让几十厘米放置耗时近一分钟。
                            # 原路径已经逐点同步求解，1 mrad 关节简化后继续做连续
                            # 碰撞检查并确认两个末端到达目标；不再用毫米级的中间
                            # FK误差把它退回数十秒的统一慢放轨迹。
                            candidate = simplified
                    if candidate is None:
                        candidate = retime(
                            projected,
                            state=state,
                            environment=environment,
                            speed_scale=effective_speed_scale,
                        )
                    if not reaches_targets(candidate, state=state, targets=targets):
                        raise PlanningError("同步末端重定时轨迹未到达目标")
                    plan = candidate

            projector = getattr(self.kinematics, "project_fixed_end_effector_path", None)
            if fixed_end_effectors and not allow_fixed_projection:
                keeps_fixed = getattr(self.kinematics, "path_keeps_fixed_end_effectors", None)
                if not callable(keeps_fixed) or not keeps_fixed(
                    plan,
                    state=state,
                    targets=targets,
                    fixed_end_effectors=fixed_end_effectors,
                ):
                    # 关节直达只在运动侧独立关节自然保持承载端时使用。
                    # 若需要共享关节补偿，应交给下面的同步约束路点；不能
                    # 先对密集时间采样逐点投影，把一次clearance放大到数分钟。
                    raise PlanningError("连续关节路径需要移动承载侧共享关节")
            if fixed_end_effectors and callable(projector):
                # 固定侧保持是每一种候选路径的完成条件，而不是选定路径后的
                # 附加修补。投影或重定时失败必须让规划器尝试下一种通用几何
                # 路径；否则一个端点可达但中途扭动承载侧的关节直达解，会在
                # 这里耗时数百秒后直接终止，Cartesian回退永远没有机会运行。
                projected = projector(
                    plan,
                    state=state,
                    targets=targets,
                    fixed_end_effectors=fixed_end_effectors,
                    environment=environment,
                )
                if projected is not plan:
                    retime = getattr(self.motion, "retime_projected_path", None)
                    keeps_fixed = getattr(self.kinematics, "path_keeps_fixed_end_effectors", None)
                    if not callable(retime) or not callable(keeps_fixed):
                        raise PlanningError("固定末端轨迹缺少重定时或完成校验能力")
                    # 先移除密集IK采样中的数值冗余，再用真实FK检查重定时
                    # 轨迹。简化结果若越过RobotDeployment声明的固定端容差，
                    # 只细化几何路点；不能通过放宽接触侧误差换取执行速度。
                    for waypoint_tolerance in (2e-4, 1e-4, 1e-5):
                        candidate = retime(
                            projected,
                            state=state,
                            environment=environment,
                            speed_scale=effective_speed_scale,
                            waypoint_tolerance_rad=waypoint_tolerance,
                        )
                        if keeps_fixed(
                            candidate,
                            state=state,
                            targets=targets,
                            fixed_end_effectors=fixed_end_effectors,
                        ):
                            projected = candidate
                            break
                    else:
                        raise PlanningError("固定末端重定时轨迹超出Robot Profile允许误差")
                plan = projected
            return plan

        collision_fallback = False
        elevated_detour = False
        constrained_path = bool(fixed_end_effectors)
        if (
            preserve_relative_geometry
            and not fixed_end_effectors
            and callable(
                getattr(
                    self.kinematics,
                    "project_synchronized_end_effector_path",
                    None,
                )
            )
        ):
            # 已形成双工具刚性耦合后，不能先求一个远端IK终点再靠关节直插值
            # 猜测中间姿态。Local Provider按连续笛卡尔几何生成少量路点，再对
            # 路点间轨迹做相对变换投影；Fake/远端Provider没有该能力时保持
            # 原有直接规划契约，不把本地实现细节扩散到公共SDK接口。
            preserve_cartesian_path = True
        if fixed_end_effectors and preserve_cartesian_path:
            # 调用方显式要求末端几何路径时，固定端约束不能把该请求降级为
            # “先试关节直达”。密集工位中直达轨迹的终点虽然可达，下钩却会
            # 在中途向外扫过邻箱；同步路点同时保持运动端路径与承载端锚点。
            guide_joint_goal = None
            try:
                guide_joint_goal = self.kinematics.solve_many(
                    robot_id=self.robot_id,
                    targets=targets,
                    state=state,
                    environment=environment,
                    fixed_end_effectors=fixed_end_effectors,
                )
            except PlanningError:
                # 终点多起点解只用于引导冗余分支；连续短路点本身仍可能
                # 从当前构型逐步可达，因此不能把引导失败当成动作失败。
                guide_joint_goal = None
            waypoint_goals = self._cartesian_end_effector_goals(
                targets,
                state=state,
                environment=environment,
                fixed_end_effectors=fixed_end_effectors,
                guide_joint_goal=guide_joint_goal,
            )
            plan = self.motion.plan_joint_waypoints(
                robot_id=self.robot_id,
                state=state,
                goals=waypoint_goals,
                environment=environment,
                speed_scale=speed_scale,
            )
            plan = finalize_path(plan, waypoint_goals=waypoint_goals)
        elif fixed_end_effectors:
            # clearance/transfer首先使用连续关节轨迹：当运动侧独立关节能够
            # 到达目标时，承载侧的躯干和机械臂根本不会进入轨迹，既能保持
            # 真实接触，也不会被笛卡尔直线困在中途的冗余IK死端。整条关节
            # 轨迹仍由Motion Provider逐点检查Robot和环境碰撞；只有它确实
            # 不可用时，才进入同步笛卡尔和升高绕行。
            try:
                goal = self.kinematics.solve_many(
                    robot_id=self.robot_id,
                    targets=targets,
                    state=state,
                    environment=environment,
                    fixed_end_effectors=fixed_end_effectors,
                )
                plan = self.motion.plan_joints(
                    robot_id=self.robot_id,
                    state=state,
                    goal=goal,
                    environment=environment,
                    speed_scale=speed_scale,
                )
                try:
                    plan = finalize_path(plan, joint_goal=goal, allow_fixed_projection=False)
                except PlanningError:
                    # 终点需要共享关节时，先沿已通过碰撞检查的关节路径
                    # 连续投影固定端。投影使用上一采样解作为唯一连续种子；
                    # 若Provider没有投影能力或不能稳定保持承载侧，则回退到
                    # 同步笛卡尔几何路径，不能把无约束直达当成成功。
                    if not callable(
                        getattr(
                            self.kinematics,
                            "project_fixed_end_effector_path",
                            None,
                        )
                    ):
                        raise PlanningError("连续关节路径需要固定末端投影能力")
                    plan = finalize_path(plan, joint_goal=goal, allow_fixed_projection=True)
                    collision_fallback = True
            except PlanningError as direct_error:
                try:
                    waypoint_goals = self._cartesian_end_effector_goals(
                        targets,
                        state=state,
                        environment=environment,
                        fixed_end_effectors=fixed_end_effectors,
                    )
                    plan = self.motion.plan_joint_waypoints(
                        robot_id=self.robot_id,
                        state=state,
                        goals=waypoint_goals,
                        environment=environment,
                        speed_scale=speed_scale,
                    )
                    plan = finalize_path(plan, waypoint_goals=waypoint_goals)
                    collision_fallback = True
                except PlanningError as constrained_error:
                    detour_errors = []
                    for rotate_during_lift in (False, True):
                        try:
                            waypoint_goals = self._elevated_end_effector_goals(
                                targets,
                                state=state,
                                environment=environment,
                                fixed_end_effectors=fixed_end_effectors,
                                rotate_during_lift=rotate_during_lift,
                            )
                            plan = self.motion.plan_joint_waypoints(
                                robot_id=self.robot_id,
                                state=state,
                                goals=waypoint_goals,
                                environment=environment,
                                speed_scale=speed_scale,
                            )
                            plan = finalize_path(plan, waypoint_goals=waypoint_goals)
                            collision_fallback = True
                            elevated_detour = True
                            break
                        except PlanningError as detour_error:
                            detour_errors.append(str(detour_error))
                    else:
                        detour_detail = "；".join(detour_errors)
                        raise PlanningError(
                            f"固定末端关节路径失败：{direct_error}；"
                            f"同步约束路径失败：{constrained_error}；"
                            f"升高后转移回退失败：{detour_detail}"
                        ) from constrained_error
        elif preserve_cartesian_path:
            guide_joint_goal = None
            # 多末端笛卡尔运动存在多条冗余IK分支。逐路点只使用上一个解，
            # 可能在前半程选中一个局部自然、但无法到达最终目标的分支；这既
            # 会影响双臂携物，也会影响释放后双手同步抬升。先求一次可选终点
            # 解来引导整条路径，逻辑只依赖通用末端目标，不包含场景或箱型。
            try:
                guide_joint_goal = self.kinematics.solve_many(
                    robot_id=self.robot_id,
                    targets=targets,
                    state=state,
                    environment=environment,
                    fixed_end_effectors=fixed_end_effectors,
                )
            except PlanningError:
                # 终点解只用于选择冗余分支，不是连续路径可达的先决条件；
                # 失败时仍从当前构型逐路点求解并保留原有碰撞检查。
                guide_joint_goal = None
            waypoint_goals = self._cartesian_end_effector_goals(
                targets,
                state=state,
                environment=environment,
                fixed_end_effectors=fixed_end_effectors,
                guide_joint_goal=guide_joint_goal,
            )
            plan = self.motion.plan_joint_waypoints(
                robot_id=self.robot_id,
                state=state,
                goals=waypoint_goals,
                environment=environment,
                speed_scale=speed_scale,
            )
            plan = finalize_path(plan, waypoint_goals=waypoint_goals)
        else:
            direct_error = None
            try:
                goal = self.kinematics.solve_many(
                    robot_id=self.robot_id,
                    targets=targets,
                    state=state,
                    environment=environment,
                    fixed_end_effectors=fixed_end_effectors,
                )
                plan = self.motion.plan_joints(
                    robot_id=self.robot_id,
                    state=state,
                    goal=goal,
                    environment=environment,
                    speed_scale=speed_scale,
                )
                plan = finalize_path(plan, joint_goal=goal)
            except PlanningError as error:
                direct_error = error
                try:
                    waypoint_goals = self._cartesian_end_effector_goals(
                        targets,
                        state=state,
                        environment=environment,
                        fixed_end_effectors=fixed_end_effectors,
                    )
                    plan = self.motion.plan_joint_waypoints(
                        robot_id=self.robot_id,
                        state=state,
                        goals=waypoint_goals,
                        environment=environment,
                        speed_scale=speed_scale,
                    )
                    plan = finalize_path(plan, waypoint_goals=waypoint_goals)
                    collision_fallback = True
                except PlanningError as fallback_error:
                    detour_errors = []
                    for rotate_during_lift in (False, True):
                        try:
                            waypoint_goals = self._elevated_end_effector_goals(
                                targets,
                                state=state,
                                environment=environment,
                                fixed_end_effectors=fixed_end_effectors,
                                rotate_during_lift=rotate_during_lift,
                            )
                            plan = self.motion.plan_joint_waypoints(
                                robot_id=self.robot_id,
                                state=state,
                                goals=waypoint_goals,
                                environment=environment,
                                speed_scale=speed_scale,
                            )
                            plan = finalize_path(plan, waypoint_goals=waypoint_goals)
                            collision_fallback = True
                            elevated_detour = True
                            break
                        except PlanningError as detour_error:
                            detour_errors.append(str(detour_error))
                    else:
                        detour_detail = "；".join(detour_errors)
                        raise PlanningError(
                            f"关节直达路径失败：{direct_error}；"
                            f"同步笛卡尔回退失败：{fallback_error}；"
                            f"升高后转移回退失败：{detour_detail}"
                        ) from fallback_error

        return plan.model_copy(
            update={
                "goal": {
                    "targets": {
                        name: target.model_dump(mode="json")
                        for name, target in requested_targets.items()
                    }
                },
                "planner": (
                    f"{self.kinematics.name}+{plan.planner}"
                    + ("+fixed-constraint" if constrained_path else "")
                    + ("+collision-fallback" if collision_fallback else "")
                    + ("+elevated-detour" if elevated_detour else "")
                ),
            }
        )

    def _cartesian_end_effector_goals(
        self,
        targets,
        *,
        state,
        environment,
        fixed_end_effectors,
        guide_joint_goal=None,
    ):
        """以少量同步末端路点生成一条连续关节路径。"""

        return self._cartesian_end_effector_goals_through(
            [targets],
            state=state,
            environment=environment,
            fixed_end_effectors=fixed_end_effectors,
            guide_joint_goal=guide_joint_goal,
        )

    def _elevated_end_effector_goals(
        self,
        targets,
        *,
        state,
        environment,
        fixed_end_effectors,
        rotate_during_lift=False,
    ):
        """直线路径受阻时，先在当前水平位置升高，再向目标转移。

        这是只依赖起终点和碰撞环境的通用几何回退。它不理解抓取对象或
        层高；固定末端仍参与每个同步IK路点，最终路径仍由Motion Provider
        逐点检查碰撞并重新进行时间参数化。
        """

        current = {
            name: (targets[name] if name in fixed_end_effectors else state.end_effectors[name])
            for name in targets
        }
        moving = [name for name in targets if name not in fixed_end_effectors]
        if not moving or not any(
            targets[name].position[2] > current[name].position[2] + 1e-4 for name in moving
        ):
            raise PlanningError("目标没有可用的升高绕行空间")
        via_targets = dict(targets)
        for name in moving:
            start = current[name]
            target = targets[name]
            via_targets[name] = Pose(
                position=(
                    start.position[0],
                    start.position[1],
                    max(start.position[2], target.position[2]),
                ),
                quaternion_xyzw=(
                    target.quaternion_xyzw if rotate_during_lift else start.quaternion_xyzw
                ),
                frame_id=target.frame_id,
            )
        path_targets = [via_targets]
        if not rotate_during_lift:
            # 高位转移期间先保持当前工具朝向，抵达目标XY后再原地转向。
            # 同时平移和翻腕会让长钩形成很大的扫掠包络：靠近身体时容易
            # 撞躯干，靠近密集堆垛时又会扫到相邻箱体。这里仍然只使用
            # 起终点实时几何，不引入箱型或场景专用关节姿态。
            aligned_targets = dict(targets)
            for name in moving:
                start = current[name]
                target = targets[name]
                aligned_targets[name] = Pose(
                    position=target.position,
                    quaternion_xyzw=start.quaternion_xyzw,
                    frame_id=target.frame_id,
                )
            path_targets.append(aligned_targets)
        path_targets.append(targets)
        return self._cartesian_end_effector_goals_through(
            path_targets,
            state=state,
            environment=environment,
            fixed_end_effectors=fixed_end_effectors,
        )

    def _cartesian_end_effector_goals_through(
        self,
        target_sets,
        *,
        state,
        environment,
        fixed_end_effectors,
        guide_joint_goal=None,
    ):
        """把多个末端几何段连续求解成同一条关节路点路径。"""

        working_state = state
        goals = []
        guide_start_joints = dict(state.joint_positions)
        current = {
            name: (
                target_sets[0][name] if name in fixed_end_effectors else state.end_effectors[name]
            )
            for name in target_sets[0]
        }
        for targets in target_sets:
            steps = max(
                _pose_steps(
                    current[name],
                    target,
                    # 双末端搬运时，左右目标必须在中间路点继续联合求解。
                    # 若把 8 cm 抬升退化成一次终点 IK，后续关节插值虽能到达
                    # 正确终点，却可能在途中改变两只工具的相对几何，使钩脚
                    # 短暂离开侧面凹槽。2 cm 是通用 Cartesian 离散精度，
                    # 不依赖箱型、仿真 timestep 或接触帧数。
                    maximum_translation_step_m=0.02,
                )
                for name, target in targets.items()
            )
            if fixed_end_effectors and steps == 1:
                # 固定一侧时，短距离接触动作也不能退化成“只求一个Cartesian
                # 终点再做关节插值”。12mm外拉的真实Gate证明，这种单段轨迹
                # 会在中途把运动端向上带出侧面凹槽，尽管终点本身满足约束。
                # 三段只约束几何路径，不涉及仿真帧数或周转箱业务参数。
                steps = 3
            for index in range(1, steps + 1):
                ratio = index / steps
                waypoint_targets = {
                    name: _interpolate_pose(current[name], target, ratio)
                    for name, target in targets.items()
                }
                solve_state = working_state
                if guide_joint_goal is not None and len(target_sets) == 1:
                    # 种子沿终点分支略微领先当前笛卡尔路点半个采样间隔。
                    # 仅使用相同进度的线性种子，在冗余系统的分支拐点仍可能
                    # 被局部IK吸入无法到达下一路点的死端；直接使用终点解又
                    # 会让第一段产生过大的关节跳变。半步前瞻兼顾连续性和
                    # 终点可达性，步长来自本次几何离散而非设备经验参数。
                    guide_ratio = min(1.0, ratio + 0.5 / steps)
                    guide_joints = dict(working_state.joint_positions)
                    for name, start_position in guide_start_joints.items():
                        target_position = guide_joint_goal.get(name, start_position)
                        guide_joints[name] = (
                            start_position + (target_position - start_position) * guide_ratio
                        )
                    solve_state = working_state.model_copy(update={"joint_positions": guide_joints})
                goal = self.kinematics.solve_many(
                    robot_id=self.robot_id,
                    targets=waypoint_targets,
                    state=solve_state,
                    environment=environment,
                    fixed_end_effectors=fixed_end_effectors,
                    # 固定接触侧存在时，第一个几何路点也必须从动作开始的
                    # 当前关节解连续推进。若允许首点多起点搜索，一个只有几
                    # 毫米的外拉可能切到远端IK分支，关节轨迹便会先抬起钩爪
                    # 再回到同一末端目标。自由空间路径的首点仍可有限择优。
                    continuous_seed=(
                        guide_joint_goal is not None or bool(goals) or bool(fixed_end_effectors)
                    ),
                )
                goals.append(goal)
                joints = dict(working_state.joint_positions)
                joints.update(goal)
                end_effectors = dict(working_state.end_effectors)
                end_effectors.update(waypoint_targets)
                working_state = working_state.model_copy(
                    update={"joint_positions": joints, "end_effectors": end_effectors}
                )
            current = dict(targets)

        # 固定侧优先IK可能在前半程只使用运动臂，接近工作区边界后才启用
        # 共享关节补偿；时间参数化前必须补齐集合并保持上一段真实值。
        return _complete_joint_waypoints(goals, state.joint_positions)

    def move_end_effectors(
        self,
        targets,
        *,
        command_id,
        environment=None,
        speed_scale=1.0,
        maximum_speed_mps=None,
        preserve_cartesian_path=False,
        preserve_relative_geometry=False,
        stop_on_contact=False,
        contact_tool_refs=(),
        max_contact_force_n=None,
        position_tolerance_rad=None,
        fixed_end_effectors=(),
    ):
        plan = self.plan_end_effectors(
            targets,
            environment=environment,
            speed_scale=speed_scale,
            maximum_speed_mps=maximum_speed_mps,
            preserve_cartesian_path=preserve_cartesian_path,
            preserve_relative_geometry=preserve_relative_geometry,
            fixed_end_effectors=fixed_end_effectors,
        )
        plan = plan.model_copy(
            update={
                "stop_on_contact": stop_on_contact,
                "contact_tool_refs": list(contact_tool_refs),
                "max_contact_force_n": max_contact_force_n,
                "position_tolerance_rad": position_tolerance_rad,
            }
        )
        return self.backend.execute_plan(plan, command_id)

    def move_end_effector(
        self,
        end_effector,
        target,
        *,
        command_id,
        environment=None,
        speed_scale=1.0,
        maximum_speed_mps=None,
        preserve_cartesian_path=False,
        stop_on_contact=False,
        contact_tool_refs=(),
        max_contact_force_n=None,
        position_tolerance_rad=None,
    ):
        plan = self.plan_end_effector(
            end_effector,
            target,
            environment=environment,
            speed_scale=speed_scale,
            maximum_speed_mps=maximum_speed_mps,
            preserve_cartesian_path=preserve_cartesian_path,
        )
        plan = plan.model_copy(
            update={
                "stop_on_contact": stop_on_contact,
                "contact_tool_refs": list(contact_tool_refs),
                "max_contact_force_n": max_contact_force_n,
                "position_tolerance_rad": position_tolerance_rad,
            }
        )
        return self.backend.execute_plan(plan, command_id)


def _complete_joint_waypoints(
    goals: list[dict[str, float]], current: dict[str, float]
) -> list[dict[str, float]]:
    """补齐分段IK返回的关节集合，并保留每段之前的真实关节值。"""

    names = sorted({name for goal in goals for name in goal})
    if not names:
        return goals
    unknown = set(names) - set(current)
    if unknown:
        raise PlanningError(f"Robot 当前状态不包含关节：{sorted(unknown)}")
    carried = {name: float(current[name]) for name in names}
    completed = []
    for goal in goals:
        carried.update({name: float(value) for name, value in goal.items()})
        completed.append(dict(carried))
    return completed


def _pose_steps(
    start: Pose,
    target: Pose,
    *,
    maximum_translation_step_m: float = 0.08,
) -> int:
    if maximum_translation_step_m <= 0:
        raise ValueError("末端路径平移步长必须大于零")
    translation = math.dist(start.position, target.position)
    dot = abs(
        sum(left * right for left, right in zip(start.quaternion_xyzw, target.quaternion_xyzw))
    )
    angle = 2.0 * math.acos(min(1.0, max(0.0, dot)))
    # 短距离插入保留2 cm粒度；几十厘米的无接触抬升若仍固定2 cm，会让
    # 数值IK的微小抖动在每个路点反复减速，生成上百秒轨迹。长路径把平移
    # 路点自适应放宽到最多8 cm，而LocalMotion仍对Ruckig产生的每个关节
    # 轨迹点做碰撞检查，因此减少的是冗余IK路点，不是跳过路径安全验证。
    translation_step = min(
        maximum_translation_step_m,
        max(0.02, translation / 8.0),
    )
    # 真实Pose运算常把精确8cm表示为0.08000000000000007。直接ceil会把
    # 本应4段的路径错误拆成5段，改变冗余IK的分支引导并增加无意义路点。
    # 这里只消除机器精度量级的整数边界误差，不放宽几何离散上限。
    translation_steps = math.ceil(translation / translation_step - 1e-9)
    rotation_steps = math.ceil(angle / math.radians(10) - 1e-9)
    return max(1, translation_steps, rotation_steps)


def _cartesian_speed_scale(
    starts,
    targets,
    planned_duration_s,
    maximum_speed_mps,
    current_scale,
):
    """只在规划轨迹确实超过笛卡尔速度上限时拉长时间轴。

    maximum_speed_mps 是末端线速度上限，并不是关节速度比例。过去直接
    用 maximum_speed_mps / 0.5 当作 speed_scale，会把一条本来已经
    受关节速度、加速度和 jerk 限制的双臂轨迹再次放慢十倍。8 cm 抬升因此
    被时间参数化为两分钟。这里先使用真实 Robot Profile 生成受限轨迹，再
    根据末端位移所需的最短时间决定是否需要进一步减速。
    """
    if maximum_speed_mps is None:
        return current_scale
    if maximum_speed_mps <= 0:
        raise ValueError("maximum_speed_mps 必须大于零")
    distance = max(
        math.dist(starts[name].position, target.position) for name, target in targets.items()
    )
    minimum_duration = distance / maximum_speed_mps
    if minimum_duration <= planned_duration_s or minimum_duration <= 1e-9:
        return current_scale
    return max(1e-6, current_scale * planned_duration_s / minimum_duration)


def _interpolate_pose(start: Pose, target: Pose, ratio: float) -> Pose:
    position = tuple(
        left + (right - left) * ratio for left, right in zip(start.position, target.position)
    )
    target_quaternion = target.quaternion_xyzw
    if sum(left * right for left, right in zip(start.quaternion_xyzw, target_quaternion)) < 0:
        target_quaternion = tuple(-value for value in target_quaternion)
    quaternion = tuple(
        left + (right - left) * ratio
        for left, right in zip(start.quaternion_xyzw, target_quaternion)
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise PlanningError("末端姿态插值结果无效")
    return Pose(
        position=position,
        quaternion_xyzw=tuple(value / norm for value in quaternion),
        frame_id=start.frame_id,
    )
