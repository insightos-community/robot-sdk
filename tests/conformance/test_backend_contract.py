import json
import math
from pathlib import Path

import pytest

from semantic_robot_sdk_core import CommandState, PlanningError, Pose, load_robot_deployment
from semantic_robot_sdk_core.errors import BackendRequestError
from semantic_robot_sdk_r1pro import R1ProSDK
from semantic_robot_sdk_r1pro.providers.local_navigation_algorithms import (
    OccupancyGrid,
    astar,
    smooth_path,
)
from semantic_robot_sdk_r1pro.providers.local_navigation import LocalNavigationProvider
from semantic_robot_sdk_r1pro.providers.navigation_map import RuntimeSceneNavigationMapSource
from semantic_robot_sdk_r1pro.providers.navigation_timing import base_segment_targets
from semantic_robot_sdk_r1pro.providers.vendor import VendorNavigationProvider
from semantic_robot_sdk_r1pro.profile import capabilities


def sdk(monkeypatch, *, auto_complete=True, stop_confirmed=True):
    monkeypatch.setenv(
        "SEMANTIC_ROBOT_CONFIG", str(Path("examples/robot-deployment.fake.yaml").resolve())
    )
    deployment = load_robot_deployment()
    deployment.robot.sdk.options.update(
        {
            "auto_complete": auto_complete,
            "stop_confirmed": stop_confirmed,
        }
    )
    return R1ProSDK.from_deployment(deployment)


def grid():
    return OccupancyGrid(
        width=40,
        height=40,
        resolution_m=0.25,
        origin_xy=(-5, -5),
        frame_id="world",
        occupied=frozenset(),
    )


def target(x=1.0):
    return Pose(position=(x, 0, 0), quaternion_xyzw=(0, 0, 0, 1), frame_id="world")


def test_modules_execute_a_planned_route_and_update_state(monkeypatch):
    robot = sdk(monkeypatch)
    initial_effectors = robot.state.snapshot().end_effectors
    plan = robot.base.plan_route(target(), grid())
    first = robot.base.follow_route(plan, command_id="route-1")
    duplicate = robot.base.follow_route(plan, command_id="route-1")
    assert first == duplicate
    assert first.status is CommandState.SUCCEEDED
    assert robot.state.snapshot().base_pose.position[:2] == pytest.approx((1, 0))
    assert all(
        pose.position[:2] == pytest.approx((initial_effectors[name].position[0] + 1, 0))
        for name, pose in robot.state.snapshot().end_effectors.items()
    )
    assert robot.backend.executions == ["route-1"]


def test_fake_sdk_resolves_its_test_map_without_ability_building_a_grid(monkeypatch):
    robot = sdk(monkeypatch)
    plan = robot.base.plan_route(target())
    assert plan.goal["x"] == 1.0


def test_fake_state_has_initial_world_pose_for_each_end_effector(monkeypatch):
    state = sdk(monkeypatch).state.snapshot()
    assert set(state.end_effectors) == {"left", "right"}
    assert all(pose.frame_id == "world" for pose in state.end_effectors.values())


def test_fake_joint_execution_reports_planned_end_effector_targets(monkeypatch):
    robot = sdk(monkeypatch)
    targets = {
        "left": Pose(
            position=(0.5, 0.2, 0.6),
            quaternion_xyzw=(0, 0, 0, 1),
            frame_id="world",
        ),
        "right": Pose(
            position=(0.5, -0.2, 0.6),
            quaternion_xyzw=(0, 0, 0, 1),
            frame_id="world",
        ),
    }

    command = robot.upper_body.move_end_effectors(targets, command_id="carry-posture")

    assert command.status is CommandState.SUCCEEDED
    assert robot.state.snapshot().end_effectors == targets


def test_local_navigation_respects_requested_clearance_over_footprint(monkeypatch):
    """请求的 clearance 大于底盘包络时必须额外收紧可通行区域。"""

    robot = sdk(monkeypatch)
    goal = Pose(position=(0.0, 2.5, 0), quaternion_xyzw=(0, 0, 0, 1), frame_id="world")
    # 两段墙体在 x=0 附近留 1.5m 工位开口。clearance=0.1 时开口有效；
    # clearance=0.8 时墙体各外扩 0.8m，开口完全封闭，规划必须失败。
    occupancy = OccupancyGrid(
        width=100,
        height=80,
        resolution_m=0.1,
        origin_xy=(-5.0, -5.0),
        frame_id="world",
        occupied=frozenset(),
        collision_boxes=(
            (-5.0, 2.0, 0.0, -0.75, 6.0, 1.0),
            (0.75, 2.0, 0.0, 5.0, 6.0, 1.0),
        ),
    )

    with pytest.raises(PlanningError, match="膨胀区|未找到可行"):
        robot.base.plan_route(goal, occupancy, minimum_clearance_m=0.8)

    plan = robot.base.plan_route(goal, occupancy, minimum_clearance_m=0.1)
    assert plan.goal["y"] == pytest.approx(2.5, abs=0.2)


def test_local_navigation_carrying_clearance_is_total_not_additive(monkeypatch):
    """携物 minimum_clearance_m 是总净空：托盘南侧 0.45m 工位必须可达。

    回归用例：携物导航曾把 clearance 与方形半边相加（0.4+0.42=0.82m），
    导致 pallet-b 四周 0.82m 全是禁区，放置工位 (1.19, 0.45) 被误判为
    “终点位于安全膨胀区”，nav-with-tote 连续 5 次规划失败。
    """

    robot = sdk(monkeypatch)
    goal = Pose(position=(1.19, 0.45, 0), quaternion_xyzw=(0, 0, 0, 1), frame_id="world")
    occupancy = OccupancyGrid(
        width=100,
        height=80,
        resolution_m=0.05,
        origin_xy=(-2.0, -2.0),
        frame_id="world",
        occupied=frozenset(),
        collision_boxes=(
            (-0.6, 1.0, 0.0, 0.6, 2.0, 0.15),
            (0.9, 1.0, 0.0, 2.1, 2.0, 0.15),
        ),
    )
    plan = robot.base.plan_route(
        goal,
        occupancy,
        maximum_speed_mps=0.15,
        minimum_clearance_m=0.4,
        carrying_object_ref="tote-large-l3-r1-c1",
        carrying_object_pose=Pose(
            position=(-0.31, 1.08, 1.08), quaternion_xyzw=(0, 0, 0, 1), frame_id="world"
        ),
        carrying_object_extent_m=(0.6, 0.4, 0.34),
    )
    assert plan.goal["x"] == pytest.approx(1.19, abs=0.2)
    assert plan.goal["y"] == pytest.approx(0.45, abs=0.2)


def test_navigation_shortcut_does_not_touch_an_inflated_obstacle_corner():
    """平滑后的斜线不能擦过已膨胀障碍的角点。"""

    occupancy = OccupancyGrid(
        width=16,
        height=16,
        resolution_m=1.0,
        origin_xy=(0.0, 0.0),
        frame_id="world",
        occupied=frozenset({(7, 7), (7, 8), (8, 7), (8, 8)}),
    )
    start = (0.5, 7.5)
    goal = (9.5, 8.5)

    route = smooth_path(astar(occupancy, start, goal), occupancy, occupancy.occupied)

    assert route != [start, (8.5, 9.5), goal]
    assert len(route) >= 3


def test_local_navigation_rejects_negative_clearance(monkeypatch):
    robot = sdk(monkeypatch)
    with pytest.raises(PlanningError, match="minimum_clearance_m"):
        robot.base.plan_route(target(), grid(), minimum_clearance_m=-0.1)


def test_local_navigation_does_not_infer_carrying_from_tool_contacts(monkeypatch):
    deployment = load_robot_deployment(Path("examples/robot-deployment.fake.yaml").resolve())
    deployment.robot.safety.carrying_motion_scale = 0.1
    robot = R1ProSDK.from_deployment(deployment)

    empty_plan = robot.base.plan_route(target(1.0), grid())
    for side in ("left", "right"):
        robot.backend.configure_grasp_fixture(
            side,
            "box-17",
            contact_opening_m=0.021,
            contact_force_n=7,
            minimum_holding_force_n=2,
        )
        robot.end_effector.close_until_contact(
            side,
            command_id=f"grasp-box-17-{side}",
            force_limit_n=5,
        )

    carrying_plan = robot.base.plan_route(target(1.0), grid())

    assert all(
        item.hook_contact and item.clamp_contact
        for item in robot.state.snapshot().tool_states.values()
    )
    assert carrying_plan.estimated_duration_s == pytest.approx(empty_plan.estimated_duration_s)


def test_local_navigation_uses_explicit_carrying_context(monkeypatch):
    deployment = load_robot_deployment(Path("examples/robot-deployment.fake.yaml").resolve())
    deployment.robot.safety.carrying_motion_scale = 0.25
    robot = R1ProSDK.from_deployment(deployment)

    empty_plan = robot.base.plan_route(target(1.0), grid())
    carrying_plan = robot.base.plan_route(
        target(1.0),
        grid(),
        carrying_object_ref="object://pallet-a/box-17",
    )

    assert carrying_plan.estimated_duration_s > empty_plan.estimated_duration_s


def test_local_navigation_routes_carried_aabb_around_same_height_obstacle(
    monkeypatch,
):
    """底盘本体可直行时，前伸箱体仍必须先退出障碍包络。"""

    robot = sdk(monkeypatch)
    resolution = 0.05
    origin = (-1.0, -1.0)
    obstacle = (0.3, 0.65, 0.8, 0.7, 1.05, 1.2)
    occupied = frozenset((x, y) for x in range(26, 35) for y in range(33, 42))
    occupancy = OccupancyGrid(
        width=70,
        height=60,
        resolution_m=resolution,
        origin_xy=origin,
        frame_id="world",
        occupied=occupied,
        collision_boxes=(obstacle,),
    )
    plan = robot.base.plan_route(
        target(1.2),
        occupancy,
        maximum_speed_mps=0.1,
        carrying_object_ref="object://box-17",
        carrying_object_pose=Pose(
            position=(0.0, 0.55, 1.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        ),
        carrying_object_extent_m=(0.6, 0.4, 0.34),
    )

    assert min(point.y for point in plan.base_trajectory) < -0.05


def test_carrying_limit_does_not_scale_an_already_lower_speed_request(monkeypatch):
    """携物比例限制Robot能力上限，不把调用方的保守低速再乘一次。"""

    deployment = load_robot_deployment(Path("examples/robot-deployment.fake.yaml").resolve())
    deployment.robot.safety.carrying_motion_scale = 0.25
    robot = R1ProSDK.from_deployment(deployment)
    plan = robot.base.plan_route(
        target(2.0),
        grid(),
        maximum_speed_mps=0.05,
        carrying_object_ref="object://pallet-a/box-17",
    )

    peak_speed = max(
        abs(current.x - previous.x) / (current.time_from_start_s - previous.time_from_start_s)
        for previous, current in zip(plan.base_trajectory, plan.base_trajectory[1:])
        if current.time_from_start_s > previous.time_from_start_s
    )
    assert peak_speed == pytest.approx(0.05, rel=0.05)


def test_carried_load_translation_preserves_yaw_before_final_turn():
    targets = base_segment_targets(
        [(0.0, 0.0), (0.5, 0.2), (1.0, 0.5)],
        initial_yaw=0.0,
        final_yaw=1.0,
        preserve_yaw=True,
    )

    assert [item["base_yaw"] for item in targets[:-1]] == [0.0, 0.0]
    assert targets[-1] == {"base_x": 1.0, "base_y": 0.5, "base_yaw": 1.0}


def test_runtime_navigation_excludes_only_the_explicitly_carried_object(monkeypatch):
    robot = sdk(monkeypatch)
    state = robot.state.snapshot()
    snapshot = {
        "generation": state.generation,
        "coordinate_frame": "world",
        "objects": [
            {
                "source_id": "box-17",
                "pose": {"position": [0.5, 0.0, 0.2]},
                "extent": [0.6, 0.4, 0.34],
            }
        ],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(snapshot).encode()

    monkeypatch.setattr(
        "semantic_robot_sdk_r1pro.providers.navigation_map.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    source = RuntimeSceneNavigationMapSource("http://runtime", "scene-1")

    occupied = source.get_occupancy(robot_id=robot.robot_id, state=state, goal=target())
    carrying = source.get_occupancy(
        robot_id=robot.robot_id,
        state=state,
        goal=target(),
        excluded_source_refs=frozenset({"object://pallet-a/box-17"}),
    )

    assert occupied.occupied
    for cell_x, cell_y in occupied.occupied:
        cell_min_x = occupied.origin_xy[0] + cell_x * occupied.resolution_m
        cell_max_x = cell_min_x + occupied.resolution_m
        cell_min_y = occupied.origin_xy[1] + cell_y * occupied.resolution_m
        cell_max_y = cell_min_y + occupied.resolution_m
        assert cell_min_x < 0.8 and cell_max_x > 0.2
        assert cell_min_y < 0.2 and cell_max_y > -0.2

    assert not carrying.occupied


def test_local_navigation_can_leave_a_discretized_occupied_start_cell(monkeypatch):
    robot = sdk(monkeypatch)
    state = robot.state.snapshot()
    start_cell = grid().world_to_cell(*state.base_pose.position[:2])
    obstacle_cell = (start_cell[0] + 2, start_cell[1])
    boundary_grid = OccupancyGrid(
        width=40,
        height=40,
        resolution_m=0.25,
        origin_xy=(-5, -5),
        frame_id="world",
        occupied=frozenset({obstacle_cell}),
    )

    plan = robot.base.plan_route(target(-1), boundary_grid)

    assert plan.start["x"] == pytest.approx(state.base_pose.position[0])


def test_local_navigation_projects_only_an_inflated_goal_edge(monkeypatch):
    robot = sdk(monkeypatch)
    occupancy = OccupancyGrid(
        width=120,
        height=100,
        resolution_m=0.1,
        origin_xy=(-5.0, -5.0),
        frame_id="world",
        occupied=frozenset({(60, 50)}),
    )

    plan = robot.base.plan_route(target(1.45), occupancy)
    resolved = (plan.goal["x"], plan.goal["y"])
    inflated = occupancy.inflated(capabilities().base_footprint_radius_m)

    assert resolved != pytest.approx((1.45, 0.0))
    assert math.dist(resolved, (1.45, 0.0)) <= occupancy.resolution_m * 2
    assert inflated.world_to_cell(*resolved) not in inflated.occupied


def test_local_navigation_does_not_project_a_raw_occupied_goal(monkeypatch):
    robot = sdk(monkeypatch)
    occupancy = OccupancyGrid(
        width=120,
        height=100,
        resolution_m=0.1,
        origin_xy=(-5.0, -5.0),
        frame_id="world",
        occupied=frozenset({(60, 50)}),
    )

    with pytest.raises(PlanningError, match="终点位于障碍物中"):
        robot.base.plan_route(target(1.05), occupancy)


def test_vendor_navigation_omits_unspecified_clearance_for_older_driver(monkeypatch):
    robot = sdk(monkeypatch)

    class ExistingVendorDriver:
        def plan_route(self, *, robot_id, state, goal, occupancy, maximum_speed_mps):
            return (robot_id, goal, maximum_speed_mps)

    provider = VendorNavigationProvider(ExistingVendorDriver())
    result = provider.plan_route(
        robot_id=robot.robot_id,
        state=robot.state.snapshot(),
        goal=target(),
        maximum_speed_mps=0.3,
    )
    assert result == (robot.robot_id, target(), 0.3)


def test_local_navigation_without_a_map_source_fails_explicitly(monkeypatch):
    robot = sdk(monkeypatch)
    provider = LocalNavigationProvider(capabilities(), allow_debug_fallback=True)
    with pytest.raises(PlanningError, match="navigation_map_source"):
        provider.plan_route(
            robot_id=robot.robot_id,
            state=robot.state.snapshot(),
            goal=target(),
            occupancy=None,
            maximum_speed_mps=0.4,
        )


def test_same_command_id_cannot_change_content(monkeypatch):
    robot = sdk(monkeypatch)
    robot.base.follow_route(robot.base.plan_route(target(), grid()), command_id="route-1")
    with pytest.raises(BackendRequestError, match="command_id"):
        robot.base.follow_route(robot.base.plan_route(target(2), grid()), command_id="route-1")


def test_running_resources_are_mutually_exclusive(monkeypatch):
    robot = sdk(monkeypatch, auto_complete=False)
    robot.base.follow_route(robot.base.plan_route(target(), grid()), command_id="route-1")
    with pytest.raises(BackendRequestError, match="资源"):
        robot.base.follow_route(robot.base.plan_route(target(2), grid()), command_id="route-2")


def test_stop_requires_hold_evidence(monkeypatch):
    robot = sdk(monkeypatch, auto_complete=False)
    robot.base.follow_route(robot.base.plan_route(target(), grid()), command_id="route-stop")
    result = robot.safety.stop_and_hold("route-stop")
    assert result.status is CommandState.STOPPED
    assert robot.state.snapshot().in_hold


def test_completed_command_still_enters_hold(monkeypatch):
    robot = sdk(monkeypatch)
    command = robot.base.follow_route(
        robot.base.plan_route(target(), grid()), command_id="route-completed"
    )
    assert command.status is CommandState.SUCCEEDED

    result = robot.safety.stop_and_hold("route-completed")

    assert result.status is CommandState.SUCCEEDED
    assert robot.state.snapshot().in_hold


def test_unconfirmed_stop_is_interrupted(monkeypatch):
    robot = sdk(monkeypatch, auto_complete=False, stop_confirmed=False)
    robot.base.follow_route(robot.base.plan_route(target(), grid()), command_id="route-stop")
    assert robot.safety.stop_and_hold("route-stop").status is CommandState.INTERRUPTED


def test_end_effector_rejects_force_above_tool_profile(monkeypatch):
    robot = sdk(monkeypatch)
    descriptor = next(tool for tool in robot.state.capabilities().tools if tool.side == "left")

    with pytest.raises(PlanningError, match="超过Profile峰值"):
        robot.end_effector.close_until_contact(
            "left",
            command_id="unsafe-force",
            force_limit_n=descriptor.maximum_force_n + 1.0,
        )


def test_fake_grasp_fixture_updates_raw_contact_and_release_observations(monkeypatch):
    robot = sdk(monkeypatch)
    for side in ("left", "right"):
        robot.backend.configure_grasp_fixture(
            side,
            "box-17",
            contact_opening_m=0.021,
            contact_force_n=7,
            minimum_holding_force_n=2,
        )

    robot.end_effector.close_until_contact("left", command_id="grasp-box-17", force_limit_n=5)
    assert robot.state.snapshot().tool_states["component://tool/left"].hook_contact
    robot.end_effector.close_until_contact(
        "right", command_id="grasp-box-17-right", force_limit_n=5
    )
    grasped = robot.state.snapshot()
    assert all(item.hook_contact and item.clamp_contact for item in grasped.tool_states.values())
    assert grasped.gripper_openings["left"] == pytest.approx(0.021)
    assert robot.sensors.latest("contact").payload == {
        "gripper": "right",
        "contact": True,
        "force_n": 5,
        "object_id": "box-17",
    }

    robot.end_effector.release("left", command_id="release-box-17")
    released = robot.state.snapshot()
    assert not released.tool_states["component://tool/left"].hook_contact
    assert released.tool_states["component://tool/right"].hook_contact
    robot.end_effector.release("right", command_id="release-box-17-right")
    assert robot.sensors.latest("contact").payload["contact"] is False


def test_fake_close_without_a_fixture_does_not_invent_a_held_object(monkeypatch):
    robot = sdk(monkeypatch)
    robot.end_effector.close_until_contact("right", command_id="empty-close", force_limit_n=5)
    assert not robot.state.snapshot().tool_states["component://tool/right"].hook_contact
    assert robot.sensors.latest("contact").payload["contact"] is False


def test_isaac_never_returns_fake_success(monkeypatch):
    monkeypatch.setenv(
        "SEMANTIC_ROBOT_CONFIG", str(Path("examples/robot-deployment.fake.yaml").resolve())
    )
    deployment = load_robot_deployment()
    deployment.robot.backend = "isaac"
    with pytest.raises(Exception, match="拆码垛型号不支持 Isaac"):
        R1ProSDK.from_deployment(deployment)


def test_real_backend_requires_a_vendor_driver(monkeypatch):
    monkeypatch.setenv(
        "SEMANTIC_ROBOT_CONFIG", str(Path("examples/robot-deployment.fake.yaml").resolve())
    )
    deployment = load_robot_deployment()
    deployment.robot.backend = "real"
    deployment.robot.sdk.providers.kinematics = "vendor"
    deployment.robot.sdk.providers.motion = "vendor"
    deployment.robot.sdk.providers.navigation = "vendor"
    with pytest.raises(Exception, match="厂商驱动"):
        R1ProSDK.from_deployment(deployment)
