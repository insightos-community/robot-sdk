"""底盘路线规划和执行入口。"""


class BaseModule:
    def __init__(self, robot_id, backend, navigation):
        self.robot_id = robot_id
        self.backend = backend
        self.navigation = navigation

    def plan_route(
        self,
        goal,
        occupancy=None,
        *,
        maximum_speed_mps=0.4,
        minimum_clearance_m=None,
        carrying_object_ref=None,
        carrying_object_pose=None,
        carrying_object_extent_m=None,
    ):
        return self.navigation.plan_route(
            robot_id=self.robot_id,
            state=self.backend.state(self.robot_id),
            goal=goal,
            occupancy=occupancy,
            maximum_speed_mps=maximum_speed_mps,
            minimum_clearance_m=minimum_clearance_m,
            carrying_object_ref=carrying_object_ref,
            carrying_object_pose=carrying_object_pose,
            carrying_object_extent_m=carrying_object_extent_m,
        )

    def follow_route(self, plan, *, command_id):
        return self.backend.execute_plan(plan, command_id)

    def navigate(
        self,
        goal,
        occupancy=None,
        *,
        command_id,
        maximum_speed_mps=0.4,
        minimum_clearance_m=None,
        carrying_object_ref=None,
        carrying_object_pose=None,
        carrying_object_extent_m=None,
    ):
        return self.follow_route(
            self.plan_route(
                goal,
                occupancy,
                maximum_speed_mps=maximum_speed_mps,
                minimum_clearance_m=minimum_clearance_m,
                carrying_object_ref=carrying_object_ref,
                carrying_object_pose=carrying_object_pose,
                carrying_object_extent_m=carrying_object_extent_m,
            ),
            command_id=command_id,
        )
