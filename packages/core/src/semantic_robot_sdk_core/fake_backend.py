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

"""共同测试使用的确定性异步 Fake Backend。"""

from __future__ import annotations

from collections.abc import Iterator

from .errors import BackendRequestError, GenerationMismatch
from .models import (
    Command,
    CommandState,
    Feedback,
    MotionPlan,
    PlanKind,
    Pose,
    RobotCapabilities,
    RobotState,
    SensorFrame,
    StrictModel,
    utc_now,
)
from .sensors import RobotStateSubscription, SensorSubscription
from .transforms import quaternion_multiply, relative_pose, rotate_vector

_ACTIVE = {CommandState.ACCEPTED, CommandState.RUNNING}


class FakeGraspFixture(StrictModel):
    """Fake 场景中明确放到夹爪接触范围内的物体。"""

    object_id: str
    contact_opening_m: float
    contact_force_n: float
    minimum_holding_force_n: float


class FakeBackendSnapshot(StrictModel):
    """Fake Backend 的完整可恢复状态。

    这个模型只描述机器人侧事实，不包含 Ability、Skill 或 Gate 流程。共享 Fake
    Backend 会在短 SQLite 事务中读写它，使独立 Ability 进程看到同一台机器人。
    """

    state: RobotState
    commands: dict[str, Command]
    plans: dict[str, MotionPlan]
    feedback: dict[str, list[Feedback]]
    sensor_frames: dict[str, SensorFrame]
    active_resources: dict[str, str]
    grasp_fixtures: dict[str, FakeGraspFixture]
    holding_gripper: str | None = None
    executions: list[str]


class FakeBackend:
    """模拟命令、状态、资源互斥和真实 stop/hold，不冒充生产 Backend。"""

    def __init__(
        self,
        capabilities: RobotCapabilities,
        state: RobotState,
        *,
        auto_complete=True,
        stop_confirmed=True,
    ) -> None:
        self._capabilities = capabilities
        self._state = state
        self.auto_complete = auto_complete
        self.stop_confirmed = stop_confirmed
        self._commands: dict[str, Command] = {}
        self._plans: dict[str, MotionPlan] = {}
        self._feedback: dict[str, list[Feedback]] = {}
        self._sensor_frames: dict[str, SensorFrame] = {}
        self._active_resources: dict[str, str] = {}
        self._grasp_fixtures: dict[str, FakeGraspFixture] = {}
        self._holding_gripper: str | None = None
        self.executions: list[str] = []

    def configure_grasp_fixture(
        self,
        gripper: str,
        object_id: str,
        *,
        contact_opening_m: float = 0.02,
        contact_force_n: float = 8,
        minimum_holding_force_n: float = 2,
    ) -> None:
        """设置可抓取物体；未显式设置时，Fake 关闭夹爪也不会伪造持物成功。"""

        if gripper not in self._capabilities.grippers:
            raise KeyError(gripper)
        if not object_id:
            raise ValueError("object_id 不能为空")
        if contact_opening_m < 0 or contact_force_n <= 0 or minimum_holding_force_n <= 0:
            raise ValueError("接触开度不能为负，接触力和最小夹持力必须大于零")
        self._grasp_fixtures[gripper] = FakeGraspFixture(
            object_id=object_id,
            contact_opening_m=contact_opening_m,
            contact_force_n=contact_force_n,
            minimum_holding_force_n=minimum_holding_force_n,
        )

    def clear_grasp_fixture(self, gripper: str) -> None:
        self._grasp_fixtures.pop(gripper, None)

    def set_tool_contact_state(
        self,
        object_id: str | None,
        *,
        gripper: str = "left",
        contact_force_n: float = 8,
    ) -> None:
        """为测试设置一侧工具的当前接触与力传感状态。"""

        if gripper not in self._capabilities.grippers:
            raise KeyError(gripper)
        if object_id is not None and not object_id:
            raise ValueError("object_id 不能为空字符串")
        descriptor = next(
            (item for item in self._capabilities.tools if item.side == gripper),
            None,
        )
        if descriptor is not None:
            previous = self._state.tool_states[descriptor.tool_ref]
            self._state.tool_states[descriptor.tool_ref] = previous.model_copy(
                update={
                    "hook_contact": object_id is not None,
                    "clamp_contact": object_id is not None,
                    "effort": contact_force_n if object_id is not None else 0.0,
                    "hook_force_n": contact_force_n if object_id is not None else 0.0,
                    "clamp_force_n": contact_force_n if object_id is not None else 0.0,
                    "hook_support_ratio": 1.0 if object_id is not None else 0.0,
                    "hook_tangential_speed_m_s": 0.0,
                    "sensor_fault": False,
                }
            )
        self._holding_gripper = gripper if object_id is not None else None
        self._state.observed_at = utc_now()
        self._publish_grasp_observation(
            gripper,
            contact=object_id is not None,
            force_n=contact_force_n if object_id is not None else 0,
            object_id=object_id,
        )

    def capabilities(self) -> RobotCapabilities:
        return self._capabilities.model_copy(deep=True)

    def export_snapshot(self) -> FakeBackendSnapshot:
        """导出运行状态，供共享状态存储使用。"""

        return FakeBackendSnapshot(
            state=self._state.model_copy(deep=True),
            commands={key: value.model_copy(deep=True) for key, value in self._commands.items()},
            plans={key: value.model_copy(deep=True) for key, value in self._plans.items()},
            feedback={
                key: [item.model_copy(deep=True) for item in values]
                for key, values in self._feedback.items()
            },
            sensor_frames={
                key: value.model_copy(deep=True) for key, value in self._sensor_frames.items()
            },
            active_resources=dict(self._active_resources),
            grasp_fixtures={
                key: value.model_copy(deep=True) for key, value in self._grasp_fixtures.items()
            },
            holding_gripper=self._holding_gripper,
            executions=list(self.executions),
        )

    def restore_snapshot(self, snapshot: FakeBackendSnapshot) -> None:
        """恢复状态；Ability 无需知道 Robot 状态存放在内存还是 SQLite。"""

        if snapshot.state.robot_id != self._state.robot_id:
            raise BackendRequestError(
                f"Fake 状态属于 Robot {snapshot.state.robot_id}，不能用于 {self._state.robot_id}"
            )
        self._state = snapshot.state.model_copy(deep=True)
        self._commands = {
            key: value.model_copy(deep=True) for key, value in snapshot.commands.items()
        }
        self._plans = {key: value.model_copy(deep=True) for key, value in snapshot.plans.items()}
        self._feedback = {
            key: [item.model_copy(deep=True) for item in values]
            for key, values in snapshot.feedback.items()
        }
        self._sensor_frames = {
            key: value.model_copy(deep=True) for key, value in snapshot.sensor_frames.items()
        }
        self._active_resources = dict(snapshot.active_resources)
        self._grasp_fixtures = {
            key: value.model_copy(deep=True) for key, value in snapshot.grasp_fixtures.items()
        }
        self._holding_gripper = snapshot.holding_gripper
        self.executions = list(snapshot.executions)

    def state(self, robot_id: str) -> RobotState:
        self._check_robot(robot_id)
        return self._state.model_copy(deep=True)

    def execute_plan(self, plan: MotionPlan, command_id: str) -> Command:
        previous = self._commands.get(command_id)
        if previous is not None:
            if self._plans[command_id] != plan:
                raise BackendRequestError("同一 command_id 不能对应不同 MotionPlan")
            return previous.model_copy(deep=True)
        self._check_robot(plan.robot_id)
        if plan.generation != self._state.generation:
            raise GenerationMismatch("计划 generation 已失效")
        busy = [resource for resource in plan.resources if resource in self._active_resources]
        if busy:
            raise BackendRequestError(f"Robot 资源正在使用：{busy}")
        command = Command(
            command_id=command_id,
            robot_id=plan.robot_id,
            generation=plan.generation,
            kind=plan.kind,
            status=CommandState.ACCEPTED,
        )
        self._commands[command_id] = command
        self._plans[command_id] = plan.model_copy(deep=True)
        self._feedback[command_id] = [
            Feedback(command_id=command_id, sequence=1, progress=0, message="accepted")
        ]
        self._active_resources.update({resource: command_id for resource in plan.resources})
        self.executions.append(command_id)
        if self.auto_complete:
            self.complete_command(command_id)
        else:
            self._set_status(command_id, CommandState.RUNNING, 0.5, "running")
        return self._commands[command_id].model_copy(deep=True)

    def complete_command(self, command_id: str) -> Command:
        command = self._commands[command_id]
        if command.status not in _ACTIVE:
            return command.model_copy(deep=True)
        self._apply_plan(self._plans[command_id])
        self._set_status(command_id, CommandState.SUCCEEDED, 1, "succeeded")
        self._release(command_id)
        return self._commands[command_id].model_copy(deep=True)

    def interrupt_command(self, command_id: str, reason="connection lost") -> Command:
        self._set_status(
            command_id, CommandState.INTERRUPTED, self._commands[command_id].progress, reason
        )
        return self._commands[command_id].model_copy(deep=True)

    def command(self, robot_id: str, command_id: str) -> Command:
        self._check_robot(robot_id)
        value = self._commands[command_id]
        return value.model_copy(deep=True)

    def feedback(self, robot_id: str, command_id: str) -> Iterator[Feedback]:
        self.command(robot_id, command_id)
        yield from (item.model_copy(deep=True) for item in self._feedback[command_id])

    def stop(self, robot_id: str, command_id: str) -> Command:
        command = self.command(robot_id, command_id)
        if command.status not in _ACTIVE:
            return command
        if not self.stop_confirmed:
            self._set_status(command_id, CommandState.INTERRUPTED, command.progress, "停止无法确认")
            return self._commands[command_id].model_copy(deep=True)
        self._set_status(command_id, CommandState.STOPPED, command.progress, "stopped")
        self._state.in_hold = True
        self._release(command_id)
        return self._commands[command_id].model_copy(deep=True)

    def hold(self, robot_id: str) -> None:
        self._check_robot(robot_id)
        self._state.in_hold = True

    def sensors(self, robot_id: str) -> list[str]:
        self._check_robot(robot_id)
        return list(self._capabilities.sensors)

    def latest_sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame:
        self._check_robot(robot_id)
        if sensor_id not in self._capabilities.sensors:
            raise KeyError(sensor_id)
        frame = self._sensor_frames.get(sensor_id) or SensorFrame(
            sensor_id=sensor_id,
            sequence=1,
            generation=self._state.generation,
            frame_id=sensor_id,
            encoding="json",
            payload={"ok": True},
        )
        return frame.model_copy(deep=True)

    def publish_sensor_frame(self, frame: SensorFrame) -> None:
        if frame.sensor_id not in self._capabilities.sensors:
            raise KeyError(frame.sensor_id)
        if frame.generation != self._state.generation:
            raise GenerationMismatch("不能发布其他 generation 的传感器帧")
        previous = self._sensor_frames.get(frame.sensor_id)
        if previous and frame.sequence <= previous.sequence:
            raise BackendRequestError("传感器帧序号必须递增")
        self._sensor_frames[frame.sensor_id] = frame.model_copy(deep=True)

    def subscribe_sensor(self, robot_id: str, sensor_id: str, *, poll_interval_seconds=0.05):
        state = self.state(robot_id)
        return SensorSubscription(
            lambda: self.latest_sensor_frame(robot_id, sensor_id),
            expected_generation=state.generation,
            poll_interval_seconds=poll_interval_seconds,
        )

    def subscribe_state(self, robot_id: str, *, poll_interval_seconds=0.1):
        state = self.state(robot_id)
        return RobotStateSubscription(
            lambda: self.state(robot_id),
            expected_generation=state.generation,
            poll_interval_seconds=poll_interval_seconds,
        )

    def _set_status(self, command_id: str, status: CommandState, progress: float, message: str):
        command = self._commands[command_id]
        command.status = status
        command.progress = progress
        command.updated_at = utc_now()
        command.reason = None if status is CommandState.SUCCEEDED else message
        self._feedback[command_id].append(
            Feedback(
                command_id=command_id,
                sequence=len(self._feedback[command_id]) + 1,
                progress=progress,
                message=message,
            )
        )

    def _apply_plan(self, plan: MotionPlan) -> None:
        if plan.kind is PlanKind.JOINT:
            self._state.joint_positions.update(plan.joint_trajectory[-1].positions)
            # 末端目标已经是 Robot SDK 规划结果的一部分。Fake Backend 完成关节
            # 计划时同步这个观测值，才能让后续 Ability 读取“当前状态”，而不是
            # 继续看到上一条运动前的末端位姿。普通关节姿态计划没有这些字段，
            # 因此不会凭空推导 FK，也不会把测试行为带入真实 Backend。
            target_poses: dict[str, Pose] = {}
            if isinstance(plan.goal.get("targets"), dict):
                target_poses = {
                    str(name): Pose.model_validate(value)
                    for name, value in plan.goal["targets"].items()
                }
            elif plan.goal.get("end_effector"):
                name = str(plan.goal["end_effector"])
                target_poses[name] = Pose.model_validate(
                    {key: value for key, value in plan.goal.items() if key != "end_effector"}
                )
            self._state.end_effectors.update(target_poses)
        elif plan.kind is PlanKind.BASE and self._state.base_pose is not None:
            point = plan.base_trajectory[-1]
            half = point.yaw / 2
            previous_base_pose = self._state.base_pose
            next_base_pose = previous_base_pose.model_copy(
                update={
                    "position": (point.x, point.y, self._state.base_pose.position[2]),
                    "quaternion_xyzw": (
                        0,
                        0,
                        __import__("math").sin(half),
                        __import__("math").cos(half),
                    ),
                }
            )
            # Fake Backend 仍应遵守世界坐标系的基本运动关系：底盘移动后，安装在
            # Robot 上的末端必须随底盘做同一刚体变换。否则下一项 Ability 读取到
            # 的实时持物位姿会停留在导航前，测试会把一条正确的放置安全校验误判
            # 为“目标太远”。这只是 Robot 状态一致性，不在 SDK 中推导持物业务结论。
            self._state.end_effectors = {
                name: self._move_world_pose_with_base(
                    pose,
                    previous_base_pose=previous_base_pose,
                    next_base_pose=next_base_pose,
                )
                for name, pose in self._state.end_effectors.items()
            }
            self._state.base_pose = next_base_pose
        elif plan.gripper_command is not None:
            target = plan.gripper_command
            fixture = self._grasp_fixtures.get(target.gripper)
            contact = fixture is not None and target.opening_m <= fixture.contact_opening_m
            load_object: str | None = None
            if contact:
                force_n = min(target.force_limit_n, fixture.contact_force_n)
                holding = force_n >= fixture.minimum_holding_force_n
                self._state.gripper_openings[target.gripper] = fixture.contact_opening_m
                load_object = fixture.object_id if holding else None
                self._holding_gripper = target.gripper if holding else None
                self._publish_grasp_observation(
                    target.gripper,
                    contact=True,
                    force_n=force_n,
                    object_id=fixture.object_id if holding else None,
                )
            else:
                self._state.gripper_openings[target.gripper] = target.opening_m
                self._publish_grasp_observation(
                    target.gripper,
                    contact=False,
                    force_n=0,
                    object_id=None,
                )
            self._holding_gripper = target.gripper if load_object is not None else None
            measured_force_n = force_n if contact else 0.0
            descriptor = next(
                (item for item in self._capabilities.tools if item.side == target.gripper), None
            )
            if descriptor is not None:
                self._state.tool_states[descriptor.tool_ref] = self._state.tool_states[
                    descriptor.tool_ref
                ].model_copy(
                    update={
                        "position": self._state.gripper_openings[target.gripper],
                        "target_position": target.opening_m,
                        "reached_target": True,
                        "hook_contact": load_object is not None,
                        "clamp_contact": load_object is not None,
                        "effort": measured_force_n,
                        "hook_force_n": measured_force_n,
                        "clamp_force_n": measured_force_n,
                        "hook_support_ratio": 1.0 if load_object is not None else 0.0,
                        "hook_tangential_speed_m_s": 0.0,
                    }
                )
        self._state.observed_at = utc_now()
        self._state.in_hold = False

    @staticmethod
    def _move_world_pose_with_base(
        pose,
        *,
        previous_base_pose,
        next_base_pose,
    ):
        if pose.frame_id != previous_base_pose.frame_id:
            return pose
        relative = relative_pose(previous_base_pose, pose, result_frame_id="base")
        offset = rotate_vector(next_base_pose.quaternion_xyzw, relative.position)
        return pose.model_copy(
            update={
                "position": tuple(
                    next_base_pose.position[index] + offset[index] for index in range(3)
                ),
                "quaternion_xyzw": quaternion_multiply(
                    next_base_pose.quaternion_xyzw,
                    relative.quaternion_xyzw,
                ),
            }
        )

    def _publish_grasp_observation(
        self,
        gripper: str,
        *,
        contact: bool,
        force_n: float,
        object_id: str | None,
    ) -> None:
        self._publish_structured_sensor(
            "contact",
            {
                "gripper": gripper,
                "contact": contact,
                "force_n": force_n,
                "object_id": object_id,
            },
        )

    def _publish_structured_sensor(self, sensor_id: str, payload: dict) -> None:
        if sensor_id not in self._capabilities.sensors:
            return
        previous = self._sensor_frames.get(sensor_id)
        self._sensor_frames[sensor_id] = SensorFrame(
            sensor_id=sensor_id,
            sequence=1 if previous is None else previous.sequence + 1,
            generation=self._state.generation,
            frame_id=sensor_id,
            encoding="json",
            payload=payload,
        )

    def _release(self, command_id: str) -> None:
        for resource in [
            name for name, owner in self._active_resources.items() if owner == command_id
        ]:
            del self._active_resources[resource]

    def _check_robot(self, robot_id: str) -> None:
        if robot_id != self._state.robot_id:
            raise KeyError(robot_id)
