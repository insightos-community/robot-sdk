"""可选的定时控制序列：物理单位、控制组与执行身份，不携带模型原始数组。"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from .models import Command, RobotState, SensorFrame, StrictModel


class SynchronizedObservation(StrictModel):
    """同一采样时刻的原始图像和机器人状态，不复用 Web JPEG 或显示方向。"""

    state: RobotState
    images: list[SensorFrame]
    gripper_joint_positions: dict[str, float]
    sim_time: float = Field(ge=0)

    @model_validator(mode="after")
    def same_generation(self) -> SynchronizedObservation:
        if any(frame.generation != self.state.generation for frame in self.images):
            raise ValueError("同步观测的图像与状态必须属于同一 generation")
        return self


class JointPositionControl(StrictModel):
    kind: Literal["joint_position"] = "joint_position"
    group: str = Field(min_length=1)
    positions_rad: dict[str, float] = Field(min_length=1)


class EndEffectorDeltaControl(StrictModel):
    kind: Literal["end_effector_delta"] = "end_effector_delta"
    group: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    translation_m: tuple[float, float, float]
    rotation_axis_angle_rad: tuple[float, float, float]


class BaseVelocityControl(StrictModel):
    kind: Literal["base_velocity"] = "base_velocity"
    group: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    linear_mps: tuple[float, float]
    angular_rps: float


class GripperDirectionControl(StrictModel):
    """归一化开合方向：正数关闭、负数打开、零保持，不代表实际米制开度。"""

    kind: Literal["gripper_direction"] = "gripper_direction"
    group: str = Field(min_length=1)
    closing_direction: float = Field(ge=-1, le=1)


class GripperPositionControl(StrictModel):
    """目标开度比例：0 全闭、1 全开；中间值是位置目标，不是保持指令。

    适配采用连续位置夹爪的策略。原 gripper_direction 的零保持语义保持不变，
    各后端只声明并实现实际支持的控制类型。
    """

    kind: Literal["gripper_position"] = "gripper_position"
    group: str = Field(min_length=1)
    opening_ratio: float = Field(ge=0, le=1)


Control = Annotated[
    JointPositionControl
    | EndEffectorDeltaControl
    | BaseVelocityControl
    | GripperDirectionControl
    | GripperPositionControl,
    Field(discriminator="kind"),
]


class ControlSample(StrictModel):
    controls: list[Control] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_groups(self) -> ControlSample:
        if len({item.group for item in self.controls}) != len(self.controls):
            raise ValueError("同一采样时刻不能向同一控制组下发多个目标")
        return self


class ControlSequence(StrictModel):
    """有限序列，不隐式循环。每个 sample 恰好占用一个 control_period_s。

    generation 隔离 reset 前后的控制，execution_id 隔离同场景不同技能执行。
    Runtime 取消某次执行后必须拒绝该身份的迟到序列；不能仅靠停止单个
    command_id，因为模型推理可能在取消之后才生成下一个 command_id。
    序列完成只证明控制周期已执行，不证明抓取或原生任务成功。
    """

    robot_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    control_period_s: float = Field(gt=0)
    samples: list[ControlSample] = Field(min_length=1)

    @model_validator(mode="after")
    def stable_groups(self) -> ControlSequence:
        expected = {(item.group, item.kind) for item in self.samples[0].controls}
        if any(
            {(item.group, item.kind) for item in sample.controls} != expected
            for sample in self.samples[1:]
        ):
            raise ValueError("一个序列内控制组和控制语义必须保持一致")
        return self


class TimedControlBackend(Protocol):
    """独立可选能力；不要求已有轨迹 Backend 实现或模拟此接口。"""

    def execute_sequence(self, sequence: ControlSequence, command_id: str) -> Command: ...

    def cancel_execution(self, robot_id: str, execution_id: str, generation: int) -> None: ...

    def synchronized_observation(self, robot_id: str) -> SynchronizedObservation: ...
