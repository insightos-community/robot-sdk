"""R1 Pro 固定型号能力；实例连接信息只来自统一部署配置。"""

from semantic_robot_sdk_core import JointLimit, RobotCapabilities, ToolDescriptor

LEFT_ARM = [f"left_arm_joint{index}" for index in range(1, 8)]
RIGHT_ARM = [f"right_arm_joint{index}" for index in range(1, 8)]
TORSO = [f"torso_joint{index}" for index in range(1, 5)]
DEFAULT_TOOLS = [
    ToolDescriptor(
        tool_ref=f"component://tool/{side}",
        side=side,
        kind="tote_clamp",
        frame=f"{side}_tote_load_frame",
        joint=f"{side}_tote_clamp_joint",
        travel_m=0.039,
        normal_force_n=60.0,
        maximum_force_n=120.0,
    )
    for side in ("left", "right")
]
_POSITION_LIMITS = {
    "torso_joint1": (-1.1345, 1.8326),
    "torso_joint2": (-2.7925, 2.5307),
    "torso_joint3": (-1.8326, 1.5708),
    "torso_joint4": (-3.0543, 3.0543),
    "left_arm_joint1": (-4.4506, 1.3090),
    "left_arm_joint2": (-0.1745, 3.1416),
    "left_arm_joint3": (-2.356196, 2.356196),
    "left_arm_joint4": (-2.0944, 0.3491),
    "left_arm_joint5": (-2.356196, 2.356196),
    "left_arm_joint6": (-1.047198, 1.047198),
    "left_arm_joint7": (-1.5708, 1.5708),
    "right_arm_joint1": (-4.4506, 1.3090),
    "right_arm_joint2": (-3.1416, 0.1745),
    "right_arm_joint3": (-2.356196, 2.356196),
    "right_arm_joint4": (-2.0944, 0.3491),
    "right_arm_joint5": (-2.356196, 2.356196),
    "right_arm_joint6": (-1.047198, 1.047198),
    "right_arm_joint7": (-1.5708, 1.5708),
}


def _limit(name: str) -> JointLimit:
    lower, upper = _POSITION_LIMITS[name]
    arm_index = int(name.rsplit("joint", 1)[1]) if "arm_joint" in name else 0
    velocity = 1.5 if name.startswith("torso") else (3.0 if arm_index <= 4 else 5.0)
    return JointLimit(
        lower=lower, upper=upper, max_velocity=velocity, max_acceleration=1.5, max_jerk=1.5
    )


def capabilities(
    tools: list[ToolDescriptor] | None = None,
    *,
    base_footprint_radius_m: float = 0.42,
) -> RobotCapabilities:
    joints = [*TORSO, *LEFT_ARM, *RIGHT_ARM]
    return RobotCapabilities(
        model="r1_pro_chassis",
        kind="mobile_manipulator",
        joint_names=joints,
        joint_groups={"torso": TORSO, "left_arm": LEFT_ARM, "right_arm": RIGHT_ARM},
        joint_limits={name: _limit(name) for name in joints},
        end_effectors=["left", "right"],
        grippers=["left", "right"],
        tools=list(tools or DEFAULT_TOOLS),
        sensors=["rgb", "depth", "contact", "robot_state"],
        supports_base=True,
        base_footprint_radius_m=base_footprint_radius_m,
    )
