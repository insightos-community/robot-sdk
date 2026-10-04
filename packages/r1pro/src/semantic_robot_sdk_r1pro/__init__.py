"""R1 Pro 真机与仿真统一 Robot SDK。"""

from .backends import FakeBackend, IsaacBackend, RealBackend, SharedFakeBackend
from .profile import LEFT_ARM, RIGHT_ARM, TORSO, capabilities
from .sdk import R1ProSDK, create_mujoco_sdk
from .providers.local_navigation_algorithms import OccupancyGrid

__all__ = [
    "FakeBackend",
    "IsaacBackend",
    "LEFT_ARM",
    "MujocoBackend",
    "OccupancyGrid",
    "R1ProSDK",
    "SharedFakeBackend",
    "create_mujoco_sdk",
    "RIGHT_ARM",
    "RealBackend",
    "TORSO",
    "capabilities",
]


def __getattr__(name: str):
    if name == "MujocoBackend":
        from .backends.mujoco import MujocoBackend

        return MujocoBackend
    raise AttributeError(name)
