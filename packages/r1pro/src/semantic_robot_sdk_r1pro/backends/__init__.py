from .fake import FakeBackend, SharedFakeBackend
from .isaac import IsaacBackend
from .real import RealBackend

__all__ = ["FakeBackend", "SharedFakeBackend", "IsaacBackend", "MujocoBackend", "RealBackend"]


def __getattr__(name: str):
    if name == "MujocoBackend":
        from .mujoco import MujocoBackend

        return MujocoBackend
    raise AttributeError(name)
