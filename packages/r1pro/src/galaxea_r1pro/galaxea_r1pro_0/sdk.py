"""Existing R1Pro Ability facade bound to a Semantic-owned scene deployment."""

from .backends.isaac import IsaacBackend
from .eef_control import EEFControl
from .gripper_control import GripperControl
from .joint_control import JointControl
from .sensors import SensorsClient
from .state import StateClient
from .torso_control import TorsoControl
from .woosh_move import WooshMove
from .utils.safety import CommandClient, SafetyClient
from .utils.types import BackendKind


class R1ProSDK:
    def __init__(self, profile, backend):
        self.profile, self._backend = profile, backend
        self._instance = {
            "instance_id": profile.options["scene_instance_id"],
            "run_id": f"{profile.options['scene_instance_id']}:{profile.options['generation']}",
        }
        self.state, self.sensors = StateClient(backend), SensorsClient(backend)
        self.command, self.safety = CommandClient(backend), SafetyClient(backend)
        self.joint, self.gripper = JointControl(backend), GripperControl(backend)
        self.torso, self.eef = TorsoControl(backend, self.sensors), EEFControl(backend)
        self.woosh_move = WooshMove(backend)

    @classmethod
    def open(cls, profile):
        if profile.backend != BackendKind.ISAAC:
            raise ValueError("此门面需要 Semantic Isaac Runtime 部署")
        sdk = cls(profile, IsaacBackend(profile))
        sdk.connect()
        return sdk

    @property
    def instance_id(self):
        """Semantic scene UUID; generation identifies a reset of this scene."""
        return self.profile.options["scene_instance_id"]

    @property
    def generation(self):
        return self.profile.options["generation"]

    def connect(self):
        if not self.is_connected():
            self._backend.connect()

    def is_connected(self):
        return self._backend.is_connected()

    def close(self):
        """Disconnect this client; scene lifecycle is managed by Semantic."""
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
