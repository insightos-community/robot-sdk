from semantic_robot_sdk_core import CommandState


class SafetyModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def stop_and_hold(self, command_id):
        command = self.backend.stop(self.robot_id, command_id)
        if command.status in {
            CommandState.SUCCEEDED,
            CommandState.FAILED,
            CommandState.STOPPED,
            CommandState.CANCELLED,
        }:
            # stop 与命令自然结束可能同时发生。无论底层返回 stopped 还是
            # succeeded，都必须重新下发 hold 并读取状态，不能把枚举竞态
            # 误判为未安全停止。
            self.backend.hold(self.robot_id)
            if not self.backend.state(self.robot_id).in_hold:
                return command.model_copy(
                    update={
                        "status": CommandState.INTERRUPTED,
                        "reason": "Backend 未能提供 hold 证据",
                    }
                )
        return command

    def hold(self):
        self.backend.hold(self.robot_id)
        return self.backend.state(self.robot_id)
