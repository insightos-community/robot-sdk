class CommandsModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def get(self, command_id):
        return self.backend.command(self.robot_id, command_id)

    def feedback(self, command_id):
        return self.backend.feedback(self.robot_id, command_id)

    def stop(self, command_id):
        return self.backend.stop(self.robot_id, command_id)
