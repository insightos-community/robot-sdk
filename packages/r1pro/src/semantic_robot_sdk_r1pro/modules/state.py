class StateModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def capabilities(self):
        return self.backend.capabilities()

    def snapshot(self):
        return self.backend.state(self.robot_id)

    def scene_snapshot(self):
        """返回 Backend 提供的场景事实；真机未配置场景来源时明确失败。"""

        operation = getattr(self.backend, "scene_snapshot", None)
        if operation is None:
            raise RuntimeError("当前 Robot Backend 不提供 SceneSnapshot")
        return operation()
