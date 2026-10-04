class SensorsModule:
    def __init__(self, robot_id, backend):
        self.robot_id = robot_id
        self.backend = backend

    def list(self):
        return self.backend.sensors(self.robot_id)

    def latest(self, sensor_id):
        return self.backend.latest_sensor_frame(self.robot_id, sensor_id)

    def subscribe(self, sensor_id, *, poll_interval_seconds=0.05):
        return self.backend.subscribe_sensor(
            self.robot_id, sensor_id, poll_interval_seconds=poll_interval_seconds
        )

    def subscribe_state(self, *, poll_interval_seconds=0.1):
        return self.backend.subscribe_state(
            self.robot_id, poll_interval_seconds=poll_interval_seconds
        )
