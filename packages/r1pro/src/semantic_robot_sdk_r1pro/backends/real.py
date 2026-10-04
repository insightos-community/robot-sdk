"""R1 Pro 真机 Backend 边界。"""

from semantic_robot_sdk_core import BackendUnavailable


class RealBackend:
    def __init__(self, *_args, driver=None, **_kwargs):
        if driver is None:
            raise BackendUnavailable("R1 Pro 真机 Backend 尚未配置厂商驱动")
        self._driver = driver

    def __getattr__(self, name):
        return getattr(self._driver, name)
