"""Robot SDK 可由调用方稳定判断的公共错误。"""


class RobotSDKError(RuntimeError):
    pass


class RobotModelError(RobotSDKError):
    """Robot 模型包缺失或与型号定义不一致。"""


class PlanningError(RobotSDKError):
    pass


class DependencyUnavailable(PlanningError):
    """必需规划依赖未安装；不得伪造成功计划。"""


class GenerationMismatch(RobotSDKError):
    pass


class BackendUnavailable(RobotSDKError):
    """底层连接不可用；调用方必须把在途命令视为 interrupted。"""


class ConfigurationError(RobotSDKError):
    """统一部署配置缺失或不合法。"""


class BackendRequestError(RobotSDKError):
    """底层明确拒绝请求。"""


class StreamProtocolError(BackendRequestError):
    """Runtime 二进制流元数据或载荷不符合公共帧协议。"""
