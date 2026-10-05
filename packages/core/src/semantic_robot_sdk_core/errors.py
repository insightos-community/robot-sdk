# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
