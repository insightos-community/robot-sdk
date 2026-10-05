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

"""R1 Pro 真机 Backend 边界。"""

from semantic_robot_sdk_core import BackendUnavailable


class RealBackend:
    def __init__(self, *_args, driver=None, **_kwargs):
        if driver is None:
            raise BackendUnavailable("R1 Pro 真机 Backend 尚未配置厂商驱动")
        self._driver = driver

    def __getattr__(self, name):
        return getattr(self._driver, name)
