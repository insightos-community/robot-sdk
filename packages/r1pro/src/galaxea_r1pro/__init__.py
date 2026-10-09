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

"""Galaxea R1 Pro Robot SDK 的稳定公开入口。

实现位于 ``galaxea_r1pro.galaxea_r1pro_0``，这里转发其公开 API，
外部统一使用 ``galaxea_r1pro``。
"""

from __future__ import annotations

from .galaxea_r1pro_0 import *  # noqa: F401,F403
from .galaxea_r1pro_0 import __all__ as _all

__all__ = list(_all)
