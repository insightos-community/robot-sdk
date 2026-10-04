"""Galaxea R1 Pro Robot SDK 的稳定公开入口。

实现位于 ``galaxea_r1pro.galaxea_r1pro_0``，这里转发其公开 API，
外部统一使用 ``galaxea_r1pro``。
"""

from __future__ import annotations

from .galaxea_r1pro_0 import *  # noqa: F401,F403
from .galaxea_r1pro_0 import __all__ as _all

__all__ = list(_all)
