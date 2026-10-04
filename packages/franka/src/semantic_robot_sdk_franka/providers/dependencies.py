"""Pinocchio、Ruckig 与 CasADi 的依赖边界。

依赖未安装时返回明确状态。调用方只能显式启用调试回退，不能在生产执行中
静默换成不同算法。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

from semantic_robot_sdk_core.errors import DependencyUnavailable


@dataclass(frozen=True)
class SolverAvailability:
    pinocchio: bool
    ruckig: bool
    casadi: bool


def solver_availability() -> SolverAvailability:
    return SolverAvailability(
        pinocchio=find_spec("pinocchio") is not None,
        ruckig=find_spec("ruckig") is not None,
        casadi=find_spec("casadi") is not None,
    )


def require_pinocchio() -> None:
    if not solver_availability().pinocchio:
        raise DependencyUnavailable("Pinocchio 未安装，不能执行末端 IK 和碰撞检查")


def require_ruckig() -> None:
    if not solver_availability().ruckig:
        raise DependencyUnavailable("Ruckig 未安装，不能生成生产级时间参数化轨迹")


def require_casadi() -> None:
    if not solver_availability().casadi:
        raise DependencyUnavailable("CasADi 未安装，不能执行双臂复杂约束优化")
