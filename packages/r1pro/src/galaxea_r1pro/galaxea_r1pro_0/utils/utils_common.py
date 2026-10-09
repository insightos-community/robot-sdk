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

"""功能函数层：跨模块复用的通用工具函数。"""

from __future__ import annotations

import logging
import numpy as np

from scipy.spatial.transform import Rotation as R
from typing import Any

logger = logging.getLogger(__name__)


def finite_vector(values, size: int, field: str) -> np.ndarray:
    """校验控制输入，拒绝广播、NaN 和无穷大。"""
    try:
        value = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须包含 {size} 个有限数值") from exc
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{field} 必须包含 {size} 个有限数值")
    return value


def pose_matrix(position, orientation) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a position and an xyzw quaternion."""
    matrix = np.eye(4)
    matrix[:3, 3] = finite_vector(position, 3, "position")
    matrix[:3, :3] = R.from_quat(finite_vector(orientation, 4, "orientation_xyzw")).as_matrix()
    return matrix


def camera_look_at(position, target) -> list[float]:
    """USD 相机的 xyzw 朝向：局部 -Z 看向目标，+Y 向上。"""
    forward = finite_vector(target, 3, "camera.target") - finite_vector(
        position, 3, "camera.position"
    )
    length = np.linalg.norm(forward)
    if length < 1e-8:
        raise ValueError("相机位置与观察目标不能重合")
    forward /= length
    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("相机视线不能与世界向上方向平行")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return R.from_matrix(np.column_stack([right, up, -forward])).as_quat().tolist()


def matrix_transform(matrix: np.ndarray) -> dict:
    """Decompose a 4x4 homogeneous transform into ``{"position", "orientation"}`` (inverse of :func:`pose_matrix`)."""
    return {
        "position": tuple(float(x) for x in matrix[:3, 3]),
        "orientation": tuple(float(x) for x in R.from_matrix(matrix[:3, :3]).as_quat()),
    }


def pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """同一坐标系内的位姿误差：位置欧氏距离（米）、旋转夹角（弧度）。"""
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    cosine = (np.trace(target[:3, :3] @ actual[:3, :3].T) - 1) / 2
    return position, float(np.arccos(np.clip(cosine, -1, 1)))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the half-open interval ``[-pi, pi)``."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def integrate_body_velocity(pose: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    """由机体 vx/vy/wz 积分 SE(2) 里程计，yaw 不折返，支持超过一圈的旋转。"""
    x, y, yaw = pose
    vx, vy, wz = velocity
    angle = wz * dt
    if abs(angle) < 1e-8:
        dx, dy = vx * dt, vy * dt
    else:
        a, b = np.sin(angle) / wz, (1 - np.cos(angle)) / wz
        dx, dy = a * vx - b * vy, b * vx + a * vy
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x + c * dx - s * dy, y + s * dx + c * dy, yaw + angle])


def euler_to_quaternionx(euler_angles: list[float], seq: str = "xyz") -> "np.ndarray":
    """欧拉角（度）转四元数（对齐 Atom Utils.euler_to_quaternion，scipy 实现）。

    输入 [roll, pitch, yaw]（度），按 seq 顺序（默认 xyz，EEF 使用 yxz）转四元数；
    返回 numpy 数组 (qx, qy, qz, qw)，并做与目标四元数一致的符号归一化。
    """

    euler_angles = np.asarray(euler_angles, dtype=np.float64)
    euler_rad = np.radians(euler_angles)
    rot = R.from_euler(seq.upper(), euler_rad, degrees=False)
    quat = rot.as_quat()

    logger.warning("欧拉角转四元数，输入：%s，输出：%s", euler_angles, quat)

    target_quat = np.array([0.0, -0.70682518, 0.0, 0.70738827], dtype=np.float64)
    if np.dot(quat, target_quat) < 0:
        quat = -quat
    return quat


def quaternion_to_euler(
    qx: float, qy: float, qz: float, qw: float, seq: str = "xyz"
) -> tuple[float, float, float]:
    """四元数（xyzw）转欧拉角，返回 (roll, pitch, yaw)，单位为弧度（scipy 实现）。"""

    roll, pitch, yaw = R.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_euler(
        seq, degrees=False
    )
    return float(roll), float(pitch), float(yaw)


def safe_close(connection: Any, method: str = "close", name: str = "连接") -> bool:
    """安全调用连接对象的关闭方法；异常只记录不抛出。"""

    closer = getattr(connection, method, None)
    if closer is None:
        logger.warning("%s 没有 %s 方法", name, method)
        return False
    try:
        result = closer()
        logger.info("%s 已关闭", name)
        return result is not False
    except Exception as exc:
        logger.error("%s 关闭失败: %s", name, exc)
        return False
