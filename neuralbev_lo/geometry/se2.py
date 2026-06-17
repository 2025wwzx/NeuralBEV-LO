#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SE(2) 几何工具。

提供 2D 平面位姿的矩阵构造、逆变换、组合和角度归一化。
"""

from __future__ import annotations

from math import atan2, cos, pi, sin

import numpy as np
from numpy.typing import NDArray

Transform2D = NDArray[np.float64]


def normalize_yaw(yaw: float) -> float:
    """将 yaw 归一化到 `[-pi, pi]`."""

    value = float(yaw)
    while value > pi:
        value -= 2.0 * pi
    while value < -pi:
        value += 2.0 * pi
    return value


def se2_from_xyyaw(x: float, y: float, yaw: float) -> Transform2D:
    """构造平面 3DoF 齐次矩阵。"""

    matrix = np.eye(3, dtype=np.float64)
    c = cos(yaw)
    s = sin(yaw)
    matrix[0, 0] = c
    matrix[0, 1] = -s
    matrix[1, 0] = s
    matrix[1, 1] = c
    matrix[0, 2] = float(x)
    matrix[1, 2] = float(y)
    return matrix


def invert_se2(transform: object) -> Transform2D:
    """求 3x3 SE(2) 变换逆。"""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"SE2 transform must have shape [3, 3], got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("SE2 transform must contain only finite values")
    inverse = np.eye(3, dtype=np.float64)
    rotation = matrix[:2, :2]
    translation = matrix[:2, 2]
    inverse[:2, :2] = rotation.T
    inverse[:2, 2] = -rotation.T @ translation
    return inverse


def compose_se2(left: object, right: object) -> Transform2D:
    """组合两个 SE(2) 变换。"""

    left_matrix = np.asarray(left, dtype=np.float64)
    right_matrix = np.asarray(right, dtype=np.float64)
    if left_matrix.shape != (3, 3) or right_matrix.shape != (3, 3):
        raise ValueError("SE2 transforms must have shape [3, 3]")
    return left_matrix @ right_matrix


def extract_xyyaw(transform: object) -> tuple[float, float, float]:
    """从 3x3 SE(2) 矩阵提取 `x, y, yaw`."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"SE2 transform must have shape [3, 3], got {matrix.shape}")
    yaw = atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return float(matrix[0, 2]), float(matrix[1, 2]), normalize_yaw(yaw)
