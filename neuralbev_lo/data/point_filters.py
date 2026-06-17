#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""点云过滤工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PointFilterConfig:
    """点云过滤配置。

    参数:
        z_range_m: 可选高度范围，按闭区间 `[min_z, max_z]` 保留点。
        distance_range_m: 可选水平距离范围，按闭区间 `[min_r, max_r]` 保留点。
    """

    z_range_m: tuple[float, float] | None = None
    distance_range_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """校验范围配置。"""

        if self.z_range_m is not None:
            _validate_range(self.z_range_m, "z_range_m", allow_zero_lower=True)
        if self.distance_range_m is not None:
            _validate_range(self.distance_range_m, "distance_range_m", allow_zero_lower=True)
            if self.distance_range_m[0] < 0.0:
                raise ValueError("distance_range_m lower bound must be non-negative")


def filter_point_cloud_for_bev(
    points: object,
    config: PointFilterConfig | None = None,
) -> NDArray[np.float32]:
    """按 Week 10 robustness 配置过滤 LiDAR 点云。

    参数:
        points: `float[N, 4]` 点云，列顺序为 `x, y, z, intensity`。
        config: 可选高度和水平距离过滤配置；为空时仅做输入校验并返回副本。
    返回:
        过滤后的 `float32[M, 4]` 点云副本。
    异常:
        输入形状错误、含非有限数值、范围非法时抛出 `ValueError`。
    """

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"point cloud must have shape [N, 4], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("point cloud must contain only finite values")
    if array.shape[0] == 0:
        return array.copy()

    filter_config = config or PointFilterConfig()
    mask = np.ones(array.shape[0], dtype=bool)
    if filter_config.z_range_m is not None:
        z_min, z_max = filter_config.z_range_m
        mask &= (array[:, 2] >= z_min) & (array[:, 2] <= z_max)
    if filter_config.distance_range_m is not None:
        distance_min, distance_max = filter_config.distance_range_m
        distances = np.linalg.norm(array[:, :2], axis=1)
        mask &= (distances >= distance_min) & (distances <= distance_max)
    return array[mask].copy()


def _validate_range(
    range_m: tuple[float, float],
    name: str,
    *,
    allow_zero_lower: bool = False,
) -> None:
    """校验二元范围配置。"""

    if len(range_m) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = float(range_m[0]), float(range_m[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} must contain finite values")
    if lower > upper or (lower == upper and not allow_zero_lower):
        raise ValueError(f"{name} lower bound must be smaller than upper bound")
