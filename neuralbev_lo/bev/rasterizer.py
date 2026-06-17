#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""点云到 BEV 张量的基础栅格化工具。

本模块实现 Week 3 的最小 BEV rasterizer：范围过滤、x/y 到 row/column 的
确定性索引，以及 density、max_height、mean_height、intensity 四个通道。
坐标约定与 `docs/coordinate_system.md` 一致：x 向前、y 向左，BEV 张量形状为
`[C, H, W]`，其中 H 对应 x bins，W 对应 y bins。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_CHANNELS: tuple[str, ...] = ("density", "max_height", "mean_height", "intensity")


@dataclass(frozen=True)
class BevGridConfig:
    """BEV 栅格配置。

    参数:
        x_range_m: x 前向范围，左闭右开。
        y_range_m: y 左向范围，左闭右开。
        z_range_m: 可选高度过滤范围，左闭右闭。
        resolution_m: BEV 分辨率，单位米。
        channels: 输出通道顺序。
        density_normalization: density 的线性归一化分母。
    """

    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    z_range_m: tuple[float, float] | None = None
    resolution_m: float = 0.5
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    density_normalization: float = 16.0

    def __post_init__(self) -> None:
        """校验配置字段。"""

        _validate_range(self.x_range_m, "x_range_m")
        _validate_range(self.y_range_m, "y_range_m")
        if self.z_range_m is not None:
            _validate_range(self.z_range_m, "z_range_m")
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if self.density_normalization <= 0:
            raise ValueError("density_normalization must be positive")
        unsupported = set(self.channels) - set(DEFAULT_CHANNELS)
        if unsupported:
            raise ValueError(f"unsupported BEV channels: {sorted(unsupported)}")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """返回 `(height, width)`，即 `(x_bins, y_bins)`。"""

        x_bins = _num_bins(self.x_range_m, self.resolution_m)
        y_bins = _num_bins(self.y_range_m, self.resolution_m)
        return x_bins, y_bins

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        """返回输出张量形状 `[C, H, W]`。"""

        height, width = self.grid_shape
        return len(self.channels), height, width


def _validate_range(range_m: tuple[float, float], name: str) -> None:
    """校验范围是有限递增二元组。"""

    if len(range_m) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = float(range_m[0]), float(range_m[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} must contain finite values")
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound")


def _num_bins(range_m: tuple[float, float], resolution_m: float) -> int:
    """计算栅格数量。"""

    size = (float(range_m[1]) - float(range_m[0])) / resolution_m
    rounded = int(round(size))
    if not np.isclose(size, rounded, atol=1e-6):
        raise ValueError(f"range {range_m} is not divisible by resolution {resolution_m}")
    return rounded


def bev_config_from_mapping(config: dict[str, object]) -> BevGridConfig:
    """从配置字典构建 `BevGridConfig`。

    参数:
        config: 通常来自 YAML 中的 `bev` 小节。

    返回:
        已校验的 BEV 配置。
    """

    return BevGridConfig(
        x_range_m=_tuple2(config.get("x_range_m", (0.0, 70.0)), "x_range_m"),
        y_range_m=_tuple2(config.get("y_range_m", (-35.0, 35.0)), "y_range_m"),
        z_range_m=_optional_tuple2(config.get("z_range_m"), "z_range_m"),
        resolution_m=float(config.get("resolution_m", 0.5)),
        channels=tuple(str(item) for item in config.get("channels", DEFAULT_CHANNELS)),  # type: ignore[arg-type]
        density_normalization=float(config.get("density_normalization", 16.0)),
    )


def _tuple2(value: object, name: str) -> tuple[float, float]:
    """将配置值转为二元浮点 tuple。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value sequence")
    return float(value[0]), float(value[1])


def _optional_tuple2(value: object, name: str) -> tuple[float, float] | None:
    """将可选配置值转为二元浮点 tuple。"""

    if value is None:
        return None
    return _tuple2(value, name)


def validate_point_cloud(points: object) -> NDArray[np.float32]:
    """校验点云为有限 `float32[N, 4]`。

    参数:
        points: 点云输入，列顺序为 `x, y, z, intensity`。

    返回:
        `float32[N, 4]` 数组。
    """

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"point cloud must have shape [N, 4], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("point cloud must contain only finite values")
    return array


def filter_points_by_range(points: object, config: BevGridConfig) -> NDArray[np.float32]:
    """按 BEV x/y 和可选 z 范围过滤点云。

    x/y 范围为左闭右开，z 范围为左闭右闭。
    """

    array = validate_point_cloud(points)
    if array.shape[0] == 0:
        return array.copy()

    mask = (
        (array[:, 0] >= config.x_range_m[0])
        & (array[:, 0] < config.x_range_m[1])
        & (array[:, 1] >= config.y_range_m[0])
        & (array[:, 1] < config.y_range_m[1])
    )
    if config.z_range_m is not None:
        mask &= (array[:, 2] >= config.z_range_m[0]) & (array[:, 2] <= config.z_range_m[1])
    return array[mask].copy()


def points_to_bev_indices(
    points: object, config: BevGridConfig
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """将已过滤点云映射到 BEV row/column 索引。"""

    array = validate_point_cloud(points)
    rows = np.floor((array[:, 0] - config.x_range_m[0]) / config.resolution_m).astype(np.int64)
    cols = np.floor((array[:, 1] - config.y_range_m[0]) / config.resolution_m).astype(np.int64)
    height, width = config.grid_shape
    if np.any(rows < 0) or np.any(rows >= height) or np.any(cols < 0) or np.any(cols >= width):
        raise ValueError("points_to_bev_indices received points outside BEV range")
    return rows, cols


def rasterize_point_cloud(points: object, config: BevGridConfig) -> NDArray[np.float32]:
    """将点云栅格化为 BEV 张量。

    通道定义:
        density: `min(1, count / density_normalization)`。
        max_height: cell 内最大 z，按最大绝对高度归一化。
        mean_height: cell 内平均 z，按最大绝对高度归一化。
        intensity: cell 内平均 intensity，裁剪到 `[0, 1]`。

    返回:
        `float32[C, H, W]` BEV 张量。
    """

    filtered = filter_points_by_range(points, config)
    bev = np.zeros(config.tensor_shape, dtype=np.float32)
    if filtered.shape[0] == 0:
        return bev

    rows, cols = points_to_bev_indices(filtered, config)
    height_scale = _height_scale(config)
    counts = np.zeros(config.grid_shape, dtype=np.float32)
    z_sum = np.zeros(config.grid_shape, dtype=np.float32)
    z_max = np.full(config.grid_shape, -np.inf, dtype=np.float32)
    intensity_sum = np.zeros(config.grid_shape, dtype=np.float32)

    for point, row, col in zip(filtered, rows, cols, strict=True):
        z_value = float(point[2])
        intensity_value = float(point[3])
        counts[row, col] += 1.0
        z_sum[row, col] += z_value
        z_max[row, col] = max(z_max[row, col], z_value)
        intensity_sum[row, col] += intensity_value

    non_empty = counts > 0
    channel_data: dict[str, NDArray[np.float32]] = {
        "density": np.minimum(1.0, counts / config.density_normalization).astype(np.float32),
        "max_height": np.zeros(config.grid_shape, dtype=np.float32),
        "mean_height": np.zeros(config.grid_shape, dtype=np.float32),
        "intensity": np.zeros(config.grid_shape, dtype=np.float32),
    }
    channel_data["max_height"][non_empty] = np.clip(z_max[non_empty] / height_scale, -1.0, 1.0)
    channel_data["mean_height"][non_empty] = np.clip(
        (z_sum[non_empty] / counts[non_empty]) / height_scale,
        -1.0,
        1.0,
    )
    channel_data["intensity"][non_empty] = np.clip(
        intensity_sum[non_empty] / counts[non_empty],
        0.0,
        1.0,
    )

    for channel_index, channel_name in enumerate(config.channels):
        bev[channel_index] = channel_data[channel_name]
    return bev


def _height_scale(config: BevGridConfig) -> float:
    """计算高度归一化分母。"""

    if config.z_range_m is None:
        return 5.0
    return max(abs(float(config.z_range_m[0])), abs(float(config.z_range_m[1])), 1.0)
