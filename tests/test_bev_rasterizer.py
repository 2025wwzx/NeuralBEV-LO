#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 3 BEV rasterizer 测试。"""

from __future__ import annotations

import numpy as np
import pytest

from neuralbev_lo.bev.rasterizer import (
    BevGridConfig,
    filter_points_by_range,
    points_to_bev_indices,
    rasterize_point_cloud,
)


def test_grid_shape_is_deterministic_from_config() -> None:
    """BEV 输出形状应完全由配置决定。"""

    config = BevGridConfig(
        x_range_m=(0.0, 4.0),
        y_range_m=(-2.0, 2.0),
        resolution_m=1.0,
        channels=("density", "max_height", "mean_height", "intensity"),
    )

    assert config.grid_shape == (4, 4)
    assert config.tensor_shape == (4, 4, 4)


def test_points_to_bev_indices_use_x_rows_and_y_columns() -> None:
    """x 应映射到 row，y 应映射到 column，边界为左闭右开。"""

    config = BevGridConfig(x_range_m=(0.0, 4.0), y_range_m=(-2.0, 2.0), resolution_m=1.0)
    points = np.array(
        [
            [0.0, -2.0, 0.0, 0.1],
            [3.999, 1.999, 0.0, 0.2],
            [4.0, 0.0, 0.0, 0.3],
            [1.0, 2.0, 0.0, 0.4],
        ],
        dtype=np.float32,
    )

    filtered = filter_points_by_range(points, config)
    rows, cols = points_to_bev_indices(filtered, config)

    assert filtered.shape == (2, 4)
    np.testing.assert_array_equal(rows, np.array([0, 3]))
    np.testing.assert_array_equal(cols, np.array([0, 3]))


def test_rasterizer_aggregates_duplicate_cells() -> None:
    """重复 cell 应聚合 density、max height、mean height 和 mean intensity。"""

    config = BevGridConfig(
        x_range_m=(0.0, 2.0),
        y_range_m=(0.0, 2.0),
        z_range_m=(-2.0, 5.0),
        resolution_m=1.0,
        density_normalization=2.0,
    )
    points = np.array(
        [
            [0.2, 0.2, 1.0, 0.2],
            [0.8, 0.8, 3.0, 0.6],
            [1.2, 1.2, -1.0, 0.4],
        ],
        dtype=np.float32,
    )

    bev = rasterize_point_cloud(points, config)

    density, max_height, mean_height, intensity = bev
    assert density[0, 0] == pytest.approx(1.0)
    assert max_height[0, 0] == pytest.approx(3.0 / 5.0)
    assert mean_height[0, 0] == pytest.approx(2.0 / 5.0)
    assert intensity[0, 0] == pytest.approx(0.4)
    assert density[1, 1] == pytest.approx(0.5)
    assert max_height[1, 1] == pytest.approx(-1.0 / 5.0)


def test_rasterizer_handles_empty_point_cloud() -> None:
    """空点云应返回确定形状的全零 BEV。"""

    config = BevGridConfig(x_range_m=(0.0, 4.0), y_range_m=(-2.0, 2.0), resolution_m=1.0)
    points = np.empty((0, 4), dtype=np.float32)

    bev = rasterize_point_cloud(points, config)

    assert bev.shape == config.tensor_shape
    assert np.count_nonzero(bev) == 0


def test_rasterizer_rejects_bad_point_shape() -> None:
    """点云输入必须是 `[N, 4]`。"""

    config = BevGridConfig(x_range_m=(0.0, 4.0), y_range_m=(-2.0, 2.0), resolution_m=1.0)

    with pytest.raises(ValueError, match="\\[N, 4\\]"):
        rasterize_point_cloud(np.ones((3, 3), dtype=np.float32), config)
