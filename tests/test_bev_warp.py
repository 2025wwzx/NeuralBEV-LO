#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 4 BEV warp 测试。"""

from __future__ import annotations

from math import pi

import numpy as np
import torch

from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.bev.warp import warp_bev
from neuralbev_lo.geometry.se2 import invert_se2, se2_from_xyyaw


def _tiny_config() -> BevGridConfig:
    """构造便于检查像素移动方向的小 BEV 配置。"""

    return BevGridConfig(
        x_range_m=(0.0, 8.0),
        y_range_m=(-4.0, 4.0),
        resolution_m=1.0,
        channels=("density",),
    )


def _symmetric_config() -> BevGridConfig:
    """构造围绕物理原点对称的 BEV 配置，用于旋转测试。"""

    return BevGridConfig(
        x_range_m=(-4.0, 4.0),
        y_range_m=(-4.0, 4.0),
        resolution_m=1.0,
        channels=("density",),
    )


def _argmax_cell(bev: torch.Tensor) -> tuple[int, int]:
    """返回单通道 BEV 最大值所在的 `(row, col)`。"""

    max_index = torch.nonzero(bev[0] == bev[0].max(), as_tuple=False)[0]
    return int(max_index[0]), int(max_index[1])


def test_warp_identity_keeps_bev_values() -> None:
    """identity warp 应保持 BEV 数值不变。"""

    config = _tiny_config()
    bev = np.zeros((1, 8, 8), dtype=np.float32)
    bev[0, 3, 4] = 1.0

    warped = warp_bev(bev, se2_from_xyyaw(0.0, 0.0, 0.0), config)

    torch.testing.assert_close(warped, torch.as_tensor(bev))


def test_warp_translation_moves_density_cell() -> None:
    """1m 前向平移在 1m 分辨率下应产生约 1 个 row 的移动。"""

    config = _tiny_config()
    bev = np.zeros((1, 8, 8), dtype=np.float32)
    bev[0, 3, 4] = 1.0

    warped = warp_bev(bev, se2_from_xyyaw(1.0, 0.0, 0.0), config)

    assert _argmax_cell(warped) == (2, 4)


def test_warp_yaw_rotates_around_physical_origin() -> None:
    """yaw warp 应围绕 BEV 物理原点旋转，而不是围绕图像中心近似旋转。"""

    config = _symmetric_config()
    bev = np.zeros((1, 8, 8), dtype=np.float32)
    bev[0, 5, 4] = 1.0

    warped = warp_bev(bev, se2_from_xyyaw(0.0, 0.0, pi / 2.0), config)

    assert _argmax_cell(warped) == (4, 2)


def test_warp_inverse_transform_recovers_translated_cell() -> None:
    """对同一 BEV 先 warp 再用逆变换 warp，应恢复原始 cell。"""

    config = _tiny_config()
    bev = np.zeros((1, 8, 8), dtype=np.float32)
    bev[0, 3, 4] = 1.0
    transform = se2_from_xyyaw(1.0, 0.0, 0.0)

    moved = warp_bev(bev, transform, config)
    recovered = warp_bev(moved, invert_se2(transform), config)

    assert _argmax_cell(recovered) == (3, 4)
    assert float(recovered.max()) > 0.99


def test_warp_large_translation_pads_with_zeros() -> None:
    """移出边界的内容应被 zero padding 清空。"""

    config = _tiny_config()
    bev = np.zeros((1, 8, 8), dtype=np.float32)
    bev[0, 0, 0] = 1.0

    warped = warp_bev(bev, se2_from_xyyaw(20.0, 0.0, 0.0), config)

    assert float(warped.max()) == 0.0


def test_warp_ten_meter_forward_translation_matches_week4_gate() -> None:
    """10m 前向平移在 0.25m 分辨率下应对应约 40 个 row。"""

    config = BevGridConfig(
        x_range_m=(0.0, 30.0),
        y_range_m=(-10.0, 10.0),
        resolution_m=0.25,
        channels=("density",),
    )
    bev = np.zeros((1, 120, 80), dtype=np.float32)
    bev[0, 80, 40] = 1.0

    warped = warp_bev(bev, se2_from_xyyaw(10.0, 0.0, 0.0), config)
    max_row, max_col = _argmax_cell(warped)

    assert abs(max_row - 40) <= 1
    assert abs(max_col - 40) <= 1
