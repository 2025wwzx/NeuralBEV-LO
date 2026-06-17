#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 4 BEV warp 测试。"""

from __future__ import annotations

import numpy as np
import torch

from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.bev.warp import warp_bev
from neuralbev_lo.geometry.se2 import se2_from_xyyaw


def _tiny_config() -> BevGridConfig:
    """构造便于检查像素移动方向的小 BEV 配置。"""

    return BevGridConfig(
        x_range_m=(0.0, 8.0),
        y_range_m=(-4.0, 4.0),
        resolution_m=1.0,
        channels=("density",),
    )


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
    max_index = torch.nonzero(warped[0] == warped[0].max(), as_tuple=False)[0]

    assert tuple(int(value) for value in max_index.tolist()) == (2, 4)


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
    max_index = torch.nonzero(warped[0] == warped[0].max(), as_tuple=False)[0]

    assert abs(int(max_index[0]) - 40) <= 1
    assert abs(int(max_index[1]) - 40) <= 1
