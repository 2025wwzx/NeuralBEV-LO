#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 3 BEV 渲染测试。"""

from __future__ import annotations

import numpy as np

from neuralbev_lo.viz.render_bev import bev_to_uint8_image, save_bev_comparison, save_bev_image


def test_bev_to_uint8_image_returns_rgb_image() -> None:
    """BEV 张量应能转为 RGB uint8 图像。"""

    bev = np.zeros((4, 4, 4), dtype=np.float32)
    bev[0, 1, 1] = 1.0
    bev[1, 2, 2] = 0.5

    image = bev_to_uint8_image(bev)

    assert image.shape == (4, 4, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0


def test_save_bev_image_writes_png(tmp_path) -> None:
    """BEV 图像应保存为可检查的 PNG 文件。"""

    bev = np.zeros((4, 4, 4), dtype=np.float32)
    bev[0, 1, 1] = 1.0
    output_path = tmp_path / "bev.png"

    saved_path = save_bev_image(bev, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_bev_comparison_writes_png(tmp_path) -> None:
    """BEV 对比图应能保存多个同尺寸面板。"""

    bev = np.zeros((4, 4, 4), dtype=np.float32)
    bev[0, 1, 1] = 1.0
    output_path = tmp_path / "comparison.png"

    saved_path = save_bev_comparison([("current", bev), ("memory", bev)], output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
