#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 4 BEV memory 测试。"""

from __future__ import annotations

import numpy as np
import torch

from neuralbev_lo.bev.memory import BevMemoryConfig, initialize_memory, update_memory
from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.geometry.se2 import se2_from_xyyaw


def _tiny_config() -> BevGridConfig:
    """构造小尺寸 BEV 网格。"""

    return BevGridConfig(
        x_range_m=(0.0, 8.0),
        y_range_m=(-4.0, 4.0),
        resolution_m=1.0,
        channels=("density",),
    )


def test_initialize_memory_clones_input_bev() -> None:
    """初始化 memory 时应复制输入 BEV。"""

    bev = np.zeros((1, 8, 8), dtype=np.float32)
    bev[0, 3, 4] = 1.0

    memory = initialize_memory(bev)

    assert memory.shape == (1, 8, 8)
    torch.testing.assert_close(memory, torch.as_tensor(bev))


def test_update_memory_decay_blends_warped_history_with_current_frame() -> None:
    """decay 策略应融合 warped memory 与 current frame。"""

    grid = _tiny_config()
    memory = torch.zeros((1, 8, 8), dtype=torch.float32)
    memory[0, 3, 4] = 1.0
    current = torch.zeros((1, 8, 8), dtype=torch.float32)
    current[0, 2, 4] = 1.0

    updated = update_memory(
        memory,
        current,
        se2_from_xyyaw(1.0, 0.0, 0.0),
        grid,
        BevMemoryConfig(policy="decay", alpha=0.5),
    )

    assert updated.shape == (1, 8, 8)
    assert float(updated.max()) > 0.0


def test_update_memory_fifo_returns_current_frame() -> None:
    """fifo 策略应直接用当前帧覆盖 memory。"""

    grid = _tiny_config()
    memory = torch.zeros((1, 8, 8), dtype=torch.float32)
    memory[0, 3, 4] = 1.0
    current = torch.zeros((1, 8, 8), dtype=torch.float32)
    current[0, 2, 4] = 1.0

    updated = update_memory(
        memory,
        current,
        se2_from_xyyaw(1.0, 0.0, 0.0),
        grid,
        BevMemoryConfig(policy="fifo", alpha=0.5),
    )

    torch.testing.assert_close(updated, current)
