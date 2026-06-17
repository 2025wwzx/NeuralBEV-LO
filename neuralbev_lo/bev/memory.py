#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Temporal BEV memory 工具。

实现最小可用的 GT-pose / learned-pose 记忆更新逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.bev.warp import warp_bev


@dataclass(frozen=True)
class BevMemoryConfig:
    """BEV memory 更新策略配置。"""

    policy: str = "decay"
    alpha: float = 0.9

    def __post_init__(self) -> None:
        if self.policy not in {"decay", "fifo"}:
            raise ValueError(f"unsupported memory policy: {self.policy}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")


def initialize_memory(bev: object) -> torch.Tensor:
    """用单帧 BEV 初始化 memory。"""

    tensor = torch.as_tensor(bev, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"memory must start from [C, H, W], got {tensor.shape}")
    return tensor.clone()


def update_memory(
    memory: object,
    current_bev: object,
    relative_transform: object,
    grid: BevGridConfig,
    config: BevMemoryConfig | None = None,
) -> torch.Tensor:
    """根据相邻帧位姿更新 BEV memory。

    参数:
        memory: 上一时刻记忆 BEV。
        current_bev: 当前帧 BEV。
        relative_transform: 当前帧相对上一帧的平面变换。
        grid: BEV 网格配置。
        config: 更新策略，默认 decay。
    """

    memory_tensor = torch.as_tensor(memory, dtype=torch.float32)
    current_tensor = torch.as_tensor(current_bev, dtype=torch.float32)
    if memory_tensor.shape != current_tensor.shape:
        raise ValueError(
            f"memory and current_bev must have the same shape, got {memory_tensor.shape} and {current_tensor.shape}"
        )
    if memory_tensor.ndim != 3:
        raise ValueError(f"BEV memory must have shape [C, H, W], got {memory_tensor.shape}")

    policy = config or BevMemoryConfig()
    warped_memory = warp_bev(memory_tensor, relative_transform, grid)
    if policy.policy == "fifo":
        return current_tensor.clone()
    return policy.alpha * warped_memory + (1.0 - policy.alpha) * current_tensor
