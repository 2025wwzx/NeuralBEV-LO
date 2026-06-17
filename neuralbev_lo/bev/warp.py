#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BEV warp 工具。

本模块按 BEV 的物理坐标生成 `grid_sample` 采样网格，而不是直接套用图像
`affine_grid`。这样可以保持项目约定：LiDAR x 映射到 BEV row，LiDAR y 映射到
BEV column，并且旋转围绕 BEV 物理原点生效。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.geometry.se2 import extract_xyyaw, se2_from_xyyaw


@dataclass(frozen=True)
class BevWarpConfig:
    """BEV warp 所需的几何配置。"""

    grid: BevGridConfig


def se2_to_bev_transform(
    transform: object, config: BevGridConfig, *, device: torch.device | None = None
) -> torch.Tensor:
    """把 LiDAR SE(2) 变换转换为 BEV row/col 像素空间矩阵。

    参数:
        transform: 3x3 SE(2) 或 4x4 齐次矩阵，表示 source 坐标系到 target
            坐标系采样时使用的 `p_source = T @ p_target`。
        config: BEV 网格配置，用于把米转换成 cell。
        device: 可选 torch device。
    返回:
        3x3 矩阵，坐标顺序为 `[col, row, 1]`。
    """

    matrix = _as_se2_matrix(transform)
    dx, dy, yaw = extract_xyyaw(matrix)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    image_matrix = np.array(
        [
            [c, s, dy / config.resolution_m],
            [-s, c, dx / config.resolution_m],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return torch.as_tensor(image_matrix, dtype=torch.float32, device=device)


def warp_bev(bev: object, transform: object, config: BevGridConfig) -> torch.Tensor:
    """对单个 BEV 张量执行 SE(2) warp。

    参数:
        bev: `[C, H, W]` 或 `[1, C, H, W]` 的 BEV 张量。
        transform: `p_source = T @ p_target` 的 2D 平面变换。GT memory 中通常传入
            `T_prev_curr`，即当前帧在上一帧坐标系下的位置。
        config: BEV 网格配置。
    返回:
        变换后的 BEV 张量。若输入为 `[C,H,W]`，返回 `[C,H,W]`；若输入带 batch，
        返回 `[B,C,H,W]`。
    """

    tensor = torch.as_tensor(bev, dtype=torch.float32)
    squeeze_batch = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze_batch = True
    if tensor.ndim != 4:
        raise ValueError(f"BEV tensor must have shape [C, H, W] or [B, C, H, W], got {tensor.shape}")

    batch, _, height, width = tensor.shape
    grid = _build_sampling_grid(transform, config, height=height, width=width, device=tensor.device)
    grid = grid.unsqueeze(0).expand(batch, height, width, 2)
    warped = F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped[0] if squeeze_batch else warped


def _as_se2_matrix(transform: object) -> np.ndarray:
    """把 3x3/4x4 输入统一为 3x3 SE(2) 矩阵。"""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (3, 3):
        if not np.all(np.isfinite(matrix)):
            raise ValueError("SE2 transform must contain only finite values")
        return matrix
    if matrix.shape == (4, 4):
        if not np.all(np.isfinite(matrix)):
            raise ValueError("SE3 transform must contain only finite values")
        x = float(matrix[0, 3])
        y = float(matrix[1, 3])
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        return se2_from_xyyaw(x, y, yaw)
    raise ValueError(f"transform must have shape [3, 3] or [4, 4], got {matrix.shape}")


def _build_sampling_grid(
    transform: object,
    config: BevGridConfig,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """构建 `grid_sample` 所需的 `[H, W, 2]` 归一化采样网格。"""

    matrix = torch.as_tensor(_as_se2_matrix(transform), dtype=torch.float32, device=device)
    rows = torch.arange(height, dtype=torch.float32, device=device)
    cols = torch.arange(width, dtype=torch.float32, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

    target_x = float(config.x_range_m[0]) + (row_grid + 0.5) * float(config.resolution_m)
    target_y = float(config.y_range_m[0]) + (col_grid + 0.5) * float(config.resolution_m)

    source_x = matrix[0, 0] * target_x + matrix[0, 1] * target_y + matrix[0, 2]
    source_y = matrix[1, 0] * target_x + matrix[1, 1] * target_y + matrix[1, 2]

    norm_col = 2.0 * (source_y - float(config.y_range_m[0])) / (
        float(config.resolution_m) * max(width, 1)
    ) - 1.0
    norm_row = 2.0 * (source_x - float(config.x_range_m[0])) / (
        float(config.resolution_m) * max(height, 1)
    ) - 1.0
    return torch.stack([norm_col, norm_row], dim=-1)
