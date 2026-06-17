#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量级 3DoF PoseNet 模型。

本模块实现 Week 6 的最小学习式里程计模型：把相邻两帧 BEV tensor 沿 channel
维拼接为 `[B, 2C, H, W]`，通过一个小型 CNN 回归归一化后的 `dx, dy, yaw`。
模型只负责 normalized pose 输出；反归一化和物理指标计算放在 `models.losses`。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class PoseNet3DoF(nn.Module):
    """用于相邻帧 BEV 的小型 CNN 位姿回归网络。

    参数:
        in_channels: 输入通道数，应等于单帧 BEV 通道数的 2 倍。
        hidden_channels: 第一层卷积通道数，后续层按倍数扩展。
        dropout: 全连接层前的 dropout 概率，默认为 0。

    输入:
        `float32[B, in_channels, H, W]`，通常来自 `stack_bev_pair`。
    返回:
        `float32[B, 3]`，顺序为 normalized `dx, dy, yaw`。
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = float(dropout)
        self.features = nn.Sequential(
            _conv_block(in_channels, hidden_channels, kernel_size=5),
            _conv_block(hidden_channels, hidden_channels * 2),
            _conv_block(hidden_channels * 2, hidden_channels * 4),
            _conv_block(hidden_channels * 4, hidden_channels * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels * 4, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        """执行前向推理并返回 normalized 3DoF 位姿。"""

        _validate_model_input(x, expected_channels=self.in_channels)
        return self.head(self.features(x))


def stack_bev_pair(bev_prev: Tensor, bev_curr: Tensor) -> Tensor:
    """沿 channel 维拼接相邻两帧 BEV。

    参数:
        bev_prev: `float[B, C, H, W]` 前一帧 BEV。
        bev_curr: `float[B, C, H, W]` 当前帧 BEV。
    返回:
        `float[B, 2C, H, W]`，可直接送入 `PoseNet3DoF`。
    """

    _validate_bev_batch(bev_prev, name="bev_prev")
    _validate_bev_batch(bev_curr, name="bev_curr")
    if bev_prev.shape != bev_curr.shape:
        raise ValueError(f"BEV pair shapes must match, got {bev_prev.shape} and {bev_curr.shape}")
    return torch.cat([bev_prev, bev_curr], dim=1)


def build_posenet_from_config(config: dict[str, Any]) -> PoseNet3DoF:
    """从训练配置构造 `PoseNet3DoF`。

    `in_channels` 默认由 `len(config['bev']['channels']) * 2` 得到，避免把 KITTI
    当前 4 通道 BEV 写死在模型里。
    """

    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    bev_section = config.get("bev", {})
    if not isinstance(bev_section, dict):
        raise ValueError("config.bev must be a mapping")
    channels = bev_section.get("channels", ("density", "max_height", "mean_height", "intensity"))
    if not isinstance(channels, (list, tuple)) or not channels:
        raise ValueError("config.bev.channels must be a non-empty sequence")

    model_section = config.get("model", {})
    if not isinstance(model_section, dict):
        raise ValueError("config.model must be a mapping when provided")
    return PoseNet3DoF(
        in_channels=int(model_section.get("in_channels", len(channels) * 2)),
        hidden_channels=int(model_section.get("hidden_channels", 32)),
        dropout=float(model_section.get("dropout", 0.0)),
    )


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
) -> nn.Sequential:
    """构造 stride-2 卷积块，逐步压缩 BEV 空间分辨率。"""

    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _validate_model_input(x: Tensor, *, expected_channels: int) -> None:
    """校验模型输入 tensor。"""

    _validate_bev_batch(x, name="x")
    if int(x.shape[1]) != expected_channels:
        raise ValueError(f"x must have {expected_channels} channels, got {x.shape[1]}")


def _validate_bev_batch(tensor: Tensor, *, name: str) -> None:
    """校验 BEV batch 为有限四维 floating tensor。"""

    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be a floating point tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
