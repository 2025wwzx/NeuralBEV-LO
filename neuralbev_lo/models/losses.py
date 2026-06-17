#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PoseNet 训练损失与反归一化指标。

网络输出和训练 target 都使用 train split mean/std 归一化后的 `dx, dy, yaw`。
本模块负责 normalized 空间里的 weighted SmoothL1 loss，以及反归一化到物理单位
后的平移误差和 yaw 误差日志指标。
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WeightedSmoothL1Loss(nn.Module):
    """三维 pose 目标的加权 SmoothL1 损失。

    参数:
        weights: 长度为 3 的权重，顺序为 `dx, dy, yaw`。
        beta: SmoothL1 的二次区间宽度，传给 `torch.nn.functional.smooth_l1_loss`。
    """

    def __init__(self, weights: Iterable[float] = (1.0, 1.0, 1.0), *, beta: float = 1.0) -> None:
        super().__init__()
        weight_tensor = _as_vector3(weights, name="weights")
        if torch.any(weight_tensor <= 0):
            raise ValueError("weights must contain positive values")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.register_buffer("weights", weight_tensor)
        self.beta = float(beta)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """返回加权后的标量 loss。"""

        return weighted_smooth_l1_loss(prediction, target, weights=self.weights, beta=self.beta)


def weighted_smooth_l1_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    weights: Iterable[float] | Tensor = (1.0, 1.0, 1.0),
    beta: float = 1.0,
) -> Tensor:
    """计算 normalized 3DoF pose 的加权 SmoothL1 loss。

    参数:
        prediction: `float[B, 3]` 模型输出。
        target: `float[B, 3]` normalized 训练目标。
        weights: 三个目标维度的正权重。
        beta: SmoothL1 beta。
    返回:
        标量 tensor，保留梯度。
    """

    _validate_pose_batch(prediction, name="prediction")
    _validate_pose_batch(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes must match, got {prediction.shape} and {target.shape}")
    if beta <= 0:
        raise ValueError("beta must be positive")

    weight_tensor = _as_vector3(weights, name="weights").to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (element_loss * weight_tensor.view(1, 3)).mean()


def denormalize_pose_batch(normalized_pose: Tensor, mean: Tensor | Iterable[float], std: Tensor | Iterable[float]) -> Tensor:
    """把 normalized `[dx, dy, yaw]` 反归一化到物理单位。

    参数:
        normalized_pose: `float[B, 3]`。
        mean/std: 长度为 3 的 train split 统计量。
    返回:
        与输入 shape 相同的物理单位 pose batch。
    """

    _validate_pose_batch(normalized_pose, name="normalized_pose")
    mean_tensor = _as_vector3(mean, name="mean").to(device=normalized_pose.device, dtype=normalized_pose.dtype)
    std_tensor = _as_vector3(std, name="std").to(device=normalized_pose.device, dtype=normalized_pose.dtype)
    if torch.any(std_tensor <= 0):
        raise ValueError("std must contain positive values")
    return normalized_pose * std_tensor.view(1, 3) + mean_tensor.view(1, 3)


def pose_error_metrics(
    prediction: Tensor,
    target: Tensor,
    mean: Tensor | Iterable[float],
    std: Tensor | Iterable[float],
) -> dict[str, float]:
    """计算反归一化后的平移和 yaw 平均绝对误差。

    返回字段:
        translation_error_m: `sqrt(dx^2 + dy^2)` 的 batch 均值。
        yaw_error_rad: `abs(yaw)` 的 batch 均值。
    """

    pred_pose = denormalize_pose_batch(prediction.detach(), mean, std)
    target_pose = denormalize_pose_batch(target.detach(), mean, std)
    diff = pred_pose - target_pose
    translation_error = torch.linalg.norm(diff[:, :2], dim=1).mean()
    yaw_error = diff[:, 2].abs().mean()
    return {
        "translation_error_m": float(translation_error.detach().cpu().item()),
        "yaw_error_rad": float(yaw_error.detach().cpu().item()),
    }


def _validate_pose_batch(tensor: Tensor, *, name: str) -> None:
    """校验 pose batch 为有限 `[B, 3]` floating tensor。"""

    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B, 3], got {tuple(tensor.shape)}")
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} batch dimension must be positive")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be a floating point tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def _as_vector3(values: Tensor | Iterable[float], *, name: str) -> Tensor:
    """把输入转换为有限 `float32[3]` tensor。"""

    if isinstance(values, Tensor):
        tensor = values.detach().clone().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.shape != torch.Size([3]):
        raise ValueError(f"{name} must contain exactly 3 values")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor
