#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PoseNet 训练循环与合成 smoke 数据。

本模块提供 Week 6 的最小训练闭环：合成 pair dataset、DataLoader 构造、单 epoch
训练、验证指标，以及可在 CPU 上快速运行的 tiny overfit smoke。真实 KITTI 训练 CLI
复用同一套函数，避免脚本里堆业务逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from neuralbev_lo.models.losses import pose_error_metrics, weighted_smooth_l1_loss
from neuralbev_lo.models.posenet import PoseNet3DoF, stack_bev_pair
from neuralbev_lo.utils.seed import set_global_seed


@dataclass(frozen=True)
class EpochResult:
    """训练或验证 epoch 的聚合结果。"""

    loss: float
    batches: int
    translation_error_m: float | None = None
    yaw_error_rad: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        """转换为日志友好的字典。"""

        return {
            "loss": self.loss,
            "batches": self.batches,
            "translation_error_m": self.translation_error_m,
            "yaw_error_rad": self.yaw_error_rad,
        }


class SyntheticPairDataset(Dataset):
    """用于 CPU smoke 和 overfit 的确定性 BEV pair 数据集。

    每个样本包含两帧合成 BEV。当前帧在前三个 channel 中编码 normalized pose target，
    因此小 CNN 应能在少量 step 内降低 loss。metadata 保持与 KITTI Pair Dataset 类似，
    方便 DataLoader collate 和训练日志统一。
    """

    def __init__(
        self,
        *,
        num_samples: int = 16,
        channels: int = 4,
        height: int = 32,
        width: int = 32,
        seed: int = 20260608,
    ) -> None:
        """构造确定性合成数据。

        参数:
            num_samples: 样本数量。
            channels: 单帧 BEV channel 数，至少为 3。
            height/width: BEV 空间尺寸。
            seed: NumPy 随机种子，用于生成背景纹理。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if channels < 3:
            raise ValueError("channels must be at least 3")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        rng = np.random.default_rng(seed)
        self.bev_prev = rng.normal(0.0, 0.02, size=(num_samples, channels, height, width)).astype(np.float32)
        self.bev_curr = self.bev_prev.copy()
        targets = []
        for index in range(num_samples):
            phase = float(index) / float(max(num_samples - 1, 1))
            target = np.asarray(
                [
                    np.sin(phase * np.pi),
                    np.cos(phase * np.pi) * 0.5,
                    phase - 0.5,
                ],
                dtype=np.float32,
            )
            self.bev_curr[index, 0, :, :] += target[0]
            self.bev_curr[index, 1, :, :] += target[1]
            self.bev_curr[index, 2, :, :] += target[2]
            targets.append(target)
        self.targets = np.stack(targets, axis=0).astype(np.float32)

    def __len__(self) -> int:
        """返回合成样本数量。"""

        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """返回单个合成 BEV pair 样本。"""

        if index < 0 or index >= len(self):
            raise IndexError(f"synthetic sample index out of range: {index}")
        metadata = {
            "sequence": "synthetic",
            "layout": "synthetic",
            "prev_index": index,
            "curr_index": index + 1,
        }
        return (
            torch.as_tensor(self.bev_prev[index], dtype=torch.float32),
            torch.as_tensor(self.bev_curr[index], dtype=torch.float32),
            torch.as_tensor(self.targets[index], dtype=torch.float32),
            metadata,
        )


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """构造训练 DataLoader，并集中校验常见参数。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    *,
    device: torch.device,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    amp: bool = False,
    max_batches: int | None = None,
) -> EpochResult:
    """训练一个 epoch，并返回平均 loss。

    参数:
        model/dataloader/optimizer: 标准 PyTorch 训练对象。
        device: `cpu` 或 `cuda`。
        loss_weights: normalized `dx, dy, yaw` loss 权重。
        amp: 仅在 CUDA 上启用 autocast/GradScaler。
        max_batches: 可选 batch 上限，用于 overfit smoke。
    """

    model.train()
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        bev_prev, bev_curr, target = _move_pair_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with context:
            prediction = model(stack_bev_pair(bev_prev, bev_curr))
            loss = weighted_smooth_l1_loss(prediction, target, weights=loss_weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))
    return _epoch_result_from_losses(losses)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    pose_mean: Tensor | NDArray[np.float32] | tuple[float, float, float],
    pose_std: Tensor | NDArray[np.float32] | tuple[float, float, float],
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_batches: int | None = None,
) -> EpochResult:
    """验证模型并返回 normalized loss 与反归一化物理误差。"""

    model.eval()
    losses: list[float] = []
    translation_errors: list[float] = []
    yaw_errors: list[float] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        bev_prev, bev_curr, target = _move_pair_batch(batch, device)
        prediction = model(stack_bev_pair(bev_prev, bev_curr))
        loss = weighted_smooth_l1_loss(prediction, target, weights=loss_weights)
        metrics = pose_error_metrics(prediction, target, pose_mean, pose_std)
        losses.append(float(loss.detach().cpu().item()))
        translation_errors.append(metrics["translation_error_m"])
        yaw_errors.append(metrics["yaw_error_rad"])
    result = _epoch_result_from_losses(losses)
    return EpochResult(
        loss=result.loss,
        batches=result.batches,
        translation_error_m=float(np.mean(translation_errors)) if translation_errors else None,
        yaw_error_rad=float(np.mean(yaw_errors)) if yaw_errors else None,
    )


def run_overfit_smoke(
    *,
    num_samples: int = 8,
    height: int = 16,
    width: int = 16,
    channels: int = 4,
    steps: int = 20,
    batch_size: int = 4,
    hidden_channels: int = 8,
    learning_rate: float = 1.0e-2,
    seed: int = 20260608,
    device: str = "cpu",
) -> dict[str, float]:
    """在合成小数据上运行 tiny overfit，并返回初始/最终 loss。

    该函数用于 Week 6 的快速验收，不依赖 KITTI 数据，也不要求 CUDA。
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    set_global_seed(seed)
    torch.manual_seed(seed)
    train_device = torch.device(device)
    dataset = SyntheticPairDataset(
        num_samples=num_samples,
        channels=channels,
        height=height,
        width=width,
        seed=seed,
    )
    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = PoseNet3DoF(in_channels=channels * 2, hidden_channels=hidden_channels).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    initial_loss = _first_batch_loss(model, dataloader, train_device)
    for _ in range(steps):
        train_one_epoch(
            model,
            dataloader,
            optimizer,
            device=train_device,
            max_batches=1,
        )
    final_loss = _first_batch_loss(model, dataloader, train_device)
    return {"initial_loss": initial_loss, "final_loss": final_loss}


def _move_pair_batch(batch: object, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """把 DataLoader batch 中的 BEV pair 和 target 移到指定 device。"""

    if not isinstance(batch, (list, tuple)) or len(batch) < 3:
        raise ValueError("batch must contain bev_prev, bev_curr, and target")
    bev_prev = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
    bev_curr = batch[1].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch[2].to(device=device, dtype=torch.float32, non_blocking=True)
    return bev_prev, bev_curr, target


def _first_batch_loss(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """计算当前模型在第一个 batch 上的 loss。"""

    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        bev_prev, bev_curr, target = _move_pair_batch(batch, device)
        prediction = model(stack_bev_pair(bev_prev, bev_curr))
        loss = weighted_smooth_l1_loss(prediction, target)
    return float(loss.detach().cpu().item())


def _epoch_result_from_losses(losses: list[float]) -> EpochResult:
    """把 batch loss 列表聚合成 EpochResult。"""

    if not losses:
        raise ValueError("at least one batch is required")
    return EpochResult(loss=float(np.mean(losses)), batches=len(losses))
