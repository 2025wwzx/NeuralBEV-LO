#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PoseNet checkpoint 保存与加载工具。

checkpoint 只保存可序列化训练状态：模型参数、可选 optimizer 状态、epoch、metrics
和配置快照。加载函数负责把模型参数恢复到调用方提供的模型实例中，并返回元数据，
用于训练续跑或 round-trip 验证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    global_step: int = 0,
) -> Path:
    """保存 PoseNet checkpoint。

    参数:
        checkpoint_path: 输出 `.pt` 路径，父目录会自动创建。
        model: 待保存的 PyTorch 模型。
        optimizer: 可选优化器；为 `None` 时不保存 optimizer state。
        epoch: 当前 epoch，从 0 或 1 起均可，由调用方保持一致。
        metrics: 当前训练/验证指标，必须可 JSON 化。
        config: 当前配置快照。
        global_step: 已完成的优化步数。
    返回:
        实际写入的 checkpoint 路径。
    """

    path = Path(checkpoint_path)
    if not path.suffix:
        raise ValueError(f"checkpoint path must include a file name: {path}")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "config": dict(config),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """加载 checkpoint 并恢复模型/优化器状态。

    返回:
        不含大 tensor 的元数据字典，包括 `epoch`、`global_step`、`metrics` 和 `config`。
    """

    path = Path(checkpoint_path)
    if not path.exists():
        raise ValueError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    if "model_state" not in payload:
        raise ValueError(f"checkpoint missing model_state: {path}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "metrics": dict(payload.get("metrics", {})),
        "config": dict(payload.get("config", {})),
        "path": str(path),
    }
