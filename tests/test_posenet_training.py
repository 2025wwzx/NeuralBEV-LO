#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 6 PoseNet 训练闭环测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from neuralbev_lo.models.losses import (
    denormalize_pose_batch,
    pose_error_metrics,
    weighted_smooth_l1_loss,
)
from neuralbev_lo.models.posenet import PoseNet3DoF, stack_bev_pair
from neuralbev_lo.training.checkpoint import load_checkpoint, save_checkpoint
from neuralbev_lo.training.train_posenet import (
    SyntheticPairDataset,
    build_dataloader,
    run_overfit_smoke,
)


def test_posenet_stacks_two_bev_frames_and_outputs_3dof() -> None:
    """PoseNet 应接收两帧 BEV 拼接后的输入，并输出 normalized dx,dy,yaw。"""

    model = PoseNet3DoF(in_channels=8, hidden_channels=8)
    bev_prev = torch.zeros((2, 4, 32, 32), dtype=torch.float32)
    bev_curr = torch.ones((2, 4, 32, 32), dtype=torch.float32)

    stacked = stack_bev_pair(bev_prev, bev_curr)
    prediction = model(stacked)

    assert stacked.shape == torch.Size([2, 8, 32, 32])
    assert prediction.shape == torch.Size([2, 3])
    assert prediction.dtype == torch.float32


def test_weighted_loss_and_denormalized_metrics_are_finite() -> None:
    """Weighted SmoothL1 loss 与反归一化误差指标应稳定返回有限数值。"""

    prediction = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    mean = torch.tensor([1.0, 0.0, 0.1], dtype=torch.float32)
    std = torch.tensor([0.5, 0.2, 0.01], dtype=torch.float32)

    loss = weighted_smooth_l1_loss(prediction, target, weights=(1.0, 1.0, 2.0))
    denorm = denormalize_pose_batch(prediction, mean, std)
    metrics = pose_error_metrics(prediction, target, mean, std)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert denorm.shape == torch.Size([1, 3])
    assert metrics["translation_error_m"] >= 0.0
    assert metrics["yaw_error_rad"] >= 0.0


def test_checkpoint_round_trip_reproduces_predictions(tmp_path: Path) -> None:
    """checkpoint save/load 后，同一 batch 的预测应逐值一致。"""

    model = PoseNet3DoF(in_channels=8, hidden_channels=8)
    batch = torch.randn((2, 8, 32, 32), dtype=torch.float32)
    checkpoint_path = tmp_path / "posenet.pt"
    expected = model(batch).detach()

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        epoch=3,
        metrics={"val_loss": 0.5},
        config={"experiment": {"name": "test"}},
    )
    loaded = PoseNet3DoF(in_channels=8, hidden_channels=8)
    metadata = load_checkpoint(checkpoint_path, model=loaded)
    actual = loaded(batch).detach()

    assert metadata["epoch"] == 3
    assert metadata["metrics"]["val_loss"] == 0.5
    torch.testing.assert_close(actual, expected)


def test_synthetic_overfit_smoke_loss_decreases() -> None:
    """合成小数据集上训练数步后，loss 应小于初始 loss。"""

    result = run_overfit_smoke(
        num_samples=8,
        height=16,
        width=16,
        channels=4,
        steps=12,
        batch_size=4,
        hidden_channels=8,
        learning_rate=1.0e-2,
        seed=7,
        device="cpu",
    )

    assert result["initial_loss"] > result["final_loss"]


def test_build_dataloader_fetches_synthetic_pair_batch() -> None:
    """训练 dataloader 应返回可直接送入模型的合成 pair batch。"""

    dataset = SyntheticPairDataset(num_samples=4, channels=4, height=16, width=16, seed=11)
    loader = build_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    bev_prev, bev_curr, target, metadata = next(iter(loader))

    assert bev_prev.shape == torch.Size([2, 4, 16, 16])
    assert bev_curr.shape == torch.Size([2, 4, 16, 16])
    assert target.shape == torch.Size([2, 3])
    assert list(metadata["sequence"]) == ["synthetic", "synthetic"]


def test_run_pipeline_smoke_synthetic_executes_data_to_model_path() -> None:
    """合成 smoke 命令应执行一次 data-to-model 前向与 checkpoint round trip。"""

    completed = subprocess.run(
        [sys.executable, "scripts/run_pipeline_smoke.py", "--synthetic"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Synthetic data-to-model smoke completed" in completed.stdout
