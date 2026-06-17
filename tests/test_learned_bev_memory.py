#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 8 learned-pose BEV memory integration 测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.eval.bev_metrics import compare_memory_to_reference, memory_quality_table
from neuralbev_lo.geometry.se2 import se2_from_xyyaw
from neuralbev_lo.models.losses import denormalize_pose_batch
from neuralbev_lo.models.posenet import PoseNet3DoF
from neuralbev_lo.training.checkpoint import save_checkpoint
from neuralbev_lo.viz.render_bev import save_bev_comparison


def _tiny_grid() -> BevGridConfig:
    """构造小尺寸 BEV grid，用于快速 memory 指标测试。"""

    return BevGridConfig(
        x_range_m=(0.0, 8.0),
        y_range_m=(-4.0, 4.0),
        resolution_m=1.0,
        channels=("density",),
    )


def _toy_memory_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 current / naive / gt 三个 BEV memory。"""

    current = np.zeros((1, 8, 8), dtype=np.float32)
    current[0, 3, 4] = 1.0
    gt = current.copy()
    naive = np.zeros_like(current)
    naive[0, 5, 4] = 1.0
    return current, naive, gt


def test_memory_quality_table_ranks_gt_above_naive() -> None:
    """与 GT memory 完全一致的结果应优于 naive memory。"""

    current, naive, gt = _toy_memory_pair()
    learned = gt.copy()

    rows = memory_quality_table(
        reference_memory=gt,
        candidates={"naive": naive, "gt_pose": gt, "learned_pose": learned},
        occupancy_threshold=0.5,
    )
    by_name = {row["memory"]: row for row in rows}

    assert by_name["gt_pose"]["mean_abs_error"] == 0.0
    assert by_name["learned_pose"]["occupancy_iou"] == 1.0
    assert by_name["naive"]["mean_abs_error"] > by_name["gt_pose"]["mean_abs_error"]
    assert compare_memory_to_reference(current, gt)["occupancy_iou"] == 1.0


def test_denormalized_prediction_can_be_used_as_memory_transform() -> None:
    """PoseNet normalized 输出反归一化后应可转为 update_memory 使用的 SE2。"""

    normalized = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    mean = torch.tensor([1.0, 0.0, 0.1], dtype=torch.float32)
    std = torch.tensor([0.5, 0.2, 0.01], dtype=torch.float32)

    physical = denormalize_pose_batch(normalized, mean, std).detach().cpu().numpy()[0]
    transform = se2_from_xyyaw(float(physical[0]), float(physical[1]), float(physical[2]))

    assert transform.shape == (3, 3)
    np.testing.assert_allclose([transform[0, 2], transform[1, 2]], [1.0, 0.0])


def test_bev_comparison_accepts_four_panels(tmp_path: Path) -> None:
    """Week 8 四联图应能保存 current/naive/gt/learned 四个面板。"""

    current, naive, gt = _toy_memory_pair()
    output_path = tmp_path / "four_panel.png"

    saved = save_bev_comparison(
        [("current", current), ("naive", naive), ("gt-pose", gt), ("learned", gt)],
        output_path,
    )

    assert saved == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_bev_memory_demo_learned_synthetic_writes_artifacts(tmp_path: Path) -> None:
    """合成 learned memory demo 应写出四联图、轨迹图和 metrics JSON。"""

    config = {
        "bev": {"channels": ["density"], "x_range_m": [0.0, 8.0], "y_range_m": [-4.0, 4.0], "resolution_m": 1.0},
        "pose": {
            "pose_norm": {
                "source_split": "synthetic",
                "sequences": ["synthetic"],
                "count": 1,
                "mean": [1.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 0.0, 0.0],
            }
        },
        "model": {"hidden_channels": 4, "dropout": 0.0},
    }
    checkpoint_path = tmp_path / "synthetic_posenet.pt"
    model = PoseNet3DoF(in_channels=2, hidden_channels=4)
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        epoch=0,
        metrics={"loss": 0.0},
        config=config,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_bev_memory_demo.py",
            "--synthetic",
            "--pose-source",
            "learned",
            "--checkpoint",
            str(checkpoint_path),
            "--frames",
            "4",
            "--output-dir",
            str(tmp_path),
            "--metrics-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "BEV memory demo completed" in completed.stdout
    assert (tmp_path / "synthetic_000000_000003_memory.png").exists()
    assert (tmp_path / "synthetic_000000_000003_trajectory.png").exists()
    metrics_path = tmp_path / "synthetic_000000_000003_memory_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["pose_source"] == "learned"
    assert {row["memory"] for row in payload["metrics"]} >= {"naive", "gt_pose", "learned_pose"}
