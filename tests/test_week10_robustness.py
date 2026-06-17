#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 10 robustness、runtime logging 与 safe config 测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from neuralbev_lo.data.point_filters import PointFilterConfig, filter_point_cloud_for_bev
from neuralbev_lo.models.posenet import PoseNet3DoF
from neuralbev_lo.training.checkpoint import save_checkpoint
from neuralbev_lo.utils.config import load_yaml_config


def test_point_filter_applies_height_and_range_clipping() -> None:
    """过滤器应同时支持 z 高度裁剪和水平距离裁剪。"""

    points = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.5],
            [3.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 2.0, 0.5],
            [0.5, 0.5, -0.5, 0.5],
        ],
        dtype=np.float32,
    )
    config = PointFilterConfig(z_range_m=(-1.0, 1.0), distance_range_m=(0.0, 2.0))

    filtered = filter_point_cloud_for_bev(points, config)

    assert filtered.shape == (2, 4)
    np.testing.assert_allclose(filtered[:, :3], [[1.0, 0.0, 0.0], [0.5, 0.5, -0.5]])


def test_week10_safe_eval_configs_are_oom_conservative() -> None:
    """Week 10 safe configs 应显式记录 CPU smoke 和 RTX 5080 推荐参数。"""

    cpu_config = load_yaml_config(Path("configs/eval/kitti_cpu_smoke_safe.yaml"))
    gpu_config = load_yaml_config(Path("configs/eval/kitti_rtx5080_safe.yaml"))

    assert cpu_config["runtime"]["device"] == "cpu"
    assert cpu_config["runtime"]["max_frames"] <= 5
    assert cpu_config["bev"]["resolution_m"] >= 0.5
    assert gpu_config["runtime"]["device"] == "cuda"
    assert gpu_config["runtime"]["amp"] is True
    assert gpu_config["runtime"]["batch_size"] <= 2
    assert gpu_config["bev"]["resolution_m"] >= 0.25


def test_build_bev_memory_demo_reports_runtime_stages(tmp_path: Path) -> None:
    """开启 runtime profiling 后，demo CLI 应输出关键阶段耗时日志。"""

    config = {
        "bev": {
            "channels": ["density"],
            "x_range_m": [0.0, 8.0],
            "y_range_m": [-4.0, 4.0],
            "resolution_m": 1.0,
        },
        "pose": {
            "pose_norm": {
                "mean": [1.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
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
            "--profile-runtime",
            "--filter-z-range",
            "-1.0",
            "1.0",
            "--filter-distance-range",
            "0.0",
            "8.0",
            "--output-dir",
            str(tmp_path),
            "--metrics-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "runtime stage completed" in stdout
    assert "stage=data_loading" in stdout
    assert "stage=rasterization" in stdout
    assert "stage=inference" in stdout
    assert "stage=warp" in stdout
    assert "stage=rendering" in stdout
    assert "filter_z_range=(-1.0, 1.0)" in stdout
