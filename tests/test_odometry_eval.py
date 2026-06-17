#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 7 odometry evaluation 与 baseline 对比测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from math import pi
from pathlib import Path

import numpy as np
import pytest

from neuralbev_lo.eval.odometry_metrics import (
    build_baseline_predictions,
    evaluate_baselines,
    integrate_relative_poses,
    save_metrics_csv,
    save_metrics_json,
)
from neuralbev_lo.viz.render_trajectory import save_trajectory_overlay


def test_integrate_relative_poses_uses_previous_frame_motion() -> None:
    """相对位姿积分应把 dx/dy 解释为 previous LiDAR frame 下的运动。"""

    relative_poses = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, pi / 2.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    trajectory = integrate_relative_poses(relative_poses)

    assert trajectory.shape == (4, 3)
    np.testing.assert_allclose(trajectory[:, :2], [[0, 0], [1, 0], [2, 0], [2, 1]], atol=1e-8)
    assert trajectory[-1, 2] == pytest.approx(pi / 2.0)


def test_baseline_metrics_rank_gt_echo_above_zero_motion() -> None:
    """GT-label echo 应得到零误差，zero-motion 应明显更差。"""

    gt_relative = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.1, 0.01], [1.0, -0.1, -0.02]],
        dtype=np.float64,
    )
    baselines = build_baseline_predictions(gt_relative)

    rows = evaluate_baselines(gt_relative, baselines)
    by_name = {row["baseline"]: row for row in rows}

    assert by_name["gt_label_echo"]["ate_rmse_m"] == pytest.approx(0.0)
    assert by_name["gt_label_echo"]["relative_translation_error_m"] == pytest.approx(0.0)
    assert by_name["zero_motion"]["ate_rmse_m"] > by_name["gt_label_echo"]["ate_rmse_m"]


def test_learned_prediction_can_beat_zero_motion_on_smoke_labels() -> None:
    """当 learned relative pose 接近 GT 时，评估表应显示它优于 zero-motion。"""

    gt_relative = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.02], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    learned_relative = gt_relative + np.array(
        [[0.05, 0.0, 0.0], [-0.04, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=np.float64,
    )
    baselines = build_baseline_predictions(gt_relative, learned_relative=learned_relative)

    rows = evaluate_baselines(gt_relative, baselines)
    by_name = {row["baseline"]: row for row in rows}

    assert by_name["learned"]["ate_rmse_m"] < by_name["zero_motion"]["ate_rmse_m"]


def test_metrics_json_csv_and_trajectory_overlay_are_saved(tmp_path: Path) -> None:
    """评估结果应保存 JSON、CSV 和轨迹 overlay 图。"""

    gt_relative = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    rows = evaluate_baselines(gt_relative, build_baseline_predictions(gt_relative))
    json_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "metrics.csv"
    figure_path = tmp_path / "trajectory.png"

    save_metrics_json(rows, json_path, metadata={"sequence": "synthetic"})
    save_metrics_csv(rows, csv_path)
    save_trajectory_overlay(
        {
            "gt": integrate_relative_poses(gt_relative),
            "zero": integrate_relative_poses(np.zeros_like(gt_relative)),
        },
        figure_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["sequence"] == "synthetic"
    assert len(payload["metrics"]) == len(rows)
    assert csv_path.exists()
    assert "baseline" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_eval_posenet_synthetic_cli_writes_metrics_and_trajectory(tmp_path: Path) -> None:
    """合成 eval CLI 应无需 KITTI 数据即可写出 metrics 表和轨迹图。"""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval_posenet.py",
            "--config",
            "configs/train/posenet_3dof.yaml",
            "--synthetic",
            "--max-pairs",
            "6",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PoseNet evaluation completed" in completed.stdout
    assert (tmp_path / "synthetic_eval_metrics.json").exists()
    assert (tmp_path / "synthetic_eval_metrics.csv").exists()
    assert (tmp_path / "synthetic_eval_trajectory.png").exists()
