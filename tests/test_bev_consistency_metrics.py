#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 9 BEV consistency metrics 测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neuralbev_lo.eval.bev_metrics import (
    bev_consistency_table,
    frame_to_memory_alignment_score,
    temporal_flicker_score,
)


def _empty_bev() -> np.ndarray:
    """构造一个最小 density-only BEV，用于指标单元测试。"""

    return np.zeros((1, 4, 4), dtype=np.float32)


def test_frame_alignment_score_uses_thresholded_density_iou() -> None:
    """alignment_score 应等于当前帧和 memory density occupancy 的 IoU。"""

    current = _empty_bev()
    memory = _empty_bev()
    current[0, 1, 1] = 1.0
    current[0, 1, 2] = 1.0
    memory[0, 1, 1] = 1.0
    memory[0, 2, 2] = 1.0

    score = frame_to_memory_alignment_score(
        current,
        memory,
        occupancy_threshold=0.5,
        channel_indices=(0,),
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_temporal_flicker_score_is_mean_pixel_std_over_window() -> None:
    """flicker_score 应为短窗口内 memory 像素标准差的均值。"""

    first = _empty_bev()
    second = _empty_bev()
    third = _empty_bev()
    second[0, 1, 1] = 1.0
    third[0, 1, 1] = 1.0
    third[0, 2, 2] = 1.0

    expected = float(np.std(np.stack([first, second, third], axis=0)[:, 0], axis=0).mean())

    score = temporal_flicker_score(
        [first, second, third],
        channel_indices=(0,),
    )

    assert score == pytest.approx(expected)


def test_bev_consistency_table_writes_per_frame_rows() -> None:
    """consistency table 应为每种 memory 逐帧输出 alignment 和 flicker。"""

    current_0 = _empty_bev()
    current_1 = _empty_bev()
    current_2 = _empty_bev()
    current_0[0, 1, 1] = 1.0
    current_1[0, 1, 2] = 1.0
    current_2[0, 2, 2] = 1.0
    current_bevs = [current_0, current_1, current_2]
    memory_histories = {
        "gt_pose": [bev.copy() for bev in current_bevs],
        "naive": [current_0, current_0, current_0],
    }

    rows = bev_consistency_table(
        current_bevs=current_bevs,
        memory_histories=memory_histories,
        occupancy_threshold=0.5,
        channel_indices=(0,),
        flicker_window=2,
    )

    assert len(rows) == 6
    by_name_and_frame = {(row["memory"], row["frame_index"]): row for row in rows}
    assert by_name_and_frame[("gt_pose", 0)]["alignment_iou"] == 1.0
    assert by_name_and_frame[("gt_pose", 0)]["flicker_window_size"] == 1
    assert by_name_and_frame[("gt_pose", 2)]["flicker_window_size"] == 2
    assert by_name_and_frame[("naive", 2)]["alignment_iou"] == 0.0


def test_build_bev_memory_demo_writes_consistency_artifacts(tmp_path: Path) -> None:
    """合成 BEV memory demo 应写出 Week 9 JSON/CSV 表和 alignment 曲线。"""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_bev_memory_demo.py",
            "--synthetic",
            "--frames",
            "4",
            "--resolution-m",
            "1.0",
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
    prefix = "synthetic_000000_000003"
    consistency_json = tmp_path / f"{prefix}_consistency_metrics.json"
    consistency_csv = tmp_path / f"{prefix}_consistency_metrics.csv"
    alignment_curve = tmp_path / f"{prefix}_alignment_curve.png"

    assert consistency_json.exists()
    assert consistency_csv.exists()
    assert alignment_curve.exists()
    assert alignment_curve.stat().st_size > 0

    payload = json.loads(consistency_json.read_text(encoding="utf-8"))
    assert payload["metadata"]["resolution_m"] == 1.0
    assert payload["metadata"]["metrics_definition"]["occupancy_threshold"] == 0.1
    assert payload["metadata"]["metrics_definition"]["selected_channels"] == ["density"]
    assert {row["memory"] for row in payload["metrics"]} >= {"naive", "gt_pose"}
    assert "alignment_iou" in consistency_csv.read_text(encoding="utf-8").splitlines()[0]
