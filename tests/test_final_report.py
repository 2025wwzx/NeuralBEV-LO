#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 final report 生成测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from neuralbev_lo.utils.final_report import build_final_report


def _write_json(path: Path, payload: dict) -> None:
    """写入测试 JSON 文件。"""

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_final_report_summarizes_metrics_and_artifacts(tmp_path: Path) -> None:
    """final report 应汇总配置、设备、checkpoint、指标和 artifact 路径。"""

    memory_metrics = tmp_path / "memory_metrics.json"
    consistency_metrics = tmp_path / "consistency_metrics.json"
    demo_video = tmp_path / "demo.mp4"
    demo_video.write_bytes(b"fake-video")
    _write_json(
        memory_metrics,
        {
            "metadata": {
                "sequence": "07",
                "frames": 5,
                "device": "cpu",
                "checkpoint": "outputs/checkpoints/demo.pt",
                "resolution_m": 0.5,
            },
            "metrics": [
                {"memory": "naive", "occupancy_iou": 0.5, "mean_abs_error": 0.1},
                {"memory": "gt_pose", "occupancy_iou": 1.0, "mean_abs_error": 0.0},
                {"memory": "learned_pose", "occupancy_iou": 0.4, "mean_abs_error": 0.2},
            ],
        },
    )
    _write_json(
        consistency_metrics,
        {
            "metadata": {"metrics_definition": {"occupancy_threshold": 0.1}},
            "metrics": [
                {"memory": "learned_pose", "alignment_iou": 0.3, "flicker_score": 0.01},
                {"memory": "learned_pose", "alignment_iou": 0.5, "flicker_score": 0.03},
            ],
        },
    )

    report = build_final_report(
        memory_metrics_path=memory_metrics,
        consistency_metrics_path=consistency_metrics,
        config_path=Path("configs/eval/kitti_cpu_smoke_safe.yaml"),
        artifact_paths=[demo_video],
    )

    assert "NeuralBEV-LO v0.1 Final Report" in report
    assert "config: configs/eval/kitti_cpu_smoke_safe.yaml" in report
    assert "device: cpu" in report
    assert "checkpoint: outputs/checkpoints/demo.pt" in report
    assert "learned_pose occupancy_iou=0.400000" in report
    assert "learned_pose mean_alignment_iou=0.400000 mean_flicker_score=0.020000" in report
    assert str(demo_video).replace("\\", "/") in report


def test_final_report_cli_writes_output(tmp_path: Path) -> None:
    """CLI 应将 final report 写到指定输出路径。"""

    memory_metrics = tmp_path / "memory_metrics.json"
    consistency_metrics = tmp_path / "consistency_metrics.json"
    artifact = tmp_path / "memory.png"
    output = tmp_path / "report.txt"
    artifact.write_bytes(b"fake-image")
    _write_json(
        memory_metrics,
        {
            "metadata": {
                "sequence": "07",
                "frames": 5,
                "device": "cpu",
                "checkpoint": "outputs/checkpoints/demo.pt",
            },
            "metrics": [{"memory": "gt_pose", "occupancy_iou": 1.0, "mean_abs_error": 0.0}],
        },
    )
    _write_json(
        consistency_metrics,
        {
            "metadata": {},
            "metrics": [{"memory": "gt_pose", "alignment_iou": 0.8, "flicker_score": 0.02}],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/write_v0_1_report.py",
            "--memory-metrics",
            str(memory_metrics),
            "--consistency-metrics",
            str(consistency_metrics),
            "--artifact",
            str(artifact),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Final report written" in completed.stdout
    assert "NeuralBEV-LO v0.1 Final Report" in output.read_text(encoding="utf-8")
