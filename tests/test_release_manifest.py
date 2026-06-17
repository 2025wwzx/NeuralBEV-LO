#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 release manifest 生成测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from neuralbev_lo.utils.release_manifest import build_release_manifest


def _write_json(path: Path, payload: dict) -> None:
    """写入测试 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_release_manifest_records_git_artifacts_and_metrics(tmp_path: Path) -> None:
    """manifest 应记录 git、artifact 哈希和指标摘要。"""

    artifact = tmp_path / "demo.mp4"
    memory_metrics = tmp_path / "memory_metrics.json"
    consistency_metrics = tmp_path / "consistency_metrics.json"
    artifact.write_bytes(b"demo-bytes")
    _write_json(
        memory_metrics,
        {
            "metadata": {"sequence": "07", "frames": 5, "device": "cpu"},
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
            "metadata": {"sequence": "07", "frames": 5},
            "metrics": [
                {"memory": "learned_pose", "alignment_iou": 0.3, "flicker_score": 0.01},
                {"memory": "learned_pose", "alignment_iou": 0.5, "flicker_score": 0.03},
            ],
        },
    )

    manifest = build_release_manifest(
        repo_root=Path.cwd(),
        artifact_paths=[artifact, memory_metrics, consistency_metrics],
        memory_metrics_path=memory_metrics,
        consistency_metrics_path=consistency_metrics,
        tag_target="v0.1-research-prototype",
    )

    assert manifest["schema_version"] == 1
    assert manifest["tag_target"] == "v0.1-research-prototype"
    assert manifest["git"]["commit"]
    assert manifest["artifacts"][0]["sha256"]
    assert manifest["artifacts"][0]["size_bytes"] == len(b"demo-bytes")
    assert manifest["metrics"]["memory"]["learned_pose"]["occupancy_iou"] == 0.4
    assert manifest["metrics"]["consistency"]["learned_pose"]["mean_alignment_iou"] == 0.4


def test_release_manifest_cli_writes_json(tmp_path: Path) -> None:
    """CLI 应将 release manifest 写为 JSON 文件。"""

    artifact = tmp_path / "memory.png"
    memory_metrics = tmp_path / "memory_metrics.json"
    consistency_metrics = tmp_path / "consistency_metrics.json"
    output = tmp_path / "manifest.json"
    artifact.write_bytes(b"image")
    _write_json(
        memory_metrics,
        {
            "metadata": {"sequence": "07", "frames": 5, "device": "cpu"},
            "metrics": [{"memory": "gt_pose", "occupancy_iou": 1.0, "mean_abs_error": 0.0}],
        },
    )
    _write_json(
        consistency_metrics,
        {
            "metadata": {"sequence": "07", "frames": 5},
            "metrics": [{"memory": "gt_pose", "alignment_iou": 0.8, "flicker_score": 0.02}],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/write_release_manifest.py",
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

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "Release manifest written" in completed.stdout
    assert payload["artifacts"][0]["path"].endswith("memory.png")
