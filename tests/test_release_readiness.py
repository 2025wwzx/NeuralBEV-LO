#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 12 release readiness 测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from neuralbev_lo.utils.release_readiness import collect_release_readiness


def _write_text(path: Path, text: str = "placeholder") -> None:
    """写入测试文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    """写入测试 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_minimal_release_repo(root: Path) -> None:
    """构造满足 release readiness 的最小测试仓库。"""

    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    _write_text(root / ".gitignore", "data/kitti_odometry\noutputs\nwork\n")
    _write_text(
        root / "README.md",
        "\n".join(
            [
                "Week 12 Reproducibility Freeze",
                "Limitations",
                "data/README.md",
                "docs/release_checklist.md",
                "docs/release_notes_v0_1.md",
                "scripts/run_v0_1_release_demo.py",
                "scripts/write_v0_1_report.py",
                "scripts/write_release_manifest.py",
            ]
        ),
    )
    for path in [
        "docs/data_contract.md",
        "docs/coordinate_system.md",
        "docs/architecture.md",
        "docs/experiment_log.md",
        "docs/m3_report.md",
        "docs/resume_notes.md",
        "docs/release_checklist.md",
        "docs/release_notes_v0_1.md",
        "scripts/check_env.py",
        "scripts/run_pipeline_smoke.py",
        "scripts/train_posenet_3dof.py",
        "scripts/eval_posenet.py",
        "scripts/build_bev_memory_demo.py",
        "scripts/build_m3_demo_video.py",
        "scripts/run_v0_1_release_demo.py",
        "scripts/write_v0_1_report.py",
        "scripts/write_release_manifest.py",
        "scripts/check_release_readiness.py",
    ]:
        _write_text(root / path)
    _write_text(
        root / "docs/release_notes_v0_1.md",
        "\n".join(
            [
                "NeuralBEV-LO v0.1 Research Prototype Release Notes",
                "Reproduce The Release Demo",
                "Metrics Snapshot",
                "Limitations",
                "v0.1-research-prototype",
            ]
        ),
    )
    _write_text(root / "data/README.md")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=NeuralBEV Test",
            "-c",
            "user.email=neuralbev-test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_release_artifacts(root: Path) -> None:
    """写入最小 v0.1 release demo artifacts。"""

    for path in [
        "outputs/figures/v0_1_release_demo/07_000000_000004_memory.png",
        "outputs/figures/v0_1_release_demo/07_000000_000004_trajectory.png",
        "outputs/figures/v0_1_release_demo/07_000000_000004_alignment_curve.png",
        "outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4",
    ]:
        artifact_path = root / path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"artifact")
    _write_json(
        root / "outputs/metrics/v0_1_release_demo/07_000000_000004_memory_metrics.json",
        {
            "metadata": {
                "sequence": "07",
                "frames": 5,
                "device": "cpu",
                "checkpoint": "outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt",
            },
            "metrics": [
                {"memory": "naive", "occupancy_iou": 0.5, "mean_abs_error": 0.1},
                {"memory": "gt_pose", "occupancy_iou": 1.0, "mean_abs_error": 0.0},
                {"memory": "learned_pose", "occupancy_iou": 0.4, "mean_abs_error": 0.2},
            ],
        },
    )
    _write_json(
        root / "outputs/metrics/v0_1_release_demo/07_000000_000004_consistency_metrics.json",
        {
            "metadata": {"sequence": "07", "frames": 5},
            "metrics": [
                {"memory": "naive", "alignment_iou": 0.7, "flicker_score": 0.01},
                {"memory": "gt_pose", "alignment_iou": 0.8, "flicker_score": 0.02},
                {"memory": "learned_pose", "alignment_iou": 0.6, "flicker_score": 0.03},
            ],
        },
    )
    _write_text(
        root / "outputs/reports/v0_1_release_demo.txt",
        "\n".join(
            [
                "NeuralBEV-LO v0.1 Final Report",
                "config: configs/eval/kitti_cpu_smoke_safe.yaml",
                "device: cpu",
                "checkpoint: outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt",
                "Memory Metrics",
                "learned_pose occupancy_iou=0.400000",
                "Artifacts",
                "outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4",
                "Limitations",
            ]
        ),
    )
    _write_json(
        root / "outputs/reports/v0_1_release_manifest.json",
        {
            "schema_version": 1,
            "tag_target": "v0.1-research-prototype",
            "git": {"commit": "abc123", "branch": "test", "upstream": "", "dirty": False},
            "metrics": {
                "memory": {
                    "naive": {"occupancy_iou": 0.5, "mean_abs_error": 0.1},
                    "gt_pose": {"occupancy_iou": 1.0, "mean_abs_error": 0.0},
                    "learned_pose": {"occupancy_iou": 0.4, "mean_abs_error": 0.2},
                },
                "consistency": {
                    "naive": {"mean_alignment_iou": 0.7, "mean_flicker_score": 0.01},
                    "gt_pose": {"mean_alignment_iou": 0.8, "mean_flicker_score": 0.02},
                    "learned_pose": {"mean_alignment_iou": 0.6, "mean_flicker_score": 0.03},
                },
            },
            "artifacts": [
                {
                    "path": "outputs/figures/v0_1_release_demo/07_000000_000004_memory.png",
                    "size_bytes": 1,
                    "sha256": "abc",
                },
                {
                    "path": "outputs/figures/v0_1_release_demo/07_000000_000004_trajectory.png",
                    "size_bytes": 1,
                    "sha256": "def",
                },
                {
                    "path": "outputs/figures/v0_1_release_demo/07_000000_000004_alignment_curve.png",
                    "size_bytes": 1,
                    "sha256": "ghi",
                },
                {
                    "path": "outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4",
                    "size_bytes": 1,
                    "sha256": "jkl",
                },
                {
                    "path": "outputs/metrics/v0_1_release_demo/07_000000_000004_memory_metrics.json",
                    "size_bytes": 1,
                    "sha256": "mno",
                },
                {
                    "path": "outputs/metrics/v0_1_release_demo/07_000000_000004_consistency_metrics.json",
                    "size_bytes": 1,
                    "sha256": "pqr",
                },
                {"path": "outputs/reports/v0_1_release_demo.txt", "size_bytes": 1, "sha256": "stu"},
            ],
            "validation_commands": ["python -m pytest"],
        },
    )


def test_release_readiness_checks_current_repo() -> None:
    """当前仓库应满足 v0.1 release readiness 的静态检查。"""

    result = collect_release_readiness(Path.cwd())

    assert result.passed, result.format_text()
    assert "required_docs" in result.checks
    assert result.checks["required_docs"].detail == "9 paths present"
    assert result.checks["required_scripts"].detail == "10 paths present"
    assert result.checks["readme_release_content"].status == "pass"
    assert result.checks["release_notes_content"].status == "pass"
    assert "runtime_artifacts_ignored" in result.checks
    assert "no_tracked_runtime_artifacts" in result.checks
    assert result.checks["release_tag"].status == "pending"


def test_release_readiness_cli_outputs_summary() -> None:
    """CLI 应输出 release readiness summary。"""

    completed = subprocess.run(
        [sys.executable, "scripts/check_release_readiness.py", "--allow-pending-tag"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Release readiness summary" in completed.stdout
    assert "release_tag: pending" in completed.stdout


def test_release_readiness_checks_release_artifacts(tmp_path: Path) -> None:
    """启用 artifact 检查时应验证 v0.1 release demo 产物。"""

    _make_minimal_release_repo(tmp_path)
    _write_release_artifacts(tmp_path)

    result = collect_release_readiness(tmp_path, require_artifacts=True)

    assert result.passed, result.format_text()
    assert result.checks["release_artifacts"].status == "pass"
    assert "8 artifacts verified" in result.checks["release_artifacts"].detail


def test_release_readiness_cli_can_require_artifacts(tmp_path: Path) -> None:
    """CLI 应支持发布前强制校验 artifacts。"""

    _make_minimal_release_repo(tmp_path)
    _write_release_artifacts(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_release_readiness.py",
            "--repo-root",
            str(tmp_path),
            "--allow-pending-tag",
            "--require-artifacts",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "release_artifacts: pass" in completed.stdout


def test_release_readiness_checks_clean_git_state(tmp_path: Path) -> None:
    """启用 clean-git 检查时应验证工作区干净。"""

    _make_minimal_release_repo(tmp_path)

    result = collect_release_readiness(tmp_path, require_clean_git=True)

    assert result.passed, result.format_text()
    assert result.checks["clean_git_state"].status == "pass"
    assert "working tree clean" in result.checks["clean_git_state"].detail


def test_release_readiness_clean_git_state_fails_on_dirty_worktree(tmp_path: Path) -> None:
    """工作区有未提交修改时 clean-git 检查应失败。"""

    _make_minimal_release_repo(tmp_path)
    _write_text(
        tmp_path / "README.md",
        "\n".join(
            [
                "Week 12 Reproducibility Freeze",
                "Limitations",
                "data/README.md",
                "docs/release_checklist.md",
                "docs/release_notes_v0_1.md",
                "scripts/run_v0_1_release_demo.py",
                "scripts/write_v0_1_report.py",
                "dirty",
            ]
        ),
    )

    result = collect_release_readiness(tmp_path, require_clean_git=True)

    assert not result.passed
    assert result.checks["clean_git_state"].status == "fail"
    assert "uncommitted changes" in result.checks["clean_git_state"].detail


def test_release_readiness_cli_can_require_clean_git(tmp_path: Path) -> None:
    """CLI 应支持发布前强制校验 Git 工作区状态。"""

    _make_minimal_release_repo(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_release_readiness.py",
            "--repo-root",
            str(tmp_path),
            "--allow-pending-tag",
            "--require-clean-git",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "clean_git_state: pass" in completed.stdout
