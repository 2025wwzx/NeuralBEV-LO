#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 release readiness 静态检查工具。"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REQUIRED_DOCS: Final[tuple[str, ...]] = (
    "README.md",
    "docs/data_contract.md",
    "docs/coordinate_system.md",
    "docs/architecture.md",
    "docs/experiment_log.md",
    "docs/m3_report.md",
    "docs/resume_notes.md",
    "docs/release_checklist.md",
    "docs/release_notes_v0_1.md",
)

REQUIRED_SCRIPTS: Final[tuple[str, ...]] = (
    "scripts/check_env.py",
    "scripts/run_pipeline_smoke.py",
    "scripts/train_posenet_3dof.py",
    "scripts/eval_posenet.py",
    "scripts/build_bev_memory_demo.py",
    "scripts/build_m3_demo_video.py",
    "scripts/run_v0_1_release_demo.py",
    "scripts/write_v0_1_report.py",
    "scripts/check_release_readiness.py",
)

README_REQUIRED_PHRASES: Final[tuple[str, ...]] = (
    "Week 12 Reproducibility Freeze",
    "Limitations",
    "data/README.md",
    "docs/release_checklist.md",
    "docs/release_notes_v0_1.md",
    "scripts/run_v0_1_release_demo.py",
    "scripts/write_v0_1_report.py",
)

RELEASE_NOTES_REQUIRED_PHRASES: Final[tuple[str, ...]] = (
    "NeuralBEV-LO v0.1 Research Prototype Release Notes",
    "Reproduce The Release Demo",
    "Metrics Snapshot",
    "Limitations",
    "v0.1-research-prototype",
)

RUNTIME_PATHS: Final[tuple[str, ...]] = (
    "data/kitti_odometry",
    "outputs",
    "work",
)

ALLOWED_TRACKED_RUNTIME_DOCS: Final[set[str]] = {
    "data/README.md",
}

RELEASE_ARTIFACTS: Final[tuple[str, ...]] = (
    "outputs/figures/v0_1_release_demo/07_000000_000004_memory.png",
    "outputs/figures/v0_1_release_demo/07_000000_000004_trajectory.png",
    "outputs/figures/v0_1_release_demo/07_000000_000004_alignment_curve.png",
    "outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4",
    "outputs/metrics/v0_1_release_demo/07_000000_000004_memory_metrics.json",
    "outputs/metrics/v0_1_release_demo/07_000000_000004_consistency_metrics.json",
    "outputs/reports/v0_1_release_demo.txt",
)

RELEASE_MEMORY_METRICS: Final[str] = (
    "outputs/metrics/v0_1_release_demo/07_000000_000004_memory_metrics.json"
)
RELEASE_CONSISTENCY_METRICS: Final[str] = (
    "outputs/metrics/v0_1_release_demo/07_000000_000004_consistency_metrics.json"
)
RELEASE_REPORT: Final[str] = "outputs/reports/v0_1_release_demo.txt"
REQUIRED_MEMORY_NAMES: Final[set[str]] = {"naive", "gt_pose", "learned_pose"}
REQUIRED_REPORT_PHRASES: Final[tuple[str, ...]] = (
    "NeuralBEV-LO v0.1 Final Report",
    "config:",
    "device:",
    "checkpoint:",
    "Memory Metrics",
    "Artifacts",
    "Limitations",
)


@dataclass(frozen=True)
class ReadinessCheck:
    """单项 release readiness 检查结果。"""

    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        """返回该检查是否通过。"""

        return self.status == "pass"


@dataclass(frozen=True)
class ReadinessResult:
    """release readiness 汇总结果。"""

    checks: dict[str, ReadinessCheck]

    @property
    def passed(self) -> bool:
        """当所有非 pending 检查通过时返回 True。"""

        return all(check.status in {"pass", "pending"} for check in self.checks.values())

    def format_text(self) -> str:
        """格式化为 CLI 友好的多行摘要。"""

        lines = ["Release readiness summary"]
        for name in sorted(self.checks):
            check = self.checks[name]
            lines.append(f"- {name}: {check.status} - {check.detail}")
        return "\n".join(lines)


def collect_release_readiness(
    repo_root: str | Path,
    *,
    require_artifacts: bool = False,
) -> ReadinessResult:
    """收集当前仓库的 v0.1 release readiness 静态检查。

    参数:
        repo_root: 仓库根目录。
    返回:
        包含文档、脚本、gitignore、runtime artifact 和 tag 状态的检查结果。
    """

    root = Path(repo_root)
    checks = {
        "required_docs": _check_required_paths(root, REQUIRED_DOCS),
        "required_scripts": _check_required_paths(root, REQUIRED_SCRIPTS),
        "readme_release_content": _check_readme_content(root),
        "release_notes_content": _check_release_notes_content(root),
        "runtime_artifacts_ignored": _check_runtime_ignored(root),
        "no_tracked_runtime_artifacts": _check_no_tracked_runtime_artifacts(root),
        "release_tag": _check_release_tag(root),
    }
    if require_artifacts:
        checks["release_artifacts"] = _check_release_artifacts(root)
    return ReadinessResult(checks=checks)


def _check_required_paths(root: Path, paths: tuple[str, ...]) -> ReadinessCheck:
    """检查必备路径是否存在。"""

    missing = [path for path in paths if not (root / path).exists()]
    name = "required_docs" if paths is REQUIRED_DOCS else "required_scripts"
    if missing:
        return ReadinessCheck(name=name, status="fail", detail=f"missing: {', '.join(missing)}")
    return ReadinessCheck(name=name, status="pass", detail=f"{len(paths)} paths present")


def _check_readme_content(root: Path) -> ReadinessCheck:
    """检查 README 是否包含 Week 12 release 关键信息。"""

    readme_path = root / "README.md"
    if not readme_path.exists():
        return ReadinessCheck(name="readme_release_content", status="fail", detail="README.md missing")
    text = readme_path.read_text(encoding="utf-8")
    missing = [phrase for phrase in README_REQUIRED_PHRASES if phrase not in text]
    if missing:
        return ReadinessCheck(
            name="readme_release_content",
            status="fail",
            detail=f"missing phrases: {', '.join(missing)}",
        )
    return ReadinessCheck(name="readme_release_content", status="pass", detail="release section present")


def _check_release_notes_content(root: Path) -> ReadinessCheck:
    """检查 v0.1 release notes 是否包含可发布的关键段落。"""

    release_notes_path = root / "docs/release_notes_v0_1.md"
    if not release_notes_path.exists():
        return ReadinessCheck(
            name="release_notes_content",
            status="fail",
            detail="docs/release_notes_v0_1.md missing",
        )
    text = release_notes_path.read_text(encoding="utf-8")
    missing = [phrase for phrase in RELEASE_NOTES_REQUIRED_PHRASES if phrase not in text]
    if missing:
        return ReadinessCheck(
            name="release_notes_content",
            status="fail",
            detail=f"missing phrases: {', '.join(missing)}",
        )
    return ReadinessCheck(name="release_notes_content", status="pass", detail="release notes ready")


def _check_runtime_ignored(root: Path) -> ReadinessCheck:
    """检查 data/outputs/work 运行时目录是否被 gitignore 覆盖。"""

    missing: list[str] = []
    for path in RUNTIME_PATHS:
        completed = _run_git(root, ["check-ignore", path])
        if completed.returncode != 0:
            missing.append(path)
    if missing:
        return ReadinessCheck(
            name="runtime_artifacts_ignored",
            status="fail",
            detail=f"not ignored: {', '.join(missing)}",
        )
    return ReadinessCheck(
        name="runtime_artifacts_ignored",
        status="pass",
        detail="data/kitti_odometry, outputs, and work ignored",
    )


def _check_no_tracked_runtime_artifacts(root: Path) -> ReadinessCheck:
    """检查 runtime artifact 目录下没有被 git 跟踪的文件。"""

    completed = _run_git(root, ["ls-files", "data", "outputs", "work"])
    if completed.returncode != 0:
        return ReadinessCheck(
            name="no_tracked_runtime_artifacts",
            status="fail",
            detail=completed.stderr.strip() or "git ls-files failed",
        )
    tracked = [
        line
        for line in completed.stdout.splitlines()
        if line.strip() and line.strip().replace("\\", "/") not in ALLOWED_TRACKED_RUNTIME_DOCS
    ]
    if tracked:
        preview = ", ".join(tracked[:5])
        return ReadinessCheck(
            name="no_tracked_runtime_artifacts",
            status="fail",
            detail=f"tracked runtime files: {preview}",
        )
    return ReadinessCheck(
        name="no_tracked_runtime_artifacts",
        status="pass",
        detail="no tracked data/outputs/work files",
    )


def _check_release_tag(root: Path) -> ReadinessCheck:
    """检查 release tag 状态；未打 tag 时保持 pending，等待用户批准。"""

    tag_name = "v0.1-research-prototype"
    completed = _run_git(root, ["tag", "--list", tag_name])
    if completed.returncode != 0:
        return ReadinessCheck(
            name="release_tag",
            status="fail",
            detail=completed.stderr.strip() or "git tag failed",
        )
    if completed.stdout.strip() == tag_name:
        return ReadinessCheck(name="release_tag", status="pass", detail=f"{tag_name} exists")
    return ReadinessCheck(
        name="release_tag",
        status="pending",
        detail=f"{tag_name} requires explicit user approval before creation",
    )


def _check_release_artifacts(root: Path) -> ReadinessCheck:
    """检查 v0.1 release demo 运行产物是否存在且内容可读。"""

    missing = [path for path in RELEASE_ARTIFACTS if not (root / path).exists()]
    if missing:
        return ReadinessCheck(
            name="release_artifacts",
            status="fail",
            detail=f"missing artifacts: {', '.join(missing)}",
        )
    empty = [path for path in RELEASE_ARTIFACTS if (root / path).stat().st_size <= 0]
    if empty:
        return ReadinessCheck(
            name="release_artifacts",
            status="fail",
            detail=f"empty artifacts: {', '.join(empty)}",
        )
    try:
        _validate_memory_metrics(root / RELEASE_MEMORY_METRICS)
        _validate_consistency_metrics(root / RELEASE_CONSISTENCY_METRICS)
        _validate_final_report(root / RELEASE_REPORT)
    except ValueError as exc:
        return ReadinessCheck(name="release_artifacts", status="fail", detail=str(exc))
    return ReadinessCheck(
        name="release_artifacts",
        status="pass",
        detail=f"{len(RELEASE_ARTIFACTS)} artifacts verified",
    )


def _validate_memory_metrics(path: Path) -> None:
    """校验 release memory metrics JSON 的关键字段。"""

    payload = _read_json_object(path)
    metadata = _require_dict(payload.get("metadata"), "memory metrics metadata")
    metrics = _require_list(payload.get("metrics"), "memory metrics")
    if str(metadata.get("sequence")) != "07":
        raise ValueError("memory metrics sequence must be 07")
    if int(metadata.get("frames", 0)) < 5:
        raise ValueError("memory metrics frames must be at least 5")
    _require_non_empty_text(metadata.get("device"), "memory metrics device")
    _require_non_empty_text(metadata.get("checkpoint"), "memory metrics checkpoint")
    names = _metric_memory_names(metrics)
    missing_names = sorted(REQUIRED_MEMORY_NAMES - names)
    if missing_names:
        raise ValueError(f"memory metrics missing memories: {', '.join(missing_names)}")
    for item in metrics:
        memory_name = _require_non_empty_text(item.get("memory"), "memory")
        _require_number(item.get("occupancy_iou"), f"{memory_name}.occupancy_iou")
        _require_number(item.get("mean_abs_error"), f"{memory_name}.mean_abs_error")


def _validate_consistency_metrics(path: Path) -> None:
    """校验 release consistency metrics JSON 的关键字段。"""

    payload = _read_json_object(path)
    metrics = _require_list(payload.get("metrics"), "consistency metrics")
    names = _metric_memory_names(metrics)
    missing_names = sorted(REQUIRED_MEMORY_NAMES - names)
    if missing_names:
        raise ValueError(f"consistency metrics missing memories: {', '.join(missing_names)}")
    for item in metrics:
        memory_name = _require_non_empty_text(item.get("memory"), "memory")
        _require_number(item.get("alignment_iou"), f"{memory_name}.alignment_iou")
        _require_number(item.get("flicker_score"), f"{memory_name}.flicker_score")


def _validate_final_report(path: Path) -> None:
    """校验 final report 是否包含发布审阅所需摘要字段。"""

    text = path.read_text(encoding="utf-8")
    missing = [phrase for phrase in REQUIRED_REPORT_PHRASES if phrase not in text]
    if missing:
        raise ValueError(f"final report missing phrases: {', '.join(missing)}")


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并校验顶层类型。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    """校验字段是字典。"""

    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[dict[str, Any]]:
    """校验字段是对象列表。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        items.append(item)
    return items


def _metric_memory_names(metrics: list[dict[str, Any]]) -> set[str]:
    """收集 metrics 中的 memory 名称。"""

    return {
        memory_name
        for item in metrics
        if isinstance(memory_name := item.get("memory"), str) and memory_name.strip()
    }


def _require_non_empty_text(value: Any, name: str) -> str:
    """校验非空文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _require_number(value: Any, name: str) -> float:
    """校验数值字段。"""

    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """运行 git 命令并捕获输出。"""

    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
