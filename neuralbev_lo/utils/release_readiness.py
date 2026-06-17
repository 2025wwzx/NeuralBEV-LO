#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 release readiness 静态检查工具。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REQUIRED_DOCS: Final[tuple[str, ...]] = (
    "README.md",
    "docs/data_contract.md",
    "docs/coordinate_system.md",
    "docs/architecture.md",
    "docs/experiment_log.md",
    "docs/m3_report.md",
    "docs/resume_notes.md",
    "docs/release_checklist.md",
)

REQUIRED_SCRIPTS: Final[tuple[str, ...]] = (
    "scripts/check_env.py",
    "scripts/run_pipeline_smoke.py",
    "scripts/train_posenet_3dof.py",
    "scripts/eval_posenet.py",
    "scripts/build_bev_memory_demo.py",
    "scripts/build_m3_demo_video.py",
    "scripts/write_v0_1_report.py",
    "scripts/check_release_readiness.py",
)

README_REQUIRED_PHRASES: Final[tuple[str, ...]] = (
    "Week 12 Reproducibility Freeze",
    "Limitations",
    "data/README.md",
    "docs/release_checklist.md",
    "scripts/write_v0_1_report.py",
)

RUNTIME_PATHS: Final[tuple[str, ...]] = (
    "data/kitti_odometry",
    "outputs",
    "work",
)

ALLOWED_TRACKED_RUNTIME_DOCS: Final[set[str]] = {
    "data/README.md",
}


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


def collect_release_readiness(repo_root: str | Path) -> ReadinessResult:
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
        "runtime_artifacts_ignored": _check_runtime_ignored(root),
        "no_tracked_runtime_artifacts": _check_no_tracked_runtime_artifacts(root),
        "release_tag": _check_release_tag(root),
    }
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


def _run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """运行 git 命令并捕获输出。"""

    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
