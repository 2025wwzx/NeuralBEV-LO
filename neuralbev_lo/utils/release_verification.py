#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 pre-tag release verification 编排工具。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from neuralbev_lo.utils.logging import log_info, log_warn


def build_release_verification_commands(
    *,
    python_executable: str | Path = sys.executable,
) -> list[list[str]]:
    """构造 v0.1 tag 前最终审计命令链。

    参数:
        python_executable: 执行项目脚本和 pytest 的 Python 可执行文件。
    返回:
        依次执行 release demo、完整测试、强 readiness 的命令列表。
    """

    python_text = str(python_executable)
    return [
        [python_text, "scripts/run_v0_1_release_demo.py"],
        [python_text, "-m", "pytest"],
        [
            python_text,
            "scripts/check_release_readiness.py",
            "--allow-pending-tag",
            "--require-artifacts",
            "--require-clean-git",
        ],
    ]


def run_release_verification(
    *,
    dry_run: bool = False,
    python_executable: str | Path = sys.executable,
) -> int:
    """运行 v0.1 tag 前最终审计命令链。

    参数:
        dry_run: 为 True 时只打印命令，不执行。
        python_executable: 执行项目脚本和 pytest 的 Python 可执行文件。
    返回:
        进程退出码；任一命令失败时立即返回该命令的退出码。
    """

    commands = build_release_verification_commands(python_executable=python_executable)
    if dry_run:
        log_info("Release verification dry run", commands=len(commands))
        for index, command in enumerate(commands, start=1):
            print(f"[{index}] {_format_command(command)}")
        return 0

    for index, command in enumerate(commands, start=1):
        log_info("Release verification step started", step=index, command=_format_command(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            log_warn("Release verification step failed", step=index, returncode=completed.returncode)
            return int(completed.returncode)
        log_info("Release verification step completed", step=index)
    log_info("Release verification completed", ready_for_tag=True)
    return 0


def _format_command(command: list[str]) -> str:
    """格式化命令，便于日志和 dry-run 阅读。"""

    return " ".join(command)
