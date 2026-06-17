#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 pre-tag release verification 测试。"""

from __future__ import annotations

import subprocess
import sys

from neuralbev_lo.utils.release_verification import build_release_verification_commands


def test_build_release_verification_commands() -> None:
    """pre-tag verifier 应串起 demo、pytest 和强 readiness。"""

    commands = build_release_verification_commands(python_executable="python")
    joined = [" ".join(command) for command in commands]

    assert len(commands) == 3
    assert joined[0] == "python scripts/run_v0_1_release_demo.py"
    assert joined[1] == "python -m pytest"
    assert joined[2] == (
        "python scripts/check_release_readiness.py "
        "--allow-pending-tag --require-artifacts --require-clean-git"
    )


def test_release_verification_cli_dry_run_outputs_commands() -> None:
    """CLI dry-run 应打印完整 pre-tag 命令链。"""

    completed = subprocess.run(
        [sys.executable, "scripts/verify_v0_1_release.py", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Release verification dry run" in completed.stdout
    assert "scripts/run_v0_1_release_demo.py" in completed.stdout
    assert "-m pytest" in completed.stdout
    assert "--require-clean-git" in completed.stdout
