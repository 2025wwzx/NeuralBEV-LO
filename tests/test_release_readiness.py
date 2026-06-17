#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 12 release readiness 测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from neuralbev_lo.utils.release_readiness import collect_release_readiness


def test_release_readiness_checks_current_repo() -> None:
    """当前仓库应满足 v0.1 release readiness 的静态检查。"""

    result = collect_release_readiness(Path.cwd())

    assert result.passed, result.format_text()
    assert "required_docs" in result.checks
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
