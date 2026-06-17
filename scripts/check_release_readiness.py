#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 NeuralBEV-LO v0.1 release readiness。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.utils.release_readiness import collect_release_readiness  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Check v0.1 release readiness.")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--allow-pending-tag",
        action="store_true",
        help="Allow the release tag check to remain pending until user approval.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Require local v0.1 release demo artifacts to exist and be readable.",
    )
    return parser.parse_args()


def main() -> int:
    """运行 release readiness 检查。"""

    args = _parse_args()
    result = collect_release_readiness(args.repo_root, require_artifacts=args.require_artifacts)
    print(result.format_text())
    failing = [
        check
        for check in result.checks.values()
        if check.status == "fail" or (check.status == "pending" and not args.allow_pending_tag)
    ]
    if failing:
        log_warn("Release readiness check failed", failed=",".join(check.name for check in failing))
        return 1
    log_info("Release readiness check completed", checks=len(result.checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
