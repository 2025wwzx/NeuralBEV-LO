#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行 NeuralBEV-LO v0.1 tag 前最终审计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.release_verification import run_release_verification  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Verify NeuralBEV-LO v0.1 before tagging.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """运行 v0.1 tag 前最终审计。"""

    args = _parse_args()
    return run_release_verification(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
