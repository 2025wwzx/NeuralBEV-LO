#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""写入 NeuralBEV-LO v0.1 release manifest。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.logging import log_info  # noqa: E402
from neuralbev_lo.utils.release_manifest import (  # noqa: E402
    DEFAULT_TAG_TARGET,
    write_release_manifest,
)

DEFAULT_MEMORY_METRICS: Final[Path] = Path(
    "outputs/metrics/v0_1_release_demo/07_000000_000004_memory_metrics.json"
)
DEFAULT_CONSISTENCY_METRICS: Final[Path] = Path(
    "outputs/metrics/v0_1_release_demo/07_000000_000004_consistency_metrics.json"
)
DEFAULT_OUTPUT: Final[Path] = Path("outputs/reports/v0_1_release_manifest.json")
DEFAULT_ARTIFACTS: Final[tuple[Path, ...]] = (
    Path("outputs/figures/v0_1_release_demo/07_000000_000004_memory.png"),
    Path("outputs/figures/v0_1_release_demo/07_000000_000004_trajectory.png"),
    Path("outputs/figures/v0_1_release_demo/07_000000_000004_alignment_curve.png"),
    Path("outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4"),
    DEFAULT_MEMORY_METRICS,
    DEFAULT_CONSISTENCY_METRICS,
    Path("outputs/reports/v0_1_release_demo.txt"),
)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Write NeuralBEV-LO v0.1 release manifest.")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--memory-metrics", type=Path, default=DEFAULT_MEMORY_METRICS)
    parser.add_argument("--consistency-metrics", type=Path, default=DEFAULT_CONSISTENCY_METRICS)
    parser.add_argument("--artifact", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tag-target", type=str, default=DEFAULT_TAG_TARGET)
    return parser.parse_args()


def main() -> int:
    """生成并写入 v0.1 release manifest。"""

    args = _parse_args()
    artifact_paths = args.artifact if args.artifact is not None else list(DEFAULT_ARTIFACTS)
    output_path = write_release_manifest(
        output_path=args.output,
        repo_root=args.repo_root,
        artifact_paths=artifact_paths,
        memory_metrics_path=args.memory_metrics,
        consistency_metrics_path=args.consistency_metrics,
        tag_target=args.tag_target,
    )
    log_info("Release manifest written", output=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
