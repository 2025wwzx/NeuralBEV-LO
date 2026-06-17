#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""写入 NeuralBEV-LO v0.1 final report。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.final_report import write_final_report  # noqa: E402
from neuralbev_lo.utils.logging import log_info  # noqa: E402

DEFAULT_MEMORY_METRICS: Final[Path] = Path(
    "outputs/metrics/week12_m3_freeze/07_000000_000004_memory_metrics.json"
)
DEFAULT_CONSISTENCY_METRICS: Final[Path] = Path(
    "outputs/metrics/week12_m3_freeze/07_000000_000004_consistency_metrics.json"
)
DEFAULT_CONFIG: Final[Path] = Path("configs/eval/kitti_cpu_smoke_safe.yaml")
DEFAULT_OUTPUT: Final[Path] = Path("outputs/reports/week12_v0_1_final_report.txt")
DEFAULT_ARTIFACTS: Final[tuple[Path, ...]] = (
    Path("outputs/figures/week12_m3_freeze/07_000000_000004_memory.png"),
    Path("outputs/figures/week12_m3_freeze/07_000000_000004_trajectory.png"),
    Path("outputs/figures/week12_m3_freeze/07_000000_000004_alignment_curve.png"),
    Path("outputs/figures/week12_m3_freeze/neuralbev_lo_v0_1_demo.mp4"),
    DEFAULT_MEMORY_METRICS,
    DEFAULT_CONSISTENCY_METRICS,
)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Write NeuralBEV-LO v0.1 final report.")
    parser.add_argument("--memory-metrics", type=Path, default=DEFAULT_MEMORY_METRICS)
    parser.add_argument("--consistency-metrics", type=Path, default=DEFAULT_CONSISTENCY_METRICS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """生成 v0.1 final report 并写入磁盘。"""

    args = _parse_args()
    artifact_paths = args.artifact if args.artifact is not None else list(DEFAULT_ARTIFACTS)
    output_path = write_final_report(
        output_path=args.output,
        memory_metrics_path=args.memory_metrics,
        consistency_metrics_path=args.consistency_metrics,
        config_path=args.config,
        artifact_paths=artifact_paths,
    )
    log_info("Final report written", output=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
