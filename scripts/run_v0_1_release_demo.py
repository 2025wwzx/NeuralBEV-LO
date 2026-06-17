#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一命令运行 NeuralBEV-LO v0.1 release demo。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.release_demo import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_METRICS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_OUTPUT,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_VIDEO_OUTPUT,
    ReleaseDemoConfig,
    run_release_demo,
)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run NeuralBEV-LO v0.1 release demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sequence", type=str, default="07")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--video-output", type=Path, default=DEFAULT_VIDEO_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--video-seconds", type=float, default=6.0)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--title", type=str, default="NeuralBEV-LO v0.1 research prototype")
    parser.add_argument("--use-cuda", action="store_true", help="Do not force CPU for the demo.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """运行 v0.1 release demo wrapper。"""

    args = _parse_args()
    config = ReleaseDemoConfig(
        eval_config=args.config,
        train_config=args.train_config,
        checkpoint=args.checkpoint,
        sequence=args.sequence,
        frames=args.frames,
        start_frame=args.start_frame,
        data_root=args.data_root,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        video_output=args.video_output,
        report_output=args.report_output,
        cpu=not args.use_cuda,
        video_seconds=args.video_seconds,
        video_fps=args.video_fps,
        title=args.title,
    )
    return run_release_demo(config=config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
