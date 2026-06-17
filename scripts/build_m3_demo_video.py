#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从已有 M3 图像产物生成 demo MP4。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.viz.make_demo_video import save_demo_video  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build NeuralBEV-LO M3 demo video.")
    parser.add_argument("--memory-image", type=Path, required=True)
    parser.add_argument("--trajectory-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--title", type=str, default="NeuralBEV-LO M3 demo")
    return parser.parse_args()


def main() -> int:
    """运行 M3 demo video 生成流程。"""

    args = _parse_args()
    try:
        output_path = save_demo_video(
            memory_image=args.memory_image,
            trajectory_image=args.trajectory_image,
            output_path=args.output,
            seconds=args.seconds,
            fps=args.fps,
            title=args.title,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 需要报告上下文并返回非零。
        log_warn(
            "M3 demo video failed",
            memory_image=args.memory_image,
            trajectory_image=args.trajectory_image,
            output=args.output,
            error=exc,
        )
        return 1

    log_info(
        "M3 demo video completed",
        memory_image=args.memory_image,
        trajectory_image=args.trajectory_image,
        output=output_path,
        seconds=args.seconds,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
