#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行 Week 1 合成烟测。

该脚本只验证配置读取、随机种子、基础张量形状和 structured logging，不依赖 KITTI
数据或 CUDA。后续阶段会扩展为真实 data-to-model smoke pipeline。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.config import load_yaml_config
from neuralbev_lo.utils.logging import log_info, log_warn
from neuralbev_lo.utils.seed import set_global_seed

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "eval" / "kitti_eval.yaml"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run NeuralBEV-LO Week 1 smoke pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--synthetic", action="store_true", help="Run without KITTI data.")
    return parser.parse_args()


def _build_synthetic_bev(height: int = 16, width: int = 16, channels: int = 4) -> np.ndarray:
    """构造确定性的合成 BEV 张量。

    参数:
        height: BEV 高度。
        width: BEV 宽度。
        channels: BEV 通道数。

    返回:
        `float32[channels, height, width]` 合成 BEV。
    """

    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("height, width, and channels must be positive")

    bev = np.zeros((channels, height, width), dtype=np.float32)
    bev[0, height // 2, width // 2] = 1.0
    bev[1, :, width // 2] = 0.5
    bev[2, height // 2, :] = 0.25
    bev[3] = np.linspace(0.0, 1.0, num=height * width, dtype=np.float32).reshape(height, width)
    return bev


def main() -> int:
    """执行合成烟测。"""

    args = _parse_args()
    config = load_yaml_config(args.config)
    seed = int(config.get("experiment", {}).get("seed", 20260608))
    set_global_seed(seed)

    if not args.synthetic:
        log_warn("Only synthetic Week 1 smoke is implemented; proceeding in synthetic mode")

    bev = _build_synthetic_bev()
    log_info(
        "Synthetic BEV smoke completed",
        config=args.config,
        shape=bev.shape,
        dtype=bev.dtype,
        seed=seed,
        nonzero=int(np.count_nonzero(bev)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
