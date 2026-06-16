#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打印 KITTI Odometry 单序列预览摘要。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.data.kitti_dataset import summarize_kitti_sequence
from neuralbev_lo.utils.config import load_yaml_config
from neuralbev_lo.utils.logging import log_info, log_warn

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "dataset" / "kitti.yaml"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Preview a KITTI Odometry sequence.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    return parser.parse_args()


def _resolve_data_root(config: dict[str, Any], override: Path | None) -> Path:
    """从命令行、环境变量或配置中解析 KITTI 根目录。"""

    if override is not None:
        return override
    dataset_config = config.get("dataset", {})
    env_name = dataset_config.get("data_root_env", "KITTI_ROOT")
    env_value = os.environ.get(str(env_name), "").strip()
    if env_value:
        return Path(env_value)
    return PROJECT_ROOT / str(dataset_config.get("default_data_root", "data/kitti_odometry"))


def main() -> int:
    """打印序列摘要。"""

    args = _parse_args()
    config = load_yaml_config(args.config)
    data_root = _resolve_data_root(config, args.data_root)

    try:
        summary = summarize_kitti_sequence(data_root, args.sequence)
    except Exception as exc:  # noqa: BLE001 - CLI 需要输出上下文后返回非零。
        log_warn("KITTI preview failed", data_root=data_root, sequence=args.sequence, error=exc)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        log_info(
            "KITTI sequence preview",
            sequence=summary["sequence"],
            frames=summary["frame_count"],
            poses=summary["pose_count"],
            timestamps=summary["timestamp_count"],
            first_timestamp=summary["first_timestamp"],
            last_timestamp=summary["last_timestamp"],
            first_frame_shape=summary["first_frame_shape"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
