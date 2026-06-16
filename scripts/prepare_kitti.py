#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 KITTI Odometry 数据目录并写出本地 manifest。

脚本不会下载数据，也不会把数据写入 git 追踪目录；manifest 默认保存到
`outputs/manifests/`，该目录由 `.gitignore` 忽略。
"""

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
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "manifests"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Validate an existing KITTI Odometry layout.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    """运行 KITTI 数据校验并写出 manifest。"""

    args = _parse_args()
    config = load_yaml_config(args.config)
    data_root = _resolve_data_root(config, args.data_root)

    try:
        summary = summarize_kitti_sequence(data_root, args.sequence)
    except Exception as exc:  # noqa: BLE001 - CLI 需要输出上下文后返回非零。
        log_warn("KITTI validation failed", data_root=data_root, sequence=args.sequence, error=exc)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"kitti_sequence_{summary['sequence']}_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log_info(
        "KITTI validation completed",
        sequence=summary["sequence"],
        frames=summary["frame_count"],
        poses=summary["pose_count"],
        timestamps=summary["timestamp_count"],
        manifest=manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
