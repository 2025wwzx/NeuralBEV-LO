#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预览 KITTI 相邻帧 3DoF pose 标签分布。"""

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

from neuralbev_lo.data.pair_dataset import (  # noqa: E402
    compute_pose_label_stats,
    resolve_sequences_for_split,
)
from neuralbev_lo.utils.config import load_yaml_config  # noqa: E402
from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "train" / "posenet_3dof.yaml"
DEFAULT_OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "outputs" / "metrics" / "pose_label_stats.json"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Preview KITTI adjacent-frame 3DoF labels.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", type=str, default="train", choices=("smoke", "train", "val", "test"))
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--stats-max-pairs-per-sequence", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument(
        "--write-stats",
        type=Path,
        default=None,
        help=f"Optional path for writing train normalization stats, default example: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def _resolve_data_root(config: dict[str, Any], override: Path | None) -> Path:
    """从命令行、环境变量或 dataset config 解析 KITTI 根目录。"""

    if override is not None:
        return override

    dataset_config_path = PROJECT_ROOT / str(config.get("data", {}).get("dataset_config"))
    dataset_config = load_yaml_config(dataset_config_path)
    dataset_section = dataset_config.get("dataset", {})
    if not isinstance(dataset_section, dict):
        raise ValueError("dataset config must contain a dataset mapping")
    env_name = str(dataset_section.get("data_root_env", "KITTI_ROOT"))
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value)
    return PROJECT_ROOT / str(dataset_section.get("default_data_root", "data/kitti_odometry"))


def _format_vector(values: list[float]) -> str:
    """把三维向量格式化为短日志字符串。"""

    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写入 UTF-8 JSON 文件并创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    """执行标签统计预览。"""

    args = _parse_args()
    try:
        config = load_yaml_config(args.config)
        data_root = _resolve_data_root(config, args.data_root)

        pose_section = config.get("pose", {})
        if not isinstance(pose_section, dict):
            raise ValueError("config.pose must be a mapping")
        pose_norm_section = pose_section.get("pose_norm", {})
        if not isinstance(pose_norm_section, dict):
            raise ValueError("config.pose.pose_norm must be a mapping")
        source_split = str(pose_norm_section.get("source_split", "train"))
        norm_sequences = resolve_sequences_for_split(config, source_split)
        preview_sequences = (
            (args.sequence.strip().zfill(2),)
            if isinstance(args.sequence, str) and args.sequence.strip()
            else resolve_sequences_for_split(config, args.split)
        )

        norm_stats = compute_pose_label_stats(
            data_root,
            norm_sequences,
            source_split=source_split,
            max_pairs_per_sequence=args.stats_max_pairs_per_sequence,
        )
        preview_stats = compute_pose_label_stats(
            data_root,
            preview_sequences,
            source_split="custom" if args.sequence else args.split,
            max_pairs_per_sequence=args.max_pairs,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 需要完整上下文后返回非零。
        log_warn(
            "Pose label preview failed",
            config=args.config,
            data_root=args.data_root,
            split=args.split,
            sequence=args.sequence,
            error=exc,
        )
        return 1

    payload = {
        "pose_normalization": norm_stats.as_dict(),
        "preview_distribution": preview_stats.as_dict(),
        "label_order": ["dx_m", "dy_m", "yaw_rad"],
    }

    if args.write_stats is not None:
        _write_json(args.write_stats, payload)
        log_info("Pose label stats written", output=args.write_stats)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        log_info(
            "Pose normalization statistics",
            source_split=norm_stats.source_split,
            sequences=",".join(norm_stats.sequences),
            count=norm_stats.count,
            mean=_format_vector(norm_stats.as_dict()["mean"]),
            std=_format_vector(norm_stats.as_dict()["std"]),
        )
        log_info(
            "Pose label distribution preview",
            split=preview_stats.source_split,
            sequences=",".join(preview_stats.sequences),
            count=preview_stats.count,
            mean=_format_vector(preview_stats.as_dict()["mean"]),
            std=_format_vector(preview_stats.as_dict()["std"]),
            min=_format_vector(preview_stats.as_dict()["min"]),
            max=_format_vector(preview_stats.as_dict()["max"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
