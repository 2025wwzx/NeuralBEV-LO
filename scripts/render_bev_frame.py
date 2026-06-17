#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""渲染单帧 KITTI LiDAR 为 BEV PNG。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.bev.rasterizer import bev_config_from_mapping, rasterize_point_cloud
from neuralbev_lo.data.kitti_dataset import build_sequence_paths, list_velodyne_files, load_velodyne_frame
from neuralbev_lo.utils.config import load_yaml_config
from neuralbev_lo.utils.logging import log_info, log_warn
from neuralbev_lo.viz.render_bev import save_bev_image

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "train" / "posenet_3dof.yaml"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "figures"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Render one KITTI LiDAR frame to a BEV PNG.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve_data_root(config: dict[str, Any], override: Path | None) -> Path:
    """从命令行、环境变量或 dataset config 解析 KITTI 根目录。"""

    if override is not None:
        return override

    dataset_config_path = PROJECT_ROOT / str(config.get("data", {}).get("dataset_config"))
    dataset_config = load_yaml_config(dataset_config_path)
    dataset_section = dataset_config.get("dataset", {})
    env_name = str(dataset_section.get("data_root_env", "KITTI_ROOT"))
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value)
    return PROJECT_ROOT / str(dataset_section.get("default_data_root", "data/kitti_odometry"))


def main() -> int:
    """执行单帧 BEV 渲染。"""

    args = _parse_args()
    if args.frame_index < 0:
        log_warn("Frame index must be non-negative", frame_index=args.frame_index)
        return 1

    config = load_yaml_config(args.config)
    data_root = _resolve_data_root(config, args.data_root)
    try:
        paths = build_sequence_paths(data_root, args.sequence)
        frame_files = list_velodyne_files(paths.velodyne_dir)
        if args.frame_index >= len(frame_files):
            raise ValueError(
                f"frame_index {args.frame_index} out of range for {len(frame_files)} frames"
            )
        points = load_velodyne_frame(frame_files[args.frame_index])
        bev_config = bev_config_from_mapping(config.get("bev", {}))
        bev = rasterize_point_cloud(points, bev_config)
        output_path = args.output_dir / f"kitti_{paths.sequence}_{args.frame_index:06d}_bev.png"
        save_bev_image(bev, output_path)
    except Exception as exc:  # noqa: BLE001 - CLI 需要报告上下文并返回非零。
        log_warn("BEV frame render failed", data_root=data_root, sequence=args.sequence, error=exc)
        return 1

    log_info(
        "BEV frame rendered",
        sequence=paths.sequence,
        frame_index=args.frame_index,
        points=points.shape[0],
        bev_shape=bev.shape,
        output=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
