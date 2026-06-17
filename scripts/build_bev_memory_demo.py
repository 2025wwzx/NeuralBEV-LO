#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建 KITTI GT-pose BEV memory demo。

脚本读取真实 KITTI 序列，逐帧 rasterize LiDAR 点云，并用 GT 相对位姿 warp
上一帧 memory，最终保存 current / naive / gt-pose memory 三联图。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.bev.memory import BevMemoryConfig, initialize_memory, update_memory
from neuralbev_lo.bev.rasterizer import bev_config_from_mapping, rasterize_point_cloud
from neuralbev_lo.data.kitti_dataset import (
    build_sequence_paths,
    list_velodyne_files,
    load_velodyne_frame,
)
from neuralbev_lo.data.pose_utils import (
    camera_pose_to_lidar_pose,
    get_velo_to_cam,
    load_kitti_poses,
    parse_calibration_file,
    relative_transform,
)
from neuralbev_lo.utils.config import load_yaml_config
from neuralbev_lo.utils.logging import log_info, log_warn
from neuralbev_lo.viz.render_bev import save_bev_comparison

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "eval" / "kitti_eval.yaml"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_TRAIN_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "train" / "posenet_3dof.yaml"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build a GT-pose BEV memory demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--start-frame", type=int, default=0)
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


def _load_bev_config(eval_config: dict[str, Any]):
    """从训练配置读取 BEV 网格，保持 Week 3/4 的 BEV 定义一致。"""

    train_config_path = PROJECT_ROOT / str(
        eval_config.get("data", {}).get("train_config", DEFAULT_TRAIN_CONFIG_PATH)
    )
    if not train_config_path.exists():
        train_config_path = DEFAULT_TRAIN_CONFIG_PATH
    train_config = load_yaml_config(train_config_path)
    return bev_config_from_mapping(train_config.get("bev", {}))


def _load_lidar_poses(paths) -> np.ndarray:
    """读取 KITTI camera pose 并转换为内部 `T_world_lidar`。"""

    camera_poses = load_kitti_poses(paths.pose_path)
    calibrations = parse_calibration_file(paths.calib_path)
    velo_to_cam = get_velo_to_cam(calibrations)
    return np.stack(
        [camera_pose_to_lidar_pose(camera_pose, velo_to_cam) for camera_pose in camera_poses],
        axis=0,
    )


def main() -> int:
    """运行 GT-pose BEV memory demo。"""

    args = _parse_args()
    if args.frames <= 0:
        log_warn("frames must be positive", frames=args.frames)
        return 1
    if args.start_frame < 0:
        log_warn("start-frame must be non-negative", start_frame=args.start_frame)
        return 1

    config = load_yaml_config(args.config)
    data_root = _resolve_data_root(config, args.data_root)
    bev_config = _load_bev_config(config)
    memory_section = config.get("bev_memory", {})
    memory_config = BevMemoryConfig(
        policy=str(memory_section.get("policy", "decay")),
        alpha=float(memory_section.get("alpha", 0.9)),
    )

    try:
        paths = build_sequence_paths(data_root, args.sequence)
        frame_files = list_velodyne_files(paths.velodyne_dir)
        lidar_poses = _load_lidar_poses(paths)
        end_frame = min(args.start_frame + args.frames, len(frame_files), int(lidar_poses.shape[0]))
        if args.start_frame >= end_frame:
            raise ValueError(
                f"start_frame {args.start_frame} out of range for {len(frame_files)} frames"
            )

        first_points = load_velodyne_frame(frame_files[args.start_frame])
        first_bev = rasterize_point_cloud(first_points, bev_config)
        current_bev = first_bev
        naive_memory = torch.as_tensor(first_bev, dtype=torch.float32)
        gt_memory = initialize_memory(first_bev)

        for frame_index in range(args.start_frame + 1, end_frame):
            points = load_velodyne_frame(frame_files[frame_index])
            current_bev = rasterize_point_cloud(points, bev_config)
            current_tensor = torch.as_tensor(current_bev, dtype=torch.float32)
            relative_pose = relative_transform(lidar_poses[frame_index - 1], lidar_poses[frame_index])
            gt_memory = update_memory(
                gt_memory,
                current_tensor,
                relative_pose,
                bev_config,
                memory_config,
            )
            naive_memory = memory_config.alpha * naive_memory + (1.0 - memory_config.alpha) * current_tensor

        output_path = args.output_dir / (
            f"kitti_{paths.sequence}_{args.start_frame:06d}_{end_frame - 1:06d}_gt_memory.png"
        )
        save_bev_comparison(
            [
                ("current", current_bev),
                ("naive", naive_memory.detach().cpu().numpy()),
                ("gt-pose", gt_memory.detach().cpu().numpy()),
            ],
            output_path,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 需要报告上下文后返回非零。
        log_warn("GT-pose BEV memory demo failed", data_root=data_root, sequence=args.sequence, error=exc)
        return 1

    log_info(
        "GT-pose BEV memory demo completed",
        sequence=paths.sequence,
        layout=paths.layout,
        start_frame=args.start_frame,
        end_frame=end_frame - 1,
        frames=end_frame - args.start_frame,
        memory_policy=memory_config.policy,
        alpha=memory_config.alpha,
        output=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
