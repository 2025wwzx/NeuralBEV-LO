#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建 GT-pose / learned-pose BEV memory 对比 demo。

脚本读取 KITTI 或合成 BEV 序列，逐帧更新 naive、GT-pose 和可选 learned-pose
memory，并保存四联图、trajectory overlay 和 memory metrics JSON。Week 8 的 learned
分支会加载 PoseNet checkpoint，对相邻帧 BEV 预测 normalized 3DoF，再反归一化为
`dx, dy, yaw` 供 `update_memory` 使用。
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

from neuralbev_lo.bev.memory import BevMemoryConfig, initialize_memory, update_memory  # noqa: E402
from neuralbev_lo.bev.rasterizer import BevGridConfig, bev_config_from_mapping, rasterize_point_cloud  # noqa: E402
from neuralbev_lo.data.kitti_dataset import (  # noqa: E402
    build_sequence_paths,
    list_velodyne_files,
    load_velodyne_frame,
)
from neuralbev_lo.data.pose_utils import (  # noqa: E402
    camera_pose_to_lidar_pose,
    get_velo_to_cam,
    load_kitti_poses,
    parse_calibration_file,
    relative_transform,
)
from neuralbev_lo.eval.bev_metrics import memory_quality_table, save_memory_metrics_json  # noqa: E402
from neuralbev_lo.eval.odometry_metrics import integrate_relative_poses  # noqa: E402
from neuralbev_lo.geometry.se2 import se2_from_xyyaw  # noqa: E402
from neuralbev_lo.models.losses import denormalize_pose_batch  # noqa: E402
from neuralbev_lo.models.posenet import build_posenet_from_config, stack_bev_pair  # noqa: E402
from neuralbev_lo.training.checkpoint import load_checkpoint  # noqa: E402
from neuralbev_lo.utils.config import load_yaml_config  # noqa: E402
from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.viz.render_bev import save_bev_comparison  # noqa: E402
from neuralbev_lo.viz.render_trajectory import save_trajectory_overlay  # noqa: E402

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "eval" / "kitti_eval.yaml"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_METRICS_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_TRAIN_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "train" / "posenet_3dof.yaml"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build a BEV memory demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--pose-source", choices=("gt", "learned"), default="gt")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
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


def _load_bev_config(eval_config: dict[str, Any], train_config_path: Path) -> BevGridConfig:
    """从训练配置读取 BEV grid，保持 Week 3/4/8 的 BEV 定义一致。"""

    configured = PROJECT_ROOT / str(eval_config.get("data", {}).get("train_config", train_config_path))
    if configured.exists():
        train_config = load_yaml_config(configured)
    elif train_config_path.exists():
        train_config = load_yaml_config(train_config_path)
    else:
        train_config = {"bev": eval_config.get("bev", {})}
    return bev_config_from_mapping(train_config.get("bev", {}))


def _load_runtime_bev_config(
    eval_config: dict[str, Any],
    train_config_path: Path,
    *,
    checkpoint_path: Path | None,
    synthetic: bool,
) -> BevGridConfig:
    """解析 demo 使用的 BEV grid，synthetic checkpoint 优先使用 checkpoint config。"""

    if synthetic and checkpoint_path is not None:
        checkpoint_config = _read_checkpoint_config(checkpoint_path)
        if checkpoint_config.get("bev"):
            return bev_config_from_mapping(checkpoint_config.get("bev", {}))
    return _load_bev_config(eval_config, train_config_path)


def _load_lidar_poses(paths) -> np.ndarray:
    """读取 KITTI camera pose 并转换为内部 `T_world_lidar`。"""

    camera_poses = load_kitti_poses(paths.pose_path)
    calibrations = parse_calibration_file(paths.calib_path)
    velo_to_cam = get_velo_to_cam(calibrations)
    return np.stack(
        [camera_pose_to_lidar_pose(camera_pose, velo_to_cam) for camera_pose in camera_poses],
        axis=0,
    )


def _synthetic_bev_sequence(frames: int, config: BevGridConfig) -> tuple[list[np.ndarray], np.ndarray, str]:
    """构造合成 BEV 序列和 GT relative poses。"""

    if frames < 2:
        raise ValueError("synthetic memory demo requires at least 2 frames")
    bevs: list[np.ndarray] = []
    for index in range(frames):
        bev = np.zeros(config.tensor_shape, dtype=np.float32)
        row = min(index + 1, config.grid_shape[0] - 1)
        col = config.grid_shape[1] // 2
        bev[0, row, col] = 1.0
        bevs.append(bev)
    gt_relative = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (frames - 1, 1))
    return bevs, gt_relative, "synthetic"


def _kitti_bev_sequence(
    data_root: Path,
    sequence: str,
    *,
    start_frame: int,
    frames: int,
    config: BevGridConfig,
) -> tuple[list[np.ndarray], np.ndarray, str]:
    """读取 KITTI BEV 序列和 GT relative poses。"""

    paths = build_sequence_paths(data_root, sequence)
    frame_files = list_velodyne_files(paths.velodyne_dir)
    lidar_poses = _load_lidar_poses(paths)
    end_frame = min(start_frame + frames, len(frame_files), int(lidar_poses.shape[0]))
    if start_frame < 0 or start_frame >= end_frame:
        raise ValueError(f"start_frame {start_frame} out of range for {len(frame_files)} frames")
    if end_frame - start_frame < 2:
        raise ValueError("memory demo requires at least 2 frames")

    bevs: list[np.ndarray] = []
    for frame_index in range(start_frame, end_frame):
        points = load_velodyne_frame(frame_files[frame_index])
        bevs.append(rasterize_point_cloud(points, config))
    relatives = []
    for frame_index in range(start_frame + 1, end_frame):
        transform = relative_transform(lidar_poses[frame_index - 1], lidar_poses[frame_index])
        relatives.append([float(transform[0, 3]), float(transform[1, 3]), float(np.arctan2(transform[1, 0], transform[0, 0]))])
    return bevs, np.asarray(relatives, dtype=np.float64), paths.sequence


@torch.no_grad()
def _predict_learned_relative(
    train_config: dict[str, Any],
    checkpoint_path: Path,
    bevs: list[np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    """用 PoseNet checkpoint 预测相邻帧 learned relative poses。"""

    if checkpoint_path is None:
        raise ValueError("checkpoint is required when pose_source=learned")
    checkpoint_config = _read_checkpoint_config(checkpoint_path)
    model_config = checkpoint_config if checkpoint_config.get("model") else train_config
    model = build_posenet_from_config(model_config).to(device)
    metadata = load_checkpoint(checkpoint_path, model=model, map_location=device)
    checkpoint_config = metadata.get("config", {})
    checkpoint_pose_norm = {}
    if isinstance(checkpoint_config, dict) and checkpoint_config.get("pose"):
        checkpoint_pose_norm = checkpoint_config.get("pose", {}).get("pose_norm", {})
    pose_norm = checkpoint_pose_norm or train_config.get("pose", {}).get("pose_norm", {})
    mean = torch.as_tensor(pose_norm.get("mean", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
    std = torch.as_tensor(pose_norm.get("std", [1.0, 1.0, 1.0]), dtype=torch.float32, device=device)
    model.eval()
    predictions: list[np.ndarray] = []
    for prev_bev, curr_bev in zip(bevs[:-1], bevs[1:], strict=True):
        prev_tensor = torch.as_tensor(prev_bev, dtype=torch.float32, device=device).unsqueeze(0)
        curr_tensor = torch.as_tensor(curr_bev, dtype=torch.float32, device=device).unsqueeze(0)
        prediction_norm = model(stack_bev_pair(prev_tensor, curr_tensor))
        prediction = denormalize_pose_batch(prediction_norm, mean, std)[0].detach().cpu().numpy()
        predictions.append(prediction.astype(np.float64))
    return np.stack(predictions, axis=0)


def _read_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    """读取 checkpoint 中的 config，不加载模型权重到实例。"""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config must be a mapping: {checkpoint_path}")
    return config


def _update_memories(
    bevs: list[np.ndarray],
    gt_relative: np.ndarray,
    learned_relative: np.ndarray | None,
    *,
    bev_config: BevGridConfig,
    memory_config: BevMemoryConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """更新 naive / GT / learned 三种 memory。"""

    naive_memory = torch.as_tensor(bevs[0], dtype=torch.float32)
    gt_memory = initialize_memory(bevs[0])
    learned_memory = initialize_memory(bevs[0]) if learned_relative is not None else None
    for pair_index, current_bev in enumerate(bevs[1:]):
        current_tensor = torch.as_tensor(current_bev, dtype=torch.float32)
        gt_transform = se2_from_xyyaw(*gt_relative[pair_index].tolist())
        gt_memory = update_memory(gt_memory, current_tensor, gt_transform, bev_config, memory_config)
        naive_memory = memory_config.alpha * naive_memory + (1.0 - memory_config.alpha) * current_tensor
        if learned_memory is not None and learned_relative is not None:
            learned_transform = se2_from_xyyaw(*learned_relative[pair_index].tolist())
            learned_memory = update_memory(
                learned_memory,
                current_tensor,
                learned_transform,
                bev_config,
                memory_config,
            )
    return naive_memory, gt_memory, learned_memory


def main() -> int:
    """运行 BEV memory demo。"""

    args = _parse_args()
    if args.frames <= 1:
        log_warn("frames must be at least 2", frames=args.frames)
        return 1
    if args.start_frame < 0:
        log_warn("start-frame must be non-negative", start_frame=args.start_frame)
        return 1
    if args.pose_source == "learned" and args.checkpoint is None:
        log_warn("checkpoint is required for learned pose memory")
        return 1

    try:
        eval_config = load_yaml_config(args.config)
        train_config = load_yaml_config(args.train_config) if args.train_config.exists() else {}
        bev_config = _load_runtime_bev_config(
            eval_config,
            args.train_config,
            checkpoint_path=args.checkpoint,
            synthetic=args.synthetic,
        )
        memory_section = eval_config.get("bev_memory", {})
        memory_config = BevMemoryConfig(
            policy=str(memory_section.get("policy", "decay")),
            alpha=float(memory_section.get("alpha", 0.9)),
        )
        if args.synthetic:
            bevs, gt_relative, sequence = _synthetic_bev_sequence(args.frames, bev_config)
            start_frame = 0
            end_frame = args.frames - 1
        else:
            data_root = _resolve_data_root(eval_config, args.data_root)
            bevs, gt_relative, sequence = _kitti_bev_sequence(
                data_root,
                args.sequence,
                start_frame=args.start_frame,
                frames=args.frames,
                config=bev_config,
            )
            start_frame = args.start_frame
            end_frame = args.start_frame + len(bevs) - 1

        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        learned_relative = None
        if args.pose_source == "learned":
            learned_relative = _predict_learned_relative(
                train_config,
                args.checkpoint,
                bevs,
                device=device,
            )
        naive_memory, gt_memory, learned_memory = _update_memories(
            bevs,
            gt_relative,
            learned_relative,
            bev_config=bev_config,
            memory_config=memory_config,
        )

        panels = [
            ("current", bevs[-1]),
            ("naive", naive_memory.detach().cpu().numpy()),
            ("gt-pose", gt_memory.detach().cpu().numpy()),
        ]
        candidates = {
            "naive": naive_memory.detach().cpu().numpy(),
            "gt_pose": gt_memory.detach().cpu().numpy(),
        }
        trajectories = {
            "gt": integrate_relative_poses(gt_relative),
            "naive": integrate_relative_poses(np.zeros_like(gt_relative)),
        }
        if learned_memory is not None and learned_relative is not None:
            learned_np = learned_memory.detach().cpu().numpy()
            panels.append(("learned", learned_np))
            candidates["learned_pose"] = learned_np
            trajectories["learned"] = integrate_relative_poses(learned_relative)

        prefix = f"{sequence}_{start_frame:06d}_{end_frame:06d}"
        image_path = args.output_dir / f"{prefix}_memory.png"
        trajectory_path = args.output_dir / f"{prefix}_trajectory.png"
        metrics_path = args.metrics_dir / f"{prefix}_memory_metrics.json"
        save_bev_comparison(panels, image_path)
        save_trajectory_overlay(trajectories, trajectory_path, title=f"BEV memory {sequence}")
        metrics_rows = memory_quality_table(
            reference_memory=gt_memory.detach().cpu().numpy(),
            candidates=candidates,
            occupancy_threshold=float(eval_config.get("metrics", {}).get("occupancy_threshold", 0.1)),
        )
        save_memory_metrics_json(
            metrics_rows,
            metrics_path,
            metadata={
                "sequence": sequence,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frames": len(bevs),
                "pose_source": args.pose_source,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "memory_policy": memory_config.policy,
                "alpha": memory_config.alpha,
            },
        )
    except Exception as exc:  # noqa: BLE001 - CLI 需要上下文后返回非零。
        log_warn("BEV memory demo failed", sequence=args.sequence, pose_source=args.pose_source, error=exc)
        return 1

    log_info(
        "BEV memory demo completed",
        sequence=sequence,
        start_frame=start_frame,
        end_frame=end_frame,
        frames=len(bevs),
        pose_source=args.pose_source,
        image=image_path,
        trajectory=trajectory_path,
        metrics=metrics_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
