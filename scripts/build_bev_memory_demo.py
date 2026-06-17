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
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np
import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.bev.memory import BevMemoryConfig, initialize_memory, update_memory  # noqa: E402
from neuralbev_lo.bev.rasterizer import (  # noqa: E402
    BevGridConfig,
    bev_config_from_mapping,
    rasterize_point_cloud,
)
from neuralbev_lo.data.kitti_dataset import (  # noqa: E402
    build_sequence_paths,
    list_velodyne_files,
    load_velodyne_frame,
)
from neuralbev_lo.data.point_filters import PointFilterConfig, filter_point_cloud_for_bev  # noqa: E402
from neuralbev_lo.data.pose_utils import (  # noqa: E402
    camera_pose_to_lidar_pose,
    get_velo_to_cam,
    load_kitti_poses,
    parse_calibration_file,
    relative_transform,
)
from neuralbev_lo.eval.bev_metrics import (  # noqa: E402
    bev_consistency_table,
    memory_quality_table,
    save_memory_metrics_csv,
    save_memory_metrics_json,
)
from neuralbev_lo.eval.odometry_metrics import integrate_relative_poses  # noqa: E402
from neuralbev_lo.geometry.se2 import se2_from_xyyaw  # noqa: E402
from neuralbev_lo.models.losses import denormalize_pose_batch  # noqa: E402
from neuralbev_lo.models.posenet import build_posenet_from_config, stack_bev_pair  # noqa: E402
from neuralbev_lo.training.checkpoint import load_checkpoint  # noqa: E402
from neuralbev_lo.utils.config import load_yaml_config  # noqa: E402
from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.viz.render_bev import save_bev_comparison  # noqa: E402
from neuralbev_lo.viz.render_metrics import save_metric_curve  # noqa: E402
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
    parser.add_argument("--resolution-m", type=float, default=None)
    parser.add_argument("--filter-z-range", type=float, nargs=2, default=None, metavar=("MIN_Z", "MAX_Z"))
    parser.add_argument(
        "--filter-distance-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_R", "MAX_R"),
    )
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


class RuntimeProfiler:
    """按阶段输出 runtime profiling 日志。"""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    @contextmanager
    def stage(self, name: str, **context: Any) -> Iterator[None]:
        """记录一个运行阶段的耗时。"""

        if not self.enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            log_info(
                "runtime stage completed",
                stage=name,
                elapsed_ms=f"{elapsed_ms:.3f}",
                **context,
            )


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

    if eval_config.get("bev"):
        return bev_config_from_mapping(eval_config.get("bev", {}))
    configured = PROJECT_ROOT / str(
        eval_config.get("data", {}).get("train_config", train_config_path)
    )
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


def _resolve_selected_channel_indices(
    metrics_config: dict[str, Any],
    bev_config: BevGridConfig,
) -> tuple[list[str], tuple[int, ...]]:
    """把 metrics 配置中的通道名解析为 BEV 通道索引。"""

    raw_channels = metrics_config.get("selected_channels", ["density"])
    if not isinstance(raw_channels, list | tuple) or not raw_channels:
        raise ValueError("metrics.selected_channels must be a non-empty list")
    selected_channels = [str(channel) for channel in raw_channels]
    indices: list[int] = []
    for channel_name in selected_channels:
        if channel_name not in bev_config.channels:
            raise ValueError(
                f"selected channel {channel_name} is not in BEV channels {bev_config.channels}"
            )
        indices.append(bev_config.channels.index(channel_name))
    return selected_channels, tuple(indices)


def _resolve_point_filter_config(
    eval_config: dict[str, Any],
    args: argparse.Namespace,
) -> PointFilterConfig:
    """从 eval config 和 CLI 覆盖项解析点云过滤配置。"""

    filter_section = eval_config.get("filters", {})
    z_range = _optional_range_tuple(args.filter_z_range, filter_section.get("z_range_m"))
    distance_range = _optional_range_tuple(
        args.filter_distance_range,
        filter_section.get("distance_range_m"),
    )
    return PointFilterConfig(z_range_m=z_range, distance_range_m=distance_range)


def _optional_range_tuple(
    cli_value: list[float] | tuple[float, float] | None,
    config_value: object,
) -> tuple[float, float] | None:
    """解析 CLI 或 YAML 中的二元范围。"""

    value = cli_value if cli_value is not None else config_value
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("filter range must contain exactly two values")
    return float(value[0]), float(value[1])


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """把 BEV tensor 复制为 numpy，避免后续 inplace 或引用共享影响历史指标。"""

    return tensor.detach().cpu().numpy().copy()


def _load_lidar_poses(paths) -> np.ndarray:
    """读取 KITTI camera pose 并转换为内部 `T_world_lidar`。"""

    camera_poses = load_kitti_poses(paths.pose_path)
    calibrations = parse_calibration_file(paths.calib_path)
    velo_to_cam = get_velo_to_cam(calibrations)
    return np.stack(
        [camera_pose_to_lidar_pose(camera_pose, velo_to_cam) for camera_pose in camera_poses],
        axis=0,
    )


def _synthetic_bev_sequence(
    frames: int,
    config: BevGridConfig,
) -> tuple[list[np.ndarray], np.ndarray, str]:
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
    filter_config: PointFilterConfig,
    profiler: RuntimeProfiler,
) -> tuple[list[np.ndarray], np.ndarray, str]:
    """读取 KITTI BEV 序列和 GT relative poses。"""

    with profiler.stage("data_loading", source="kitti_metadata", sequence=sequence):
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
        with profiler.stage("data_loading", source="kitti_velodyne", frame_index=frame_index):
            points = load_velodyne_frame(frame_files[frame_index])
        with profiler.stage("rasterization", frame_index=frame_index):
            filtered_points = filter_point_cloud_for_bev(points, filter_config)
            bevs.append(rasterize_point_cloud(filtered_points, config))
        if filter_config.z_range_m is not None or filter_config.distance_range_m is not None:
            log_info(
                "point filter applied",
                frame_index=frame_index,
                points_before=points.shape[0],
                points_after=filtered_points.shape[0],
            )
    relatives = []
    for frame_index in range(start_frame + 1, end_frame):
        transform = relative_transform(lidar_poses[frame_index - 1], lidar_poses[frame_index])
        relatives.append(
            [
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(np.arctan2(transform[1, 0], transform[0, 0])),
            ]
        )
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
    mean = torch.as_tensor(
        pose_norm.get("mean", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device=device,
    )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, list[np.ndarray]]]:
    """更新 naive / GT / learned 三种 memory。"""

    naive_memory = torch.as_tensor(bevs[0], dtype=torch.float32)
    gt_memory = initialize_memory(bevs[0])
    learned_memory = initialize_memory(bevs[0]) if learned_relative is not None else None
    memory_histories: dict[str, list[np.ndarray]] = {
        "naive": [_tensor_to_numpy(naive_memory)],
        "gt_pose": [_tensor_to_numpy(gt_memory)],
    }
    if learned_memory is not None:
        memory_histories["learned_pose"] = [_tensor_to_numpy(learned_memory)]
    for pair_index, current_bev in enumerate(bevs[1:]):
        current_tensor = torch.as_tensor(current_bev, dtype=torch.float32)
        gt_transform = se2_from_xyyaw(*gt_relative[pair_index].tolist())
        gt_memory = update_memory(
            gt_memory,
            current_tensor,
            gt_transform,
            bev_config,
            memory_config,
        )
        naive_memory = (
            memory_config.alpha * naive_memory
            + (1.0 - memory_config.alpha) * current_tensor
        )
        if learned_memory is not None and learned_relative is not None:
            learned_transform = se2_from_xyyaw(*learned_relative[pair_index].tolist())
            learned_memory = update_memory(
                learned_memory,
                current_tensor,
                learned_transform,
                bev_config,
                memory_config,
            )
        memory_histories["naive"].append(_tensor_to_numpy(naive_memory))
        memory_histories["gt_pose"].append(_tensor_to_numpy(gt_memory))
        if learned_memory is not None:
            memory_histories["learned_pose"].append(_tensor_to_numpy(learned_memory))
    return naive_memory, gt_memory, learned_memory, memory_histories


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
    if args.resolution_m is not None and args.resolution_m <= 0:
        log_warn("resolution-m must be positive", resolution_m=args.resolution_m)
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
        if args.resolution_m is not None:
            bev_config = replace(bev_config, resolution_m=float(args.resolution_m))
        memory_section = eval_config.get("bev_memory", {})
        metrics_section = eval_config.get("metrics", {})
        runtime_section = eval_config.get("runtime", {})
        filter_config = _resolve_point_filter_config(eval_config, args)
        profile_runtime = args.profile_runtime or bool(runtime_section.get("profile_runtime", False))
        profiler = RuntimeProfiler(profile_runtime)
        memory_config = BevMemoryConfig(
            policy=str(memory_section.get("policy", "decay")),
            alpha=float(memory_section.get("alpha", 0.9)),
        )
        occupancy_threshold = float(metrics_section.get("occupancy_threshold", 0.1))
        flicker_window = int(
            metrics_section.get("flicker_window", memory_section.get("temporal_window", 5))
        )
        selected_channels, channel_indices = _resolve_selected_channel_indices(
            metrics_section,
            bev_config,
        )
        if args.synthetic:
            with profiler.stage("data_loading", source="synthetic"):
                with profiler.stage("rasterization", source="synthetic"):
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
                filter_config=filter_config,
                profiler=profiler,
            )
            start_frame = args.start_frame
            end_frame = args.start_frame + len(bevs) - 1

        runtime_device = str(runtime_section.get("device", "")).lower()
        force_cpu = args.cpu or runtime_device == "cpu"
        device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        learned_relative = None
        if args.pose_source == "learned":
            with profiler.stage("inference", device=device.type):
                learned_relative = _predict_learned_relative(
                    train_config,
                    args.checkpoint,
                    bevs,
                    device=device,
                )
        else:
            with profiler.stage("inference", device=device.type, skipped=True):
                learned_relative = None
        with profiler.stage("warp", memory_policy=memory_config.policy):
            naive_memory, gt_memory, learned_memory, memory_histories = _update_memories(
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
        metrics_csv_path = args.metrics_dir / f"{prefix}_memory_metrics.csv"
        consistency_metrics_path = args.metrics_dir / f"{prefix}_consistency_metrics.json"
        consistency_csv_path = args.metrics_dir / f"{prefix}_consistency_metrics.csv"
        alignment_curve_path = args.output_dir / f"{prefix}_alignment_curve.png"
        metrics_rows = memory_quality_table(
            reference_memory=gt_memory.detach().cpu().numpy(),
            candidates=candidates,
            occupancy_threshold=occupancy_threshold,
        )
        memory_metrics_metadata = {
            "sequence": sequence,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": len(bevs),
            "pose_source": args.pose_source,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "memory_policy": memory_config.policy,
            "alpha": memory_config.alpha,
            "resolution_m": bev_config.resolution_m,
            "filter_z_range": filter_config.z_range_m,
            "filter_distance_range": filter_config.distance_range_m,
            "device": device.type,
            "profile_runtime": profile_runtime,
        }
        consistency_rows = bev_consistency_table(
            current_bevs=bevs,
            memory_histories=memory_histories,
            occupancy_threshold=occupancy_threshold,
            channel_indices=channel_indices,
            flicker_window=flicker_window,
        )
        consistency_metadata = {
            "sequence": sequence,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": len(bevs),
            "pose_source": args.pose_source,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "memory_policy": memory_config.policy,
            "alpha": memory_config.alpha,
            "resolution_m": bev_config.resolution_m,
            "filter_z_range": filter_config.z_range_m,
            "filter_distance_range": filter_config.distance_range_m,
            "device": device.type,
            "profile_runtime": profile_runtime,
            "metrics_definition": {
                "alignment_score": "mean_iou(thresholded_current_bev, thresholded_memory_bev)",
                "flicker_score": "mean(pixel_std(memory_window, dim=time))",
                "occupancy_threshold": occupancy_threshold,
                "selected_channels": selected_channels,
                "selected_channel_indices": list(channel_indices),
                "flicker_window": flicker_window,
            },
        }
        with profiler.stage("rendering"):
            save_bev_comparison(panels, image_path)
            save_trajectory_overlay(trajectories, trajectory_path, title=f"BEV memory {sequence}")
            save_memory_metrics_json(
                metrics_rows,
                metrics_path,
                metadata=memory_metrics_metadata,
            )
            save_memory_metrics_csv(metrics_rows, metrics_csv_path)
            save_memory_metrics_json(
                consistency_rows,
                consistency_metrics_path,
                metadata=consistency_metadata,
            )
            save_memory_metrics_csv(consistency_rows, consistency_csv_path)
            save_metric_curve(
                consistency_rows,
                alignment_curve_path,
                metric_name="alignment_iou",
                title=f"BEV alignment IoU {sequence}",
            )
    except Exception as exc:  # noqa: BLE001 - CLI 需要上下文后返回非零。
        log_warn(
            "BEV memory demo failed",
            sequence=args.sequence,
            pose_source=args.pose_source,
            error=exc,
        )
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
        metrics_csv=metrics_csv_path,
        consistency_metrics=consistency_metrics_path,
        consistency_csv=consistency_csv_path,
        alignment_curve=alignment_curve_path,
        filter_z_range=filter_config.z_range_m,
        filter_distance_range=filter_config.distance_range_m,
        device=device.type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
