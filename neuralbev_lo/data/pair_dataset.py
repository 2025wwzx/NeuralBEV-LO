#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KITTI 相邻帧 BEV Pair Dataset 与 3DoF 标签统计。

本模块把 Week 2 的 KITTI pose/loader 和 Week 3 的 BEV rasterizer 串起来，形成
PoseNet 训练前的数据契约：每个样本由相邻两帧 BEV、上一帧 LiDAR 坐标系下的
`dx, dy, yaw` 标签，以及可追踪的元数据组成。标签统计函数只接收显式传入的
sequence 列表，调用方必须用 train split 计算归一化参数，避免 val/test 泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from neuralbev_lo.bev.rasterizer import BevGridConfig, rasterize_point_cloud
from neuralbev_lo.data.kitti_dataset import (
    build_sequence_paths,
    list_velodyne_files,
    load_timestamps,
    load_velodyne_frame,
)
from neuralbev_lo.data.pose_utils import (
    camera_pose_to_lidar_pose,
    extract_se2,
    get_velo_to_cam,
    load_kitti_poses,
    parse_calibration_file,
    relative_transform,
)
from neuralbev_lo.data.split import (
    SMOKE_SEQUENCES,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    VAL_SEQUENCES,
)

LabelArray = NDArray[np.float32]


@dataclass(frozen=True)
class KittiPairIndex:
    """相邻帧样本索引。

    参数:
        sequence: 两位 KITTI sequence id。
        layout: 当前 sequence 使用的 merged 或 split 数据布局。
        prev_index: 前一帧帧号。
        curr_index: 当前帧帧号。
        prev_frame_path: 前一帧 Velodyne `.bin` 路径。
        curr_frame_path: 当前帧 Velodyne `.bin` 路径。
        prev_timestamp: 前一帧时间戳。
        curr_timestamp: 当前帧时间戳。
        target_pose_3dof: `dx, dy, yaw`，表达在 previous LiDAR/BEV frame。
    """

    sequence: str
    layout: str
    prev_index: int
    curr_index: int
    prev_frame_path: Path
    curr_frame_path: Path
    prev_timestamp: float
    curr_timestamp: float
    target_pose_3dof: tuple[float, float, float]

    def metadata(self) -> dict[str, Any]:
        """返回 DataLoader 默认 collate 友好的元数据字典。"""

        return {
            "sequence": self.sequence,
            "layout": self.layout,
            "prev_index": self.prev_index,
            "curr_index": self.curr_index,
            "prev_frame_path": str(self.prev_frame_path),
            "curr_frame_path": str(self.curr_frame_path),
            "prev_timestamp": self.prev_timestamp,
            "curr_timestamp": self.curr_timestamp,
        }


@dataclass(frozen=True)
class PoseLabelStats:
    """3DoF pose 标签分布与归一化统计。

    参数:
        source_split: 统计来源 split，训练归一化应为 `train`。
        sequences: 实际参与统计的 sequence 列表。
        count: 标签数量。
        mean/std/min/max: 形状为 `[3]` 的 `dx, dy, yaw` 统计量。
    """

    source_split: str
    sequences: tuple[str, ...]
    count: int
    mean: LabelArray
    std: LabelArray
    minimum: LabelArray
    maximum: LabelArray

    def as_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON/YAML 的普通字典。"""

        return {
            "source_split": self.source_split,
            "sequences": list(self.sequences),
            "count": self.count,
            "mean": [float(value) for value in self.mean.tolist()],
            "std": [float(value) for value in self.std.tolist()],
            "min": [float(value) for value in self.minimum.tolist()],
            "max": [float(value) for value in self.maximum.tolist()],
        }


class KittiAdjacentPairDataset(Dataset):
    """KITTI 相邻帧 BEV pair 数据集。

    数据集在初始化时只建立索引和 3DoF 标签，不加载点云；`__getitem__` 再读取相邻
    两帧 Velodyne 点云并 rasterize 为 BEV tensor。返回值为
    `(bev_prev, bev_curr, target_pose_3dof, metadata)`，其中前三项为 CPU
    `torch.float32` tensor，metadata 为可追踪路径、帧号、时间戳和 sequence 信息。
    """

    def __init__(
        self,
        data_root: str | Path,
        sequences: Sequence[str | int],
        bev_config: BevGridConfig,
        *,
        max_pairs_per_sequence: int | None = None,
        start_index: int = 0,
        pose_stats: PoseLabelStats | None = None,
        normalize_targets: bool = False,
    ) -> None:
        """构造相邻帧数据集。

        参数:
            data_root: KITTI Odometry 根目录，支持 merged 和官方 split 布局。
            sequences: 要使用的 sequence 列表，必须由调用方按 split 显式传入。
            bev_config: BEV rasterizer 配置。
            max_pairs_per_sequence: 每个 sequence 最多取多少个相邻 pair；`None` 表示全量。
            start_index: 每个 sequence 的起始 previous frame index。
            pose_stats: 可选 train split 统计量，用于目标归一化。
            normalize_targets: 为真时返回 `(target - mean) / std`。
        """

        if normalize_targets and pose_stats is None:
            raise ValueError("pose_stats is required when normalize_targets=True")
        self.data_root = Path(data_root)
        self.bev_config = bev_config
        self.pose_stats = pose_stats
        self.normalize_targets = normalize_targets
        self.samples = build_kitti_pair_indices(
            self.data_root,
            sequences,
            max_pairs_per_sequence=max_pairs_per_sequence,
            start_index=start_index,
        )

    def __len__(self) -> int:
        """返回 pair 样本数量。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """读取并 rasterize 一个相邻帧样本。

        异常会补充 sequence、帧号和路径上下文，避免批处理时难以定位坏样本。
        """

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"pair index out of range: {index}")
        sample = self.samples[index]
        try:
            prev_points = load_velodyne_frame(sample.prev_frame_path)
            curr_points = load_velodyne_frame(sample.curr_frame_path)
            bev_prev = rasterize_point_cloud(prev_points, self.bev_config)
            bev_curr = rasterize_point_cloud(curr_points, self.bev_config)
        except Exception as exc:
            raise RuntimeError(
                "failed to load KITTI BEV pair: "
                f"sequence={sample.sequence} prev={sample.prev_index} curr={sample.curr_index} "
                f"prev_path={sample.prev_frame_path} curr_path={sample.curr_frame_path}"
            ) from exc

        target = np.asarray(sample.target_pose_3dof, dtype=np.float32)
        if self.normalize_targets:
            target = normalize_pose_label(target, self.pose_stats)

        return (
            torch.as_tensor(bev_prev, dtype=torch.float32),
            torch.as_tensor(bev_curr, dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
            sample.metadata(),
        )


def resolve_sequences_for_split(config: dict[str, Any], split: str) -> tuple[str, ...]:
    """从训练配置中解析指定 split 的 sequence 列表。

    参数:
        config: 训练或评估 YAML 的顶层字典。
        split: `smoke`、`train`、`val`、`test` 或 `demo`。
    返回:
        归一化为两位字符串的 sequence 元组。
    """

    split_name = _clean_split_name(split)
    data_section = config.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("config.data must be a mapping")
    key_by_split = {
        "smoke": "smoke_sequences",
        "train": "train_sequences",
        "val": "val_sequences",
        "test": "test_sequences",
        "demo": "demo_sequences",
    }
    default_by_split = {
        "smoke": SMOKE_SEQUENCES,
        "train": TRAIN_SEQUENCES,
        "val": VAL_SEQUENCES,
        "test": TEST_SEQUENCES,
        "demo": TEST_SEQUENCES,
    }
    raw_sequences = data_section.get(key_by_split[split_name], default_by_split[split_name])
    return _normalize_sequences(raw_sequences)


def build_kitti_pair_indices(
    data_root: str | Path,
    sequences: Sequence[str | int],
    *,
    max_pairs_per_sequence: int | None = None,
    start_index: int = 0,
) -> list[KittiPairIndex]:
    """为一个或多个 KITTI sequence 构建相邻帧 pair 索引。

    该函数会读取 pose、calib 和 timestamps，并把 KITTI camera pose 转成内部
    `T_world_lidar` 后再计算 `T_prev_curr`。它不读取 Velodyne 点云内容，因此可用于
    快速统计标签分布。
    """

    sequence_ids = _normalize_sequences(sequences)
    _validate_pair_limits(max_pairs_per_sequence=max_pairs_per_sequence, start_index=start_index)
    all_indices: list[KittiPairIndex] = []
    for sequence in sequence_ids:
        try:
            all_indices.extend(
                _build_single_sequence_pair_indices(
                    Path(data_root),
                    sequence,
                    max_pairs_per_sequence=max_pairs_per_sequence,
                    start_index=start_index,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"failed to build KITTI pair indices for sequence {sequence}") from exc
    return all_indices


def collect_pose_labels(indices: Sequence[KittiPairIndex]) -> LabelArray:
    """从 pair 索引中收集 `float32[N, 3]` pose 标签数组。"""

    if not indices:
        raise ValueError("at least one KITTI pair index is required")
    labels = np.asarray([item.target_pose_3dof for item in indices], dtype=np.float32)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(f"pose labels must have shape [N, 3], got {labels.shape}")
    if not np.all(np.isfinite(labels)):
        raise ValueError("pose labels must contain only finite values")
    return labels


def compute_pose_label_stats(
    data_root: str | Path,
    sequences: Sequence[str | int],
    *,
    source_split: str = "train",
    max_pairs_per_sequence: int | None = None,
    start_index: int = 0,
    min_std: float = 1.0e-6,
) -> PoseLabelStats:
    """计算 3DoF pose 标签统计量。

    参数:
        data_root: KITTI Odometry 根目录。
        sequences: 参与统计的 sequence 列表；训练归一化必须只传 train split。
        source_split: 写入日志/元数据的 split 名称。
        max_pairs_per_sequence: 可选 smoke 限制；正式 train stats 应使用全量 train split。
        start_index: 起始 previous frame index。
        min_std: std 下限，避免后续归一化除零。
    """

    if min_std <= 0:
        raise ValueError("min_std must be positive")
    sequence_ids = _normalize_sequences(sequences)
    indices = build_kitti_pair_indices(
        data_root,
        sequence_ids,
        max_pairs_per_sequence=max_pairs_per_sequence,
        start_index=start_index,
    )
    labels = collect_pose_labels(indices)
    std = np.maximum(labels.std(axis=0), np.float32(min_std)).astype(np.float32)
    return PoseLabelStats(
        source_split=source_split,
        sequences=sequence_ids,
        count=int(labels.shape[0]),
        mean=labels.mean(axis=0).astype(np.float32),
        std=std,
        minimum=labels.min(axis=0).astype(np.float32),
        maximum=labels.max(axis=0).astype(np.float32),
    )


def normalize_pose_label(label: object, stats: PoseLabelStats | None) -> LabelArray:
    """按 train split 统计量归一化单个 `dx, dy, yaw` 标签。"""

    if stats is None:
        raise ValueError("pose label stats are required for normalization")
    label_array = np.asarray(label, dtype=np.float32)
    if label_array.shape != (3,):
        raise ValueError(f"pose label must have shape [3], got {label_array.shape}")
    if not np.all(np.isfinite(label_array)):
        raise ValueError("pose label must contain only finite values")
    return ((label_array - stats.mean) / stats.std).astype(np.float32)


def pose_label_stats_from_config(config: dict[str, Any]) -> PoseLabelStats:
    """从训练配置的 `pose.pose_norm` 节点读取已计算的 train stats。

    该函数只解析显式写入配置的统计量，不会现场扫描数据集；如果配置缺少 mean/std
    等字段，会抛出可定位的错误，提醒调用方先运行 `scripts/preview_pose_labels.py`。
    """

    pose_section = config.get("pose", {})
    if not isinstance(pose_section, dict):
        raise ValueError("config.pose must be a mapping")
    pose_norm = pose_section.get("pose_norm", {})
    if not isinstance(pose_norm, dict):
        raise ValueError("config.pose.pose_norm must be a mapping")

    return PoseLabelStats(
        source_split=str(pose_norm.get("source_split", "train")),
        sequences=_normalize_sequences(pose_norm.get("sequences", TRAIN_SEQUENCES)),
        count=_positive_int(pose_norm.get("count"), "pose.pose_norm.count"),
        mean=_vector3(pose_norm.get("mean"), "pose.pose_norm.mean"),
        std=_positive_vector3(pose_norm.get("std"), "pose.pose_norm.std"),
        minimum=_vector3(pose_norm.get("min"), "pose.pose_norm.min"),
        maximum=_vector3(pose_norm.get("max"), "pose.pose_norm.max"),
    )


def _build_single_sequence_pair_indices(
    data_root: Path,
    sequence: str,
    *,
    max_pairs_per_sequence: int | None,
    start_index: int,
) -> list[KittiPairIndex]:
    """构建单个 sequence 的相邻帧 pair 索引。"""

    paths = build_sequence_paths(data_root, sequence)
    frame_files = list_velodyne_files(paths.velodyne_dir)
    timestamps = load_timestamps(paths.times_path)
    camera_poses = load_kitti_poses(paths.pose_path)
    _validate_sequence_counts(frame_files, timestamps, camera_poses, sequence=sequence)

    calibrations = parse_calibration_file(paths.calib_path)
    velo_to_cam = get_velo_to_cam(calibrations)
    lidar_poses = np.stack(
        [camera_pose_to_lidar_pose(camera_pose, velo_to_cam) for camera_pose in camera_poses],
        axis=0,
    )
    stop_index = len(frame_files) - 1
    if start_index >= stop_index:
        raise ValueError(
            f"start_index {start_index} leaves no adjacent pairs for sequence {sequence}"
        )
    if max_pairs_per_sequence is not None:
        stop_index = min(stop_index, start_index + max_pairs_per_sequence)

    indices: list[KittiPairIndex] = []
    for prev_index in range(start_index, stop_index):
        curr_index = prev_index + 1
        relative_pose = relative_transform(lidar_poses[prev_index], lidar_poses[curr_index])
        target = extract_se2(relative_pose)
        indices.append(
            KittiPairIndex(
                sequence=paths.sequence,
                layout=paths.layout,
                prev_index=prev_index,
                curr_index=curr_index,
                prev_frame_path=frame_files[prev_index],
                curr_frame_path=frame_files[curr_index],
                prev_timestamp=float(timestamps[prev_index]),
                curr_timestamp=float(timestamps[curr_index]),
                target_pose_3dof=target,
            )
        )
    if not indices:
        raise ValueError(f"no adjacent pairs were built for sequence {sequence}")
    return indices


def _validate_sequence_counts(
    frame_files: Sequence[Path],
    timestamps: NDArray[np.float64],
    camera_poses: NDArray[np.float64],
    *,
    sequence: str,
) -> None:
    """校验一条 sequence 的 frame/timestamp/pose 数量一致且至少两帧。"""

    frame_count = len(frame_files)
    timestamp_count = int(timestamps.shape[0])
    pose_count = int(camera_poses.shape[0])
    if len({frame_count, timestamp_count, pose_count}) != 1:
        raise ValueError(
            "KITTI sequence count mismatch before pair indexing: "
            f"sequence={sequence} frames={frame_count} "
            f"timestamps={timestamp_count} poses={pose_count}"
        )
    if frame_count < 2:
        raise ValueError(f"sequence {sequence} needs at least 2 frames for adjacent pairs")


def _validate_pair_limits(*, max_pairs_per_sequence: int | None, start_index: int) -> None:
    """校验 pair 采样边界。"""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_pairs_per_sequence is not None and max_pairs_per_sequence <= 0:
        raise ValueError("max_pairs_per_sequence must be positive when provided")


def _normalize_sequences(sequences: object) -> tuple[str, ...]:
    """把配置或参数中的 sequence 列表规范化为两位字符串元组。"""

    if isinstance(sequences, (str, bytes)):
        raw_items: Iterable[object] = [sequences]
    elif isinstance(sequences, Iterable):
        raw_items = sequences
    else:
        raise ValueError("sequences must be a sequence of ids")

    normalized: list[str] = []
    for item in raw_items:
        if isinstance(item, int):
            if item < 0:
                raise ValueError(f"sequence id must be non-negative, got {item}")
            normalized.append(f"{item:02d}")
        elif isinstance(item, str) and item.strip():
            clean = item.strip()
            normalized.append(clean.zfill(2) if clean.isdigit() else clean)
        else:
            raise ValueError(f"invalid sequence id: {item!r}")
    if not normalized:
        raise ValueError("at least one sequence id is required")
    return tuple(normalized)


def _clean_split_name(split: str) -> str:
    """校验并归一化 split 名称。"""

    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    split_name = split.strip().lower()
    if split_name not in {"smoke", "train", "val", "test", "demo"}:
        raise ValueError(f"unsupported split: {split}")
    return split_name


def _vector3(value: object, name: str) -> LabelArray:
    """读取并校验三维 float32 向量。"""

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_vector3(value: object, name: str) -> LabelArray:
    """读取并校验三维正数向量。"""

    array = _vector3(value, name)
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain positive values")
    return array


def _positive_int(value: object, name: str) -> int:
    """读取并校验正整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
