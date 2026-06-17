#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KITTI Odometry 数据读取与序列校验。

本模块负责 Week 2 的基础数据契约：构建路径、校验关键文件、读取 Velodyne
`.bin`、读取时间戳、解析 pose/calib，并输出 CLI 可打印的序列摘要。
支持两种常见目录：

1. 合并布局：`root/sequences/<seq>/...` 和 `root/poses/<seq>.txt`
2. 官方分包解压布局：`root/data_odometry_* /dataset/...`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuralbev_lo.data.pose_utils import load_kitti_poses, parse_calibration_file


@dataclass(frozen=True)
class KittiSequencePaths:
    """KITTI 单序列关键路径集合。

    参数:
        root: 用户配置的 KITTI 根目录。
        layout: 已识别的数据布局，取值为 `merged` 或 `split`.
        sequence: 两位序列号，例如 `00`.
        sequence_dir: 包含当前序列主数据的目录。
        velodyne_dir: 当前序列 Velodyne `.bin` 文件目录。
        calib_path: 当前序列 `calib.txt` 路径。
        times_path: 当前序列 `times.txt` 路径。
        pose_path: 当前序列 pose 文件路径。
    """

    root: Path
    layout: str
    sequence: str
    sequence_dir: Path
    velodyne_dir: Path
    calib_path: Path
    times_path: Path
    pose_path: Path


@dataclass(frozen=True)
class KittiSequenceInfo:
    """KITTI 单序列校验摘要。"""

    sequence: str
    root: Path
    layout: str
    frame_count: int
    timestamp_count: int
    pose_count: int
    first_frame_shape: tuple[int, int]
    first_timestamp: float
    last_timestamp: float
    calib_keys: tuple[str, ...]


def _normalize_sequence(sequence: str | int) -> str:
    """标准化 KITTI sequence id。

    参数:
        sequence: 字符串或非负整数序列号。
    返回:
        两位字符串序列号；非数字字符串会保留原值。
    """

    if isinstance(sequence, int):
        if sequence < 0:
            raise ValueError(f"sequence must be non-negative, got {sequence}")
        return f"{sequence:02d}"
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("sequence must be a non-empty string or non-negative integer")
    clean_sequence = sequence.strip()
    return clean_sequence.zfill(2) if clean_sequence.isdigit() else clean_sequence


def build_sequence_paths(data_root: str | Path, sequence: str | int) -> KittiSequencePaths:
    """按 KITTI Odometry 约定构建单序列路径。

    参数:
        data_root: KITTI Odometry 根目录。既可以是合并后的 `sequences/poses`
            根目录，也可以是官方三个 zip 分包直接解压后的父目录。
        sequence: 序列号，例如 `00`.
    返回:
        已解析布局的 `KittiSequencePaths` 路径集合。
    """

    root = Path(data_root)
    sequence_id = _normalize_sequence(sequence)
    merged_paths = _build_merged_sequence_paths(root, sequence_id)
    if merged_paths.sequence_dir.exists() or not _looks_like_split_kitti_root(root, sequence_id):
        return merged_paths
    return _build_split_sequence_paths(root, sequence_id)


def _build_merged_sequence_paths(root: Path, sequence_id: str) -> KittiSequencePaths:
    """构建标准合并布局路径。"""

    sequence_dir = root / "sequences" / sequence_id
    return KittiSequencePaths(
        root=root,
        layout="merged",
        sequence=sequence_id,
        sequence_dir=sequence_dir,
        velodyne_dir=sequence_dir / "velodyne",
        calib_path=sequence_dir / "calib.txt",
        times_path=sequence_dir / "times.txt",
        pose_path=root / "poses" / f"{sequence_id}.txt",
    )


def _build_split_sequence_paths(root: Path, sequence_id: str) -> KittiSequencePaths:
    """构建官方分包解压布局路径。

    官方 KITTI Odometry 常见下载方式会把 calib、poses、velodyne 解压到三个
    `data_odometry_*` 目录内，每个目录里再嵌套 `dataset/`。这里不移动数据，
    只在读取时把三份目录虚拟拼成一个序列路径集合。
    """

    calib_sequence_dir = root / "data_odometry_calib" / "dataset" / "sequences" / sequence_id
    velodyne_sequence_dir = (
        root / "data_odometry_velodyne" / "dataset" / "sequences" / sequence_id
    )
    pose_dir = root / "data_odometry_poses" / "dataset" / "poses"
    return KittiSequencePaths(
        root=root,
        layout="split",
        sequence=sequence_id,
        sequence_dir=velodyne_sequence_dir,
        velodyne_dir=velodyne_sequence_dir / "velodyne",
        calib_path=calib_sequence_dir / "calib.txt",
        times_path=calib_sequence_dir / "times.txt",
        pose_path=pose_dir / f"{sequence_id}.txt",
    )


def _looks_like_split_kitti_root(root: Path, sequence_id: str) -> bool:
    """判断根目录是否像官方 KITTI Odometry 分包解压目录。"""

    split_paths = _build_split_sequence_paths(root, sequence_id)
    return (
        split_paths.velodyne_dir.exists()
        or split_paths.calib_path.exists()
        or split_paths.times_path.exists()
        or split_paths.pose_path.exists()
    )


def _require_path(path: Path, *, label: str) -> None:
    """校验路径存在。

    参数:
        path: 待检查路径。
        label: 错误信息中的路径角色。
    异常:
        ValueError: 当路径不存在时抛出。
    """

    if not path.exists():
        raise ValueError(f"missing {label}: {path}")


def list_velodyne_files(velodyne_dir: str | Path) -> list[Path]:
    """列出并排序 Velodyne `.bin` 文件。"""

    path = Path(velodyne_dir)
    _require_path(path, label="velodyne directory")
    files = sorted(path.glob("*.bin"))
    if not files:
        raise ValueError(f"no .bin velodyne frames found: {path}")
    return files


def load_velodyne_frame(frame_path: str | Path) -> NDArray[np.float32]:
    """读取 KITTI Velodyne `.bin` 为 `float32[N, 4]`.

    参数:
        frame_path: 单帧 `.bin` 文件路径。
    返回:
        点云数组，列顺序为 `x, y, z, intensity`.
    异常:
        ValueError: 文件缺失、float 数量不是 4 的倍数或包含非有限值时抛出。
    """

    path = Path(frame_path)
    _require_path(path, label="velodyne frame")
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"velodyne frame float count must be a multiple of 4: {path}")
    points = raw.reshape(-1, 4)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"velodyne frame must contain only finite values: {path}")
    return points


def load_timestamps(times_path: str | Path) -> NDArray[np.float64]:
    """读取 KITTI `times.txt` 为有限浮点时间戳。"""

    path = Path(times_path)
    _require_path(path, label="timestamps file")
    timestamps: list[float] = []
    for line_index, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(f"timestamp line {line_index} is not numeric: {path}") from exc
        if not np.isfinite(value):
            raise ValueError(f"timestamp line {line_index} must be finite: {path}")
        timestamps.append(value)

    if not timestamps:
        raise ValueError(f"timestamps file is empty: {path}")
    return np.asarray(timestamps, dtype=np.float64)


def validate_kitti_sequence(data_root: str | Path, sequence: str | int) -> KittiSequenceInfo:
    """校验 KITTI 单序列并返回摘要。

    校验内容包括关键路径存在、Velodyne 文件数量、timestamp 数量、pose 数量、
    标定文件可解析，以及首帧点云可读取。该函数只读数据，不会创建或移动数据文件。
    """

    paths = build_sequence_paths(data_root, sequence)
    _require_path(paths.sequence_dir, label="sequence directory")
    _require_path(paths.calib_path, label="calibration file")
    _require_path(paths.times_path, label="timestamps file")
    _require_path(paths.pose_path, label="poses file")

    velodyne_files = list_velodyne_files(paths.velodyne_dir)
    timestamps = load_timestamps(paths.times_path)
    poses = load_kitti_poses(paths.pose_path)
    calibrations = parse_calibration_file(paths.calib_path)

    frame_count = len(velodyne_files)
    timestamp_count = int(timestamps.shape[0])
    pose_count = int(poses.shape[0])
    if len({frame_count, timestamp_count, pose_count}) != 1:
        raise ValueError(
            "KITTI sequence count mismatch: "
            f"frames={frame_count}, timestamps={timestamp_count}, poses={pose_count}, "
            f"sequence={paths.sequence}, layout={paths.layout}"
        )

    first_frame = load_velodyne_frame(velodyne_files[0])
    return KittiSequenceInfo(
        sequence=paths.sequence,
        root=paths.root,
        layout=paths.layout,
        frame_count=frame_count,
        timestamp_count=timestamp_count,
        pose_count=pose_count,
        first_frame_shape=tuple(int(value) for value in first_frame.shape),
        first_timestamp=float(timestamps[0]),
        last_timestamp=float(timestamps[-1]),
        calib_keys=tuple(sorted(calibrations.keys())),
    )


def summarize_kitti_sequence(data_root: str | Path, sequence: str | int) -> dict[str, Any]:
    """返回适合 CLI 打印和 JSON manifest 写入的序列摘要。"""

    info = validate_kitti_sequence(data_root, sequence)
    summary = asdict(info)
    summary["root"] = str(info.root)
    summary["first_frame_shape"] = list(info.first_frame_shape)
    summary["calib_keys"] = list(info.calib_keys)
    return summary
