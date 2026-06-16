#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KITTI Odometry 数据读取与序列校验。

本模块只负责 Week 2 的基础数据契约：路径构建、文件存在性校验、Velodyne
`.bin` 读取、时间戳读取、pose/calib 计数校验和预览摘要。它不做 BEV
栅格化，也不触碰训练逻辑。
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
    """KITTI 单序列关键路径集合。"""

    root: Path
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
    frame_count: int
    timestamp_count: int
    pose_count: int
    first_frame_shape: tuple[int, int]
    first_timestamp: float
    last_timestamp: float
    calib_keys: tuple[str, ...]


def _normalize_sequence(sequence: str | int) -> str:
    """标准化 KITTI sequence id。"""

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
        data_root: KITTI Odometry 根目录。
        sequence: 序列号，例如 `00`。

    返回:
        `KittiSequencePaths` 路径集合。
    """

    root = Path(data_root)
    sequence_id = _normalize_sequence(sequence)
    sequence_dir = root / "sequences" / sequence_id
    return KittiSequencePaths(
        root=root,
        sequence=sequence_id,
        sequence_dir=sequence_dir,
        velodyne_dir=sequence_dir / "velodyne",
        calib_path=sequence_dir / "calib.txt",
        times_path=sequence_dir / "times.txt",
        pose_path=root / "poses" / f"{sequence_id}.txt",
    )


def _require_path(path: Path, *, label: str) -> None:
    """校验路径存在。"""

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
    """读取 KITTI Velodyne `.bin` 为 `float32[N, 4]`。

    参数:
        frame_path: 单帧 `.bin` 文件路径。

    返回:
        点云数组，列为 `x, y, z, intensity`。
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
    标定文件可解析，以及首帧点云可读取。
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
            f"sequence={paths.sequence}"
        )

    first_frame = load_velodyne_frame(velodyne_files[0])
    return KittiSequenceInfo(
        sequence=paths.sequence,
        root=paths.root,
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
