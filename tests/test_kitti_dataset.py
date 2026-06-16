#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 2 KITTI 数据集测试。"""

from __future__ import annotations

import numpy as np
import pytest

from neuralbev_lo.data.kitti_dataset import (
    build_sequence_paths,
    load_timestamps,
    load_velodyne_frame,
    summarize_kitti_sequence,
    validate_kitti_sequence,
)


def _write_fake_kitti_sequence(root, sequence: str = "00", frame_count: int = 2) -> None:
    """写入最小 KITTI Odometry 测试目录。"""

    sequence_dir = root / "sequences" / sequence
    velodyne_dir = sequence_dir / "velodyne"
    poses_dir = root / "poses"
    velodyne_dir.mkdir(parents=True)
    poses_dir.mkdir(parents=True)

    (sequence_dir / "calib.txt").write_text(
        "Tr: 1 0 0 1 0 1 0 2 0 0 1 3\n",
        encoding="utf-8",
    )
    (sequence_dir / "times.txt").write_text(
        "\n".join(f"{index * 0.1:.1f}" for index in range(frame_count)),
        encoding="utf-8",
    )
    (poses_dir / f"{sequence}.txt").write_text(
        "\n".join("1 0 0 0 0 1 0 0 0 0 1 0" for _ in range(frame_count)),
        encoding="utf-8",
    )

    for index in range(frame_count):
        points = np.array(
            [
                [index + 1.0, 0.0, 0.5, 0.1],
                [index + 2.0, 1.0, 0.6, 0.2],
            ],
            dtype=np.float32,
        )
        points.tofile(velodyne_dir / f"{index:06d}.bin")


def test_build_sequence_paths_uses_expected_kitti_layout(tmp_path) -> None:
    """路径构建应匹配 KITTI Odometry 目录结构。"""

    paths = build_sequence_paths(tmp_path, "00")

    assert paths.sequence_dir == tmp_path / "sequences" / "00"
    assert paths.velodyne_dir == tmp_path / "sequences" / "00" / "velodyne"
    assert paths.calib_path == tmp_path / "sequences" / "00" / "calib.txt"
    assert paths.times_path == tmp_path / "sequences" / "00" / "times.txt"
    assert paths.pose_path == tmp_path / "poses" / "00.txt"


def test_load_velodyne_frame_reads_float32_n_by_4(tmp_path) -> None:
    """Velodyne bin 文件应读取为 float32[N, 4]。"""

    frame_path = tmp_path / "000000.bin"
    expected = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.6]], dtype=np.float32)
    expected.tofile(frame_path)

    loaded = load_velodyne_frame(frame_path)

    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 4)
    np.testing.assert_allclose(loaded, expected)


def test_load_velodyne_frame_rejects_malformed_binary(tmp_path) -> None:
    """不是 4 列倍数的 bin 文件必须被拒绝。"""

    frame_path = tmp_path / "bad.bin"
    np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(frame_path)

    with pytest.raises(ValueError, match="multiple of 4"):
        load_velodyne_frame(frame_path)


def test_validate_kitti_sequence_counts_frames_timestamps_and_poses(tmp_path) -> None:
    """序列校验应返回 frame/timestamp/pose 计数和首帧形状。"""

    _write_fake_kitti_sequence(tmp_path, frame_count=2)

    info = validate_kitti_sequence(tmp_path, "00")

    assert info.frame_count == 2
    assert info.timestamp_count == 2
    assert info.pose_count == 2
    assert info.first_frame_shape == (2, 4)
    assert info.first_timestamp == pytest.approx(0.0)
    assert info.last_timestamp == pytest.approx(0.1)


def test_validate_kitti_sequence_rejects_count_mismatch(tmp_path) -> None:
    """timestamps、poses、velodyne 文件数量不一致时必须报错。"""

    _write_fake_kitti_sequence(tmp_path, frame_count=2)
    (tmp_path / "sequences" / "00" / "times.txt").write_text("0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="count mismatch"):
        validate_kitti_sequence(tmp_path, "00")


def test_summarize_kitti_sequence_contains_preview_fields(tmp_path) -> None:
    """preview 脚本所需摘要字段必须稳定存在。"""

    _write_fake_kitti_sequence(tmp_path, frame_count=2)

    summary = summarize_kitti_sequence(tmp_path, "00")

    assert summary["sequence"] == "00"
    assert summary["frame_count"] == 2
    assert summary["pose_count"] == 2
    assert summary["timestamp_count"] == 2
    assert summary["first_frame_shape"] == [2, 4]


def test_load_timestamps_rejects_non_finite_values(tmp_path) -> None:
    """时间戳必须是有限数值。"""

    timestamp_path = tmp_path / "times.txt"
    timestamp_path.write_text("0.0\nnan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_timestamps(timestamp_path)


def test_load_timestamps_accepts_utf8_bom(tmp_path) -> None:
    """Windows 工具写出的 UTF-8 BOM 不应破坏时间戳解析。"""

    timestamp_path = tmp_path / "times.txt"
    timestamp_path.write_text("0.0\n0.1\n", encoding="utf-8-sig")

    timestamps = load_timestamps(timestamp_path)

    np.testing.assert_allclose(timestamps, [0.0, 0.1])
