#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 5 相邻帧 BEV Pair Dataset 测试。"""

from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neuralbev_lo.bev.rasterizer import BevGridConfig
from neuralbev_lo.data.pair_dataset import (
    KittiAdjacentPairDataset,
    build_kitti_pair_indices,
    compute_pose_label_stats,
    normalize_pose_label,
    pose_label_stats_from_config,
    resolve_sequences_for_split,
)


def _bev_config() -> BevGridConfig:
    """构造小尺寸 BEV 配置，保证单测运行很快。"""

    return BevGridConfig(
        x_range_m=(0.0, 5.0),
        y_range_m=(-2.5, 2.5),
        resolution_m=0.5,
        channels=("density",),
    )


def _pose_line(x: float, y: float, yaw: float) -> str:
    """生成 KITTI poses.txt 兼容的 3x4 位姿行。"""

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    values = [
        cos_yaw,
        -sin_yaw,
        0.0,
        x,
        sin_yaw,
        cos_yaw,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return " ".join(f"{value:.12f}" for value in values)


def _write_sequence(
    root: Path,
    sequence: str,
    poses: list[tuple[float, float, float]],
) -> None:
    """写入最小 KITTI merged 布局。"""

    sequence_dir = root / "sequences" / sequence
    velodyne_dir = sequence_dir / "velodyne"
    poses_dir = root / "poses"
    velodyne_dir.mkdir(parents=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    (sequence_dir / "calib.txt").write_text(
        "Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )
    (sequence_dir / "times.txt").write_text(
        "\n".join(f"{index * 0.1:.1f}" for index in range(len(poses))),
        encoding="utf-8",
    )
    (poses_dir / f"{sequence}.txt").write_text(
        "\n".join(_pose_line(x, y, yaw) for x, y, yaw in poses),
        encoding="utf-8",
    )

    for index in range(len(poses)):
        points = np.array(
            [
                [1.0 + index * 0.1, 0.0, 0.2, 0.5],
                [2.0 + index * 0.1, 1.0, 0.3, 0.6],
            ],
            dtype=np.float32,
        )
        points.tofile(velodyne_dir / f"{index:06d}.bin")


def test_known_transform_fixture_recovers_expected_3dof_label(tmp_path) -> None:
    """已知相邻位姿应恢复 previous LiDAR frame 下的 dx、dy、yaw。"""

    _write_sequence(tmp_path, "00", [(0.0, 0.0, 0.0), (1.0, 2.0, pi / 4.0)])

    indices = build_kitti_pair_indices(tmp_path, ["00"])

    assert len(indices) == 1
    dx, dy, yaw = indices[0].target_pose_3dof
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(2.0)
    assert yaw == pytest.approx(pi / 4.0)


def test_pair_dataset_returns_bev_tensors_target_and_metadata(tmp_path) -> None:
    """Dataset 样本应返回两帧 BEV、3DoF target 和可追踪 metadata。"""

    _write_sequence(
        tmp_path,
        "00",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    )
    dataset = KittiAdjacentPairDataset(
        tmp_path,
        ["00"],
        _bev_config(),
        max_pairs_per_sequence=1,
    )

    bev_prev, bev_curr, target, metadata = dataset[0]

    assert len(dataset) == 1
    assert bev_prev.shape == torch.Size([1, 10, 10])
    assert bev_curr.shape == torch.Size([1, 10, 10])
    assert target.shape == torch.Size([3])
    assert target.dtype == torch.float32
    assert metadata["sequence"] == "00"
    assert metadata["prev_index"] == 0
    assert metadata["curr_index"] == 1
    assert metadata["prev_frame_path"].endswith("000000.bin")


def test_dataloader_smoke_batch_fetches_pair_dataset(tmp_path) -> None:
    """PyTorch DataLoader 应能批量 collate BEV pair 和 metadata。"""

    _write_sequence(
        tmp_path,
        "00",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    )
    dataset = KittiAdjacentPairDataset(tmp_path, ["00"], _bev_config())
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    bev_prev, bev_curr, targets, metadata = batch
    assert bev_prev.shape == torch.Size([2, 1, 10, 10])
    assert bev_curr.shape == torch.Size([2, 1, 10, 10])
    assert targets.shape == torch.Size([2, 3])
    assert list(metadata["sequence"]) == ["00", "00"]
    if torch.cuda.is_available():
        assert bev_prev.cuda().is_cuda


def test_pose_label_stats_are_computed_from_explicit_train_sequences_only(tmp_path) -> None:
    """归一化统计只使用调用方传入的 train sequence，不混入 val/test。"""

    _write_sequence(
        tmp_path,
        "00",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    )
    _write_sequence(
        tmp_path,
        "07",
        [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)],
    )

    stats = compute_pose_label_stats(tmp_path, ["00"], source_split="train")

    assert stats.source_split == "train"
    assert stats.sequences == ("00",)
    assert stats.count == 2
    np.testing.assert_allclose(stats.mean, [1.0, 0.0, 0.0], atol=1e-6)
    assert float(stats.std[0]) == pytest.approx(1.0e-6)


def test_normalize_pose_label_uses_provided_train_stats(tmp_path) -> None:
    """标签归一化应显式依赖传入的 train stats。"""

    _write_sequence(
        tmp_path,
        "00",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
    )
    stats = compute_pose_label_stats(tmp_path, ["00"], source_split="train")

    normalized = normalize_pose_label(np.array([1.5, 0.0, 0.0], dtype=np.float32), stats)

    assert normalized.shape == (3,)
    assert np.all(np.isfinite(normalized))


def test_resolve_sequences_for_split_reads_config_and_defaults() -> None:
    """split 解析应优先使用配置，缺省时回退到项目默认 split。"""

    config = {"data": {"train_sequences": ["03", 4]}}

    assert resolve_sequences_for_split(config, "train") == ("03", "04")
    assert resolve_sequences_for_split({"data": {}}, "smoke") == ("00",)


def test_pose_label_stats_from_config_reads_explicit_norm_values() -> None:
    """训练配置中写入的 pose_norm 应可还原为 PoseLabelStats。"""

    config = {
        "pose": {
            "pose_norm": {
                "source_split": "train",
                "sequences": ["00"],
                "count": 2,
                "mean": [1.0, 0.0, 0.0],
                "std": [0.5, 0.1, 0.01],
                "min": [0.5, -0.1, -0.01],
                "max": [1.5, 0.1, 0.01],
            }
        }
    }

    stats = pose_label_stats_from_config(config)

    assert stats.source_split == "train"
    assert stats.sequences == ("00",)
    assert stats.count == 2
    np.testing.assert_allclose(stats.mean, [1.0, 0.0, 0.0])
