#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 2 位姿工具测试。"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from neuralbev_lo.data.pose_utils import (
    camera_pose_to_lidar_pose,
    compose_transforms,
    extract_se2,
    invert_transform,
    load_kitti_poses,
    relative_transform,
    validate_transform_matrix,
)


def _make_transform(x: float, y: float, yaw: float) -> np.ndarray:
    """构造用于测试的平面 4x4 位姿矩阵。"""

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = cos_yaw
    matrix[0, 1] = -sin_yaw
    matrix[1, 0] = sin_yaw
    matrix[1, 1] = cos_yaw
    matrix[0, 3] = x
    matrix[1, 3] = y
    return matrix


def test_transform_inverse_and_composition_return_identity() -> None:
    """矩阵求逆和组合应恢复单位阵。"""

    transform = _make_transform(3.0, -2.0, pi / 4.0)
    inverse = invert_transform(transform)
    identity = compose_transforms(transform, inverse)

    np.testing.assert_allclose(identity, np.eye(4), atol=1e-8)


def test_relative_transform_and_se2_extraction_are_in_previous_frame() -> None:
    """相对位姿应等于 inv(T_world_prev) @ T_world_curr。"""

    previous = _make_transform(1.0, 0.0, 0.0)
    current = _make_transform(4.0, 2.0, pi / 2.0)

    relative = relative_transform(previous, current)
    dx, dy, yaw = extract_se2(relative)

    assert dx == pytest.approx(3.0)
    assert dy == pytest.approx(2.0)
    assert yaw == pytest.approx(pi / 2.0)


def test_camera_pose_to_lidar_pose_uses_velo_to_cam_calibration() -> None:
    """KITTI camera 位姿应通过 Tr_velo_to_cam 转成 LiDAR 位姿。"""

    world_cam0 = np.eye(4, dtype=np.float64)
    world_cam0[:3, 3] = [10.0, 20.0, 30.0]
    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, 3] = [1.0, 2.0, 3.0]

    world_lidar = camera_pose_to_lidar_pose(world_cam0, velo_to_cam)

    np.testing.assert_allclose(world_lidar[:3, 3], [11.0, 22.0, 33.0])


def test_validate_transform_matrix_rejects_bad_shape() -> None:
    """非 4x4 矩阵必须被拒绝。"""

    with pytest.raises(ValueError, match="4x4"):
        validate_transform_matrix(np.eye(3), name="bad_pose")


def test_load_kitti_poses_reads_3x4_lines_as_4x4_matrices(tmp_path) -> None:
    """KITTI poses.txt 的 12 数值行应被扩展为 4x4 齐次矩阵。"""

    pose_path = tmp_path / "00.txt"
    pose_path.write_text(
        "\n".join(
            [
                "1 0 0 0 0 1 0 0 0 0 1 0",
                "1 0 0 1 0 1 0 2 0 0 1 3",
            ]
        ),
        encoding="utf-8",
    )

    poses = load_kitti_poses(pose_path)

    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[1, :3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(poses[:, 3, :], np.array([[0, 0, 0, 1], [0, 0, 0, 1]]))
