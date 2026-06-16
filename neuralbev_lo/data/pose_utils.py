#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KITTI 位姿和齐次变换工具。

本模块统一处理 4x4 齐次矩阵校验、求逆、组合、相对位姿、KITTI `poses.txt`
读取，以及从 camera frame 到 LiDAR frame 的标定转换。内部约定使用
`T_world_lidar`，相邻帧标签使用 `inv(T_world_prev) @ T_world_curr`。
"""

from __future__ import annotations

from math import atan2, pi
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Transform = NDArray[np.float64]


def _as_float_array(values: Iterable[float], *, name: str) -> NDArray[np.float64]:
    """将输入转成有限 `float64` 数组。"""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _homogeneous_from_values(values: list[float], *, name: str) -> Transform:
    """从 KITTI 12/16 个数值构造 4x4 齐次矩阵。"""

    if len(values) == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        return validate_transform_matrix(matrix, name=name)
    if len(values) == 16:
        matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
        return validate_transform_matrix(matrix, name=name)
    raise ValueError(f"{name} must contain 12 or 16 numeric values, got {len(values)}")


def validate_transform_matrix(matrix: object, *, name: str = "transform") -> Transform:
    """校验并返回 4x4 齐次变换矩阵。

    参数:
        matrix: 任意可转成 NumPy 数组的矩阵输入。
        name: 错误信息中的矩阵名称。

    返回:
        `float64[4, 4]` 矩阵副本。

    异常:
        `ValueError`: 当矩阵不是 4x4、含非有限值，或最后一行不是齐次行时抛出。
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform matrix, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    return array.copy()


def invert_transform(transform: object) -> Transform:
    """求齐次变换矩阵的逆矩阵。

    参数:
        transform: `T_a_b` 形式的 4x4 齐次矩阵。

    返回:
        `T_b_a` 形式的 4x4 齐次矩阵。
    """

    matrix = validate_transform_matrix(transform)
    inverse = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose_transforms(left: object, right: object) -> Transform:
    """组合两个齐次变换矩阵。

    参数:
        left: 左侧 4x4 齐次矩阵。
        right: 右侧 4x4 齐次矩阵。

    返回:
        `left @ right` 的校验后结果。
    """

    result = validate_transform_matrix(left, name="left") @ validate_transform_matrix(
        right, name="right"
    )
    return validate_transform_matrix(result, name="composed_transform")


def relative_transform(previous_world: object, current_world: object) -> Transform:
    """计算当前帧相对上一帧的位姿。

    参数:
        previous_world: `T_world_prev`。
        current_world: `T_world_curr`。

    返回:
        `T_prev_curr = inv(T_world_prev) @ T_world_curr`。
    """

    previous = validate_transform_matrix(previous_world, name="previous_world")
    current = validate_transform_matrix(current_world, name="current_world")
    return compose_transforms(invert_transform(previous), current)


def camera_pose_to_lidar_pose(world_camera: object, velo_to_cam: object) -> Transform:
    """将 KITTI camera 位姿转换为内部 LiDAR 位姿。

    参数:
        world_camera: KITTI `poses.txt` 中的 `T_world_cam0`。
        velo_to_cam: `calib.txt` 中的 `Tr_velo_to_cam`，即 `T_cam_lidar`。

    返回:
        `T_world_lidar = T_world_cam0 @ T_cam_lidar`。
    """

    return compose_transforms(
        validate_transform_matrix(world_camera, name="world_camera"),
        validate_transform_matrix(velo_to_cam, name="velo_to_cam"),
    )


def extract_se2(transform: object) -> tuple[float, float, float]:
    """从 4x4 位姿矩阵提取平面 `dx, dy, yaw`。

    参数:
        transform: 已表达在目标参考系下的 4x4 齐次矩阵。

    返回:
        `(dx, dy, yaw)`，其中 yaw 归一化到 `[-pi, pi]`。
    """

    matrix = validate_transform_matrix(transform)
    yaw = atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    if yaw > pi:
        yaw -= 2.0 * pi
    if yaw < -pi:
        yaw += 2.0 * pi
    return float(matrix[0, 3]), float(matrix[1, 3]), float(yaw)


def load_kitti_poses(pose_path: str | Path) -> NDArray[np.float64]:
    """读取 KITTI Odometry `poses/<seq>.txt`。

    参数:
        pose_path: pose 文件路径，每个非空行应包含 12 或 16 个数值。

    返回:
        `float64[N, 4, 4]` 位姿数组。

    异常:
        `ValueError`: 文件不存在、为空、行格式错误或含非有限值时抛出。
    """

    path = Path(pose_path)
    if not path.exists():
        raise ValueError(f"pose file does not exist: {path}")

    poses: list[Transform] = []
    for line_index, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            values = [float(item) for item in line.split()]
        except ValueError as exc:
            raise ValueError(f"pose line {line_index} contains non-numeric values: {path}") from exc
        _as_float_array(values, name=f"pose line {line_index}")
        poses.append(_homogeneous_from_values(values, name=f"pose line {line_index}"))

    if not poses:
        raise ValueError(f"pose file is empty: {path}")
    return np.stack(poses, axis=0)


def parse_calibration_file(calib_path: str | Path) -> dict[str, Transform]:
    """解析 KITTI `calib.txt` 中的 3x4/4x4 标定矩阵。

    参数:
        calib_path: KITTI sequence 下的 `calib.txt`。

    返回:
        key 到 4x4 齐次矩阵的映射。常见 key 包括 `Tr` 和 `Tr_velo_to_cam`。
    """

    path = Path(calib_path)
    if not path.exists():
        raise ValueError(f"calibration file does not exist: {path}")

    matrices: dict[str, Transform] = {}
    for line_index, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, raw_values = line.split(":", 1)
        tokens = raw_values.split()
        if len(tokens) not in {12, 16}:
            continue
        try:
            values = [float(item) for item in tokens]
        except ValueError as exc:
            raise ValueError(f"calibration line {line_index} contains non-numeric values") from exc
        matrices[key.strip()] = _homogeneous_from_values(values, name=f"calibration {key.strip()}")

    if not matrices:
        raise ValueError(f"no 3x4 or 4x4 calibration matrices found: {path}")
    return matrices


def get_velo_to_cam(calibrations: dict[str, Transform]) -> Transform:
    """从标定字典中取出 `Tr_velo_to_cam`。

    参数:
        calibrations: `parse_calibration_file` 的返回值。

    返回:
        4x4 `T_cam_lidar` 矩阵。
    """

    for key in ("Tr_velo_to_cam", "Tr"):
        if key in calibrations:
            return validate_transform_matrix(calibrations[key], name=key)
    raise ValueError("calibration must contain Tr_velo_to_cam or Tr")
