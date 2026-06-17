#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""短序列 odometry 评估指标与 baseline 对比。

Week 7 先评估 3DoF `dx, dy, yaw` 相邻帧预测。相对标签表达在 previous
LiDAR/BEV frame，因此轨迹积分时先用上一帧世界 yaw 把 dx/dy 旋到世界坐标，再更新
世界 yaw。这里的 ATE 是短序列 XY 轨迹 RMSE，用作工程 sanity check。
"""

from __future__ import annotations

import csv
import json
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

PoseArray = NDArray[np.float64]


def integrate_relative_poses(relative_poses: object) -> PoseArray:
    """把相邻帧 3DoF 相对位姿积分为短轨迹。

    参数:
        relative_poses: `float[N, 3]`，列顺序为 `dx, dy, yaw`，表达在 previous frame。
    返回:
        `float[N + 1, 3]` 轨迹，第一行为原点 `[0, 0, 0]`。
    """

    relatives = _as_pose_array(relative_poses, name="relative_poses")
    trajectory = np.zeros((relatives.shape[0] + 1, 3), dtype=np.float64)
    for index, (dx, dy, dyaw) in enumerate(relatives, start=1):
        prev_x, prev_y, prev_yaw = trajectory[index - 1]
        world_dx = cos(prev_yaw) * dx - sin(prev_yaw) * dy
        world_dy = sin(prev_yaw) * dx + cos(prev_yaw) * dy
        trajectory[index, 0] = prev_x + world_dx
        trajectory[index, 1] = prev_y + world_dy
        trajectory[index, 2] = wrap_angle(prev_yaw + dyaw)
    return trajectory


def build_baseline_predictions(
    gt_relative: object,
    *,
    learned_relative: object | None = None,
) -> dict[str, PoseArray]:
    """构造 Week 7 baseline 的相对位姿预测。

    baseline:
        zero_motion: 所有相邻运动为 0。
        constant_velocity: 第一帧为 0，之后重复上一帧 GT 相对运动。
        gt_label_echo: 直接回放 GT 标签，作为上界 sanity check。
        learned: 可选模型预测。
    """

    gt = _as_pose_array(gt_relative, name="gt_relative")
    predictions: dict[str, PoseArray] = {
        "zero_motion": np.zeros_like(gt),
        "constant_velocity": _constant_velocity_prediction(gt),
        "gt_label_echo": gt.copy(),
    }
    if learned_relative is not None:
        learned = _as_pose_array(learned_relative, name="learned_relative")
        if learned.shape != gt.shape:
            raise ValueError(f"learned_relative shape must match GT, got {learned.shape} vs {gt.shape}")
        predictions["learned"] = learned
    return predictions


def evaluate_baselines(
    gt_relative: object,
    baseline_predictions: Mapping[str, object],
) -> list[dict[str, float | int | str]]:
    """评估多个 baseline 并返回表格行。

    每行包含相对 pose error、短轨迹 ATE RMSE 和最终 XY 漂移误差。
    """

    gt = _as_pose_array(gt_relative, name="gt_relative")
    gt_trajectory = integrate_relative_poses(gt)
    rows: list[dict[str, float | int | str]] = []
    for name, raw_prediction in baseline_predictions.items():
        predicted = _as_pose_array(raw_prediction, name=f"{name}_prediction")
        if predicted.shape != gt.shape:
            raise ValueError(f"prediction shape for {name} must match GT, got {predicted.shape} vs {gt.shape}")
        predicted_trajectory = integrate_relative_poses(predicted)
        relative_errors = relative_pose_errors(predicted, gt)
        rows.append(
            {
                "baseline": str(name),
                "num_pairs": int(gt.shape[0]),
                "relative_translation_error_m": relative_errors["relative_translation_error_m"],
                "relative_yaw_error_rad": relative_errors["relative_yaw_error_rad"],
                "ate_rmse_m": trajectory_ate_rmse(predicted_trajectory, gt_trajectory),
                "final_translation_error_m": final_translation_error(predicted_trajectory, gt_trajectory),
            }
        )
    return rows


def relative_pose_errors(predicted_relative: object, gt_relative: object) -> dict[str, float]:
    """计算相邻相对 pose 的平均平移误差和 yaw 误差。"""

    predicted = _as_pose_array(predicted_relative, name="predicted_relative")
    gt = _as_pose_array(gt_relative, name="gt_relative")
    if predicted.shape != gt.shape:
        raise ValueError(f"predicted_relative shape must match GT, got {predicted.shape} vs {gt.shape}")
    diff_xy = predicted[:, :2] - gt[:, :2]
    translation_error = np.linalg.norm(diff_xy, axis=1).mean()
    yaw_error = np.asarray([abs(wrap_angle(value)) for value in predicted[:, 2] - gt[:, 2]], dtype=np.float64).mean()
    return {
        "relative_translation_error_m": float(translation_error),
        "relative_yaw_error_rad": float(yaw_error),
    }


def trajectory_ate_rmse(predicted_trajectory: object, gt_trajectory: object) -> float:
    """计算短轨迹 XY ATE RMSE。"""

    predicted = _as_trajectory_array(predicted_trajectory, name="predicted_trajectory")
    gt = _as_trajectory_array(gt_trajectory, name="gt_trajectory")
    if predicted.shape != gt.shape:
        raise ValueError(f"trajectory shapes must match, got {predicted.shape} vs {gt.shape}")
    squared = np.sum((predicted[:, :2] - gt[:, :2]) ** 2, axis=1)
    return float(sqrt(float(np.mean(squared))))


def final_translation_error(predicted_trajectory: object, gt_trajectory: object) -> float:
    """计算轨迹终点 XY 平移误差。"""

    predicted = _as_trajectory_array(predicted_trajectory, name="predicted_trajectory")
    gt = _as_trajectory_array(gt_trajectory, name="gt_trajectory")
    if predicted.shape != gt.shape:
        raise ValueError(f"trajectory shapes must match, got {predicted.shape} vs {gt.shape}")
    return float(np.linalg.norm(predicted[-1, :2] - gt[-1, :2]))


def save_metrics_json(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """保存 metrics JSON，包含 metadata 和 metrics 两个顶层字段。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "metrics": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_metrics_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """保存 metrics CSV 表。"""

    if not rows:
        raise ValueError("at least one metrics row is required")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def wrap_angle(angle_rad: float) -> float:
    """把角度归一化到 `[-pi, pi]`。"""

    return atan2(sin(float(angle_rad)), cos(float(angle_rad)))


def _constant_velocity_prediction(gt_relative: PoseArray) -> PoseArray:
    """用上一帧 GT 相对运动预测当前帧运动。"""

    predicted = np.zeros_like(gt_relative)
    if gt_relative.shape[0] > 1:
        predicted[1:] = gt_relative[:-1]
    return predicted


def _as_pose_array(values: object, *, name: str) -> PoseArray:
    """校验 `[N, 3]` pose 数组。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {array.shape}")
    if array.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one pose")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_trajectory_array(values: object, *, name: str) -> PoseArray:
    """校验 `[N, 3]` trajectory 数组。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {array.shape}")
    if array.shape[0] <= 1:
        raise ValueError(f"{name} must contain at least two poses")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
