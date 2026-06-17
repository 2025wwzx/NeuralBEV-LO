#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BEV memory 质量对比指标。

Week 8 先把 GT-pose memory 作为参考上界，比较 naive 与 learned-pose memory 的
均值绝对误差和 occupancy IoU。这里的指标是 smoke-scale 工程 sanity check，不代表
全局地图质量。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def compare_memory_to_reference(
    candidate_memory: object,
    reference_memory: object,
    *,
    occupancy_threshold: float = 0.1,
) -> dict[str, float]:
    """比较一个 BEV memory 和参考 memory。

    参数:
        candidate_memory: 待评估 `[C, H, W]` BEV memory。
        reference_memory: 参考 `[C, H, W]` BEV memory，通常为 GT-pose memory。
        occupancy_threshold: density channel 的二值 occupancy 阈值。
    返回:
        包含 `mean_abs_error`、`occupancy_iou`、`candidate_occupied` 和
        `reference_occupied` 的指标字典。
    """

    candidate = _as_bev_memory(candidate_memory, name="candidate_memory")
    reference = _as_bev_memory(reference_memory, name="reference_memory")
    if candidate.shape != reference.shape:
        raise ValueError(f"memory shapes must match, got {candidate.shape} and {reference.shape}")
    if occupancy_threshold < 0:
        raise ValueError("occupancy_threshold must be non-negative")

    candidate_occ = candidate[0] > occupancy_threshold
    reference_occ = reference[0] > occupancy_threshold
    union = np.logical_or(candidate_occ, reference_occ)
    intersection = np.logical_and(candidate_occ, reference_occ)
    iou = (
        1.0
        if not np.any(union)
        else float(np.count_nonzero(intersection) / np.count_nonzero(union))
    )
    return {
        "mean_abs_error": float(np.mean(np.abs(candidate - reference))),
        "occupancy_iou": iou,
        "candidate_occupied": float(np.count_nonzero(candidate_occ)),
        "reference_occupied": float(np.count_nonzero(reference_occ)),
    }


def memory_quality_table(
    *,
    reference_memory: object,
    candidates: Mapping[str, object],
    occupancy_threshold: float = 0.1,
) -> list[dict[str, float | str]]:
    """为多个 memory candidate 生成对比表。"""

    if not candidates:
        raise ValueError("at least one memory candidate is required")
    rows: list[dict[str, float | str]] = []
    for name, candidate in candidates.items():
        metrics = compare_memory_to_reference(
            candidate,
            reference_memory,
            occupancy_threshold=occupancy_threshold,
        )
        rows.append({"memory": str(name), **metrics})
    return rows


def frame_to_memory_alignment_score(
    current_bev: object,
    memory_bev: object,
    *,
    occupancy_threshold: float = 0.1,
    channel_indices: Sequence[int] | None = (0,),
) -> float:
    """计算当前 BEV 与 memory BEV 的阈值化 occupancy mean IoU。

    参数:
        current_bev: 当前帧 BEV，形状为 `[C, H, W]`。
        memory_bev: 与当前帧对齐后的 memory BEV，形状必须一致。
        occupancy_threshold: 大于该阈值的像素视为 occupied。
        channel_indices: 参与 IoU 的通道索引，默认只使用 density 通道。
    返回:
        所选通道 IoU 的平均值；当某通道两侧都为空时，该通道 IoU 记为 1.0。
    """

    current = _as_bev_memory(current_bev, name="current_bev")
    memory = _as_bev_memory(memory_bev, name="memory_bev")
    if current.shape != memory.shape:
        raise ValueError(f"BEV shapes must match, got {current.shape} and {memory.shape}")
    if occupancy_threshold < 0:
        raise ValueError("occupancy_threshold must be non-negative")

    indices = _normalize_channel_indices(channel_indices, current.shape[0])
    scores: list[float] = []
    for channel_index in indices:
        current_occ = current[channel_index] > occupancy_threshold
        memory_occ = memory[channel_index] > occupancy_threshold
        union = np.logical_or(current_occ, memory_occ)
        intersection = np.logical_and(current_occ, memory_occ)
        score = (
            1.0
            if not np.any(union)
            else float(np.count_nonzero(intersection) / np.count_nonzero(union))
        )
        scores.append(score)
    return float(np.mean(scores))


def temporal_flicker_score(
    memory_window: Sequence[object],
    *,
    channel_indices: Sequence[int] | None = (0,),
) -> float:
    """计算短窗口 memory 的平均像素标准差。

    参数:
        memory_window: 时间顺序排列的 BEV memory 列表，每项形状为 `[C, H, W]`。
        channel_indices: 参与统计的通道索引，默认只使用 density 通道。
    返回:
        在时间维做 `std` 后，对所选通道和空间像素求均值。窗口只有一帧时返回 0。
    """

    if not memory_window:
        raise ValueError("memory_window must contain at least one BEV")
    arrays = [
        _as_bev_memory(memory, name=f"memory_window[{index}]")
        for index, memory in enumerate(memory_window)
    ]
    reference_shape = arrays[0].shape
    for index, array in enumerate(arrays[1:], start=1):
        if array.shape != reference_shape:
            raise ValueError(
                f"memory_window[{index}] shape must match {reference_shape}, got {array.shape}"
            )
    indices = _normalize_channel_indices(channel_indices, reference_shape[0])
    stacked = np.stack([array[list(indices)] for array in arrays], axis=0)
    return float(np.std(stacked, axis=0).mean())


def bev_consistency_table(
    *,
    current_bevs: Sequence[object],
    memory_histories: Mapping[str, Sequence[object]],
    occupancy_threshold: float = 0.1,
    channel_indices: Sequence[int] | None = (0,),
    flicker_window: int = 5,
) -> list[dict[str, float | int | str]]:
    """生成逐帧 BEV consistency 指标表。

    参数:
        current_bevs: 当前帧 BEV 序列，长度为帧数。
        memory_histories: memory 名称到逐帧 memory BEV 序列的映射。
        occupancy_threshold: alignment IoU 使用的 occupancy 阈值。
        channel_indices: 指标使用的通道索引。
        flicker_window: flicker 统计使用的最多历史帧数。
    返回:
        每行包含 `memory`、`frame_index`、`alignment_iou`、`flicker_score`
        和 `flicker_window_size`。
    """

    if not current_bevs:
        raise ValueError("current_bevs must contain at least one BEV")
    if not memory_histories:
        raise ValueError("at least one memory history is required")
    if flicker_window <= 0:
        raise ValueError("flicker_window must be positive")

    currents = [
        _as_bev_memory(bev, name=f"current_bevs[{index}]")
        for index, bev in enumerate(current_bevs)
    ]
    reference_shape = currents[0].shape
    indices = _normalize_channel_indices(channel_indices, reference_shape[0])
    for index, current in enumerate(currents[1:], start=1):
        if current.shape != reference_shape:
            raise ValueError(
                f"current_bevs[{index}] shape must match {reference_shape}, got {current.shape}"
            )

    rows: list[dict[str, float | int | str]] = []
    for memory_name, raw_history in memory_histories.items():
        history = [
            _as_bev_memory(memory, name=f"{memory_name}[{index}]")
            for index, memory in enumerate(raw_history)
        ]
        if len(history) != len(currents):
            raise ValueError(
                f"memory history {memory_name} must have {len(currents)} frames, got {len(history)}"
            )
        for frame_index, (current, memory) in enumerate(zip(currents, history, strict=True)):
            if memory.shape != reference_shape:
                raise ValueError(
                    f"memory history {memory_name}[{frame_index}] shape must match "
                    f"{reference_shape}, got {memory.shape}"
                )
            window_start = max(0, frame_index - flicker_window + 1)
            window = history[window_start : frame_index + 1]
            rows.append(
                {
                    "memory": str(memory_name),
                    "frame_index": int(frame_index),
                    "alignment_iou": frame_to_memory_alignment_score(
                        current,
                        memory,
                        occupancy_threshold=occupancy_threshold,
                        channel_indices=indices,
                    ),
                    "flicker_score": temporal_flicker_score(window, channel_indices=indices),
                    "flicker_window_size": int(len(window)),
                }
            )
    return rows


def save_memory_metrics_json(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """保存 BEV memory metrics JSON。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "metrics": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_memory_metrics_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """保存 BEV memory metrics CSV 表。"""

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


def _as_bev_memory(memory: object, *, name: str) -> NDArray[np.float32]:
    """校验 BEV memory 为有限 `[C, H, W]` float32 数组。"""

    array = np.asarray(memory, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [C, H, W], got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalize_channel_indices(
    channel_indices: Sequence[int] | None,
    channel_count: int,
) -> tuple[int, ...]:
    """校验并规范化 BEV 通道索引。"""

    if channel_indices is None:
        indices = (0,)
    else:
        if isinstance(channel_indices, (str, bytes)):
            raise ValueError("channel_indices must be a sequence of integers")
        indices = tuple(int(index) for index in channel_indices)
    if not indices:
        raise ValueError("at least one channel index is required")
    for index in indices:
        if index < 0 or index >= channel_count:
            raise ValueError(f"channel index {index} out of range for {channel_count} channels")
    return indices
