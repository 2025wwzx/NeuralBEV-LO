#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BEV memory 质量对比指标。

Week 8 先把 GT-pose memory 作为参考上界，比较 naive 与 learned-pose memory 的
均值绝对误差和 occupancy IoU。这里的指标是 smoke-scale 工程 sanity check，不代表
全局地图质量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
    iou = 1.0 if not np.any(union) else float(np.count_nonzero(intersection) / np.count_nonzero(union))
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
