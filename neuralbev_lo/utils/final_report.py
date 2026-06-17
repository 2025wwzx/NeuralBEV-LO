#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NeuralBEV-LO v0.1 final report 生成工具。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

REPORT_TITLE: Final[str] = "NeuralBEV-LO v0.1 Final Report"


def build_final_report(
    memory_metrics_path: str | Path,
    consistency_metrics_path: str | Path,
    config_path: str | Path,
    artifact_paths: list[str | Path],
) -> str:
    """生成 v0.1 研究原型的最终文本报告。

    参数:
        memory_metrics_path: BEV memory final metrics JSON 路径。
        consistency_metrics_path: per-frame consistency metrics JSON 路径。
        config_path: 复现实验使用的 eval config 路径。
        artifact_paths: 本次报告需要列出的图片、视频、指标等 artifact 路径。
    返回:
        可直接打印或写入磁盘的多行 final report 文本。
    异常:
        当输入 JSON、指标字段或 artifact 路径缺失时抛出 ValueError/FileNotFoundError。
    """

    memory_path = _validate_existing_file(memory_metrics_path, "memory_metrics_path")
    consistency_path = _validate_existing_file(consistency_metrics_path, "consistency_metrics_path")
    checked_config = _validate_existing_file(config_path, "config_path")
    checked_artifacts = [_validate_existing_file(path, "artifact") for path in artifact_paths]

    memory_payload = _read_json_object(memory_path)
    consistency_payload = _read_json_object(consistency_path)
    metadata = _require_dict(memory_payload.get("metadata"), "memory metadata")
    memory_metrics = _require_list(memory_payload.get("metrics"), "memory metrics")
    consistency_metrics = _require_list(consistency_payload.get("metrics"), "consistency metrics")

    memory_lines = _format_memory_metrics(memory_metrics)
    consistency_lines = _format_consistency_metrics(consistency_metrics)

    lines = [
        REPORT_TITLE,
        "",
        "Run Context",
        f"- config: {_format_path(checked_config)}",
        f"- sequence: {metadata.get('sequence', 'unknown')}",
        f"- frames: {metadata.get('frames', 'unknown')}",
        f"- device: {metadata.get('device', 'unknown')}",
        f"- checkpoint: {_format_metadata_value(metadata.get('checkpoint', 'unknown'))}",
        f"- resolution_m: {metadata.get('resolution_m', 'unknown')}",
        "",
        "Memory Metrics",
        *memory_lines,
        "",
        "Consistency Metrics",
        *consistency_lines,
        "",
        "Artifacts",
        *[f"- {_format_path(path)}" for path in checked_artifacts],
        "",
        "Limitations",
        "- Learned-pose metrics come from a smoke-scale checkpoint and should not be read as production odometry accuracy.",
        "- 4D BEV means a 2D BEV memory evolving over time, not a dense x/y/z/t reconstruction.",
    ]
    return "\n".join(lines)


def write_final_report(
    output_path: str | Path,
    memory_metrics_path: str | Path,
    consistency_metrics_path: str | Path,
    config_path: str | Path,
    artifact_paths: list[str | Path],
) -> Path:
    """生成并写入 v0.1 final report。

    参数:
        output_path: 报告输出路径，父目录不存在时会自动创建。
        memory_metrics_path: BEV memory metrics JSON 路径。
        consistency_metrics_path: consistency metrics JSON 路径。
        config_path: 复现实验使用的 eval config 路径。
        artifact_paths: 报告中列出的 artifact 路径。
    返回:
        写入后的报告路径。
    """

    report = build_final_report(
        memory_metrics_path=memory_metrics_path,
        consistency_metrics_path=consistency_metrics_path,
        config_path=config_path,
        artifact_paths=artifact_paths,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并校验顶层类型。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top-level must be an object: {path}")
    return payload


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    """校验字段是字典。"""

    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_list(value: Any, name: str) -> list[dict[str, Any]]:
    """校验字段是对象列表。"""

    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}] must be a JSON object")
        items.append(item)
    return items


def _format_memory_metrics(metrics: list[dict[str, Any]]) -> list[str]:
    """格式化 final memory metrics。"""

    lines: list[str] = []
    for item in metrics:
        memory_name = _require_non_empty_text(item.get("memory"), "memory")
        occupancy_iou = _require_number(item.get("occupancy_iou"), f"{memory_name}.occupancy_iou")
        mean_abs_error = _require_number(item.get("mean_abs_error"), f"{memory_name}.mean_abs_error")
        lines.append(
            f"- {memory_name} occupancy_iou={occupancy_iou:.6f} "
            f"mean_abs_error={mean_abs_error:.6f}"
        )
    return lines


def _format_consistency_metrics(metrics: list[dict[str, Any]]) -> list[str]:
    """按 memory 类型汇总 consistency metrics。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metrics:
        memory_name = _require_non_empty_text(item.get("memory"), "memory")
        grouped[memory_name].append(item)

    lines: list[str] = []
    for memory_name in sorted(grouped):
        items = grouped[memory_name]
        alignment_values = [
            _require_number(item.get("alignment_iou"), f"{memory_name}.alignment_iou")
            for item in items
        ]
        flicker_values = [
            _require_number(item.get("flicker_score"), f"{memory_name}.flicker_score")
            for item in items
        ]
        lines.append(
            f"- {memory_name} mean_alignment_iou={_mean(alignment_values):.6f} "
            f"mean_flicker_score={_mean(flicker_values):.6f}"
        )
    return lines


def _validate_existing_file(path: str | Path, field_name: str) -> Path:
    """校验路径存在且是文件。"""

    candidate = Path(path)
    if not str(candidate).strip():
        raise ValueError(f"{field_name} is empty")
    if not candidate.exists():
        raise FileNotFoundError(f"{field_name} not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"{field_name} is not a file: {candidate}")
    return candidate


def _require_non_empty_text(value: Any, field_name: str) -> str:
    """校验非空字符串字段。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_number(value: Any, field_name: str) -> float:
    """校验数值字段并转换为 float。"""

    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _mean(values: list[float]) -> float:
    """计算非空列表均值。"""

    if not values:
        raise ValueError("cannot compute mean of empty values")
    return sum(values) / len(values)


def _format_path(path: str | Path) -> str:
    """将路径格式化为跨平台可读文本。"""

    return str(path).replace("\\", "/")


def _format_metadata_value(value: Any) -> str:
    """格式化 metadata 字段，避免 Windows 路径分隔符污染报告。"""

    if isinstance(value, str):
        return value.replace("\\", "/")
    return str(value)
