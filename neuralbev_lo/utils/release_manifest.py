#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 release manifest 生成工具。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

DEFAULT_TAG_TARGET: Final[str] = "v0.1-research-prototype"
VALIDATION_COMMANDS: Final[tuple[str, ...]] = (
    "python scripts/run_v0_1_release_demo.py",
    "python scripts/check_release_readiness.py --allow-pending-tag --require-artifacts --require-clean-git",
    "python -m pytest",
)


def build_release_manifest(
    repo_root: str | Path,
    artifact_paths: list[str | Path],
    memory_metrics_path: str | Path,
    consistency_metrics_path: str | Path,
    tag_target: str = DEFAULT_TAG_TARGET,
) -> dict[str, Any]:
    """构建 v0.1 release manifest 数据结构。

    参数:
        repo_root: 仓库根目录，用于读取 Git 元数据。
        artifact_paths: 需要记录 size/sha256 的 release artifact 路径列表。
        memory_metrics_path: memory metrics JSON 路径。
        consistency_metrics_path: consistency metrics JSON 路径。
        tag_target: 计划创建的 release tag 名称。
    返回:
        可序列化为 JSON 的 release manifest。
    异常:
        当 artifact 或 metrics 文件缺失、JSON 不合法、字段类型不符合预期时抛出异常。
    """

    root = Path(repo_root)
    memory_path = _resolve_existing_file(root, memory_metrics_path, "memory_metrics_path")
    consistency_path = _resolve_existing_file(
        root,
        consistency_metrics_path,
        "consistency_metrics_path",
    )
    checked_artifacts = [_artifact_entry(root, path) for path in artifact_paths]
    memory_payload = _read_json_object(memory_path)
    consistency_payload = _read_json_object(consistency_path)

    return {
        "schema_version": 1,
        "tag_target": _require_non_empty_text(tag_target, "tag_target"),
        "git": _collect_git_metadata(root),
        "metrics": {
            "memory": _summarize_memory_metrics(memory_payload),
            "consistency": _summarize_consistency_metrics(consistency_payload),
        },
        "artifacts": checked_artifacts,
        "validation_commands": list(VALIDATION_COMMANDS),
    }


def write_release_manifest(
    output_path: str | Path,
    repo_root: str | Path,
    artifact_paths: list[str | Path],
    memory_metrics_path: str | Path,
    consistency_metrics_path: str | Path,
    tag_target: str = DEFAULT_TAG_TARGET,
) -> Path:
    """生成并写入 v0.1 release manifest JSON。"""

    root = Path(repo_root)
    manifest = build_release_manifest(
        repo_root=root,
        artifact_paths=artifact_paths,
        memory_metrics_path=memory_metrics_path,
        consistency_metrics_path=consistency_metrics_path,
        tag_target=tag_target,
    )
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _collect_git_metadata(root: Path) -> dict[str, Any]:
    """读取当前 Git 元数据，失败时保留错误字段而不中断 manifest 生成。"""

    commit = _git_output(root, ["rev-parse", "HEAD"])
    branch = _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    upstream = _git_output(root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    status = _git_output(root, ["status", "--porcelain"])
    return {
        "commit": commit["stdout"],
        "branch": branch["stdout"],
        "upstream": upstream["stdout"],
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else None,
        "errors": [
            item["stderr"]
            for item in (commit, branch, upstream, status)
            if item["returncode"] != 0 and item["stderr"]
        ],
    }


def _git_output(root: Path, args: list[str]) -> dict[str, Any]:
    """运行 Git 并返回结构化输出。"""

    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _artifact_entry(root: Path, path: str | Path) -> dict[str, Any]:
    """生成单个 artifact 的 manifest 条目。"""

    artifact_path = _resolve_existing_file(root, path, "artifact")
    data = artifact_path.read_bytes()
    return {
        "path": _format_path(_relative_or_original(root, artifact_path)),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _summarize_memory_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """提取 memory metrics 摘要。"""

    metrics = _require_list(payload.get("metrics"), "memory metrics")
    summary: dict[str, dict[str, float]] = {}
    for item in metrics:
        memory_name = _require_non_empty_text(item.get("memory"), "memory")
        summary[memory_name] = {
            "occupancy_iou": _require_number(item.get("occupancy_iou"), "occupancy_iou"),
            "mean_abs_error": _require_number(item.get("mean_abs_error"), "mean_abs_error"),
        }
    return summary


def _summarize_consistency_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """按 memory 类型聚合 consistency metrics 摘要。"""

    metrics = _require_list(payload.get("metrics"), "consistency metrics")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metrics:
        memory_name = _require_non_empty_text(item.get("memory"), "memory")
        grouped[memory_name].append(item)

    summary: dict[str, dict[str, float]] = {}
    for memory_name in sorted(grouped):
        alignment_values = [
            _require_number(item.get("alignment_iou"), "alignment_iou")
            for item in grouped[memory_name]
        ]
        flicker_values = [
            _require_number(item.get("flicker_score"), "flicker_score")
            for item in grouped[memory_name]
        ]
        summary[memory_name] = {
            "mean_alignment_iou": _mean(alignment_values),
            "mean_flicker_score": _mean(flicker_values),
        }
    return summary


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并校验顶层类型。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top-level must be an object: {path}")
    return payload


def _require_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    """校验字段是对象列表。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{index}] must be an object")
        items.append(item)
    return items


def _require_non_empty_text(value: Any, field_name: str) -> str:
    """校验非空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _require_number(value: Any, field_name: str) -> float:
    """校验数值字段。"""

    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _mean(values: list[float]) -> float:
    """计算非空列表均值。"""

    if not values:
        raise ValueError("cannot compute mean of empty values")
    return sum(values) / len(values)


def _resolve_existing_file(root: Path, path: str | Path, field_name: str) -> Path:
    """解析并校验文件存在。"""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"{field_name} not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"{field_name} is not a file: {candidate}")
    return candidate


def _relative_or_original(root: Path, path: Path) -> Path:
    """尽量返回相对仓库根目录的路径。"""

    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _format_path(path: str | Path) -> str:
    """将路径格式化为跨平台可读文本。"""

    return str(path).replace("\\", "/")
