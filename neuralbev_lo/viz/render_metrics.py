#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标曲线渲染工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_metric_curve(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    metric_name: str = "alignment_iou",
    x_name: str = "frame_index",
    group_name: str = "memory",
    title: str = "BEV consistency curve",
) -> Path:
    """保存按 memory 分组的逐帧指标曲线。

    参数:
        rows: 指标表行，每行至少包含分组、横轴和纵轴字段。
        output_path: 输出 PNG 路径。
        metric_name: 纵轴指标名，例如 `alignment_iou`。
        x_name: 横轴字段名，默认帧索引。
        group_name: 曲线分组字段名，默认 memory 名称。
        title: 图标题。
    返回:
        实际保存路径。
    """

    if not rows:
        raise ValueError("at least one metric row is required")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        for key in (metric_name, x_name, group_name):
            if key not in row:
                raise ValueError(f"metric row missing required key: {key}")
        group = str(row[group_name])
        grouped.setdefault(group, []).append((float(row[x_name]), float(row[metric_name])))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for group, points in sorted(grouped.items()):
        points_sorted = sorted(points, key=lambda item: item[0])
        xs = [item[0] for item in points_sorted]
        ys = [item[1] for item in points_sorted]
        ax.plot(xs, ys, marker="o", markersize=3.0, linewidth=1.4, label=group)
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(metric_name)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
