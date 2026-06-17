#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹 overlay 渲染工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_trajectory_overlay(
    trajectories: Mapping[str, object],
    output_path: str | Path,
    *,
    title: str = "Odometry trajectory comparison",
) -> Path:
    """保存多条 XY 轨迹 overlay 图。

    参数:
        trajectories: 名称到 `[N, 3]` 轨迹数组的映射，前两列为 x/y。
        output_path: 输出 PNG 路径。
        title: 图标题。
    返回:
        实际保存路径。
    """

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    for name, raw_trajectory in trajectories.items():
        trajectory = _as_trajectory_array(raw_trajectory, name=str(name))
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker="o", markersize=2.5, linewidth=1.4, label=str(name))
        ax.scatter(trajectory[0, 0], trajectory[0, 1], marker="s", s=20)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker="x", s=30)
    ax.set_title(title)
    ax.set_xlabel("x forward / m")
    ax.set_ylabel("y left / m")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _as_trajectory_array(values: object, *, name: str) -> NDArray[np.float64]:
    """校验轨迹数组为有限 `[N, 3]`。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"trajectory {name} must have shape [N, 3], got {array.shape}")
    if array.shape[0] <= 1:
        raise ValueError(f"trajectory {name} must contain at least two poses")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"trajectory {name} must contain only finite values")
    return array
