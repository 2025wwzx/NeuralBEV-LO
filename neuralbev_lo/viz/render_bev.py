#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BEV 张量渲染工具。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def _normalize_channel(channel: NDArray[np.float32]) -> NDArray[np.float32]:
    """将单通道裁剪到 `[0, 1]`。"""

    return np.clip(channel.astype(np.float32), 0.0, 1.0)


def bev_to_uint8_image(bev: object) -> NDArray[np.uint8]:
    """将 BEV `[C, H, W]` 转为 RGB `uint8[H, W, 3]`。

    映射规则:
        R: density 通道。
        G: max_height 通道，从 `[-1, 1]` 映射到 `[0, 1]`。
        B: intensity 通道；缺失时退化为 density。
    """

    array = np.asarray(bev, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"BEV tensor must have shape [C, H, W], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("BEV tensor must contain at least one channel")

    density = _normalize_channel(array[0])
    height = _normalize_channel((array[1] + 1.0) / 2.0) if array.shape[0] > 1 else density
    intensity = _normalize_channel(array[3]) if array.shape[0] > 3 else density
    rgb = np.stack([density, height, intensity], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def save_bev_image(bev: object, output_path: str | Path) -> Path:
    """保存 BEV PNG 图像。

    参数:
        bev: `[C, H, W]` BEV 张量。
        output_path: 输出 PNG 路径。

    返回:
        实际保存路径。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = bev_to_uint8_image(bev)
    success = cv2.imwrite(str(path), image_rgb[:, :, ::-1])
    if not success:
        raise ValueError(f"failed to write BEV image: {path}")
    return path
