#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BEV 张量渲染工具。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def _normalize_channel(channel: NDArray[np.float32]) -> NDArray[np.float32]:
    """将单通道裁剪到 `[0, 1]`."""

    return np.clip(channel.astype(np.float32), 0.0, 1.0)


def bev_to_uint8_image(bev: object) -> NDArray[np.uint8]:
    """将 BEV `[C, H, W]` 转为 RGB `uint8[H, W, 3]`.

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
    """保存 BEV PNG 图像。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = bev_to_uint8_image(bev)
    success = cv2.imwrite(str(path), image_rgb[:, :, ::-1])
    if not success:
        raise ValueError(f"failed to write BEV image: {path}")
    return path


def save_bev_comparison(
    panels: list[tuple[str, object]],
    output_path: str | Path,
    *,
    label_height: int = 24,
) -> Path:
    """保存多个 BEV 面板的横向对比图。

    参数:
        panels: `(title, bev)` 列表，每个 BEV 形状为 `[C,H,W]`。
        output_path: 输出 PNG 路径。
        label_height: 顶部标签区域高度。
    返回:
        实际保存路径。
    """

    if not panels:
        raise ValueError("at least one BEV panel is required")

    images = [bev_to_uint8_image(bev) for _, bev in panels]
    heights = {image.shape[0] for image in images}
    widths = {image.shape[1] for image in images}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("all BEV panels must have the same spatial shape")

    labeled_images: list[NDArray[np.uint8]] = []
    for (title, _), image in zip(panels, images, strict=True):
        canvas = np.zeros((image.shape[0] + label_height, image.shape[1], 3), dtype=np.uint8)
        canvas[label_height:] = image
        cv2.putText(
            canvas,
            str(title),
            (4, max(16, label_height - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled_images.append(canvas)

    comparison = np.concatenate(labeled_images, axis=1)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), comparison[:, :, ::-1])
    if not success:
        raise ValueError(f"failed to write BEV comparison image: {path}")
    return path
