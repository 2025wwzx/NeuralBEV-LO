#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3 demo 视频合成工具。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def save_demo_video(
    *,
    memory_image: str | Path,
    trajectory_image: str | Path,
    output_path: str | Path,
    seconds: float = 6.0,
    fps: int = 6,
    title: str = "NeuralBEV-LO M3 demo",
) -> Path:
    """把 BEV memory panel 与 trajectory overlay 合成为 MP4 视频。

    参数:
        memory_image: 已生成的 current/naive/GT/learned BEV 对比图。
        trajectory_image: 已生成的 GT/naive/learned trajectory overlay 图。
        output_path: 输出 MP4 路径。
        seconds: 视频时长，必须为正数。
        fps: 视频帧率，必须为正整数。
        title: 顶部标题文本。
    返回:
        实际保存的视频路径。
    异常:
        输入图像不存在、无法读取或视频写入失败时抛出 `ValueError`。
    """

    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    memory = _read_image(memory_image, name="memory_image")
    trajectory = _read_image(trajectory_image, name="trajectory_image")
    frame = _compose_frame(memory, trajectory, title=title)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = max(1, int(round(seconds * fps)))
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"failed to open video writer: {path}")
    try:
        for frame_index in range(frame_count):
            rendered = frame.copy()
            cv2.putText(
                rendered,
                f"frame {frame_index + 1:02d}/{frame_count:02d}",
                (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(rendered)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError(f"failed to write demo video: {path}")
    return path


def _read_image(path_like: str | Path, *, name: str) -> NDArray[np.uint8]:
    """读取 BGR `uint8[H, W, 3]` 图像。"""

    path = Path(path_like)
    if not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a readable RGB/BGR image: {path}")
    return image


def _compose_frame(
    memory: NDArray[np.uint8],
    trajectory: NDArray[np.uint8],
    *,
    title: str,
) -> NDArray[np.uint8]:
    """组合视频单帧，左侧 memory panel，右侧 trajectory overlay。"""

    target_height = max(memory.shape[0], trajectory.shape[0], 240)
    memory_resized = _resize_to_height(memory, target_height)
    trajectory_resized = _resize_to_height(trajectory, target_height)
    divider = np.full((target_height, 8, 3), 28, dtype=np.uint8)
    content = np.concatenate([memory_resized, divider, trajectory_resized], axis=1)
    title_height = 42
    footer_height = 32
    canvas = np.zeros(
        (target_height + title_height + footer_height, content.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[title_height : title_height + target_height] = content
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "left: current/naive/GT/learned BEV memory | right: trajectory overlay",
        (12, target_height + title_height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return _ensure_even_shape(canvas)


def _resize_to_height(image: NDArray[np.uint8], target_height: int) -> NDArray[np.uint8]:
    """按目标高度等比例缩放图像。"""

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {image.shape}")
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _ensure_even_shape(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """补齐到偶数宽高，避免部分 MP4 编码器拒绝奇数尺寸。"""

    height, width = image.shape[:2]
    pad_bottom = height % 2
    pad_right = width % 2
    if pad_bottom == 0 and pad_right == 0:
        return image
    return cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
