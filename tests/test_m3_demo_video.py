#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 11 M3 demo video 测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from neuralbev_lo.viz.make_demo_video import save_demo_video


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    """写出一张小尺寸 RGB 测试图。"""

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = np.asarray(color, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image[:, :, ::-1])


def _read_video_frame_count(path: Path) -> int:
    """读取视频帧数。"""

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def test_save_demo_video_combines_memory_and_trajectory_images(tmp_path: Path) -> None:
    """demo video 应能把 memory panel 和 trajectory overlay 合成 MP4。"""

    memory_path = tmp_path / "memory.png"
    trajectory_path = tmp_path / "trajectory.png"
    output_path = tmp_path / "demo.mp4"
    _write_rgb_image(memory_path, (220, 40, 40))
    _write_rgb_image(trajectory_path, (40, 80, 220))

    saved = save_demo_video(
        memory_image=memory_path,
        trajectory_image=trajectory_path,
        output_path=output_path,
        seconds=1.0,
        fps=4,
    )

    assert saved == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert _read_video_frame_count(output_path) == 4


def test_build_m3_demo_video_cli_writes_video(tmp_path: Path) -> None:
    """M3 video CLI 应能从已有 Week 8/9 图像产物生成视频。"""

    memory_path = tmp_path / "memory.png"
    trajectory_path = tmp_path / "trajectory.png"
    output_path = tmp_path / "m3_demo.mp4"
    _write_rgb_image(memory_path, (180, 180, 40))
    _write_rgb_image(trajectory_path, (40, 180, 180))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_m3_demo_video.py",
            "--memory-image",
            str(memory_path),
            "--trajectory-image",
            str(trajectory_path),
            "--output",
            str(output_path),
            "--seconds",
            "1.0",
            "--fps",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "M3 demo video completed" in completed.stdout
    assert output_path.exists()
    assert _read_video_frame_count(output_path) == 4
