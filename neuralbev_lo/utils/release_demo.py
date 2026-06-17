#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 release demo 一命令编排工具。"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from neuralbev_lo.utils.logging import log_info

DEFAULT_EVAL_CONFIG: Final[Path] = Path("configs/eval/kitti_cpu_smoke_safe.yaml")
DEFAULT_TRAIN_CONFIG: Final[Path] = Path("configs/train/posenet_3dof.yaml")
DEFAULT_CHECKPOINT: Final[Path] = Path(
    "outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt"
)
DEFAULT_DATA_ROOT: Final[Path] = Path("data/kitti_odometry")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/figures/v0_1_release_demo")
DEFAULT_METRICS_DIR: Final[Path] = Path("outputs/metrics/v0_1_release_demo")
DEFAULT_VIDEO_OUTPUT: Final[Path] = DEFAULT_OUTPUT_DIR / "neuralbev_lo_v0_1_release_demo.mp4"
DEFAULT_REPORT_OUTPUT: Final[Path] = Path("outputs/reports/v0_1_release_demo.txt")


@dataclass(frozen=True)
class ReleaseDemoConfig:
    """v0.1 release demo 编排配置。"""

    eval_config: Path = DEFAULT_EVAL_CONFIG
    train_config: Path = DEFAULT_TRAIN_CONFIG
    checkpoint: Path = DEFAULT_CHECKPOINT
    sequence: str = "07"
    frames: int = 5
    start_frame: int = 0
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    metrics_dir: Path = DEFAULT_METRICS_DIR
    video_output: Path = DEFAULT_VIDEO_OUTPUT
    report_output: Path = DEFAULT_REPORT_OUTPUT
    cpu: bool = True
    video_seconds: float = 6.0
    video_fps: int = 6
    title: str = "NeuralBEV-LO v0.1 research prototype"

    @property
    def end_frame(self) -> int:
        """返回 demo 的闭区间结束帧索引。"""

        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        return self.start_frame + self.frames - 1

    @property
    def prefix(self) -> str:
        """返回与 build_bev_memory_demo.py 一致的 artifact 文件名前缀。"""

        sequence = self.sequence.strip()
        if not sequence:
            raise ValueError("sequence must be non-empty")
        return f"{sequence}_{self.start_frame:06d}_{self.end_frame:06d}"

    @property
    def memory_image(self) -> Path:
        """返回 memory comparison 图片路径。"""

        return self.output_dir / f"{self.prefix}_memory.png"

    @property
    def trajectory_image(self) -> Path:
        """返回 trajectory overlay 图片路径。"""

        return self.output_dir / f"{self.prefix}_trajectory.png"

    @property
    def alignment_curve(self) -> Path:
        """返回 alignment curve 图片路径。"""

        return self.output_dir / f"{self.prefix}_alignment_curve.png"

    @property
    def memory_metrics(self) -> Path:
        """返回 memory metrics JSON 路径。"""

        return self.metrics_dir / f"{self.prefix}_memory_metrics.json"

    @property
    def consistency_metrics(self) -> Path:
        """返回 consistency metrics JSON 路径。"""

        return self.metrics_dir / f"{self.prefix}_consistency_metrics.json"

    @property
    def report_artifacts(self) -> list[Path]:
        """返回 final report 中应列出的 artifact 路径。"""

        return [
            self.memory_image,
            self.trajectory_image,
            self.alignment_curve,
            self.video_output,
            self.memory_metrics,
            self.consistency_metrics,
        ]


def build_release_demo_commands(
    config: ReleaseDemoConfig,
    *,
    python_executable: str | Path = sys.executable,
) -> list[list[str]]:
    """构造 v0.1 release demo 的三段子命令。

    参数:
        config: release demo 编排配置。
        python_executable: 用于执行子脚本的 Python 可执行文件。
    返回:
        依次执行 BEV memory demo、M3 video、final report 的命令列表。
    """

    python_text = str(python_executable)
    bev_command = [
        python_text,
        "scripts/build_bev_memory_demo.py",
        "--config",
        _path_text(config.eval_config),
        "--train-config",
        _path_text(config.train_config),
        "--pose-source",
        "learned",
        "--checkpoint",
        _path_text(config.checkpoint),
        "--sequence",
        config.sequence,
        "--frames",
        str(config.frames),
        "--start-frame",
        str(config.start_frame),
        "--data-root",
        _path_text(config.data_root),
        "--output-dir",
        _path_text(config.output_dir),
        "--metrics-dir",
        _path_text(config.metrics_dir),
    ]
    if config.cpu:
        bev_command.append("--cpu")

    video_command = [
        python_text,
        "scripts/build_m3_demo_video.py",
        "--memory-image",
        _path_text(config.memory_image),
        "--trajectory-image",
        _path_text(config.trajectory_image),
        "--output",
        _path_text(config.video_output),
        "--seconds",
        str(config.video_seconds),
        "--fps",
        str(config.video_fps),
        "--title",
        config.title,
    ]

    report_command = [
        python_text,
        "scripts/write_v0_1_report.py",
        "--memory-metrics",
        _path_text(config.memory_metrics),
        "--consistency-metrics",
        _path_text(config.consistency_metrics),
        "--config",
        _path_text(config.eval_config),
        "--output",
        _path_text(config.report_output),
    ]
    for artifact_path in config.report_artifacts:
        report_command.extend(["--artifact", _path_text(artifact_path)])

    return [bev_command, video_command, report_command]


def run_release_demo(
    config: ReleaseDemoConfig,
    *,
    dry_run: bool = False,
    python_executable: str | Path = sys.executable,
) -> int:
    """运行 v0.1 release demo 子命令。

    参数:
        config: release demo 编排配置。
        dry_run: 为 True 时只打印命令，不执行。
        python_executable: 子命令使用的 Python 可执行文件。
    返回:
        进程退出码；任一子命令失败时立即返回该命令的退出码。
    """

    commands = build_release_demo_commands(config, python_executable=python_executable)
    if dry_run:
        log_info("Release demo dry run", commands=len(commands))
        for index, command in enumerate(commands, start=1):
            print(f"[{index}] {_format_command(command)}")
        return 0

    for index, command in enumerate(commands, start=1):
        log_info("Release demo step started", step=index, command=_format_command(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            log_info("Release demo step failed", step=index, returncode=completed.returncode)
            return int(completed.returncode)
        log_info("Release demo step completed", step=index)
    log_info("Release demo completed", report=config.report_output, video=config.video_output)
    return 0


def _path_text(path: str | Path) -> str:
    """格式化路径参数。"""

    return str(path).replace("\\", "/")


def _format_command(command: list[str]) -> str:
    """格式化命令，便于 dry-run 和日志阅读。"""

    return " ".join(command)
