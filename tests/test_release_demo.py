#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v0.1 release demo wrapper 测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from neuralbev_lo.utils.release_demo import ReleaseDemoConfig, build_release_demo_commands


def test_build_release_demo_commands_use_expected_paths() -> None:
    """wrapper 应按 sequence/start/end 推导 metrics、figures、video 和 report 路径。"""

    config = ReleaseDemoConfig(
        sequence="07",
        frames=5,
        start_frame=0,
        data_root=Path("data/kitti_odometry"),
        output_dir=Path("outputs/figures/demo"),
        metrics_dir=Path("outputs/metrics/demo"),
        report_output=Path("outputs/reports/demo.txt"),
        video_output=Path("outputs/figures/demo/demo.mp4"),
        cpu=True,
    )

    commands = build_release_demo_commands(config, python_executable="python")
    joined = [" ".join(command) for command in commands]

    assert len(commands) == 3
    assert commands[0][:2] == ["python", "scripts/build_bev_memory_demo.py"]
    assert "--cpu" in commands[0]
    assert "outputs/figures/demo/07_000000_000004_memory.png" in joined[1]
    assert "outputs/figures/demo/demo.mp4" in joined[1]
    assert "outputs/metrics/demo/07_000000_000004_memory_metrics.json" in joined[2]
    assert "outputs/reports/demo.txt" in joined[2]


def test_release_demo_cli_dry_run_outputs_commands() -> None:
    """CLI dry-run 应打印将要执行的三段命令但不生成 artifact。"""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_v0_1_release_demo.py",
            "--dry-run",
            "--sequence",
            "07",
            "--frames",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Release demo dry run" in completed.stdout
    assert "scripts/build_bev_memory_demo.py" in completed.stdout
    assert "scripts/build_m3_demo_video.py" in completed.stdout
    assert "scripts/write_v0_1_report.py" in completed.stdout
