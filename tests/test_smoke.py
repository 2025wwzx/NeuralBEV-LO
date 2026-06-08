#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Week 1 smoke tests."""

from __future__ import annotations

from pathlib import Path

import neuralbev_lo
from neuralbev_lo.utils.config import load_yaml_config
from neuralbev_lo.utils.seed import set_global_seed


def test_package_imports() -> None:
    """验证包可以被导入并暴露版本号。"""

    assert neuralbev_lo.__version__ == "0.1.0"


def test_dataset_config_loads() -> None:
    """验证默认 KITTI 配置文件可读取。"""

    config = load_yaml_config(Path("configs/dataset/kitti.yaml"))
    assert config["dataset"]["name"] == "kitti_odometry"
    assert config["dataset"]["data_root_env"] == "KITTI_ROOT"


def test_seed_validation() -> None:
    """验证随机种子工具接受非负整数。"""

    assert set_global_seed(20260608) == 20260608
