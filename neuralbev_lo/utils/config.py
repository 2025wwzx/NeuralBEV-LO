#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配置读取工具。

提供统一的 YAML 配置读取入口，负责路径校验、基础类型校验和异常信息整理。
所有调用方应传入明确路径，避免在业务代码中硬编码数据集或实验目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并返回字典。

    参数:
        config_path: YAML 配置文件路径，可以是字符串或 `Path`。

    返回:
        解析后的配置字典。空文件会返回空字典。

    异常:
        `ValueError`: 路径为空、文件不存在、后缀不是 YAML，或 YAML 顶层不是字典。
        `yaml.YAMLError`: YAML 语法错误。
    """

    path = Path(config_path)
    if not str(path).strip():
        raise ValueError("config_path must not be empty")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"config must be a YAML file: {path}")
    if not path.exists():
        raise ValueError(f"config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return loaded
