#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 NeuralBEV-LO 本地运行环境。

脚本只输出非敏感环境信息，包括 Python、平台、PyTorch、CUDA、GPU、关键包版本
和配置路径。它不读取 `.env`，不打印任何 secret，也不要求 KITTI 数据已经存在。
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.utils.config import load_yaml_config
from neuralbev_lo.utils.logging import log_info, log_warn

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "dataset" / "kitti.yaml"
PACKAGE_NAMES: Final[tuple[str, ...]] = (
    "numpy",
    "yaml",
    "cv2",
    "matplotlib",
    "tqdm",
    "torch",
)


def _get_package_version(import_name: str) -> str:
    """读取包版本，缺失时返回 `not-installed`。

    参数:
        import_name: 导入名或包名。

    返回:
        版本字符串或 `not-installed`。
    """

    package_map = {
        "yaml": "PyYAML",
        "cv2": "opencv-python-headless",
    }
    metadata_name = package_map.get(import_name, import_name)
    try:
        return importlib.metadata.version(metadata_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _collect_torch_info() -> dict[str, Any]:
    """收集 PyTorch 和 CUDA 信息。

    返回:
        包含 torch 版本、CUDA 是否可用、设备名和算力的字典。未安装 PyTorch 时
        返回明确的 `installed=False`。
    """

    try:
        import torch
    except ImportError:
        return {"installed": False}

    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        try:
            device_index = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(device_index)
            info["device_capability"] = torch.cuda.get_device_capability(device_index)
        except Exception as exc:  # noqa: BLE001 - 环境检查必须尽量不中断。
            info["device_error"] = str(exc)
    return info


def main() -> int:
    """运行环境检查。

    返回:
        进程退出码。环境检查本身尽量返回 0，除非配置文件缺失或解析失败。
    """

    log_info("NeuralBEV-LO environment check started", project_root=PROJECT_ROOT)
    log_info("Python runtime", version=sys.version.split()[0], executable=sys.executable)
    log_info("Platform", system=platform.system(), release=platform.release())

    try:
        dataset_config = load_yaml_config(DEFAULT_CONFIG_PATH)
        log_info("Dataset config loaded", path=DEFAULT_CONFIG_PATH)
        log_info(
            "Dataset root config",
            env=dataset_config.get("dataset", {}).get("data_root_env"),
            default=dataset_config.get("dataset", {}).get("default_data_root"),
        )
    except Exception as exc:  # noqa: BLE001 - 环境检查报告异常上下文。
        log_warn("Dataset config check failed", error=exc)
        return 1

    for package_name in PACKAGE_NAMES:
        log_info("Package version", package=package_name, version=_get_package_version(package_name))

    torch_info = _collect_torch_info()
    if not torch_info.get("installed"):
        log_warn("PyTorch is not installed; CPU-only synthetic smoke remains available")
    else:
        log_info("PyTorch status", **torch_info)

    log_info("NeuralBEV-LO environment check completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
