#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""随机种子工具。"""

from __future__ import annotations

import random
from typing import Final

import numpy as np

DEFAULT_SEED: Final[int] = 20260608


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """设置可用库的随机种子。

    参数:
        seed: 非负整数随机种子。

    返回:
        实际设置的随机种子。

    异常:
        `ValueError`: 当种子不是非负整数时抛出。

    说明:
        Week 1 只强制设置 Python 和 NumPy。PyTorch 种子会在训练阶段根据
        是否安装 PyTorch 再补充。
    """

    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    return seed
