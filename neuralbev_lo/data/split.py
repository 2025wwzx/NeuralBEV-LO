#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集划分常量。"""

from __future__ import annotations

SMOKE_SEQUENCES: tuple[str, ...] = ("00",)
TRAIN_SEQUENCES: tuple[str, ...] = ("00", "01", "02", "03", "04", "05", "06")
VAL_SEQUENCES: tuple[str, ...] = ("07", "08")
TEST_SEQUENCES: tuple[str, ...] = ("09", "10")
