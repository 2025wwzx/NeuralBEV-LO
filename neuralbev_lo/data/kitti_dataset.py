#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KITTI 数据集占位模块。

本模块在 Week 1 只保留包结构和文档化入口，真实 KITTI 读取逻辑将在 Week 2
实现。保留独立模块可以让后续 agent 在明确边界内添加数据验证、点云读取、
标定解析和位姿转换。
"""

from __future__ import annotations


class KittiDatasetNotImplementedError(NotImplementedError):
    """标记 KITTI 数据读取尚未进入实现阶段的异常。"""
