#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""结构化控制台日志工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _format_message(level: str, message: str, **context: Any) -> str:
    """格式化结构化日志行。

    参数:
        level: 日志级别，例如 INFO、WARN 或 DEBUG。
        message: 主日志内容。
        context: 可选上下文字段，会按 key=value 输出。

    返回:
        单行日志字符串。
    """

    clean_message = message.strip() if isinstance(message, str) else str(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context_text = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
    suffix = f" {context_text}" if context_text else ""
    return f"[{level}] {timestamp} {clean_message}{suffix}"


def log_info(message: str, **context: Any) -> None:
    """输出 INFO 日志。"""

    print(_format_message("INFO", message, **context))


def log_warn(message: str, **context: Any) -> None:
    """输出 WARN 日志。"""

    print(_format_message("WARN", message, **context))


def log_debug(message: str, **context: Any) -> None:
    """输出 DEBUG 日志。"""

    print(_format_message("DEBUG", message, **context))
