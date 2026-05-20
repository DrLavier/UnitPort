"""Severity classifier for Isaac Sim subprocess stdout.

Isaac Sim / carb / hydra emit their own ``[Fatal]`` / ``[Error]`` /
``[Critical]`` and ``[Warning]`` tagged lines (carbOnPluginStartup
crashes, RTX init failures, USD parse errors). Without classification
these all flow through ``log_info`` and silently drown in the boot-spam
INFO stream — so a STATUS_ACCESS_VIOLATION backtrace looks the same as
a benign warm-up message.

Note: carb's crashreporter tags its category as ``[crash]`` on both
``[Warning]`` (metadata dump) and ``[Fatal]`` (backtrace) lines, so
``crash`` cannot be used as a severity signal — only the leading
severity bracket counts.
"""
from __future__ import annotations

import re

_SEVERITY_RE = re.compile(r"\[(Fatal|Error|Critical)\]", re.IGNORECASE)
_WARNING_RE = re.compile(r"\[Warning\]", re.IGNORECASE)


def classify_severity(line: str) -> str:
    """Return ``"error"`` / ``"warning"`` / ``"info"`` for a subprocess line."""
    if _SEVERITY_RE.search(line):
        return "error"
    if _WARNING_RE.search(line):
        return "warning"
    return "info"


__all__ = ["classify_severity"]
