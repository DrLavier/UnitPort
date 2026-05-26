# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Termination preset aggregator — SB3 + Isaac Lab.

Each preset lives in its own file under ``sb3/`` or ``isaac_lab/`` and
exports a single ``ENTRY: TaskModuleItem``. This module scans both
sub-packages at import time and materialises ``TERMINATION_REGISTRY`` /
``IL_TERMINATION_REGISTRY``. The dicts are module-level so downstream
in-place mutation (UI editor's ``dataclasses.replace`` path) lands on a
stable object identity.

Per-key scoping is achieved by each entry passing
``backends=frozenset({BACKEND_SB3 or BACKEND_ISAAC})`` directly — the
private ``_SB3_ONLY`` / ``_IL_ONLY`` aggregator shortcuts are inlined.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from scripts.task_module import TaskModuleItem


def _collect(subpkg: str) -> Dict[str, TaskModuleItem]:
    pkg = importlib.import_module(f"scripts.terminations.{subpkg}")
    out: Dict[str, TaskModuleItem] = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"scripts.terminations.{subpkg}.{m.name}")
        entry = getattr(mod, "ENTRY", None)
        if entry is None:
            continue
        if entry.key in out:
            raise RuntimeError(
                f"[scripts.terminations.{subpkg}] duplicate ENTRY.key "
                f"{entry.key!r} in {mod.__name__}"
            )
        out[entry.key] = entry
    return out


TERMINATION_REGISTRY:    Dict[str, TaskModuleItem] = _collect("sb3")
IL_TERMINATION_REGISTRY: Dict[str, TaskModuleItem] = _collect("isaac_lab")


def termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(TERMINATION_REGISTRY)


def default_termination_conditions() -> Dict[str, float]:
    return {
        key: TERMINATION_REGISTRY[key].default
        for key in (
            "fall_threshold_roll",
            "fall_threshold_pitch",
            "min_height",
            "max_contact_impulse",
            "joint_limit_violation",
        )
    }


def il_termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_TERMINATION_REGISTRY)


def default_il_termination_conditions() -> Dict[str, float]:
    """Default IL termination items for a fresh canvas — keep timeout and
    illegal-contact on by default to mirror the previous toggle defaults
    (enable_timeout=true, enable_illegal_contact=true, enable_base_height=false).
    """
    return {
        "time_out": IL_TERMINATION_REGISTRY["time_out"].default,
        "illegal_contact": IL_TERMINATION_REGISTRY["illegal_contact"].default,
    }


__all__ = [
    "TERMINATION_REGISTRY",
    "IL_TERMINATION_REGISTRY",
    "termination_registry",
    "il_termination_registry",
    "default_termination_conditions",
    "default_il_termination_conditions",
]
