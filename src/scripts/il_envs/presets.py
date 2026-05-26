# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab environment presets — parameter templates for common tasks.

Each preset is a dict mapping ``node_type`` → ``{param_key: value}``.
Applying a preset batch-updates every matching node on the canvas.

Each preset lives in its own file under ``presets_data/`` and exports
``NAME: str``, ``TASK_NAME: str``, ``ORDER: int``, and ``ENTRY: ILPreset``.
The aggregator scans the sub-package at import time and materialises
``IL_PRESETS`` as an ordered dict (display order driven by ``ORDER``).

Preset values are derived from the official Isaac Lab source:
  isaaclab_tasks/manager_based/locomotion/velocity/config/go2/
  - velocity_env_cfg.py  (base)
  - rough_env_cfg.py     (Go2 rough override)
  - flat_env_cfg.py      (Go2 flat override)
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Tuple


# Type alias kept for callers that annotate against ILPreset.
ILPreset = Dict[str, Dict[str, str]]


def _collect_presets() -> Dict[str, ILPreset]:
    pkg = importlib.import_module("scripts.il_envs.presets_data")
    items: List[Tuple[int, str, ILPreset]] = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"scripts.il_envs.presets_data.{m.name}")
        name = getattr(mod, "NAME", None)
        entry = getattr(mod, "ENTRY", None)
        if not name or not entry:
            continue
        order = int(getattr(mod, "ORDER", 1_000_000))
        items.append((order, name, entry))
    items.sort(key=lambda iao: (iao[0], iao[1]))
    return {name: entry for _, name, entry in items}


IL_PRESETS: Dict[str, ILPreset] = _collect_presets()


def list_il_presets() -> List[str]:
    """Return available preset names."""
    return list(IL_PRESETS.keys())


def get_il_preset(name: str) -> ILPreset:
    """Return preset dict by name. Raises KeyError if not found."""
    return IL_PRESETS[name]


def get_il_preset_task_name(name: str) -> str:
    """Return the Isaac Lab registered task name for a preset."""
    preset = IL_PRESETS[name]
    meta = preset.get("_meta", {})
    return meta.get("task_name", "Isaac-Velocity-Flat-Unitree-Go2-v0")


__all__ = [
    "ILPreset",
    "IL_PRESETS",
    "list_il_presets",
    "get_il_preset",
    "get_il_preset_task_name",
]
