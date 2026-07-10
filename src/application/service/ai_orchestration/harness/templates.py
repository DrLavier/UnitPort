# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Same-family template discovery — the orchestrator's soft reference.

A human engineer authors a new canvas by cloning a known-good same-family
template and patching it; the blueprint feeds that template to the orchestrator
as a **worked example** (soft reference, §5), with **loud degradation** to
pure-semantic construction when none exists. This module finds the nearest
shipped template for a ``(backend, family)`` from
``custom_mods/canvas/<canvas_subdir>/*.canvas.json``.

Templates are factory references, never the file that trains (``training_config.md``
§2 / footgun #8) — callers use the returned dict as a prompt example or as the
seed to clone into the live project canvas, never edit it in place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from unitport_sdk import Paths, log_warning, read_data

__all__ = [
    "list_template_paths",
    "find_same_family_template",
    "template_family",
]


def _template_root(backend: str) -> Path:
    from registers import backends as _backends

    subdir = _backends.canvas_subdir(backend)
    return Paths.CUSTOM_MODS_DIR / "canvas" / subdir


def list_template_paths(backend: str) -> List[Path]:
    """All shipped template canvases for a backend (sorted by name)."""
    root = _template_root(backend)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(root.glob("*.canvas.json"))


def template_family(canvas: Dict[str, Any]) -> FrozenSet[str]:
    """Resolve the robot family set a template targets (from its robot SKU)."""
    from registers import robots as _robots

    sku = str(canvas.get("robot_id") or "").strip()
    if not sku:
        # Fall back to the robot node's asset_id param.
        for n in canvas.get("nodes", []):
            if str(n.get("schema_id")) == "robot":
                cell = (n.get("params") or {}).get("asset_id")
                sku = str((cell or {}).get("value") if isinstance(cell, dict) else "")
                break
    if not sku:
        return frozenset()
    spec = _robots.get_robot_spec(sku)
    return frozenset(spec.families) if spec is not None else frozenset()


def find_same_family_template(
    backend: str, families: FrozenSet[str]
) -> Optional[Dict[str, Any]]:
    """Return the nearest same-family template canvas dict, or None.

    Preference: a template whose family set shares the most tokens with
    ``families`` (an exact-family match wins over a broader-tag overlap). None
    means no reference exists → the caller announces loud degradation (§5).
    """
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for path in list_template_paths(backend):
        try:
            canvas = read_data(path)
        except Exception as exc:  # a corrupt shipped template must not crash us
            log_warning(f"[ai.templates] skip unreadable template {path.name}: {exc!r}")
            continue
        if not isinstance(canvas, dict):
            continue
        overlap = len(template_family(canvas) & families)
        if overlap > best_score:
            best, best_score = canvas, overlap
    return best
