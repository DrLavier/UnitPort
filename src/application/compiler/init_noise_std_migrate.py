# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Bug#1 migration — fold legacy log-space ``init_noise_std`` to direct std.

Before 2026-06 the Isaac Lab config compiler treated ``il_policy_network``'s
``init_noise_std`` as ``log_std`` and applied ``exp()`` (so a user's ``1.0``
silently became std ``e≈2.718``, and ``-1.0`` became ``0.368``). The convention
is now DIRECT std (legged_gym: ``1.0`` == std ``1.0``), emitted verbatim, with a
fail-loud (§8) on any non-positive value.

This migrator reconciles canvases authored under the old convention:

* ``init_noise_std <= 0`` — a value that is only meaningful as log-space (a
  direct std can never be ≤ 0). Convert to ``exp(value)`` so the policy keeps
  the EXACT std it actually trained with (§8(c) legacy compatibility) and emits
  a loud WARN.
* ``init_noise_std > 0`` — already a valid direct std; left untouched. (Its
  realized std DOES change vs the old build — that is precisely the bug fix the
  framework now honours: the user's typed value is taken at face value.)

Shared by the one-shot bootstrap migrator
(``bootstrap/migrate_canvas_init_noise_std_direct.py``) and the canvas load-time
hook (``CanvasPage.from_workflow_dict``). Idempotent: converting a negative
yields a positive, which never re-converts; a stamped flag short-circuits.
"""
from __future__ import annotations

import math
from typing import Any, Dict

INIT_STD_DIRECT_FLAG = "init_noise_std_direct_v1"


def _warn(msg: str) -> None:
    try:
        from unitport_sdk import log_warning
        log_warning(msg)
    except Exception:  # WHY KEPT: (a) sdk may be absent in bare bootstrap sweeps
        print(f"WARNING: {msg}")


def _param_spec(node: Dict[str, Any], key: str):
    return (node.get("params") or {}).get(key)


def _param_value(node: Dict[str, Any], key: str) -> Any:
    spec = _param_spec(node, key)
    return spec.get("value") if isinstance(spec, dict) else spec


def _set_value(node: Dict[str, Any], key: str, value: Any) -> None:
    spec = _param_spec(node, key)
    if isinstance(spec, dict):
        spec["value"] = value
    else:
        node.setdefault("params", {})[key] = {
            "name": key, "value": value, "param_type": "float",
        }


def migrate_init_noise_std_direct(canvas: Dict[str, Any]) -> bool:
    """In-place reconcile a canvas dict to the direct-std convention.

    Returns True iff it changed (a legacy ``<= 0`` value was converted).
    """
    if not isinstance(canvas, dict):
        return False
    meta = canvas.setdefault("metadata", {})
    if isinstance(meta, dict) and meta.get(INIT_STD_DIRECT_FLAG) is True:
        return False

    changed = False
    for node in (canvas.get("nodes") or []):
        if node.get("schema_id") != "il_policy_network":
            continue
        raw = _param_value(node, "init_noise_std")
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v <= 0.0:
            new = math.exp(v)
            # Preserve the stored type flavour (string presets vs numeric canvas).
            _set_value(node, "init_noise_std",
                       f"{new:.6f}" if isinstance(raw, str) else round(new, 6))
            _warn(
                f"[init_noise_std migration] node {node.get('id')!r}: legacy "
                f"log-space init_noise_std={v} → direct std exp({v})={new:.6f} "
                f"(preserves the realized training std). New convention: the "
                f"value is the direct std (1.0 = std 1.0)."
            )
            changed = True

    if changed and isinstance(meta, dict):
        meta[INIT_STD_DIRECT_FLAG] = True
    return changed
