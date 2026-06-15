# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.sim2sim_ranges — measured cross-engine randomization range tables.

Unlike the other registries, this domain holds **measured** (data-driven)
state, not hand-authored factory data: the Stage-1 measurement infrastructure
writes one ``<sku>.json`` per robot under
``USER_CONFIG_DIR/registers/sim2sim_ranges/`` (via ``Storage.push_data``), and
Stage-2 cross-engine domain randomization reads it back here. There is therefore
NO ``registers/data/`` factory file (§4 — factory data is never written at
runtime) and NO alias/overlay merge (the table IS the user state).

This loader is queried ON DEMAND by the training compiler (Stage 2), NOT loaded
into the global ``RegistryHub.load_all`` graph — a per-SKU measured artifact has
no place in the dependency-ordered boot. It returns the raw parsed dict; the
application side wraps it in ``…sim2sim_measurement.range_table.RangeTable`` so
this registry module stays free of any ``application`` import (no layer
inversion).

Fail-loud (§8): ``get_ranges`` raises on an unknown SKU (a Stage-2 consumer
asking for ranges that were never measured is a bug, not a soft default).

API::

    load() -> int
    reload() -> int
    list_skus() -> list[str]
    has(sku) -> bool
    get_ranges(sku) -> dict        # raw measured table for the SKU
    validate() -> list[str]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from unitport_sdk import Paths, log_warning, read_data

_EXPECTED_SCHEMA_VERSION = 1


class Sim2SimRangesError(RuntimeError):
    """Raised by the runtime API on an unknown SKU or a malformed table."""


def _ranges_dir() -> Path:
    """Live measured-table directory inside the current USER_CONFIG_DIR.

    Resolved at call time so a workspace hot-switch is picked up (mirrors the
    other registries' lazy user-overlay path). Tests monkeypatch this."""
    return Paths.USER_CONFIG_DIR / "registers" / "sim2sim_ranges"


_state: Dict[str, Any] = {"loaded": False, "tables": {}}


def load() -> int:
    """Scan the measured-table dir and cache one parsed dict per SKU.

    Idempotent. When ``USER_CONFIG_DIR`` is not configured yet (pre-wizard) or
    the dir is absent, there simply are no measured tables — load nothing,
    return 0 (NOT an error: a robot with no Stage-1 measurement is a normal
    pre-Stage-2 state)."""
    if _state["loaded"]:
        return len(_state["tables"])
    tables: Dict[str, Dict[str, Any]] = {}
    if Paths.is_user_config_dir_set():
        d = _ranges_dir()
        if d.is_dir():
            for fp in sorted(d.glob("*.json")):
                sku = fp.stem
                payload = read_data(fp)
                if not isinstance(payload, dict):
                    log_warning(
                        f"[sim2sim_ranges] {fp} did not parse to a dict; skipping"
                    )
                    continue
                tables[sku] = payload
    _state["tables"] = tables
    _state["loaded"] = True
    return len(tables)


def reload() -> int:
    """Drop the cache and re-scan — call after a new measurement is persisted."""
    _state["loaded"] = False
    _state["tables"] = {}
    return load()


def list_skus() -> List[str]:
    if not _state["loaded"]:
        load()
    return list(_state["tables"].keys())


def has(sku: str) -> bool:
    if not _state["loaded"]:
        load()
    return str(sku) in _state["tables"]


def get_ranges(sku: str) -> Dict[str, Any]:
    """Return the raw measured range-table dict for ``sku``.

    Raises :class:`Sim2SimRangesError` when no table was measured for the SKU —
    Stage-2 randomization must not silently proceed with an absent range (§8).
    """
    if not _state["loaded"]:
        load()
    key = str(sku)
    if key not in _state["tables"]:
        raise Sim2SimRangesError(
            f"sim2sim_ranges.get_ranges: no measured range table for sku={sku!r} "
            f"(known: {sorted(_state['tables'].keys())}). Run the Stage-1 "
            f"measurement (coordinator) for this robot before enabling "
            f"cross-engine domain randomization."
        )
    return _state["tables"][key]


def validate() -> List[str]:
    """Light per-table schema check. Returns problem strings; ``[]`` = OK.

    These tables are measured artifacts, so corruption is the only failure mode
    — schema_version + a ``dimensions`` list. Cross-registry rules don't apply
    (no factory catalog to align against)."""
    if not _state["loaded"]:
        load()
    problems: List[str] = []
    for sku, tbl in _state["tables"].items():
        ver = tbl.get("schema_version")
        if ver != _EXPECTED_SCHEMA_VERSION:
            problems.append(
                f"  sim2sim_ranges[{sku}]: schema_version {ver!r} != "
                f"{_EXPECTED_SCHEMA_VERSION} — re-measure with the current tool"
            )
        if not isinstance(tbl.get("dimensions"), list):
            problems.append(
                f"  sim2sim_ranges[{sku}]: 'dimensions' missing or not a list"
            )
    return problems


__all__ = [
    "Sim2SimRangesError",
    "load",
    "reload",
    "list_skus",
    "has",
    "get_ranges",
    "validate",
]
