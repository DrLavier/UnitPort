#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor parameter source helpers — Step 3.

Pure-Python; no Qt dependency.  Provides source-resolution and IO-mapping
mutation utilities used by the Movement Settings panel and its tests.

Public API
----------
STRUCTURAL_KEYS          — frozenset of params that never support source-switching.
ParamSourceInfo          — NamedTuple(source, signal).
get_param_source(...)    — resolve current source for a track/param pair.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple


# Timing/structural params that are always constant and must not be bound
# to external signals.
STRUCTURAL_KEYS: frozenset = frozenset({"start_time", "duration"})


class ParamSourceInfo(NamedTuple):
    """Resolved source status for a single motor parameter."""
    source: str   # "constant" | "external"
    signal: str   # external signal name when source=="external", else ""


def get_param_source(
    track_name: str,
    param_key: str,
    io_mappings: "List[Any]",  # List[HBIOMapping] — duck-typed
) -> ParamSourceInfo:
    """Return the resolved source for a *track_name* / *param_key* pair.

    Checks whether any inbound IO mapping targets exactly
    ``"{track_name}.{param_key}"``.  Returns ``ParamSourceInfo("external",
    signal)`` if found, else ``ParamSourceInfo("constant", "")``.

    Only *inbound* mappings are considered; outbound mappings do not change
    the incoming source for the parameter.
    """
    if param_key in STRUCTURAL_KEYS:
        return ParamSourceInfo("constant", "")

    exact_target = f"{track_name}.{param_key}"
    for m in io_mappings:
        if getattr(m, "direction", "") != "inbound":
            continue
        if getattr(m, "target_param", "") == exact_target:
            return ParamSourceInfo("external", getattr(m, "signal", ""))
    return ParamSourceInfo("constant", "")
