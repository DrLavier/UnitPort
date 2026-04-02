"""Helpers for interpreting node result payloads."""

from __future__ import annotations

from typing import Any, Set


_FAILURE_STATUSES = {"failed", "error"}


def is_failure_result(result: Any) -> bool:
    """Return True when *result* encodes a failure at any nested level.

    Canvas/runtime nodes do not share a single flat output schema. Some emit:

    - ``{"error": "..."}``
    - ``{"status": "error"}``
    - ``{"checkpoint": {"status": "error", ...}}``
    - ``{"result": {"status": "failed", ...}}``

    The runtime/UI failure contract needs to treat all of these as failures.
    """

    return _is_failure_result_inner(result, seen=set())


def _is_failure_result_inner(result: Any, seen: Set[int]) -> bool:
    if not isinstance(result, dict):
        return False

    marker = id(result)
    if marker in seen:
        return False
    seen.add(marker)

    error_value = result.get("error")
    if isinstance(error_value, str) and error_value.strip():
        return True
    if error_value not in (None, "", False):
        return True

    status = str(result.get("status", "") or "").strip().lower()
    if status in _FAILURE_STATUSES:
        return True

    for value in result.values():
        if isinstance(value, dict) and _is_failure_result_inner(value, seen):
            return True

    return False
