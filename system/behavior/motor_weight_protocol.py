#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor Weight Protocol ingest — Step 1.

Validates, parses, and routes structured motor-parameter protocol payloads
from Mission into the Behavior execution layer.

Design rules (from instructions.txt Step 1)
-------------------------------------------
- Mission computes; Behavior executes. No conflict-resolution logic here.
- No silent fallback from invalid protocol to arbitrary values.
- No implicit unit conversion without explicit config.
- Stale / invalid / out-of-range payloads must not silently execute.
- One motor-param final value per tick.

Public API
----------
validate_protocol_payload(payload, now_ms=None)
    → (is_valid: bool, diagnostics: List[BehaviorDiagnostic])

validate_protocol_structure(payload)
    → (is_valid: bool, errors: List[str])   — structural check only (no staleness)

parse_protocol_targets(payload)
    → List[ParsedTarget]  — only call on a payload that passed validation

Diagnostic code constants
-------------------------
ProtocolDiagCode.INVALID_SCHEMA    "protocol.invalid_schema"
ProtocolDiagCode.STALE             "protocol.stale"
ProtocolDiagCode.UNKNOWN_ADDRESS   "protocol.unknown_address"
ProtocolDiagCode.CLAMPED_VALUE     "protocol.clamped_value"
ProtocolDiagCode.VALID             "protocol.valid"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from system.behavior.behavior_artifact import BehaviorDiagnostic

# ---------------------------------------------------------------------------
# Diagnostic code constants
# ---------------------------------------------------------------------------

class ProtocolDiagCode:
    """Machine-readable diagnostic code constants for protocol ingest."""

    INVALID_SCHEMA  = "protocol.invalid_schema"   # structural validation failed
    STALE           = "protocol.stale"             # timestamp + max_age_ms exceeded
    UNKNOWN_ADDRESS = "protocol.unknown_address"   # address not resolvable at runtime
    CLAMPED_VALUE   = "protocol.clamped_value"     # source value was clamped to limits
    VALID           = "protocol.valid"             # payload passed all checks (info)
    # Circle 1 Step 1: strict mode unknown-field rejection
    UNKNOWN_FIELD   = "protocol.unknown_field"     # unknown top-level or target field
    # Circle 1 Step 2: apply-path result codes
    APPLY_SUCCESS   = "protocol.apply_success"     # target value written to actuator
    APPLY_SKIPPED   = "protocol.apply_skipped"     # target skipped (adapter returned False)
    APPLY_FAILED    = "protocol.apply_failed"      # target apply raised an exception


# ---------------------------------------------------------------------------
# ParsedTarget — resolved write intent for one motor param
# ---------------------------------------------------------------------------

@dataclass
class ParsedTarget:
    """Resolved write intent produced by parse_protocol_targets().

    Fields
    ------
    motor_key   : Logical motor id (e.g. "leg_fl_hip").
    param_key   : Controlled parameter name (e.g. "gain").
    address     : Runtime route key (numeric index or symbolic string).
    dtype       : "float32" | "float64".
    shape       : List of positive integers (e.g. [1] for scalar).
    final_value : Resolved numeric value(s) after source selection and optional clamp.
    was_clamped : True if the value was clamped by limits.
    source_kind : "constant" | "external".
    note        : Optional annotation from the payload target entry.
    """

    motor_key:   str
    param_key:   str
    address:     Union[int, str]
    dtype:       str
    shape:       List[int]
    final_value: Union[float, List[float]]
    was_clamped: bool = False
    source_kind: str = "constant"
    note:        str = ""


# ---------------------------------------------------------------------------
# Internal structural validator (no jsonschema dependency)
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = {"protocol_version", "trace_id", "timestamp_ms", "mode", "max_age_ms", "targets"}
# Circle 1 Step 1: complete set of all known top-level keys.
# In strict mode any key outside this set is rejected with UNKNOWN_FIELD.
_ALLOWED_TOP_LEVEL  = frozenset(_REQUIRED_TOP_LEVEL)
_VALID_MODES        = {"constant", "external"}
_VALID_DTYPES       = {"float32", "float64"}
_VALID_SOURCE_KINDS = {"constant", "external"}
_PROTOCOL_VERSION   = "1.0"
# Circle 1 Step 1: complete set of all known per-target keys.
_ALLOWED_TARGET_FIELDS = frozenset({
    "motor_key", "param_key", "address", "dtype", "shape", "source", "limits", "note",
})


def _err(msg: str, trace_id: str = "") -> BehaviorDiagnostic:
    return BehaviorDiagnostic.error(
        code=ProtocolDiagCode.INVALID_SCHEMA,
        message=msg,
        trace_id=trace_id,
    )


def _validate_numeric_value(v: Any, label: str) -> Optional[str]:
    """Return an error string if v is not a valid NumericValue, else None."""
    if isinstance(v, (int, float)):
        return None
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return None
    return f"{label} must be a number or non-empty array of numbers, got {type(v).__name__}"


def _validate_source(src: Any, idx: int, trace_id: str) -> List[BehaviorDiagnostic]:
    diags: List[BehaviorDiagnostic] = []
    if not isinstance(src, dict):
        diags.append(_err(f"targets[{idx}].source must be an object", trace_id))
        return diags
    kind = src.get("kind")
    if kind not in _VALID_SOURCE_KINDS:
        diags.append(_err(
            f"targets[{idx}].source.kind must be one of {sorted(_VALID_SOURCE_KINDS)}, got {kind!r}",
            trace_id,
        ))
        return diags
    if kind == "constant":
        if "constant" not in src:
            diags.append(_err(f"targets[{idx}].source.constant is required when kind='constant'", trace_id))
        else:
            err = _validate_numeric_value(src["constant"], f"targets[{idx}].source.constant")
            if err:
                diags.append(_err(err, trace_id))
    else:  # external
        for req_key in ("external_key", "external_value"):
            if req_key not in src:
                diags.append(_err(
                    f"targets[{idx}].source.{req_key} is required when kind='external'",
                    trace_id,
                ))
        if "external_key" in src and not isinstance(src["external_key"], str):
            diags.append(_err(f"targets[{idx}].source.external_key must be a string", trace_id))
        if "external_value" in src:
            err = _validate_numeric_value(src["external_value"], f"targets[{idx}].source.external_value")
            if err:
                diags.append(_err(err, trace_id))
    return diags


def _validate_limits(limits: Any, idx: int, trace_id: str) -> List[BehaviorDiagnostic]:
    diags: List[BehaviorDiagnostic] = []
    if limits is None:
        return diags
    if not isinstance(limits, dict):
        diags.append(_err(f"targets[{idx}].limits must be an object", trace_id))
        return diags
    for k in ("min", "max"):
        if k in limits and not isinstance(limits[k], (int, float)):
            diags.append(_err(f"targets[{idx}].limits.{k} must be a number", trace_id))
    if "clamp" in limits and not isinstance(limits["clamp"], bool):
        diags.append(_err(f"targets[{idx}].limits.clamp must be a boolean", trace_id))
    return diags


def _validate_target(
    tgt: Any, idx: int, trace_id: str, strict_mode: bool = True
) -> List[BehaviorDiagnostic]:
    diags: List[BehaviorDiagnostic] = []
    if not isinstance(tgt, dict):
        diags.append(_err(f"targets[{idx}] must be an object", trace_id))
        return diags

    # Circle 1 Step 1: unknown target-field gate
    unknown_keys = set(tgt.keys()) - _ALLOWED_TARGET_FIELDS
    if unknown_keys:
        for k in sorted(unknown_keys):
            msg = f"targets[{idx}]: unknown field '{k}'"
            if strict_mode:
                diags.append(BehaviorDiagnostic.error(
                    code=ProtocolDiagCode.UNKNOWN_FIELD, message=msg, trace_id=trace_id,
                ))
            else:
                diags.append(BehaviorDiagnostic.warning(
                    code=ProtocolDiagCode.UNKNOWN_FIELD, message=msg, trace_id=trace_id,
                ))
        if strict_mode:
            return diags

    for req in ("motor_key", "param_key", "address", "dtype", "shape", "source"):
        if req not in tgt:
            diags.append(_err(f"targets[{idx}].{req} is required", trace_id))
    if not isinstance(tgt.get("motor_key", None), str) or not tgt.get("motor_key"):
        diags.append(_err(f"targets[{idx}].motor_key must be a non-empty string", trace_id))
    if not isinstance(tgt.get("param_key", None), str) or not tgt.get("param_key"):
        diags.append(_err(f"targets[{idx}].param_key must be a non-empty string", trace_id))
    addr = tgt.get("address")
    if addr is not None:
        if not isinstance(addr, (int, str)) or (isinstance(addr, int) and addr < 0):
            diags.append(_err(
                f"targets[{idx}].address must be a non-negative integer or non-empty string",
                trace_id,
            ))
        if isinstance(addr, str) and not addr:
            diags.append(_err(f"targets[{idx}].address string must be non-empty", trace_id))
    dtype = tgt.get("dtype")
    if dtype not in _VALID_DTYPES:
        diags.append(_err(
            f"targets[{idx}].dtype must be one of {sorted(_VALID_DTYPES)}, got {dtype!r}",
            trace_id,
        ))
    shape = tgt.get("shape")
    if shape is not None:
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(d, int) and d >= 1 for d in shape)
        ):
            diags.append(_err(
                f"targets[{idx}].shape must be a non-empty array of positive integers",
                trace_id,
            ))
    if "source" in tgt:
        diags.extend(_validate_source(tgt["source"], idx, trace_id))
    if "limits" in tgt:
        diags.extend(_validate_limits(tgt.get("limits"), idx, trace_id))
    return diags


def _unknown_field_diag(
    key: str, trace_id: str, strict_mode: bool
) -> BehaviorDiagnostic:
    """Return error (strict) or warning (compat) for an unknown top-level field."""
    msg = f"unknown top-level field '{key}'"
    if strict_mode:
        return BehaviorDiagnostic.error(
            code=ProtocolDiagCode.UNKNOWN_FIELD, message=msg, trace_id=trace_id,
        )
    return BehaviorDiagnostic.warning(
        code=ProtocolDiagCode.UNKNOWN_FIELD, message=msg, trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# Public API — validate_protocol_payload
# ---------------------------------------------------------------------------

def validate_protocol_payload(
    payload: Any,
    now_ms: Optional[int] = None,
    trace_id: str = "",
    strict_mode: bool = True,
) -> Tuple[bool, List[BehaviorDiagnostic]]:
    """Validate a motor weight protocol payload.

    Parameters
    ----------
    payload : The payload dict (or any value) received on the protocol port.
    now_ms  : Current time in milliseconds; defaults to time.time() * 1000.
              Inject for deterministic testing.
    trace_id: Propagated trace ID for diagnostics.

    Returns
    -------
    (is_valid, diagnostics)
      is_valid   — True only if ALL structural checks and staleness check pass.
      diagnostics — List of BehaviorDiagnostic entries; non-empty on any failure.
    """
    diags: List[BehaviorDiagnostic] = []

    # 1 — Must be a dict
    if not isinstance(payload, dict):
        diags.append(_err(
            f"protocol payload must be a JSON object, got {type(payload).__name__}",
            trace_id,
        ))
        return False, diags

    tid = payload.get("trace_id") or trace_id  # prefer payload's trace_id for diagnostics

    # 2 — Required top-level fields
    missing = _REQUIRED_TOP_LEVEL - set(payload.keys())
    if missing:
        for f_name in sorted(missing):
            diags.append(_err(f"required field '{f_name}' is missing", tid))
        return False, diags

    # 2b — Circle 1 Step 1: unknown top-level field gate
    unknown_top = set(payload.keys()) - _ALLOWED_TOP_LEVEL
    if unknown_top:
        for k in sorted(unknown_top):
            diags.append(_unknown_field_diag(k, tid, strict_mode))
        if strict_mode:
            return False, diags
        # compat mode: warnings accumulated, validation continues

    # 3 — protocol_version
    if payload.get("protocol_version") != _PROTOCOL_VERSION:
        diags.append(_err(
            f"unsupported protocol_version {payload.get('protocol_version')!r}; "
            f"expected '{_PROTOCOL_VERSION}'",
            tid,
        ))
        return False, diags

    # 4 — trace_id
    if not isinstance(payload.get("trace_id"), str) or not payload["trace_id"]:
        diags.append(_err("trace_id must be a non-empty string", tid))
        return False, diags

    # 5 — timestamp_ms
    if not isinstance(payload.get("timestamp_ms"), int) or payload["timestamp_ms"] < 0:
        diags.append(_err("timestamp_ms must be a non-negative integer", tid))
        return False, diags

    # 6 — mode
    if payload.get("mode") not in _VALID_MODES:
        diags.append(_err(
            f"mode must be one of {sorted(_VALID_MODES)}, got {payload.get('mode')!r}",
            tid,
        ))
        return False, diags

    # 7 — max_age_ms
    if not isinstance(payload.get("max_age_ms"), int) or payload["max_age_ms"] < 1:
        diags.append(_err("max_age_ms must be a positive integer", tid))
        return False, diags

    # 8 — targets non-empty array
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        diags.append(_err("targets must be a non-empty array", tid))
        return False, diags

    # 9 — per-target structural validation (Circle 1: pass strict_mode)
    for i, tgt in enumerate(targets):
        diags.extend(_validate_target(tgt, i, tid, strict_mode=strict_mode))

    # Block only on error-level diagnostics (warnings from compat mode must not block).
    if any(d.level == "error" for d in diags):
        return False, diags

    # 10 — staleness check (non-fatal schema-wise; emits STALE diagnostic)
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    age_ms = now_ms - payload["timestamp_ms"]
    if age_ms > payload["max_age_ms"]:
        diags.append(BehaviorDiagnostic.warning(
            code=ProtocolDiagCode.STALE,
            message=(
                f"protocol payload is stale: age={age_ms}ms exceeds max_age_ms={payload['max_age_ms']}"
            ),
            trace_id=tid,
        ))
        # Stale is its own distinct outcome — return False so callers can gate on it.
        return False, diags

    # All checks passed
    diags.append(BehaviorDiagnostic.info(
        code=ProtocolDiagCode.VALID,
        message=f"protocol payload valid: {len(targets)} target(s), mode={payload['mode']}",
        trace_id=tid,
    ))
    return True, diags


# ---------------------------------------------------------------------------
# Public API — validate_protocol_structure  (design-time, no staleness check)
# ---------------------------------------------------------------------------

def validate_protocol_structure(payload: Any) -> Tuple[bool, List[str]]:
    """Structural-only validation for use at design / canvas time.

    Checks required top-level fields, protocol_version, non-empty targets list
    and per-target required fields.  Does NOT check timestamps or staleness —
    those are runtime concerns handled by validate_protocol_payload().

    Returns
    -------
    (is_valid, errors)
      is_valid — True when all structural checks pass.
      errors   — Human-readable error strings; empty when is_valid is True.
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        errors.append(f"protocol payload must be a JSON object, got {type(payload).__name__}")
        return False, errors

    missing = _REQUIRED_TOP_LEVEL - set(payload.keys())
    if missing:
        for f_name in sorted(missing):
            errors.append(f"required field '{f_name}' is missing")
        return False, errors

    if payload.get("protocol_version") != _PROTOCOL_VERSION:
        errors.append(
            f"unsupported protocol_version {payload.get('protocol_version')!r}; "
            f"expected '{_PROTOCOL_VERSION}'"
        )
        return False, errors

    if not isinstance(payload.get("trace_id"), str) or not payload["trace_id"]:
        errors.append("trace_id must be a non-empty string")

    if not isinstance(payload.get("timestamp_ms"), int) or payload["timestamp_ms"] < 0:
        errors.append("timestamp_ms must be a non-negative integer")

    if payload.get("mode") not in _VALID_MODES:
        errors.append(
            f"mode must be one of {sorted(_VALID_MODES)}, got {payload.get('mode')!r}"
        )

    if not isinstance(payload.get("max_age_ms"), int) or payload["max_age_ms"] < 1:
        errors.append("max_age_ms must be a positive integer")

    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append("targets must be a non-empty array")
        return len(errors) == 0, errors

    for i, tgt in enumerate(targets):
        diags = _validate_target(tgt, i, "")
        for d in diags:
            errors.append(d.message)

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Public API — parse_protocol_targets
# ---------------------------------------------------------------------------

def parse_protocol_targets(
    payload: Dict[str, Any],
    trace_id: str = "",
    known_addresses: Optional[Any] = None,
) -> Tuple[List[ParsedTarget], List[BehaviorDiagnostic]]:
    """Parse validated protocol payload into resolved ParsedTarget list.

    Only call this on payloads that have already passed validate_protocol_payload().

    For each target:
    - Selects the final value from source (constant or external_value).
    - Applies limits clamping if limits.clamp=True (default True).
    - Emits CLAMPED_VALUE diagnostic when clamping occurs.
    - Emits UNKNOWN_ADDRESS warning when known_addresses is provided and the
      target address is not in that set.

    Parameters
    ----------
    payload         : Validated protocol payload dict.
    trace_id        : Propagated trace ID for diagnostics.
    known_addresses : Optional collection (set/frozenset/list) of addresses
                      the runtime adapter can resolve.  When provided, any
                      target address not found in this collection emits a
                      UNKNOWN_ADDRESS warning-level diagnostic.  When None
                      (default), address resolvability is not checked at parse
                      time and must be validated by the calling adapter.

    Returns
    -------
    (targets, diagnostics)
      targets     — List of ParsedTarget (one per source target entry).
      diagnostics — Warning/info-level diagnostics (clamp, unknown address).
                    Never contains errors; callers should only call this on
                    payloads that passed validate_protocol_payload().
    """
    tid = payload.get("trace_id") or trace_id
    parsed: List[ParsedTarget] = []
    diags:  List[BehaviorDiagnostic] = []

    for tgt in payload.get("targets", []):
        src      = tgt["source"]
        kind     = src["kind"]
        raw_val  = src["constant"] if kind == "constant" else src["external_value"]
        limits   = tgt.get("limits")
        clamped  = False

        # Unknown address check — emitted before value resolution so that
        # callers can detect routing errors without applying any value.
        addr = tgt["address"]
        if known_addresses is not None and addr not in known_addresses:
            diags.append(BehaviorDiagnostic.warning(
                code=ProtocolDiagCode.UNKNOWN_ADDRESS,
                message=(
                    f"motor_key={tgt['motor_key']} param_key={tgt['param_key']}: "
                    f"address {addr!r} not found in known_addresses — target will be skipped"
                ),
                trace_id=tid,
            ))
            # Do not append a ParsedTarget for this entry; it cannot be routed.
            continue

        final_val = _apply_limits(raw_val, limits, tgt, diags, tid)
        if final_val is not raw_val:
            clamped = True

        parsed.append(ParsedTarget(
            motor_key=tgt["motor_key"],
            param_key=tgt["param_key"],
            address=addr,
            dtype=tgt["dtype"],
            shape=tgt["shape"],
            final_value=final_val,
            was_clamped=clamped,
            source_kind=kind,
            note=tgt.get("note", ""),
        ))

    return parsed, diags


def _apply_limits(
    raw_val: Any,
    limits: Optional[Dict[str, Any]],
    tgt: Dict[str, Any],
    diags: List[BehaviorDiagnostic],
    trace_id: str,
) -> Any:
    """Apply limits.clamp to raw_val if applicable; append CLAMPED_VALUE diag."""
    if limits is None:
        return raw_val
    should_clamp = limits.get("clamp", True)  # default clamp=True per schema
    if not should_clamp:
        return raw_val

    lo = limits.get("min")
    hi = limits.get("max")
    if lo is None and hi is None:
        return raw_val

    def _clamp_scalar(v: float) -> Tuple[float, bool]:
        orig = v
        if lo is not None:
            v = max(float(lo), v)
        if hi is not None:
            v = min(float(hi), v)
        return v, (v != orig)

    if isinstance(raw_val, list):
        result = []
        any_clamped = False
        for x in raw_val:
            cx, was = _clamp_scalar(float(x))
            result.append(cx)
            any_clamped = any_clamped or was
        if any_clamped:
            diags.append(BehaviorDiagnostic.info(
                code=ProtocolDiagCode.CLAMPED_VALUE,
                message=(
                    f"motor_key={tgt['motor_key']} param_key={tgt['param_key']}: "
                    f"vector value clamped to [{lo}, {hi}]"
                ),
                trace_id=trace_id,
            ))
            return result
        return raw_val
    else:
        clamped, was = _clamp_scalar(float(raw_val))
        if was:
            diags.append(BehaviorDiagnostic.info(
                code=ProtocolDiagCode.CLAMPED_VALUE,
                message=(
                    f"motor_key={tgt['motor_key']} param_key={tgt['param_key']}: "
                    f"value {raw_val} clamped to {clamped} (limits=[{lo},{hi}])"
                ),
                trace_id=trace_id,
            ))
        return clamped if was else raw_val
