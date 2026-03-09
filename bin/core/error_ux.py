"""Operator-facing error message mapping and diagnostic formatting for Mission UI.

Separates friendly UI text from raw technical diagnostic payloads.
Lives in bin/core/ to keep ui.py thin (§C1-4 guardrail 4).

Usage::

    from bin.core.error_ux import (
        get_operator_text,
        get_error_category,
        classify_run_result,
        format_node_diagnostics,
        format_settings_validation_error,
        extract_failed_nodes_info,
    )

Localization contract
---------------------
All string values returned by this module are plain Python strings with
no tr() / localisation calls embedded.  Localisation of *labels* (keys)
is the responsibility of the display layer (DiagnosticsPanel, ui.py).
This keeps the raw-diagnostics pathway fully engineering-readable in any
locale: raw JSON is always the original technical payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Stage category labels ──────────────────────────────────────────────────
# Maps a lifecycle stage string → user-visible UI category label.

STAGE_CATEGORY: Dict[str, str] = {
    "acquire":       "Setup",
    "open_session":  "Connection",
    "preflight":     "Preflight / Safety",
    "execute":       "Execution",
    "close_session": "Cleanup",
    "any":           "General",
}


# ── Reason code → operator text ────────────────────────────────────────────
# Canonical codes first, then per-adapter alphabetically within each adapter.
# Stable identifiers — do not rename once published.
# Values are plain strings; do NOT wrap with tr() here (localization contract).

REASON_OPERATOR_TEXT: Dict[str, str] = {
    # Settings validation (Cycle 2 STAGE-05)
    "settings_validation_failed": (
        "SDK settings are incomplete or invalid. "
        "Fill in the required fields before running."
    ),
    # Canonical
    "adapter_not_found":      "Robot adapter not found. Check connection settings.",
    "capability_unavailable": "Required capability is not available on this robot.",
    "execute_failed":         "Action execution failed.",
    "preflight_failed":       "Pre-flight check failed. Robot may not be ready.",
    "retry_budget_exhausted": "Maximum retries exceeded. Action could not complete.",
    "safety_gate_blocked":    "Blocked by safety gate. Check safety conditions.",
    "session_close_failed":   "Session close failed. Check robot connection.",
    "session_open_failed":    "Could not open robot session. Check connection.",
    # Unitree
    "unitree_action_in_progress": "Another Unitree action is already running.",
    "unitree_control_conflict":   "Control conflict: SDK and simulation both active.",
    "unitree_model_unavailable":  "Unitree robot model is not initialised.",
    # Spot
    "spot_action_failed":    "Spot action failed.",
    "spot_auth_failed":      "Spot authentication failed. Check credentials.",
    "spot_command_timeout":  "Spot command timed out. Check robot responsiveness.",
    "spot_estop_active":     "Spot E-stop is engaged. Release before proceeding.",
    "spot_lease_failed":     "Could not acquire Spot lease.",
    "spot_no_session":       "Spot session not open. Re-connect required.",
    "spot_sdk_unavailable":  "Spot SDK not installed.",
    "spot_time_sync_failed": "Spot time synchronisation failed.",
    # CyberDog
    "cyberdog_action_failed":            "CyberDog action failed.",
    "cyberdog_endpoint_unavailable":     "CyberDog motion topics not found.",
    "cyberdog_mode_precondition_failed": "CyberDog is in an uncontrollable mode.",
    "cyberdog_namespace_mismatch":       "CyberDog namespace does not match configuration.",
    "cyberdog_node_init_failed":         "CyberDog ROS2 node initialisation failed.",
    "cyberdog_no_session":               "CyberDog session not open. Re-connect required.",
    "cyberdog_ros2_unavailable":         "ROS2 not installed or not sourced.",
    "cyberdog_stale_timestamp":          "CyberDog clock freshness check failed.",
}


# ── Error category system (STAGE-06) ──────────────────────────────────────
# Four mutually-exclusive categories that drive the panel's visual distinction
# and the summary bar's category label.
#
# validation      — adapter/SDK setup issues; the system cannot even start
# preflight_safety — safety gates and session-level blocking before execution
# runtime         — action-time failures during execution
# compat_warning  — informational; a compatibility path was activated
#
# These are plain constants — do NOT translate them; they are keys, not labels.

ERROR_CATEGORY_VALIDATION:  str = "validation"
ERROR_CATEGORY_PREFLIGHT:   str = "preflight_safety"
ERROR_CATEGORY_RUNTIME:     str = "runtime"
ERROR_CATEGORY_COMPAT:      str = "compat_warning"

# Maps every canonical reason code to exactly one category.
# Every code present in REASON_OPERATOR_TEXT must have an entry here.
REASON_CATEGORY: Dict[str, str] = {
    # ── validation ────────────────────────────────────────────────────────
    # Adapter/SDK not found or not initialised; cannot proceed at all.
    # Cycle 2 STAGE-05: settings pre-run validation failure.
    "settings_validation_failed": ERROR_CATEGORY_VALIDATION,
    "adapter_not_found":      ERROR_CATEGORY_VALIDATION,
    "capability_unavailable": ERROR_CATEGORY_VALIDATION,
    "spot_sdk_unavailable":   ERROR_CATEGORY_VALIDATION,
    "cyberdog_ros2_unavailable": ERROR_CATEGORY_VALIDATION,
    "unitree_model_unavailable": ERROR_CATEGORY_VALIDATION,

    # ── preflight_safety ─────────────────────────────────────────────────
    # Safety gate or session-level blocking before/during open_session/preflight.
    "preflight_failed":       ERROR_CATEGORY_PREFLIGHT,
    "safety_gate_blocked":    ERROR_CATEGORY_PREFLIGHT,
    "session_open_failed":    ERROR_CATEGORY_PREFLIGHT,
    # Spot session / safety
    "spot_auth_failed":       ERROR_CATEGORY_PREFLIGHT,
    "spot_estop_active":      ERROR_CATEGORY_PREFLIGHT,
    "spot_lease_failed":      ERROR_CATEGORY_PREFLIGHT,
    "spot_no_session":        ERROR_CATEGORY_PREFLIGHT,
    "spot_time_sync_failed":  ERROR_CATEGORY_PREFLIGHT,
    # CyberDog session / health checks
    "cyberdog_no_session":               ERROR_CATEGORY_PREFLIGHT,
    "cyberdog_node_init_failed":         ERROR_CATEGORY_PREFLIGHT,
    "cyberdog_namespace_mismatch":       ERROR_CATEGORY_PREFLIGHT,
    "cyberdog_stale_timestamp":          ERROR_CATEGORY_PREFLIGHT,

    # ── runtime ───────────────────────────────────────────────────────────
    # Action-time execution failures.
    "execute_failed":         ERROR_CATEGORY_RUNTIME,
    "retry_budget_exhausted": ERROR_CATEGORY_RUNTIME,
    "session_close_failed":   ERROR_CATEGORY_RUNTIME,
    # Unitree runtime
    "unitree_action_in_progress": ERROR_CATEGORY_RUNTIME,
    "unitree_control_conflict":   ERROR_CATEGORY_RUNTIME,
    # Spot runtime
    "spot_action_failed":    ERROR_CATEGORY_RUNTIME,
    "spot_command_timeout":  ERROR_CATEGORY_RUNTIME,
    # CyberDog runtime
    "cyberdog_action_failed":            ERROR_CATEGORY_RUNTIME,
    "cyberdog_endpoint_unavailable":     ERROR_CATEGORY_RUNTIME,
    "cyberdog_mode_precondition_failed": ERROR_CATEGORY_RUNTIME,
}

# Maps ExecutionPath.BLOCKED_* constants → error category.
# Used by classify_run_result() for blocked results.
BLOCKED_PATH_CATEGORY: Dict[str, str] = {
    "blocked_compile": ERROR_CATEGORY_VALIDATION,   # IR/schema compile error
    "blocked_execute": ERROR_CATEGORY_PREFLIGHT,    # execution-guard check
    "blocked_safety":  ERROR_CATEGORY_PREFLIGHT,    # safety gate
}


# ── Deterministic display key order ───────────────────────────────────────
# Keys listed here appear first in this exact order when rendering diagnostics.
# Keys not in this list appear afterwards, sorted alphabetically.
#
# error_category immediately follows stage so the operator sees both the
# lifecycle position and the severity class together.
# Telemetry pointer keys appear last so they are visible but non-intrusive.

DISPLAY_KEY_ORDER: List[str] = [
    "node_id",
    "node_name",
    "status",
    "stage",
    "error_category",   # STAGE-06: explicit error category
    "reason",
    "operator_text",
    "message",
    "adapter_name",
    "retryable",
    "context",
    # ── Telemetry pointers ────────────────────────────────────────────────
    "trace_id",
    "mission_trace_id",
]


# ── Public functions ───────────────────────────────────────────────────────

def get_operator_text(reason_code: str) -> str:
    """Return user-friendly text for *reason_code*, or a generic fallback.

    Returns a plain Python string — no tr() wrapping here (localization
    contract: display layer is responsible for locale-specific rendering).
    """
    return REASON_OPERATOR_TEXT.get(reason_code, f"An error occurred ({reason_code}).")


def get_stage_category(stage: str) -> str:
    """Return the UI category label for a lifecycle *stage* string."""
    return STAGE_CATEGORY.get(stage, stage.replace("_", " ").title())


def get_error_category(reason_code: str) -> str:
    """Return the error category constant for *reason_code*.

    Returns one of ERROR_CATEGORY_VALIDATION, ERROR_CATEGORY_PREFLIGHT,
    ERROR_CATEGORY_RUNTIME, or ERROR_CATEGORY_COMPAT.

    Unknown codes default to ``ERROR_CATEGORY_RUNTIME`` (safest assumption
    for unexpected adapter errors).
    """
    return REASON_CATEGORY.get(reason_code, ERROR_CATEGORY_RUNTIME)


def classify_run_result(run_result: Dict[str, Any]) -> str:
    """Return the primary error category for a complete *run_result* dict.

    Decision tree
    -------------
    1. status == "blocked"  → map diagnostics.path via BLOCKED_PATH_CATEGORY
    2. diagnostics.compat_path is True  → ERROR_CATEGORY_COMPAT
    3. reason code present  → get_error_category(reason)
    4. failed_nodes present → ERROR_CATEGORY_RUNTIME (generic execution failure)
    5. fallback             → ERROR_CATEGORY_RUNTIME

    Returns one of ERROR_CATEGORY_* constants.
    """
    status = run_result.get("status", "")

    if status == "blocked":
        path = run_result.get("diagnostics", {}).get("path", "")
        return BLOCKED_PATH_CATEGORY.get(path, ERROR_CATEGORY_VALIDATION)

    diag = run_result.get("diagnostics", {})
    if diag.get("compat_path"):
        return ERROR_CATEGORY_COMPAT

    reason = run_result.get("reason", "")
    if reason:
        return get_error_category(reason)

    if diag.get("failed_nodes"):
        return ERROR_CATEGORY_RUNTIME

    return ERROR_CATEGORY_RUNTIME


def format_node_diagnostics(
    node_id: Any,
    node_result: Dict[str, Any],
    run_result: Optional[Dict[str, Any]] = None,
    node_name: str = "",
) -> Dict[str, Any]:
    """Build a normalised, deterministically-ordered display dict for a node result.

    Args:
        node_id:     The node identifier (int or str).
        node_result: The per-node dict from run_result["results"][node_id].
        run_result:  Full run_result dict for cross-referencing diagnostics.
        node_name:   Optional display name.

    Returns a dict with keys in DISPLAY_KEY_ORDER order; remaining keys follow
    alphabetically.  Suitable for direct rendering in the DiagnosticsPanel.

    Localization contract: all values are raw Python strings — the display
    layer (DiagnosticsPanel) is responsible for any locale-specific labels.
    """
    reason       = node_result.get("reason", node_result.get("error", ""))
    stage        = node_result.get("stage", "execute")
    status       = node_result.get("status", "failed" if "error" in node_result else "success")
    message      = node_result.get("message", node_result.get("error", ""))
    context      = node_result.get("context", node_result.get("diagnostics", {}))
    adapter_name = node_result.get("adapter_name", "")

    # Retryability from REASON_MAP (best-effort; import may fail in test context)
    retryable = False
    try:
        from system.service.reason_codes import REASON_MAP  # noqa: PLC0415
        retryable = REASON_MAP.get(reason, {}).get("retryable", False)
    except ImportError:
        pass

    raw: Dict[str, Any] = {
        "node_id":        str(node_id),
        "node_name":      node_name or str(node_id),
        "status":         status,
        "stage":          stage,
        "error_category": get_error_category(reason),   # STAGE-06
        "reason":         reason,
        "operator_text":  get_operator_text(reason),
        "message":        message,
        "adapter_name":   adapter_name,
        "retryable":      retryable,
        "context":        context,
    }

    # ── Telemetry pointers ────────────────────────────────────────────────
    # trace_id: node_result carries it when the node is a behavior node
    # (set by BehaviorSubgraphInvoker).  Fall back to the per-execution
    # behavior_trace_ids map in run_result.diagnostics for cross-referencing.
    trace_id: str = node_result.get("trace_id", "")
    mission_trace_id: str = ""

    if run_result:
        diag = run_result.get("diagnostics", {})

        if not trace_id:
            btids = diag.get("behavior_trace_ids", {})
            trace_id = btids.get(str(node_id), btids.get(node_id, ""))

        mission_trace_id = str(diag.get("mission_trace_id", "") or "")

        # Circle 5: carry package metadata trace for advanced/package-expanded
        # diagnostics visibility in UI drill-down.
        pkg_trace = diag.get("package_metadata_trace")
        if isinstance(pkg_trace, dict):
            raw["package_metadata_trace"] = {
                "package_id": str(pkg_trace.get("package_id", "") or ""),
                "package_version": str(pkg_trace.get("package_version", "") or ""),
                "schema_version": str(pkg_trace.get("schema_version", "") or ""),
            }

        # Compat path info
        if diag.get("compat_path"):
            raw["compat_path"]   = True
            raw["compat_reason"] = diag.get("compat_reason", "")

    # Only include telemetry pointers when they carry actual data
    if trace_id:
        raw["trace_id"] = trace_id
    if mission_trace_id:
        raw["mission_trace_id"] = mission_trace_id

    # Build ordered dict: DISPLAY_KEY_ORDER first, then remaining keys alphabetically
    ordered: Dict[str, Any] = {}
    for k in DISPLAY_KEY_ORDER:
        if k in raw:
            ordered[k] = raw[k]
    for k in sorted(raw.keys()):
        if k not in ordered:
            ordered[k] = raw[k]

    return ordered


def format_settings_validation_error(
    brand: str,
    validation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a normalised diagnostic dict for a settings pre-run validation failure.

    Returns a dict shaped identically to those produced by
    :func:`format_node_diagnostics` so it can be passed directly to
    ``DiagnosticsPanel.show_diagnostics()`` without adaptation.

    The raw *validation_result* is preserved in ``context`` for engineering
    inspection (localization contract: no tr() calls here).

    Args:
        brand:             Current robot brand key (e.g. ``"unitree"``).
        validation_result: The dict returned by ``validate_settings()``.
                           Expected keys: ``status``, ``missing``, ``invalid``.

    Returns:
        Ordered diagnostic info dict with all standard display keys.
    """
    missing: list = validation_result.get("missing", [])
    invalid: list = validation_result.get("invalid", [])

    parts: list = []
    if missing:
        parts.append(f"Missing: {', '.join(missing)}")
    if invalid:
        parts.append(f"Invalid: {', '.join(invalid)}")
    message: str = " | ".join(parts) or "Settings validation failed"

    raw: Dict[str, Any] = {
        "node_id":        "settings",
        "node_name":      f"SDK Settings ({brand})",
        "status":         "failed",
        "stage":          "preflight",
        "error_category": ERROR_CATEGORY_VALIDATION,
        "reason":         "settings_validation_failed",
        "operator_text":  get_operator_text("settings_validation_failed"),
        "message":        message,
        "adapter_name":   brand,
        "retryable":      False,
        "context": {
            "missing_fields":    missing,
            "invalid_fields":    invalid,
            "validation_result": validation_result,
        },
    }

    # Build ordered dict: DISPLAY_KEY_ORDER first, then remaining keys alphabetically
    ordered: Dict[str, Any] = {}
    for k in DISPLAY_KEY_ORDER:
        if k in raw:
            ordered[k] = raw[k]
    for k in sorted(raw.keys()):
        if k not in ordered:
            ordered[k] = raw[k]
    return ordered


def extract_failed_nodes_info(
    run_result: Dict[str, Any],
    node_names: Optional[Dict[Any, str]] = None,
) -> List[Dict[str, Any]]:
    """Return a list of formatted diagnostic dicts for all failed nodes.

    Args:
        run_result:  The full run_result dict from RuntimeEngine.execute().
        node_names:  Optional mapping of node_id → display name.

    Nodes appear in the order they are listed in diagnostics.failed_nodes.
    """
    results = run_result.get("results", {})
    diag    = run_result.get("diagnostics", {})
    failed  = diag.get("failed_nodes", [])
    names   = node_names or {}

    infos = []
    for nid in failed:
        nid_str = str(nid)
        node_result = results.get(nid_str, results.get(nid, {"error": "unknown", "status": "failed"}))
        name = names.get(nid, names.get(nid_str, ""))
        infos.append(format_node_diagnostics(nid, node_result, run_result, node_name=name))
    return infos
