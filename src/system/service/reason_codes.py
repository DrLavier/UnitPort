"""Cross-brand lifecycle reason code registry (Phase 4 STAGE-06 taxonomy;
Phase 6 STAGE-04 hardening: +2 canonical codes, retryable classification).

Two tiers of reason codes are used across the UnitPort service layer:

  Canonical   — LifecycleReason class constants shared by all adapters and the
                ServiceRouter.  Stable identifiers for lifecycle-stage failures.

  Adapter     — Brand-specific reason strings prefixed by adapter name.
                Surfaced inside LifecycleResult.diagnostics["reason"] or as
                the top-level "reason" key after wrapping by the router.

All reason codes are stable strings — do not rename once published.

Usage::

    from src.system.service.reason_codes import ALL_REASON_CODES, REASON_MAP
    assert "spot_sdk_unavailable" in ALL_REASON_CODES

    from src.system.service.lifecycle import LifecycleReason
    # Use constants in code; strings are for log/result inspection only.
    if result["reason"] == LifecycleReason.EXECUTE_FAILED: ...
"""

from .lifecycle import LifecycleReason

# ── Canonical tier ──────────────────────────────────────────────────────────
# LifecycleReason class constants used by all adapters and the router.
# Phase 4: 6 codes.  Phase 6 STAGE-04: +SAFETY_GATE_BLOCKED, +RETRY_BUDGET_EXHAUSTED → 8.

CANONICAL_REASONS: frozenset = frozenset({
    LifecycleReason.SESSION_OPEN_FAILED,
    LifecycleReason.PREFLIGHT_FAILED,
    LifecycleReason.SESSION_CLOSE_FAILED,
    LifecycleReason.CAPABILITY_UNAVAILABLE,
    LifecycleReason.ADAPTER_NOT_FOUND,
    LifecycleReason.EXECUTE_FAILED,
    LifecycleReason.SAFETY_GATE_BLOCKED,       # Phase 6
    LifecycleReason.RETRY_BUDGET_EXHAUSTED,    # Phase 6
})

# ── Unitree adapter tier (adapter_name = "unitree_sdk2") ────────────────────

UNITREE_REASONS: frozenset = frozenset({
    "unitree_model_unavailable",    # open_session / preflight: model not init'd
    "unitree_action_in_progress",   # preflight: simulation action already running
    "unitree_control_conflict",     # preflight: SDK high-level + MuJoCo both active
})

# ── Spot adapter tier (adapter_name = "spot_sdk") ───────────────────────────

SPOT_REASONS: frozenset = frozenset({
    "spot_sdk_unavailable",    # open_session: bosdyn-client not installed
    "spot_auth_failed",        # open_session: authentication / robot init failed
    "spot_time_sync_failed",   # open_session: time sync wait failed
    "spot_lease_failed",       # open_session: lease acquisition failed
    "spot_estop_active",       # preflight: E-stop is engaged
    "spot_no_session",         # preflight: open_session not called
    "spot_action_failed",      # execute: topic publish / command error
    "spot_command_timeout",    # execute: command did not complete in time (retryable)
})

# ── CyberDog adapter tier (adapter_name = "cyberdog_sdk") ──────────────────

CYBERDOG_REASONS: frozenset = frozenset({
    "cyberdog_ros2_unavailable",          # open_session: rclpy not installed / ROS2 not sourced
    "cyberdog_node_init_failed",          # open_session: rclpy.init or create_node failed
    "cyberdog_endpoint_unavailable",      # open_session: motion topics not found (reserved)
    "cyberdog_mode_precondition_failed",  # open_session: robot in uncontrollable mode (reserved)
    "cyberdog_no_session",                # preflight: open_session not called
    "cyberdog_stale_timestamp",           # preflight: clock freshness check failed (reserved)
    "cyberdog_namespace_mismatch",        # preflight: node namespace ≠ configured namespace
    "cyberdog_action_failed",             # execute: topic publish error (retryable)
})

# ── Complete cross-brand taxonomy ───────────────────────────────────────────
# Phase 4: 6+3+8+8 = 25.  Phase 6 STAGE-04: 8+3+8+8 = 27.

ALL_REASON_CODES: frozenset = (
    CANONICAL_REASONS | UNITREE_REASONS | SPOT_REASONS | CYBERDOG_REASONS
)

# ── Structured REASON_MAP: reason_code → metadata ───────────────────────────
# Provides structured lookup for tooling, dashboards, and test assertions.
#
# Keys:
#   adapter   — "all" = every adapter, "router" = ServiceRouter only
#   stage     — lifecycle stage that produces this code
#   tier      — "canonical" | "adapter"
#   retryable — True iff a subsequent attempt may succeed (Phase 6 STAGE-04)
#
# Retryable classification:
#   True  — transient execution errors: execute_failed, spot_command_timeout,
#            cyberdog_action_failed.
#   False — structural / precondition / budget-exhausted / session-management
#            failures where retrying the same call cannot succeed.

REASON_MAP: dict = {
    # ── Canonical ─────────────────────────────────────────────────────────
    "session_open_failed":    {"adapter": "all",    "stage": "open_session",          "tier": "canonical", "retryable": False},
    "preflight_failed":       {"adapter": "all",    "stage": "preflight",             "tier": "canonical", "retryable": False},
    "session_close_failed":   {"adapter": "all",    "stage": "close_session",         "tier": "canonical", "retryable": False},
    "capability_unavailable": {"adapter": "all",    "stage": "any",                   "tier": "canonical", "retryable": False},
    "adapter_not_found":      {"adapter": "router", "stage": "acquire",               "tier": "canonical", "retryable": False},
    "execute_failed":         {"adapter": "router", "stage": "execute",               "tier": "canonical", "retryable": True},
    "safety_gate_blocked":    {"adapter": "router", "stage": "preflight",             "tier": "canonical", "retryable": False},
    "retry_budget_exhausted": {"adapter": "router", "stage": "execute",               "tier": "canonical", "retryable": False},
    # ── Unitree ───────────────────────────────────────────────────────────
    "unitree_model_unavailable":  {"adapter": "unitree_sdk2", "stage": "open_session/preflight", "tier": "adapter", "retryable": False},
    "unitree_action_in_progress": {"adapter": "unitree_sdk2", "stage": "preflight",             "tier": "adapter", "retryable": False},
    "unitree_control_conflict":   {"adapter": "unitree_sdk2", "stage": "preflight",             "tier": "adapter", "retryable": False},
    # ── Spot ──────────────────────────────────────────────────────────────
    "spot_sdk_unavailable":  {"adapter": "spot_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "spot_auth_failed":      {"adapter": "spot_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "spot_time_sync_failed": {"adapter": "spot_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "spot_lease_failed":     {"adapter": "spot_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "spot_estop_active":     {"adapter": "spot_sdk", "stage": "preflight",    "tier": "adapter", "retryable": False},
    "spot_no_session":       {"adapter": "spot_sdk", "stage": "preflight",    "tier": "adapter", "retryable": False},
    "spot_action_failed":    {"adapter": "spot_sdk", "stage": "execute",      "tier": "adapter", "retryable": False},
    "spot_command_timeout":  {"adapter": "spot_sdk", "stage": "execute",      "tier": "adapter", "retryable": True},
    # ── CyberDog ──────────────────────────────────────────────────────────
    "cyberdog_ros2_unavailable":         {"adapter": "cyberdog_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "cyberdog_node_init_failed":         {"adapter": "cyberdog_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "cyberdog_endpoint_unavailable":     {"adapter": "cyberdog_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "cyberdog_mode_precondition_failed": {"adapter": "cyberdog_sdk", "stage": "open_session", "tier": "adapter", "retryable": False},
    "cyberdog_no_session":               {"adapter": "cyberdog_sdk", "stage": "preflight",    "tier": "adapter", "retryable": False},
    "cyberdog_stale_timestamp":          {"adapter": "cyberdog_sdk", "stage": "preflight",    "tier": "adapter", "retryable": False},
    "cyberdog_namespace_mismatch":       {"adapter": "cyberdog_sdk", "stage": "preflight",    "tier": "adapter", "retryable": False},
    "cyberdog_action_failed":            {"adapter": "cyberdog_sdk", "stage": "execute",      "tier": "adapter", "retryable": True},
}

# ── Diagnostics key schema (by adapter × stage) ─────────────────────────────
# Documents the standard keys each adapter injects into diagnostics dicts on
# lifecycle failures.  Used for contract validation in tests and tooling.

DIAGNOSTIC_KEY_SCHEMA: dict = {
    "unitree_sdk2": {
        "open_session_fail":  ["reason", "message", "robot_type", "sdk_available", "mujoco_available"],
        "preflight_fail":     ["reason", "message", "robot_type", "sdk_client_active", "mujoco_available"],
        "close_session_fail": ["message", "robot_type"],
    },
    "spot_sdk": {
        "open_session_fail":  ["reason", "message", "hostname", "sdk_available"],
        "preflight_fail":     ["reason", "message", "session_open", "lease_held"],
        "close_session_fail": ["errors"],
    },
    "cyberdog_sdk": {
        "open_session_fail":  ["reason", "message", "namespace", "ros2_available"],
        "preflight_fail":     ["reason", "message", "session_open", "node_alive"],
        "close_session_fail": ["errors"],
    },
}
