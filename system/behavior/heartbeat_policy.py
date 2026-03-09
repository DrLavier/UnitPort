#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heartbeat policy schema — Circle 2 Step 2.2.

Defines the validated configuration that controls how the heartbeat loop
runs during behavior execution:

  HeartbeatFailPolicy  — valid fail_policy string constants
  HeartbeatSafetyMode  — valid safety_mode string constants
  HeartbeatPolicy      — full policy dataclass with validation

Design constraints
------------------
- No SDK calls; pure data + validation.
- validate() returns BehaviorDiagnostic list so callers can surface
  detailed reason codes without parsing strings.
- default() always produces a policy that passes validate() with zero errors.
- from_dict() is lenient (uses defaults for missing keys) to support
  partial policy overrides in mission files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .behavior_artifact import BehaviorDiagnostic


# ---------------------------------------------------------------------------
# Constant pools — string values accepted for enum-like fields
# ---------------------------------------------------------------------------

class HeartbeatFailPolicy:
    """Valid values for HeartbeatPolicy.fail_policy.

    STOP_BEHAVIOR    — terminate the behavior on timeout/error (default; safe).
    LOG_AND_CONTINUE — log a warning but allow the behavior loop to continue.
    ABORT_MISSION    — escalate to a full mission abort.
    """
    STOP_BEHAVIOR    = "stop_behavior"
    LOG_AND_CONTINUE = "log_and_continue"
    ABORT_MISSION    = "abort_mission"

    @classmethod
    def valid_values(cls) -> frozenset:
        return frozenset({cls.STOP_BEHAVIOR, cls.LOG_AND_CONTINUE, cls.ABORT_MISSION})


class HeartbeatSafetyMode:
    """Valid values for HeartbeatPolicy.safety_mode.

    STRICT  — block execution if the policy fails validation (default).
    RELAXED — emit a warning but allow execution with the invalid policy.
    """
    STRICT  = "strict"
    RELAXED = "relaxed"

    @classmethod
    def valid_values(cls) -> frozenset:
        return frozenset({cls.STRICT, cls.RELAXED})


# ---------------------------------------------------------------------------
# Validated range constants
# ---------------------------------------------------------------------------

TICK_MS_MIN     = 10        # 10 ms minimum tick interval
TICK_MS_MAX     = 10_000    # 10 s maximum tick interval
TIMEOUT_MS_MIN  = 100       # 100 ms minimum timeout
TIMEOUT_MS_MAX  = 300_000   # 5 min maximum timeout
MAX_OVERRIDE_MIN = 0.0      # 0 % (no override)
MAX_OVERRIDE_MAX = 1.0      # 100 % override


# ---------------------------------------------------------------------------
# HeartbeatPolicy
# ---------------------------------------------------------------------------

@dataclass
class HeartbeatPolicy:
    """Validated heartbeat loop configuration.

    Fields
    ------
    tick_ms      : Milliseconds between heartbeat ticks.
                   Range: [10, 10 000].  Default: 100 ms (10 Hz).
    timeout_ms   : Milliseconds after which the heartbeat loop is considered
                   timed-out if the behavior has not completed.
                   Range: [100, 300 000].  Default: 5 000 ms.
    fail_policy  : What to do on timeout or heartbeat error.
                   One of HeartbeatFailPolicy.*  Default: "stop_behavior".
    max_override : Maximum fractional movement override the heartbeat can
                   apply to the base robot command (0.0 = disabled, 1.0 = full).
                   Range: [0.0, 1.0].  Default: 0.2 (20 %).
    safety_mode  : Whether an invalid policy blocks execution ("strict") or
                   only emits a warning ("relaxed").
                   One of HeartbeatSafetyMode.*  Default: "strict".
    """

    tick_ms:      int   = 100
    timeout_ms:   int   = 5_000
    fail_policy:  str   = HeartbeatFailPolicy.STOP_BEHAVIOR
    max_override: float = 0.2
    safety_mode:  str   = HeartbeatSafetyMode.STRICT

    # ------------------------------------------------------------------
    # Factory / serialization
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "HeartbeatPolicy":
        """Return the canonical default policy (always passes validate())."""
        return cls()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeartbeatPolicy":
        """Construct from a dict; missing keys fall back to field defaults.

        Never raises — lenient parsing so partial policy overrides in mission
        files work without specifying every field.
        """
        try:
            tick_ms = int(d.get("tick_ms", 100))
        except (TypeError, ValueError):
            tick_ms = 100

        try:
            timeout_ms = int(d.get("timeout_ms", 5_000))
        except (TypeError, ValueError):
            timeout_ms = 5_000

        fail_policy = str(d.get("fail_policy", HeartbeatFailPolicy.STOP_BEHAVIOR))

        try:
            max_override = float(d.get("max_override", 0.2))
        except (TypeError, ValueError):
            max_override = 0.2

        safety_mode = str(d.get("safety_mode", HeartbeatSafetyMode.STRICT))

        return cls(
            tick_ms=tick_ms,
            timeout_ms=timeout_ms,
            fail_policy=fail_policy,
            max_override=max_override,
            safety_mode=safety_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain JSON-safe dict."""
        return {
            "tick_ms":      self.tick_ms,
            "timeout_ms":   self.timeout_ms,
            "fail_policy":  self.fail_policy,
            "max_override": self.max_override,
            "safety_mode":  self.safety_mode,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, trace_id: str = "") -> List[BehaviorDiagnostic]:
        """Return a list of BehaviorDiagnostic errors for invalid fields.

        An empty list means the policy is valid.  Does not raise.

        Each error carries a machine-readable code of the form
        ``heartbeat_policy.<field>_<reason>`` so callers can match
        specific violations without parsing message strings.
        """
        diags: List[BehaviorDiagnostic] = []

        if not (TICK_MS_MIN <= self.tick_ms <= TICK_MS_MAX):
            diags.append(BehaviorDiagnostic.error(
                code="heartbeat_policy.tick_ms_out_of_range",
                message=(
                    f"tick_ms={self.tick_ms} is out of range "
                    f"[{TICK_MS_MIN}, {TICK_MS_MAX}]"
                ),
                trace_id=trace_id,
            ))

        if not (TIMEOUT_MS_MIN <= self.timeout_ms <= TIMEOUT_MS_MAX):
            diags.append(BehaviorDiagnostic.error(
                code="heartbeat_policy.timeout_ms_out_of_range",
                message=(
                    f"timeout_ms={self.timeout_ms} is out of range "
                    f"[{TIMEOUT_MS_MIN}, {TIMEOUT_MS_MAX}]"
                ),
                trace_id=trace_id,
            ))

        if self.fail_policy not in HeartbeatFailPolicy.valid_values():
            diags.append(BehaviorDiagnostic.error(
                code="heartbeat_policy.invalid_fail_policy",
                message=(
                    f"fail_policy='{self.fail_policy}' is not one of "
                    f"{sorted(HeartbeatFailPolicy.valid_values())}"
                ),
                trace_id=trace_id,
            ))

        if not (MAX_OVERRIDE_MIN <= self.max_override <= MAX_OVERRIDE_MAX):
            diags.append(BehaviorDiagnostic.error(
                code="heartbeat_policy.max_override_out_of_range",
                message=(
                    f"max_override={self.max_override} is out of range "
                    f"[{MAX_OVERRIDE_MIN:.1f}, {MAX_OVERRIDE_MAX:.1f}]"
                ),
                trace_id=trace_id,
            ))

        if self.safety_mode not in HeartbeatSafetyMode.valid_values():
            diags.append(BehaviorDiagnostic.error(
                code="heartbeat_policy.invalid_safety_mode",
                message=(
                    f"safety_mode='{self.safety_mode}' is not one of "
                    f"{sorted(HeartbeatSafetyMode.valid_values())}"
                ),
                trace_id=trace_id,
            ))

        return diags

    @property
    def is_valid(self) -> bool:
        """True iff validate() returns no errors."""
        return len(self.validate()) == 0


# ---------------------------------------------------------------------------
# Standalone validator function
# ---------------------------------------------------------------------------

def validate_heartbeat_policy(
    policy_dict: Dict[str, Any],
    trace_id: str = "",
) -> Tuple[bool, List[BehaviorDiagnostic]]:
    """Parse and validate a policy dict in one call.

    Returns
    -------
    (is_valid, diagnostics)
        is_valid    : True iff no error-level diagnostics were emitted.
        diagnostics : List[BehaviorDiagnostic]; empty on success.

    Never raises — all parsing errors are captured as diagnostics.
    """
    policy = HeartbeatPolicy.from_dict(policy_dict)
    diags = policy.validate(trace_id=trace_id)
    return len(diags) == 0, diags
