#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol apply engine — Circle 1 Step 2.

Provides an explicit apply call point that writes ParsedTarget values to an
adapter's motor channel after invoker execution.  This upgrades the protocol
path from "validation/parsing + context injection only" to a real write path.

Design rules
------------
- No silent fallback: each target that cannot be applied is counted in
  apply_skipped_count and its reason is recorded in apply_failure_reasons.
- When no adapter is supplied (or the adapter lacks apply_motor_param), all
  targets are recorded as context-inject-only successes — the protocol payload
  is still valid and the values remain available via sub_context.
- Adapter interface: adapter.apply_motor_param(target: ParsedTarget) -> bool
    Returns True on success, False when the target was intentionally skipped.
    Raises on hard failures (address not found, out-of-range after secondary
    bounds check, hardware error, etc.).

Public API
----------
ProtocolApplyResult  — dataclass; apply_success_count / apply_skipped_count /
                        apply_failure_reasons / diagnostics
ProtocolApplyEngine  — stateless; .apply(parsed_targets, adapter, trace_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from src.system.behavior.behavior_artifact import BehaviorDiagnostic
from src.system.behavior.motor_weight_protocol import ParsedTarget, ProtocolDiagCode


# ---------------------------------------------------------------------------
# ProtocolApplyResult
# ---------------------------------------------------------------------------

@dataclass
class ProtocolApplyResult:
    """Aggregated outcome from one ProtocolApplyEngine.apply() call.

    Fields
    ------
    apply_success_count   : Targets successfully written (or context-injected).
    apply_skipped_count   : Targets skipped / failed (adapter returned False or raised).
    apply_failure_reasons : Human-readable reason string per skipped/failed target.
    diagnostics           : BehaviorDiagnostic entries (APPLY_SUCCESS/SKIPPED/FAILED).
    """

    apply_success_count: int = 0
    apply_skipped_count: int = 0
    apply_failure_reasons: List[str] = field(default_factory=list)
    diagnostics: List[BehaviorDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "apply_success_count": self.apply_success_count,
            "apply_skipped_count": self.apply_skipped_count,
            "apply_failure_reasons": self.apply_failure_reasons,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# ---------------------------------------------------------------------------
# ProtocolApplyEngine
# ---------------------------------------------------------------------------

class ProtocolApplyEngine:
    """Stateless engine that applies ParsedTarget values to an adapter.

    Instantiate once and call .apply() per execution tick; there is no
    internal state between calls.
    """

    def apply(
        self,
        parsed_targets: List[ParsedTarget],
        adapter: Any = None,
        trace_id: str = "",
    ) -> ProtocolApplyResult:
        """Apply each parsed target to the adapter (or record as context-injected).

        Parameters
        ----------
        parsed_targets : List of ParsedTarget produced by parse_protocol_targets().
                         Must be the already-validated list (unknown-address targets
                         have already been excluded by the parser).
        adapter        : Optional robot adapter.  When it exposes apply_motor_param()
                         each target is written via that method.  When None (or the
                         method is absent), targets are counted as context-inject
                         successes — values remain available in sub_context but no
                         hardware write occurs.
        trace_id       : Propagated trace ID for all emitted diagnostics.

        Returns
        -------
        ProtocolApplyResult
        """
        success_count = 0
        skipped_count = 0
        failure_reasons: List[str] = []
        diags: List[BehaviorDiagnostic] = []

        can_apply = adapter is not None and hasattr(adapter, "apply_motor_param")

        for target in parsed_targets:
            if can_apply:
                try:
                    ok = adapter.apply_motor_param(target)
                    if ok:
                        success_count += 1
                        diags.append(BehaviorDiagnostic.info(
                            code=ProtocolDiagCode.APPLY_SUCCESS,
                            message=(
                                f"motor_key={target.motor_key} "
                                f"param_key={target.param_key}: "
                                f"applied value={target.final_value}"
                            ),
                            trace_id=trace_id,
                        ))
                    else:
                        skipped_count += 1
                        reason = (
                            f"motor_key={target.motor_key} "
                            f"param_key={target.param_key}: adapter returned False"
                        )
                        failure_reasons.append(reason)
                        diags.append(BehaviorDiagnostic.warning(
                            code=ProtocolDiagCode.APPLY_SKIPPED,
                            message=reason,
                            trace_id=trace_id,
                        ))
                except Exception as exc:
                    skipped_count += 1
                    reason = (
                        f"motor_key={target.motor_key} "
                        f"param_key={target.param_key}: {exc}"
                    )
                    failure_reasons.append(reason)
                    diags.append(BehaviorDiagnostic.warning(
                        code=ProtocolDiagCode.APPLY_FAILED,
                        message=reason,
                        trace_id=trace_id,
                    ))
            else:
                # No adapter / no apply_motor_param: context-inject-only success.
                success_count += 1
                diags.append(BehaviorDiagnostic.info(
                    code=ProtocolDiagCode.APPLY_SUCCESS,
                    message=(
                        f"motor_key={target.motor_key} "
                        f"param_key={target.param_key}: "
                        f"context-injected (no adapter apply), value={target.final_value}"
                    ),
                    trace_id=trace_id,
                ))

        return ProtocolApplyResult(
            apply_success_count=success_count,
            apply_skipped_count=skipped_count,
            apply_failure_reasons=failure_reasons,
            diagnostics=diags,
        )
