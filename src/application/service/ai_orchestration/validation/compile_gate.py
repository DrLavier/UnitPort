# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""B4 tier-2 Compile Gate — the authoritative ERROR gate (blueprint §4.B4.2).

Reuses the existing verifier (``canvas_to_ir`` → ``compile_training_spec``);
it does NOT reimplement validation (blueprint §7 "reuse, do not reimplement").
The pipeline runs in the app venv with **no engine / Kit / GPU**, so the gate is
cheap to run offline against the assembled canvas snapshot dict.

``compile_training_spec`` already runs ``check_topology`` + the spec checks and
returns every finding (it never raises — the caller decides). The gate surfaces
a structurally-broken canvas (``canvas_to_ir`` raising) as a single ERROR issue
so the repair loop can report it rather than crashing the run (fail-loud as a
visible issue, §8).
"""
from __future__ import annotations

from typing import Any, Dict, List

from unitport_sdk import log_warning

from ..contracts import IssueCode, Severity, ValidationIssue

__all__ = ["compile_gate", "errors_of", "has_errors"]


def compile_gate(canvas_dict: Dict[str, Any]) -> List[ValidationIssue]:
    """Compile the canvas snapshot and return ALL validation issues.

    A malformed canvas dict (``canvas_to_ir`` raising) becomes one GENERIC
    ERROR carrying the message — the repair signal — instead of propagating.
    """
    # Local imports: the training compiler pulls the registry/SDK stack; keep
    # it out of module import so the service package stays lightweight to import.
    from application.compiler.lowering import canvas_to_ir
    from application.training.spec_compiler import compile_training_spec

    try:
        ir = canvas_to_ir(canvas_dict)
    except (ValueError, KeyError, TypeError) as exc:
        log_warning(f"[ai.compile_gate] canvas_to_ir failed: {exc!r}")
        return [
            ValidationIssue(
                code=IssueCode.GENERIC,
                severity=Severity.ERROR,
                message=f"canvas could not be lowered to IR: {exc}",
            )
        ]

    try:
        _spec, issues = compile_training_spec(ir)
    except Exception as exc:
        # A malformed param a populator can't consume (e.g. obs_terms written as
        # a JSON list instead of an object → dict([...]) raises) must surface as a
        # BLOCKING issue the repair loop can act on — NEVER crash the AI Build run
        # (§8: fail loud as a reported issue, not an unhandled task exception).
        log_warning(f"[ai.compile_gate] compile_training_spec raised: {exc!r}")
        return [
            ValidationIssue(
                code=IssueCode.GENERIC,
                severity=Severity.ERROR,
                message=f"canvas could not be compiled: {exc}",
            )
        ]
    return list(issues)


def errors_of(issues: List[ValidationIssue]) -> List[ValidationIssue]:
    """The ERROR-severity subset (the blocking issues)."""
    return [i for i in issues if i.severity == Severity.ERROR]


def has_errors(issues: List[ValidationIssue]) -> bool:
    """True if any issue is ERROR severity (the gate is NOT clean)."""
    return any(i.severity == Severity.ERROR for i in issues)
