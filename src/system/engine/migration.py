#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime execution-strategy compat gate — Phase 7 policy.

Controls whether the NodeExecutor uses the default flow-aware DFS execution
(STEP-04) or the retained topological-sort compatibility path.

Phase 7 default-path policy
----------------------------
The flow-aware DFS execution (``use_flow_aware_execution=True``) is the
**stable default** for all WorkflowIR execution.  It has been the default
since Phase 1 STEP-04 and remains so throughout Phase 7 and beyond.

The topological-sort path is retained as an **explicit compatibility gate**
for field rollback scenarios only.  It must never become the default again.
When it is activated, RuntimeEngine emits a ``log_warning`` so the non-default
path is always visible in logs.

Override via environment variable
----------------------------------
Set  UNITPORT_FLOW_AWARE_EXECUTION=0   to activate the compat (topological) path.
Set  UNITPORT_FLOW_AWARE_EXECUTION=1   (or unset) to use the default path.

Scope
-----
This switch is intentionally narrow: it only affects the WorkflowIR execution
path inside NodeExecutor.  The execution-graph path (WorkflowRunner) is
unaffected.  The switch state is recorded in the audit log on each execution
so that field issues can be correlated with the active strategy.

Removal criteria (Phase 7 §12.4)
---------------------------------
Remove this module when flow-aware DFS has 3+ months production stability
with no regressions requiring topological-sort fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_ENV_KEY = "UNITPORT_FLOW_AWARE_EXECUTION"
_BEHAVIOR_ENV_KEY = "UNITPORT_BEHAVIOR_ENABLED"
_FALSY = frozenset({"0", "false", "off", "no"})


@dataclass
class MigrationFlags:
    """Runtime migration flags for Phase 1.

    Attributes
    ----------
    use_flow_aware_execution:
        True  → NodeExecutor uses flow-aware DFS (STEP-04, new behaviour).
        False → NodeExecutor falls back to topological sort (pre-STEP-04,
                compat/legacy behaviour preserved as controlled fallback).
    """

    use_flow_aware_execution: bool = True

    @classmethod
    def from_env(cls) -> "MigrationFlags":
        """Build flags from environment variables (or defaults)."""
        flags = cls()
        raw = os.environ.get(_ENV_KEY, "1").strip().lower()
        flags.use_flow_aware_execution = raw not in _FALSY
        return flags

    def to_dict(self) -> dict:
        """Serialise to a plain dict for audit logging."""
        return {
            "use_flow_aware_execution": self.use_flow_aware_execution,
            "source_env_key": _ENV_KEY,
        }


@dataclass
class BehaviorRunFlags:
    """Circle 1 Step 1.1 — controls whether behavior-capable missions use WorkflowIR path.

    Default: False (exec_graph compat path).
    Set UNITPORT_BEHAVIOR_ENABLED=1 to activate WorkflowIR path for all runs,
    or the UI will auto-activate it when behavior nodes are detected.

    Attributes
    ----------
    use_workflowir_for_behavior:
        True  → UI compiles canvas → WorkflowIR and routes through NodeExecutor.
        False → exec_graph dict path (WorkflowRunner) used as before.
    """

    use_workflowir_for_behavior: bool = False

    @classmethod
    def from_env(cls) -> "BehaviorRunFlags":
        """Build flags from environment variables (or defaults)."""
        raw = os.environ.get(_BEHAVIOR_ENV_KEY, "0").strip().lower()
        return cls(use_workflowir_for_behavior=raw not in _FALSY and raw != "0")

    def to_dict(self) -> dict:
        """Serialise to a plain dict for audit logging."""
        return {
            "use_workflowir_for_behavior": self.use_workflowir_for_behavior,
            "source_env_key": _BEHAVIOR_ENV_KEY,
        }
