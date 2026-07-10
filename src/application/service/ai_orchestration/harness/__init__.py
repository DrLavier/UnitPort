# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""B2 System Harness for AI Build (blueprint §4.B2 / §5 / §6).

- ``roles``       — RoleSpec + the v1 active roster (Design / Orchestrator / Critic).
- ``agent``       — generic tool-calling Agent driving B3 through the C2 tool schemas.
- ``intent_gate`` — silent context-free pre-step: is this prompt a canvas build
  needing the knowledge base? False → the scheduler answers directly instead.
- ``scheduler``   — the deterministic §6 control flow + bounded repair + commit decision.
- ``session``   — the thin B1-facing session seam (build a scheduler, run, result).
- ``templates`` — same-family template discovery (the orchestrator's soft reference).
- ``run_task``  — the Qt worker entry (Wave 2; marshals mutations to the GUI thread).
"""
