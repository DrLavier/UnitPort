# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Sim2sim Stage 1 — §4 cross-engine difference measurement infrastructure.

The data-driven foundation shared by Layer 2 (cross-engine domain
randomization) and Layer 3 (residual characterization). It drives the SAME
``(state, action)`` open-loop on BOTH engines — **after Layer-1 alignment**, so
the measured residual is pure Type-II/III (no Type-I leakage) — and a
systematic-vs-chaotic discriminator decides whether each residual dimension is
randomizable (Layer 2) or only diagnosable (Layer 3).

Architecture (batch-file contract — no live IPC; the PhysX side runs manually
inside the Kit venv):

    coordinator (.venv311)                      il_sim2sim_launcher (Kit venv)
      generate_probes  ──► sim2sim_probes.json ───────────────┐
      run_mujoco       ──► mujoco_results.jsonl                │
                                                               ▼
                              read probes → PhysX open-loop → physx_results.jsonl
      analyze  ◄───────────────────────────────────────────────┘
        → discriminator → range table (push_data) + residual report (md+json)

Public surface is intentionally small; the heavy lifting lives in the
sub-modules (``protocol`` / ``scenarios`` / ``mujoco_driver`` /
``discriminator`` / ``range_table`` / ``residual_report`` / ``coordinator`` /
``mock_physx``).
"""

from __future__ import annotations

from .protocol import (
    AlignedPlant,
    ContactCfg,
    EngineResult,
    Probe,
    ProbeSet,
    StepRecord,
)

__all__ = [
    "AlignedPlant",
    "ContactCfg",
    "EngineResult",
    "Probe",
    "ProbeSet",
    "StepRecord",
]
