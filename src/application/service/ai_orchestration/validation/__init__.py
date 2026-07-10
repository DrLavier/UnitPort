# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""B4 Validation & Calibration tiers for AI Build (blueprint §4.B4).

- ``invariants``   — tier-3 Cross-Domain Invariant Catalog (C3) + integrator.
- ``compile_gate`` — tier-2 authoritative ERROR gate (Wave 1).
- ``critic``       — tier-4 baseline-relative pathology review (Wave 1).

Tier-1 (per-op interface alignment) lives in :mod:`..mutation_api`.
"""
