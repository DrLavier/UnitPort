# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 output A — the measured cross-engine randomization range table.

For each Type-II dimension judged **systematic** by the discriminator, the table
records the cross-engine effective-response **factor envelope** ``[low, high]``
that Stage-2 domain randomization must span so the policy experiences BOTH
engines' realized behaviour. **Chaotic** dimensions get NO range (they cannot be
covered by finite randomization) — recorded with ``factor=None`` so Stage 2
fail-louds rather than silently randomizing a non-coverable axis.

The table is a MEASURED artifact (data-driven), so it lives under
``USER_CONFIG_DIR`` (never ``registers/data/``, §4) and is read back by the
``registers.sim2sim_ranges`` loader. §7.2 MuJoCo-lean: the envelope can be
re-centred toward MuJoCo (the preparatory truth end) — recorded as provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .discriminator import MeasurementAnalysis

RANGE_TABLE_SCHEMA_VERSION = 1

# Stage-2 (收窄版) cross-engine friction residual reference, from the 2026-06-04
# block-slide campaign (the CLEAN single-rigid-body friction probe). This is the
# campaign-level finding shared by the bundle DR-coverage provenance and the
# residual report — single source so the two cannot drift. ``factor`` is the
# PhysX/MuJoCo box stopping-distance ratio at matched μ; the realistic band is
# μ∈[0.6,1.0] (~0–12% stickier in PhysX), the extreme tail is μ≥1.5 (~1.78×).
FRICTION_RESIDUAL_REFERENCE: Dict[str, Any] = {
    "dimension": "friction_cone",
    "method": "block_slide",
    "measured_at": "2026-06-04",
    "factor_low": 0.56,
    "factor_high": 1.02,
    "realistic_band_pct": [0, 12],
    "extreme_factor": 1.78,
}

DR_PROVENANCE_SCHEMA_VERSION = 1


@dataclass
class DimensionRange:
    dimension: str
    verdict: str                          # "systematic" | "chaotic"
    channel: str                          # effective-measure channel
    factor_low: Optional[float]           # None ⇒ chaotic (no randomizable range)
    factor_high: Optional[float]
    n_probes: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "verdict": self.verdict,
            "channel": self.channel,
            "factor_low": (None if self.factor_low is None else float(self.factor_low)),
            "factor_high": (None if self.factor_high is None else float(self.factor_high)),
            "n_probes": int(self.n_probes),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DimensionRange":
        lo = d.get("factor_low")
        hi = d.get("factor_high")
        return cls(
            dimension=str(d["dimension"]),
            verdict=str(d["verdict"]),
            channel=str(d.get("channel", "")),
            factor_low=(None if lo is None else float(lo)),
            factor_high=(None if hi is None else float(hi)),
            n_probes=int(d.get("n_probes", 0)),
            notes=list(d.get("notes", []) or []),
        )


@dataclass
class RangeTable:
    sku: str
    dimensions: List[DimensionRange]
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = RANGE_TABLE_SCHEMA_VERSION

    def systematic(self) -> List[DimensionRange]:
        return [d for d in self.dimensions if d.verdict == "systematic"]

    def get(self, dimension: str) -> Optional[DimensionRange]:
        for d in self.dimensions:
            if d.dimension == dimension:
                return d
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "sku": str(self.sku),
            "provenance": dict(self.provenance),
            "dimensions": [d.to_dict() for d in self.dimensions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RangeTable":
        if not isinstance(d, dict):
            raise ValueError("RangeTable.from_dict: expected dict")
        ver = int(d.get("schema_version", 0))
        if ver != RANGE_TABLE_SCHEMA_VERSION:
            raise ValueError(
                f"RangeTable.from_dict: schema_version {ver} != "
                f"{RANGE_TABLE_SCHEMA_VERSION} (re-measure with the current tool)"
            )
        dims_raw = d.get("dimensions")
        if not isinstance(dims_raw, list):
            raise ValueError("RangeTable.from_dict: 'dimensions' must be a list")
        return cls(
            sku=str(d.get("sku", "")),
            dimensions=[DimensionRange.from_dict(x) for x in dims_raw],
            provenance=dict(d.get("provenance", {}) or {}),
            schema_version=ver,
        )

    # ---- low-level JSON I/O (no USER_CONFIG_DIR dependency; used by tests) ----

    def write_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        import os
        os.replace(str(tmp), str(path))

    @classmethod
    def read_json(cls, path: Path | str) -> "RangeTable":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_range_table(
    analysis: MeasurementAnalysis,
    *,
    timestamp: str = "",
    mujoco_lean: bool = True,
    engine_info: Optional[Dict[str, Any]] = None,
) -> RangeTable:
    """Assemble the range table from a discriminator analysis.

    ``timestamp`` is passed in (the workflow/runtime forbids ``Date.now()`` in
    deterministic contexts) so callers stamp it after the analysis returns.
    ``mujoco_lean`` records the §7.2 preference; the envelope itself is the raw
    cross-engine span (Stage 2 applies the lean when sampling)."""
    dims: List[DimensionRange] = []
    for dv in analysis.per_dimension:
        # Pull the channel from the first probe of this dimension.
        channel = next(
            (p.channel for p in analysis.per_probe if p.dimension == dv.dimension),
            "",
        )
        dims.append(DimensionRange(
            dimension=dv.dimension,
            verdict=dv.verdict,
            channel=channel,
            factor_low=dv.factor_low,
            factor_high=dv.factor_high,
            n_probes=dv.n_probes,
            notes=list(dv.notes),
        ))
    provenance = {
        "measured_at": str(timestamp),
        "n_probes": int(len(analysis.per_probe)),
        "chaos_sensitivity_threshold": float(analysis.chaos_sensitivity_threshold),
        "mujoco_lean": bool(mujoco_lean),
        "engine_info": dict(engine_info or {}),
    }
    return RangeTable(sku=analysis.sku, dimensions=dims, provenance=provenance)


def build_dr_coverage_provenance(
    *,
    training_backend: str,
    friction_dr_enabled: bool,
    friction_dr_range: Sequence[float],
    sku: str = "",
    measured_factor: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Stage-2 (收窄版) bundle provenance: record that the cross-engine friction
    residual is COVERED by the existing ground-friction DR — no new channel.

    The 2026-06-04 block-slide campaign showed the only surviving cross-engine
    knob is friction (box factor [0.56, 1.02]); contact-stiffness is a ~5–10%
    tight bias (zero-signal) and nothing graded chaotic. The residual sits
    INSIDE the existing ground-friction DR envelope (SB3 [0.5,1.5] /
    IsaacLab [0.5,1.25]), so a policy trained with friction DR has already seen
    wider friction variation — no friction-μ channel, no stiffness channel.

    The ``coverage.friction_dr_enabled`` flag is what the load-time fail-loud
    check keys on: the "covered" guarantee holds ONLY while friction DR is on,
    so a bundle trained with it OFF must warn at load (§8 — a switch-dependent
    guarantee must fail loud when the switch is off, not silently).
    """
    residual = dict(FRICTION_RESIDUAL_REFERENCE)
    if measured_factor is not None and len(measured_factor) == 2:
        residual["factor_low"] = float(measured_factor[0])
        residual["factor_high"] = float(measured_factor[1])
        residual["measured_for_this_sku"] = True
    else:
        residual["measured_for_this_sku"] = False
    return {
        "schema_version": DR_PROVENANCE_SCHEMA_VERSION,
        "sku": str(sku),
        "friction_residual": residual,
        "coverage": {
            "covered_by": "existing_ground_friction_dr",
            "friction_dr_enabled": bool(friction_dr_enabled),
            "friction_dr_range": [float(friction_dr_range[0]), float(friction_dr_range[1])],
            "training_backend": str(training_backend),
        },
        "known_gaps": {
            # realized==designed item (same design → different realized friction
            # envelope per engine; same class as the armature gap). Deferred for
            # disproportion, NOT because the training side is off-limits.
            "dr_envelope_asymmetry": (
                "Two engines randomize friction over different envelopes from "
                "one canvas (SB3 friction_range [0.5,1.5] vs IsaacLab "
                "static/dynamic_friction_range [0.5,1.25]). This is a "
                "realized==designed item — in principle it should be aligned. "
                "Deferred because it has zero consequence here (both envelopes "
                "amply cover the [0.56,1.02] residual) and aligning would "
                "perturb every training distribution (disproportionate), NOT "
                "because the training side is off-limits. To be handled when "
                "training-side DR is unified as a standalone effort."
            ),
        },
        "no_new_channel_rationale": (
            "Cross-engine friction residual (realistic band ~0-12%, extreme "
            "<=1.78x at mu>=1.5) falls inside the existing ground-friction DR "
            "envelope; no new friction-mu channel. Contact-stiffness factor "
            "[1.05,1.10] is zero-signal; no stiffness channel. Nothing chaotic."
        ),
    }


def persist_to_user_config(table: RangeTable) -> bool:
    """Write the table under ``USER_CONFIG_DIR/registers/sim2sim_ranges/<sku>.json``
    via the SDK storage channel (atomic). Production path — tests use
    ``RangeTable.write_json`` to a temp dir instead."""
    from unitport_sdk import push_data

    if not table.sku:
        raise ValueError("persist_to_user_config: range table has no sku (§8)")
    rel = f"registers/sim2sim_ranges/{table.sku}.json"
    return bool(push_data(rel, table.to_dict(), channel="local"))


__all__ = [
    "RANGE_TABLE_SCHEMA_VERSION",
    "DR_PROVENANCE_SCHEMA_VERSION",
    "FRICTION_RESIDUAL_REFERENCE",
    "DimensionRange",
    "RangeTable",
    "build_range_table",
    "build_dr_coverage_provenance",
    "persist_to_user_config",
]
