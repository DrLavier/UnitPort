# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 output B — the residual structure report (Layer-3 feed).

Per Type-II/III dimension: a systematic/chaotic verdict + the IC-sensitivity
distribution. Systematic dimensions carry their cross-engine effective-response
factor span (the same envelope the range table records); chaotic dimensions
carry a fail-loud "non-convergent zone" declaration — an honest statement that
sim2sim cannot reproduce that dimension, NOT a silent omission.

Written as a markdown report + a machine-readable JSON sidecar (the
``CalibrationReport.write_markdown`` atomic-write pattern), under
``USER_CONFIG_DIR/registers/sim2sim_residuals/<sku>_report.{md,json}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .discriminator import MeasurementAnalysis


@dataclass(frozen=True)
class CampaignSummary:
    """The cross-round campaign archive folded into the report (Stage-3 feed).

    A single residual report normally documents ONE probe set; this consolidates
    the whole measurement campaign — the methodology, the headline empirical law,
    and the key contrast — so the six-dimension verdict is self-explaining and
    rework-proof. Faithfully records that every apparent large cross-engine
    difference dissolved into a probe/measure artifact, and that nothing graded
    chaotic. Rendered as a leading narrative section + a JSON sidecar block."""

    methodology: List[str]      # how the clean measure was obtained
    key_contrast: str           # the legged-vs-box collapse-confound contrast
    law_memo: str               # the headline empirical law (rework memo)
    layer_routing: List[str]    # what each of the 3 layers becomes, per data
    known_gaps: List[str] = field(default_factory=list)  # honest residual gaps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "methodology": list(self.methodology),
            "key_contrast": self.key_contrast,
            "law_memo": self.law_memo,
            "layer_routing": list(self.layer_routing),
            "known_gaps": list(self.known_gaps),
        }


@dataclass(frozen=True)
class ResidualReport:
    sku: str
    markdown: str
    json_data: Dict[str, Any]

    def write_artifacts(self, base_path: Path | str) -> None:
        """Write ``<base>.md`` + ``<base>.json`` atomically (tmp → os.replace)."""
        import os

        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        md_path = base.with_suffix(".md")
        tmp = md_path.with_suffix(".md.tmp")
        tmp.write_text(self.markdown, encoding="utf-8")
        os.replace(str(tmp), str(md_path))
        json_path = base.with_suffix(".json")
        tmpj = json_path.with_suffix(".json.tmp")
        tmpj.write_text(json.dumps(self.json_data, indent=2), encoding="utf-8")
        os.replace(str(tmpj), str(json_path))


def build_residual_report(
    analysis: MeasurementAnalysis, *, timestamp: str = "",
    campaign: "CampaignSummary | None" = None,
) -> ResidualReport:
    thr = analysis.chaos_sensitivity_threshold
    n_clean = sum(1 for d in analysis.per_dimension if d.verdict == "clean")
    n_param = sum(1 for d in analysis.per_dimension if d.verdict == "parameterizable")
    n_chaotic = sum(1 for d in analysis.per_dimension if d.verdict == "chaotic")

    lines: List[str] = [
        "# Sim2Sim Residual Structure Report",
        "",
        f"- **SKU**: `{analysis.sku or '(unspecified)'}`",
        f"- **Measured at**: {timestamp or '(unstamped)'}",
        f"- **Probes**: {len(analysis.per_probe)} · "
        f"**dimensions**: {len(analysis.per_dimension)} "
        f"({n_clean} clean, {n_param} parameterizable, {n_chaotic} chaotic)",
        "",
    ]

    # Campaign archive (the cross-round 规律 + methodology + key contrast). Goes
    # FIRST so the six-dimension verdict below reads as a conclusion, not a
    # bare table — and so the rework memo is impossible to miss.
    if campaign is not None:
        lines += [
            "## Campaign law (rework memo — read first)",
            "",
            f"> {campaign.law_memo}",
            "",
            "**Key contrast.** " + campaign.key_contrast,
            "",
            "### Methodology (how the clean measure was obtained)",
            "",
        ]
        lines += [f"- {m}" for m in campaign.methodology]
        lines += [
            "",
            "### Three-layer routing implied by the data",
            "",
        ]
        lines += [f"- {r}" for r in campaign.layer_routing]
        if campaign.known_gaps:
            lines += ["", "### Known gaps (honest residual — not covered here)", ""]
            lines += [f"- {g}" for g in campaign.known_gaps]
        lines += [""]

    lines += [
        "## Grading (continuous S, NOT a single binary cut)",
        "",
        "S = coefficient-of-variation of the cross-engine residual across "
        "IC-perturbed repeats. Real data clusters in a middle band with no "
        "natural bimodal gap, so the verdict is a 3-level GRADE over the "
        "continuous S, not a binary threshold:",
        "",
        "- **clean** — `S < 0.1`: residual ~deterministic in IC → no DR needed.",
        "- **parameterizable** — `0.1 ≤ S < 1.0`: bounded IC sensitivity → "
        "Layer-2 domain randomization absorbs it.",
        "- **chaotic** — `S ≥ 1.0`: residual spread exceeds its mean "
        "(exponential-ish divergence) → Layer-3 fail-loud, not coverable.",
        "",
        ("> **No dimension reached S ≥ 1.0 (chaotic) in this measurement.** "
         "Recorded honestly — the middle band is NOT force-fit into 'chaotic' "
         "to satisfy a binary model." if n_chaotic == 0 else
         f"> {n_chaotic} dimension(s) graded chaotic (S ≥ 1.0)."),
        "",
        "## Per-dimension verdict",
        "",
        "| Dimension | Grade | Probes | median S | factor span | Routes to |",
        "|---|---|---|---|---|---|",
    ]
    _routes = {
        "clean": "Layer 2 optional (factor≈1 ⇒ unnecessary)",
        "parameterizable": "Layer 2 (DR range)",
        "chaotic": "Layer 3 (diagnose only)",
    }
    for d in analysis.per_dimension:
        med = float(np.median(d.sensitivities)) if d.sensitivities else 0.0
        if d.verdict != "chaotic" and d.factor_low is not None:
            span = f"[{d.factor_low:.3f}, {d.factor_high:.3f}]"
        else:
            span = "— (non-coverable)"
        lines.append(
            f"| `{d.dimension}` | **{d.verdict}** | {d.n_probes} | "
            f"{med:.3f} | {span} | {_routes.get(d.verdict, '?')} |"
        )

    # Chaotic non-convergent-zone declarations (fail-loud honesty).
    chaotic = [d for d in analysis.per_dimension if d.verdict == "chaotic"]
    if chaotic:
        lines += ["", "## Non-convergent zones (chaotic — sim2sim irreproducible)", ""]
        for d in chaotic:
            lines.append(
                f"- **`{d.dimension}`**: residual is chaotic (IC-sensitive) — "
                "cannot be covered by finite randomization or stably compensated. "
                "The policy must NOT depend on critical behaviour in this regime; "
                "Layer 3 records it as a non-convergent zone (no false compensation)."
            )

    # Sensitivity distribution per dimension (for threshold calibration).
    lines += ["", "## Sensitivity distribution (per probe)", ""]
    for d in analysis.per_dimension:
        svals = ", ".join(f"{s:.3f}" for s in d.sensitivities)
        lines.append(f"- `{d.dimension}`: S = [{svals}]")

    # Sampling-convergence curve (硬化二): S using the first n repeats. A flat
    # tail ⇒ the verdict is sampling-converged; a still-drifting tail ⇒ MORE
    # repeats are needed before the systematic/chaotic call can be trusted.
    lines += ["", "## Sensitivity convergence (S vs #repeats)", ""]
    any_unconverged = False
    for p in analysis.per_probe:
        conv = p.sensitivity_convergence
        if not conv:
            lines.append(f"- `{p.probe_id}`: (need ≥3 repeats for a convergence curve)")
            continue
        # "Converged" heuristic: last step changes S by < 10% of its value.
        converged = (
            len(conv) >= 3
            and abs(conv[-1] - conv[-2]) <= 0.1 * (abs(conv[-1]) + 1e-9)
        )
        any_unconverged = any_unconverged or not converged
        curve = " → ".join(f"{s:.3f}" for s in conv)
        flag = "CONVERGED" if converged else "STILL DRIFTING — add repeats"
        lines.append(f"- `{p.probe_id}` (n=2..{len(conv) + 1}): {curve}   [{flag}]")
    if any_unconverged:
        lines.append("")
        lines.append("> ⚠ At least one probe's S is still drifting — the "
                     "systematic/chaotic verdict is NOT yet sampling-converged. "
                     "Increase `n_repeats` and re-measure before trusting it.")

    lines += [
        "",
        "## Method (v1)",
        "",
        "- Residual measured AFTER Layer-1 alignment → pure Type-II/III.",
        "- Sensitivity `S = std(R_r) / (mean(R_r) + ε)` across IC-perturbed "
        "repeats; `S ≥ threshold` ⇒ chaotic. Finite-time Lyapunov exponent is a "
        "documented future refinement, not v1.",
        "- Systematic dimensions export a cross-engine effective-response factor "
        "span to the range table (Layer 2); chaotic dimensions are diagnosed "
        "only (Layer 3).",
    ]
    markdown = "\n".join(lines) + "\n"

    json_data: Dict[str, Any] = {"timestamp": timestamp, **analysis.to_dict()}
    if campaign is not None:
        json_data["campaign"] = campaign.to_dict()
    return ResidualReport(
        sku=analysis.sku,
        markdown=markdown,
        json_data=json_data,
    )


__all__ = ["ResidualReport", "CampaignSummary", "build_residual_report"]
