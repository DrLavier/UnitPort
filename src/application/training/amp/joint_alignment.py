# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Boot-time verification that env amp_obs fields match the motion clip.

This module is the hard mitigation for AMP_design.yaml §7.risks
.amp_obs_dim_drift. The failure mode it prevents:

1. The env exposes ``get_amp_observations()`` returning a tensor whose
   columns are laid out as ``[joint_pos | joint_vel | root_z | ...]``.
2. The user loads an amp_legged_gym motion file whose transitions are
   laid out as ``[joint_pos | toe_pos | lin_vel | ...]``.
3. The discriminator trains on pairs whose columns don't line up —
   training does not crash, but the disc learns garbage and the
   policy quietly drifts.

To block this we require both sides to declare a canonical ordered
list of field names at launcher startup. ``verify_alignment`` compares
them element-wise and raises on any mismatch. ``dump_alignment_report``
writes the two orderings (and the outcome) to ``run.yaml`` so the
result is visible post-hoc.

Pure Python — no torch, no rsl_rl, no isaac. Main-venv safe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class AmpObsAlignmentError(ValueError):
    """Raised when env and motion clip amp_obs layouts disagree.

    The message contains the two orderings plus a compact diff. The
    launcher aborts training on this rather than continuing.
    """


@dataclass
class AlignmentReport:
    env_fields: List[str]
    clip_fields: List[str]
    ok: bool
    error: Optional[str] = None
    # Intra-field canonical mapping (e.g. joint permutation + foot ids
    # for quadrupeds). Populated by the launcher via
    # ``amp_obs_terms.preflight_canonical_mapping``. ``verify_alignment``
    # only catches *field-name* drift; this extra dict captures the
    # deeper intra-field ordering that previously went unchecked.
    canonical_mapping: Dict[str, Any] = field(default_factory=dict)
    # Per-term reference frame BOTH producers must emit in ("base" /
    # "world_z" / "joint"). Captured from the AMP obs term registry so
    # the audit file records the frame contract that the toe-frame /
    # leg-order class of bug (RC-1/RC-2) violated silently.
    term_frames: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "env_fields": list(self.env_fields),
            "clip_fields": list(self.clip_fields),
            "ok": self.ok,
            "error": self.error,
            "canonical_mapping": dict(self.canonical_mapping),
            "term_frames": dict(self.term_frames),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_alignment(
    env_fields: Sequence[str],
    clip_fields: Sequence[str],
) -> None:
    """Raise ``AmpObsAlignmentError`` iff the two orderings differ.

    Comparison is:
    - length (same number of fields)
    - element-wise, case-sensitive

    The intent is strict equality: any permutation, rename, or count
    mismatch is a hard error. If you need to map fields that are the
    "same thing" with different names, fix the names upstream rather
    than tolerating the drift here.
    """
    env_list = list(env_fields)
    clip_list = list(clip_fields)

    if len(env_list) != len(clip_list):
        raise AmpObsAlignmentError(_format_diff(env_list, clip_list, reason="length"))

    diffs: List[str] = []
    for i, (e, c) in enumerate(zip(env_list, clip_list)):
        if e != c:
            diffs.append(f"  [{i}] env={e!r}  clip={c!r}")
            if len(diffs) >= 8:
                diffs.append("  ...")
                break
    if diffs:
        raise AmpObsAlignmentError(
            f"AMP observation field mismatch between env and motion clip "
            f"({len(env_list)} fields each):\n" + "\n".join(diffs) +
            "\n\nThis would train the discriminator on column-misaligned "
            "transitions. Fix the env's get_amp_observations() or the motion "
            "clip's amp_obs_fields so they agree exactly."
        )


#: Discriminator LOGIT MARGIN (mean expert_pred − mean policy_pred) above
#: which the disc is separating the two distributions *confidently*. This
#: is the real health signal — NOT binary ``Disc acc``, which saturates
#: near 1.00 even for a healthy converged AMP (accuracy is the sign of the
#: logit; a flexible classifier almost always finds some consistent
#: separating boundary, so acc ≈ 1.0 says nothing). A structurally
#: separable obs (RC-1/RC-2/normalization) keeps the margin large and
#: persistent (~2.0); a healthy run's margin collapses toward 0 (the
#: policy nearly fools the disc, logits squeezed to ~±0.1) — empirically
#: a healthy Go2 run is < ~0.9 by iter 100 and ~0.25 by iter 1000, while a
#: broken run stays ~2.0 forever.
_SEPARABLE_MARGIN_THRESHOLD: float = 1.5
#: Don't evaluate before this — the margin spikes transiently in the first
#: few dozen iterations even on healthy runs (the disc trains faster than
#: the policy initially) and then collapses.
_SEPARABLE_MIN_ITER: int = 100


def amp_separability_warning(
    it: int,
    policy_pred: float,
    expert_pred: float,
    *,
    margin_threshold: float = _SEPARABLE_MARGIN_THRESHOLD,
    min_iter: int = _SEPARABLE_MIN_ITER,
) -> Optional[str]:
    """Return a loud diagnostic string when the discriminator is
    separating expert from policy with a LARGE, PERSISTENT logit margin,
    else ``None``.

    A large margin that does not collapse is the fingerprint of a
    TRIVIALLY-SEPARABLE expert-vs-policy AMP observation distribution —
    the policy cannot close a structural obs mismatch (leg order, frame,
    normalization), so the style reward becomes a near-constant per-step
    survival bonus carrying no imitation gradient (RC-1 leg swap / RC-2
    toe frame). A healthy run's margin shrinks toward 0 as the policy
    learns to fool the disc.

    ⚠️ This deliberately does NOT key on ``Disc acc``: accuracy is the
    binary sign of the logit and saturates at ~1.00 even when the disc is
    barely separating (margin ~0.2). Reading acc was the original
    confusion — a perfectly healthy run shows ``Disc acc 1.00/1.00``.

    Pure function (no torch / no isaac) so it is unit-testable and the
    Kit-side runner can import it without dragging the DL stack.
    """
    if it < min_iter:
        return None
    margin = float(expert_pred) - float(policy_pred)
    if margin < margin_threshold:
        return None
    return (
        f"[UnitPort][AMP][SEPARABILITY] iter={it}: discriminator logit "
        f"margin is large and persistent (expert_pred={expert_pred:.2f} − "
        f"policy_pred={policy_pred:.2f} = {margin:.2f} >= {margin_threshold:.1f}). "
        f"The expert and policy AMP observations are likely TRIVIALLY "
        f"SEPARABLE — the policy cannot close a structural obs mismatch, so "
        f"the style reward degenerates into a near-constant survival bonus "
        f"with no imitation gradient. This is the RC-1 (leg-order swap) / "
        f"RC-2 (toe reference frame) / normalization class of bug, NOT a "
        f"hyperparameter, and NOT the (misleading) Disc-acc=1.00 readout — "
        f"a healthy run squeezes this margin toward 0. Inspect "
        f"amp_alignment.json (per-term frames + canonical mapping) and the "
        f"expert-vs-env obs parity."
    )


def dump_alignment_report(
    run_dir: Path | str,
    env_fields: Sequence[str],
    clip_fields: Sequence[str],
    *,
    ok: bool,
    error: Optional[str] = None,
    canonical_mapping: Optional[Dict[str, Any]] = None,
    term_frames: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the alignment result to ``<run_dir>/amp_alignment.json``.

    The launcher calls this after :func:`verify_alignment` (wrapped in
    try/except) so both success and failure outcomes are preserved for
    post-mortem. Returns the written file path.

    Optional ``canonical_mapping`` captures the resolved per-DoF /
    per-foot index permutation (from
    ``amp_obs_terms.preflight_canonical_mapping``) so the audit file
    records the deeper intra-field ordering that ``verify_alignment``
    cannot check.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = AlignmentReport(
        env_fields=list(env_fields),
        clip_fields=list(clip_fields),
        ok=bool(ok),
        error=error,
        canonical_mapping=dict(canonical_mapping or {}),
        term_frames=dict(term_frames or {}),
    )
    out = run_dir / "amp_alignment.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_diff(
    env_fields: List[str], clip_fields: List[str], *, reason: str
) -> str:
    return (
        f"AMP observation field mismatch ({reason}):\n"
        f"  env  ({len(env_fields)}): {env_fields}\n"
        f"  clip ({len(clip_fields)}): {clip_fields}"
    )
