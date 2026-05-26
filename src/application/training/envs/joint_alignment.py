# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Boot-time verification that env amp_obs fields match the motion clip.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/amp/joint_alignment.py``.
The DEMO version sat under the ``amp/`` subpackage; here it lives under
``envs/`` because Stage 6 owns the env-side declarations and the AMP
launcher (Stage 8/10) is the consumer.

Failure mode prevented:
    1. The env exposes ``get_amp_observations()`` returning a tensor whose
       columns are laid out as ``[joint_pos | joint_vel | root_z | ...]``.
    2. The user loads an amp_legged_gym motion file whose transitions are
       laid out as ``[joint_pos | toe_pos | lin_vel | ...]``.
    3. The discriminator trains on column-misaligned pairs — training
       does not crash, but the disc learns garbage and the policy
       quietly drifts.

Pure Python — no torch, no rsl_rl, no isaac. Main-venv safe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class AmpObsAlignmentError(ValueError):
    """Raised when env and motion clip amp_obs layouts disagree.

    The launcher aborts training on this rather than continuing.
    """


@dataclass
class AlignmentReport:
    env_fields: List[str]
    clip_fields: List[str]
    ok: bool
    error: Optional[str] = None
    canonical_mapping: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "env_fields": list(self.env_fields),
            "clip_fields": list(self.clip_fields),
            "ok": self.ok,
            "error": self.error,
            "canonical_mapping": dict(self.canonical_mapping),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_alignment(
    env_fields: Sequence[str],
    clip_fields: Sequence[str],
) -> None:
    """Raise :exc:`AmpObsAlignmentError` iff the two orderings differ.

    Comparison is element-wise + case-sensitive + length-checked. Any
    permutation, rename, or count mismatch is a hard error.
    """
    env_list = list(env_fields)
    clip_list = list(clip_fields)

    if len(env_list) != len(clip_list):
        raise AmpObsAlignmentError(
            _format_diff(env_list, clip_list, reason="length")
        )

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
            "transitions. Fix env get_amp_observations() or the motion "
            "clip's amp_obs_fields so they agree exactly."
        )


def dump_alignment_report(
    run_dir: Union[Path, str],
    env_fields: Sequence[str],
    clip_fields: Sequence[str],
    *,
    ok: bool,
    error: Optional[str] = None,
    canonical_mapping: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the alignment result to ``<run_dir>/amp_alignment.json``.

    Routes through ``DataManager.save_data`` so the SDK contract holds
    (atomic write via tempfile + os.replace).
    """
    from unitport_sdk import save_data

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = AlignmentReport(
        env_fields=list(env_fields),
        clip_fields=list(clip_fields),
        ok=bool(ok),
        error=error,
        canonical_mapping=dict(canonical_mapping or {}),
    )
    out = run_dir / "amp_alignment.json"
    save_data(out, report.to_dict())
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


__all__ = [
    "AlignmentReport",
    "AmpObsAlignmentError",
    "verify_alignment",
    "dump_alignment_report",
]
