# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``amp_legged_gym`` — DeepMimic-style 61-float JSON loader.

Reads the JSON/text format used by AMP_for_hardware and the wider
legged_gym AMP family. Each frame has 61 floats laid out as::

    [root_pos(3) | root_quat(4, xyzw) | joint_pos(12) | toe_pos(12)
     | lin_vel(3) | ang_vel(3) | joint_vel(12) | toe_vel(12)]

Per-leg arrays (joint_pos / joint_vel / toe_pos / toe_vel) are reordered
at load time into UnitPort's canonical **FL/FR/RL/RR** leg order. The
source file's native leg order is a per-file *physical* property — it is
**not** assumed positionally. It is auto-detected from each clip's
body-frame toe geometry (front foot = +x, left foot = +y) and turned
into a role-keyed permutation. An explicit ``source_leg_order`` may be
passed for clips whose footprint is too degenerate to decode (synthetic
fixtures, near-static poses); otherwise an undecodable clip raises
(CLAUDE.md §8 — no silent positional assumption).

Why auto-detect rather than a declared ``(source_format, target_sku)``
override: the two conventions shipped under ``datasets/amp_go2/``
(``go2_motion/*`` = FL/FR/RL/RR, ``mocap_motions/*`` = FR/FL/RR/RL) share
the SAME ``format_id`` AND target the SAME robot SKU, so no static key
can tell them apart. The leg order lives in the bytes; we read it there.

Numeric helpers (``_normalize_quats`` / ``_standardize_quats``) live in
this module because they are private to the AMP layout.
``_standardize_quats`` is re-used by :mod:`augment` for the mirror op —
that is the only cross-module private import in this subpackage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from application.training.motion.clip import MotionClip
from application.training.motion.contract import (
    CONTRACT_SCHEMA_VERSION,
    ReferenceMotionContract,
    SourceInfo,
    validate_contract,
)
from application.training.motion.labels import infer_task_tag
from application.training.motion.loaders.base import LoaderError, MotionLoader
from application.training.motion.normalizer import normalize_motion_clip
from application.training.motion_ir_mapping import QUADRUPED_AMP_IR_ROLES


# UnitPort canonical quadruped leg order. Single contract shared with the
# policy-side env extractor (``amp.obs_terms.CANONICAL_QUAD_LEGS``) — both
# producers MUST emit per-leg tensors in this order or the discriminator
# trivially separates expert from policy on leg index, pinning disc_acc
# at ~1.0 regardless of policy quality. Defined locally to keep this
# loader numpy-only / main-venv safe (no torch-laden import of obs_terms).
_CANONICAL_LEG_ORDER: Tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")

#: Minimum |x| / |y| (metres) a mean foot offset must clear to be assigned
#: unambiguously to a front/rear × left/right corner. Real quadruped
#: footprints sit at ±0.1–0.2 m; anything inside this band is treated as
#: undecodable so a degenerate clip fails loud rather than mis-labelled.
_LEG_DETECT_MIN_OFFSET: float = 0.02


class AMPLegGymLoader(MotionLoader):
    """Load an amp_legged_gym ``.txt``/``.json`` motion file.

    Frame layout (61 floats; shared across the AMP_for_hardware family):

        [root_pos(3) | root_quat(4, xyzw) | joint_pos(12) | toe_pos(12)
         | lin_vel(3) | ang_vel(3) | joint_vel(12) | toe_vel(12)]

    Per-leg arrays are reordered at load time into UnitPort's canonical
    FL/FR/RL/RR. The source leg order is auto-detected from toe geometry
    (or taken from an explicit ``source_leg_order``). Downstream consumers
    receive canonical-ordered data and need not re-permute.

    ``MotionWeight`` and ``FrameDuration`` come from the JSON header;
    ``LoopMode`` ``"Wrap"`` maps to ``"wrap"``, everything else to
    ``"clamp"``.
    """

    format_id = "amp_legged_gym"

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TOE_POS_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TOE_VEL_SIZE = 12

    FRAME_DIM = (
        POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TOE_POS_SIZE
        + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + TOE_VEL_SIZE
    )  # == 61

    def __init__(
        self,
        *,
        source_leg_order: Optional[Sequence[str]] = None,
    ) -> None:
        """``source_leg_order``: the per-leg label order of the source
        file's four leg-chunks, e.g. ``("FR", "FL", "RR", "RL")``.

        - ``None`` (default, production path) — auto-detect each clip's
          leg order from its body-frame toe geometry and build a
          role-keyed permutation into canonical FL/FR/RL/RR. A clip whose
          mean footprint is undecodable raises (§8).
        - explicit tuple — used for clips whose geometry is degenerate
          (synthetic test fixtures). Validated against the canonical leg
          set; when the geometry *is* decodable it is also cross-checked
          and a disagreement warns loud.
        """
        self._source_leg_order: Optional[Tuple[str, ...]] = (
            _validate_leg_order(source_leg_order)
            if source_leg_order is not None
            else None
        )

    def load(
        self,
        path: Union[Path, str],
        *,
        target_sku: str,
        target_family: str,
    ) -> ReferenceMotionContract:
        p = Path(path)
        if not p.is_file():
            raise LoaderError(f"AMPLegGymLoader: file not found: {p}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
        except json.JSONDecodeError as exc:
            raise LoaderError(
                f"AMPLegGymLoader: {p} is not valid JSON: {exc}"
            ) from exc
        except OSError as exc:
            raise LoaderError(f"AMPLegGymLoader: cannot read {p}: {exc}") from exc

        frames_raw = j.get("Frames")
        if not isinstance(frames_raw, list) or len(frames_raw) < 2:
            raise LoaderError(
                f"AMPLegGymLoader: {p} has missing or too-short 'Frames' array"
            )

        arr = np.asarray(frames_raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.FRAME_DIM:
            raise LoaderError(
                f"AMPLegGymLoader: {p} frame shape is {arr.shape}, "
                f"expected (n, {self.FRAME_DIM})"
            )

        frame_duration = float(j.get("FrameDuration", 0.0))
        if frame_duration <= 0:
            raise LoaderError(
                f"AMPLegGymLoader: {p} has invalid FrameDuration={frame_duration}"
            )

        loop_mode = "wrap" if str(j.get("LoopMode", "Wrap")).lower() == "wrap" else "clamp"
        motion_weight = float(j.get("MotionWeight", 1.0))

        i = 0
        root_pos = arr[:, i:i + self.POS_SIZE]; i += self.POS_SIZE
        root_quat = arr[:, i:i + self.ROT_SIZE]; i += self.ROT_SIZE
        joint_pos = arr[:, i:i + self.JOINT_POS_SIZE]; i += self.JOINT_POS_SIZE
        toe_pos = arr[:, i:i + self.TOE_POS_SIZE]; i += self.TOE_POS_SIZE
        lin_vel = arr[:, i:i + self.LINEAR_VEL_SIZE]; i += self.LINEAR_VEL_SIZE
        ang_vel = arr[:, i:i + self.ANGULAR_VEL_SIZE]; i += self.ANGULAR_VEL_SIZE
        joint_vel = arr[:, i:i + self.JOINT_VEL_SIZE]; i += self.JOINT_VEL_SIZE
        toe_vel = arr[:, i:i + self.TOE_VEL_SIZE]; i += self.TOE_VEL_SIZE
        assert i == self.FRAME_DIM

        root_quat = _normalize_quats(root_quat)
        root_quat = _standardize_quats(root_quat)

        # ── Resolve the source file's leg order (role-keyed, NOT a
        # positional assumption) and reorder every per-leg array into
        # canonical FL/FR/RL/RR. Joints + feet share the same source leg
        # order in the amp_legged_gym layout, so one permutation covers
        # all four arrays. ───────────────────────────────────────────────
        src_order, leg_perm = self._resolve_leg_perm(toe_pos, clip_name=p.stem)
        if leg_perm != [0, 1, 2, 3]:
            joint_pos = _apply_leg_perm(joint_pos, leg_perm)
            toe_pos = _apply_leg_perm(toe_pos, leg_perm)
            joint_vel = _apply_leg_perm(joint_vel, leg_perm)
            toe_vel = _apply_leg_perm(toe_vel, leg_perm)

        amp_obs_dim = (
            self.JOINT_POS_SIZE + self.TOE_POS_SIZE + self.LINEAR_VEL_SIZE
            + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE + 1
        )

        clip = MotionClip(
            name=p.stem,
            format_id=self.format_id,
            fps=1.0 / frame_duration,
            loop_mode=loop_mode,
            motion_weight=motion_weight,
            task_tag=infer_task_tag(p.name),
            root_pos=root_pos.copy(),
            root_quat=root_quat.copy(),
            joint_pos=joint_pos.copy(),
            joint_vel=joint_vel.copy(),
            toe_pos_local=toe_pos.copy(),
            toe_vel_local=toe_vel.copy(),
            lin_vel=lin_vel.copy(),
            ang_vel=ang_vel.copy(),
            amp_obs_dim=amp_obs_dim,
            metadata={
                "source_path": str(p.resolve()),
                "frame_duration": frame_duration,
                "raw_loop_mode": str(j.get("LoopMode", "Wrap")),
                # Resolved provenance: which source leg order was used and
                # the permutation applied to reach canonical FL/FR/RL/RR.
                "source_leg_order": list(src_order),
                "leg_perm": list(leg_perm),
                # Derived: was a non-identity permutation applied. Kept for
                # the byte-identical baseline contract.
                "legs_reordered": leg_perm != [0, 1, 2, 3],
                # Self-describe joint channel IR roles so cross-robot
                # retargeting routes by IR rather than positional index.
                # After the canonical reorder the joint channels are pinned
                # to QUADRUPED_AMP_IR_ROLES (single source of truth — see
                # motion_ir_mapping.QUADRUPED_AMP_IR_ROLES).
                "ir_roles": list(QUADRUPED_AMP_IR_ROLES),
            },
        )
        # Identity-only normaliser today: with no override on disk for
        # this (source_format, target_sku) pair the call returns the
        # same MotionClip instance, leaving the contract byte-identical
        # to pre-normaliser output. A non-identity override (set up
        # through the alignment override directory) raises loud rather
        # than silently mis-translate.
        clip = normalize_motion_clip(clip, override=None)
        contract = ReferenceMotionContract(
            schema_version=CONTRACT_SCHEMA_VERSION,
            consumption_mode=self.default_consumption_mode,
            clip=clip,
            source_info=SourceInfo(source_path=str(p.resolve())),
            target_family=target_family,
            target_sku=target_sku,
        )
        validate_contract(contract)
        return contract

    # ------------------------------------------------------------------
    # Leg-order resolution
    # ------------------------------------------------------------------

    def _resolve_leg_perm(
        self, toe_pos: np.ndarray, *, clip_name: str
    ) -> Tuple[Tuple[str, ...], List[int]]:
        """Resolve ``(source_leg_order, perm_to_canonical)`` for one clip.

        Priority:
          1. Geometry — decode each leg-chunk's corner from mean body-frame
             toe XY (front=+x, left=+y). Authoritative for real clips.
          2. Explicit ``self._source_leg_order`` — used when geometry is
             undecodable (degenerate / synthetic clips). When geometry IS
             decodable it cross-checks the declaration and warns on
             disagreement.
          3. Neither resolves → raise (§8).
        """
        detected = _try_detect_source_leg_order(toe_pos)
        declared = self._source_leg_order

        # Explicit declaration is authoritative: trust it and skip the
        # geometry post-assert (used by callers whose clips have degenerate
        # / synthetic footprints). A decodable disagreement is surfaced as
        # a loud WARN so a wrong declaration on a real clip is still caught.
        if declared is not None:
            if detected is not None and detected != declared:
                print(
                    f"[UnitPort][AMP][WARN] {clip_name}: explicit "
                    f"source_leg_order={declared} disagrees with "
                    f"geometry-detected {detected}. Using the explicit "
                    f"declaration — verify it is correct (toe geometry is "
                    f"the ground truth for real locomotion clips).",
                    flush=True,
                )
            return declared, _leg_perm_to_canonical(declared)

        # Auto path: geometry is authoritative and required (§8).
        if detected is None:
            raise LoaderError(
                f"AMPLegGymLoader: cannot decode the source leg order of "
                f"clip {clip_name!r} from its toe geometry (mean foot "
                f"offsets do not form a clear front/rear × left/right "
                f"footprint; min |x|/|y| < {_LEG_DETECT_MIN_OFFSET} m or "
                f"two feet map to the same corner). Refusing to assume a "
                f"positional leg order (CLAUDE.md §8). Construct the loader "
                f"with an explicit source_leg_order=(...) declaring the "
                f"file's native leg order, e.g. ('FR','FL','RR','RL')."
            )

        perm = _leg_perm_to_canonical(detected)
        # §8 self-check: the reordered feet MUST land in canonical corners.
        reordered = _apply_leg_perm(toe_pos, perm) if perm != [0, 1, 2, 3] else toe_pos
        post = _try_detect_source_leg_order(reordered)
        if post != _CANONICAL_LEG_ORDER:
            raise LoaderError(
                f"AMPLegGymLoader: leg-order reorder of clip {clip_name!r} "
                f"did not yield canonical FL/FR/RL/RR (post-reorder geometry "
                f"= {post}, detected={detected}, perm={perm}). This is a "
                f"loader bug — the toe geometry after reorder must be "
                f"canonical."
            )
        return detected, perm


# ---------------------------------------------------------------------------
# Numeric helpers (private to the AMP layout)
# ---------------------------------------------------------------------------


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / (norms + 1e-12)


def _standardize_quats(quats: np.ndarray) -> np.ndarray:
    """Flip quaternions to the positive-w hemisphere (xyzw order).

    Re-exported via private import to :mod:`augment` (mirror op) — the
    only cross-module private use in this subpackage.
    """
    out = quats.copy()
    mask = out[:, 3] < 0.0
    out[mask] = -out[mask]
    return out


def _validate_leg_order(order: Sequence[str]) -> Tuple[str, ...]:
    """Validate an explicit source leg order against the canonical set.

    Raises :class:`LoaderError` if *order* is not a permutation of
    ``FL/FR/RL/RR`` (case-insensitive on input, normalised to upper).
    """
    norm = tuple(str(x).strip().upper() for x in order)
    if sorted(norm) != sorted(_CANONICAL_LEG_ORDER):
        raise LoaderError(
            f"AMPLegGymLoader: source_leg_order={tuple(order)!r} is not a "
            f"permutation of the canonical legs {_CANONICAL_LEG_ORDER}. "
            f"Provide each of FL/FR/RL/RR exactly once."
        )
    return norm


def _label_corner(x: float, y: float) -> Optional[str]:
    """Map a mean foot (x=forward, y=left) offset to a corner label.

    Returns ``None`` when either axis is inside the ambiguity band.
    """
    if abs(x) < _LEG_DETECT_MIN_OFFSET or abs(y) < _LEG_DETECT_MIN_OFFSET:
        return None
    return ("F" if x >= 0.0 else "R") + ("L" if y >= 0.0 else "R")


def _try_detect_source_leg_order(toe_pos: np.ndarray) -> Optional[Tuple[str, ...]]:
    """Decode the 4 leg-chunks' corner labels from body-frame toe geometry.

    ``toe_pos``: ``(n_frames, 12)`` = 4 feet × (x, y, z) in the clip's
    body frame (x forward, y left). Returns a 4-tuple like
    ``('FL','FR','RL','RR')`` or ``None`` when the footprint is
    undecodable (ambiguous corner or a corner used more than once).
    """
    if toe_pos.ndim != 2 or toe_pos.shape[1] != 12 or toe_pos.shape[0] < 1:
        return None
    feet = toe_pos.reshape(toe_pos.shape[0], 4, 3).mean(axis=0)  # (4, 3)
    labels: List[str] = []
    for f in range(4):
        lab = _label_corner(float(feet[f, 0]), float(feet[f, 1]))
        if lab is None:
            return None
        labels.append(lab)
    if sorted(labels) != sorted(_CANONICAL_LEG_ORDER):
        return None  # two feet collapsed onto the same corner
    return tuple(labels)


def _leg_perm_to_canonical(src_order: Sequence[str]) -> List[int]:
    """Indices that reorder the 4 source leg-chunks into canonical
    FL/FR/RL/RR order: ``out_leg[k] = src_leg[perm[k]]``."""
    src_index = {leg: i for i, leg in enumerate(src_order)}
    return [src_index[leg] for leg in _CANONICAL_LEG_ORDER]


def _apply_leg_perm(arr: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Apply a 4-leg permutation to a ``(n, 12)`` per-leg array.

    Reshapes to ``(n, 4, 3)`` (leg-major, 3 = within-leg channel:
    hip/thigh/calf for joints, x/y/z for feet), reorders the leg axis,
    and flattens back. Within-leg order is preserved.
    """
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise LoaderError(
            f"_apply_leg_perm: expected (n, 12) per-leg array, got "
            f"shape={arr.shape}"
        )
    n = arr.shape[0]
    return arr.reshape(n, 4, 3)[:, list(perm), :].reshape(n, 12)


__all__ = ["AMPLegGymLoader"]
