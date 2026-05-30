# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``amp_legged_gym`` — DeepMimic-style 61-float JSON loader.

Reads the JSON/text format used by AMP_for_hardware and the wider
legged_gym AMP family. Each frame has 61 floats laid out as::

    [root_pos(3) | root_quat(4, xyzw) | joint_pos(12) | toe_pos(12)
     | lin_vel(3) | ang_vel(3) | joint_vel(12) | toe_vel(12)]

Legs are reordered at load time from the source's FR/FL/RR/RL
convention to UnitPort's canonical FL/FR/RL/RR. Downstream consumers
that need a different ordering must re-permute.

Numeric helpers (``_normalize_quats`` / ``_standardize_quats`` /
``_reorder_quad_legs`` / ``_reorder_quad_feet``) live in this module
because they are private to the AMP layout. ``_standardize_quats`` is
re-used by :mod:`augment` for the mirror op — that is the only
cross-module private import in this subpackage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

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


class AMPLegGymLoader(MotionLoader):
    """Load an amp_legged_gym ``.txt``/``.json`` motion file.

    Frame layout (61 floats; shared across the AMP_for_hardware family):

        [root_pos(3) | root_quat(4, xyzw) | joint_pos(12) | toe_pos(12)
         | lin_vel(3) | ang_vel(3) | joint_vel(12) | toe_vel(12)]

    Legs are reordered at load time from the source's FR/FL/RR/RL
    convention to UnitPort's canonical FL/FR/RL/RR. Downstream consumers
    that need a different ordering must re-permute.

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

    def __init__(self, *, reorder_legs: bool = True) -> None:
        self._reorder_legs = reorder_legs

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

        if self._reorder_legs:
            joint_pos = _reorder_quad_legs(joint_pos)
            toe_pos = _reorder_quad_feet(toe_pos)
            joint_vel = _reorder_quad_legs(joint_vel)
            toe_vel = _reorder_quad_feet(toe_vel)

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
                "legs_reordered": self._reorder_legs,
                # Self-describe joint channel IR roles so cross-robot
                # retargeting routes by IR rather than positional index.
                # amp_legged_gym pins to QUADRUPED_AMP_IR_ROLES after
                # the leg reorder; sourced from motion_ir_mapping (single
                # source of truth — see motion_ir_mapping.QUADRUPED_AMP_IR_ROLES).
                "ir_roles": list(QUADRUPED_AMP_IR_ROLES) if self._reorder_legs else [],
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


def _reorder_quad_legs(arr: np.ndarray) -> np.ndarray:
    """Permute ``(n, 12)`` joint array from FR/FL/RR/RL to FL/FR/RL/RR.

    Stays bit-identical to the policy-side env extractor (Stage 8 will
    pin the same permutation in ``amp.obs_terms``). Strict shape gate
    raises :class:`LoaderError` rather than asserting — the
    :func:`validate_clip_against_robot` IR-role gate covers the
    semantic match against the canvas robot (CLAUDE.md §1.2).
    """
    if arr.ndim != 2:
        raise LoaderError(
            f"_reorder_quad_legs: expected 2-D (n_frames, 12), got shape={arr.shape}"
        )
    if arr.shape[1] != 12:
        raise LoaderError(
            f"_reorder_quad_legs: expected 12 dof (4 legs × 3 joints, "
            f"hip/thigh/calf), got {arr.shape[1]}; non-quadruped clips must "
            f"not pass through this reorder"
        )
    if arr.shape[0] < 1:
        raise LoaderError(f"_reorder_quad_legs: empty frame axis (shape={arr.shape})")
    fr, fl, rr, rl = np.split(arr, 4, axis=1)
    return np.hstack([fl, fr, rl, rr])


def _reorder_quad_feet(arr: np.ndarray) -> np.ndarray:
    """Same permutation for 4-feet × 3-axis foot arrays (shape ``(n, 12)``)."""
    if arr.ndim != 2:
        raise LoaderError(
            f"_reorder_quad_feet: expected 2-D (n_frames, 12), got shape={arr.shape}"
        )
    if arr.shape[1] != 12:
        raise LoaderError(
            f"_reorder_quad_feet: expected 12 (= 4 feet × 3 axes), got "
            f"{arr.shape[1]}"
        )
    fr, fl, rr, rl = np.split(arr, 4, axis=1)
    return np.hstack([fl, fr, rl, rr])


__all__ = ["AMPLegGymLoader"]
