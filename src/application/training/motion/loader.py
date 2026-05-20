"""Motion loaders — ``format_id → MotionClip``.

DIRECT-MIGRATE from DEMO ``src/system/training/motion/loader.py``. Two
formats are supported:

- ``unitport_npy`` — legacy ``.npy`` rows of joint positions; produces a
  tracking-only ``MotionClip`` with no AMP payload.
- ``amp_legged_gym`` — DeepMimic-style 61-float JSON/txt (root pose +
  joints + toes + velocities). Produces an AMP-capable clip and reorders
  legs FR/FL/RR/RL → FL/FR/RL/RR to match UnitPort's canonical ordering.

I/O note: these loaders read user-supplied paths (project tree / user
data dir). ``np.load`` and ``json.load`` are the right tools — wrapping
them in ``DataManager`` would round-trip arrays through dtype conversions
the SDK doesn't model. Stays consistent with DataManager's intent
(per-extension dispatch for SDK-managed files; arbitrary user paths can
use the canonical Python loaders).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from application.training.motion.clip import MotionClip
from application.training.motion.labels import infer_task_tag


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LoaderError(ValueError):
    """Raised when a motion file cannot be parsed by the requested loader."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MotionLoader(ABC):
    """Abstract ``(path) → MotionClip`` contract."""

    #: Stable identifier used by :func:`get_loader` and
    #: ``ReferenceMotionConfig.format_id``.
    format_id: str = ""

    @abstractmethod
    def load(self, path: Union[Path, str]) -> MotionClip:
        """Read ``path`` and return a populated :class:`MotionClip`.

        Raises :exc:`LoaderError` on any parse or structural failure.
        Does NOT run :func:`validate_clip` — that is the caller's
        responsibility (separation of concerns: loaders produce data,
        the validator cross-checks against a robot spec).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# unitport_npy — legacy tracking format
# ---------------------------------------------------------------------------


class NpyLoader(MotionLoader):
    """Load a ``.npy`` file of shape ``(n_frames, dof)`` joint positions.

    The resulting clip has **no AMP payload** (``has_amp_payload() ==
    False``) — only the tracking reward consumes it. The caller supplies
    ``fps`` explicitly because .npy files don't carry timing metadata.
    """

    format_id = "unitport_npy"

    def __init__(self, fps: float = 50.0) -> None:
        self._fps = float(fps)

    def load(self, path: Union[Path, str]) -> MotionClip:
        p = Path(path)
        if not p.is_file():
            raise LoaderError(f"NpyLoader: file not found: {p}")
        try:
            arr = np.load(str(p))
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(f"NpyLoader: np.load failed for {p}: {exc}") from exc

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise LoaderError(
                f"NpyLoader: expected 2-D array (n_frames, dof), "
                f"got shape {arr.shape} in {p}"
            )
        if arr.shape[0] < 2:
            raise LoaderError(
                f"NpyLoader: motion too short ({arr.shape[0]} frames) in {p}"
            )

        return MotionClip(
            name=p.stem,
            format_id=self.format_id,
            fps=self._fps,
            loop_mode="wrap",
            motion_weight=1.0,
            task_tag=infer_task_tag(p.name),
            joint_pos=arr,
            amp_obs_dim=0,
            metadata={"source_path": str(p.resolve())},
        )


# ---------------------------------------------------------------------------
# amp_legged_gym — DeepMimic-style JSON used by AMP_for_hardware etc.
# ---------------------------------------------------------------------------


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

    # Stage 4: clip joint channel → IR role mapping (post leg-reorder).
    # Pinned with motion_ir_mapping.QUADRUPED_AMP_IR_ROLES; kept inline
    # here to keep loader.py importable without the IR mapping module.
    _AMP_LEGGED_GYM_IR_ROLES = [
        "hip_FL",   "thigh_FL", "calf_FL",
        "hip_FR",   "thigh_FR", "calf_FR",
        "hip_RL",   "thigh_RL", "calf_RL",
        "hip_RR",   "thigh_RR", "calf_RR",
    ]

    FRAME_DIM = (
        POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TOE_POS_SIZE
        + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + TOE_VEL_SIZE
    )  # == 61

    def __init__(self, *, reorder_legs: bool = True) -> None:
        self._reorder_legs = reorder_legs

    def load(self, path: Union[Path, str]) -> MotionClip:
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

        return MotionClip(
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
                # Stage 4: self-describe joint channel IR roles so
                # cross-robot retargeting can route by IR rather than
                # by positional index. amp_legged_gym format pins to
                # QUADRUPED_AMP_IR_ROLES after the leg reorder.
                "ir_roles": list(self._AMP_LEGGED_GYM_IR_ROLES) if self._reorder_legs else [],
            },
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_LOADER_CLASSES = {
    "unitport_npy": NpyLoader,
    "amp_legged_gym": AMPLegGymLoader,
}


def get_loader(format_id: str, **kwargs) -> MotionLoader:
    """Factory: return a loader instance for ``format_id``.

    ``**kwargs`` are forwarded to the concrete loader's constructor.
    Unknown ``format_id`` raises :exc:`LoaderError`.
    """
    cls = _LOADER_CLASSES.get(format_id)
    if cls is None:
        raise LoaderError(
            f"Unknown motion format_id {format_id!r}. "
            f"Known formats: {sorted(_LOADER_CLASSES)}"
        )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / (norms + 1e-12)


def _standardize_quats(quats: np.ndarray) -> np.ndarray:
    """Flip quaternions to the positive-w hemisphere (xyzw order)."""
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


# ---------------------------------------------------------------------------
# Left/right mirror augmentation
# ---------------------------------------------------------------------------

_QUAD_LR_SWAP_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
_QUAD_HIP_INDICES = (0, 3, 6, 9)
_QUAD_FOOT_Y_INDICES = (1, 4, 7, 10)


def mirror_motion_clip(clip: MotionClip) -> MotionClip:
    """Reflect a quadruped MotionClip across the sagittal plane.

    Returns a new clip suffixed ``"_mirror"`` — input is not mutated.
    Only valid for 12-DoF FL/FR/RL/RR clips (typically produced by
    :class:`AMPLegGymLoader`).
    """
    if clip.joint_pos.shape[1] != 12:
        raise LoaderError(
            f"mirror_motion_clip: only 12-dof quadruped clips are supported, "
            f"got dof={clip.joint_pos.shape[1]} for {clip.name!r}."
        )

    swap = _QUAD_LR_SWAP_PERM
    hip_idx = list(_QUAD_HIP_INDICES)
    foot_y_idx = list(_QUAD_FOOT_Y_INDICES)

    new_joint_pos = clip.joint_pos[:, swap].copy()
    new_joint_pos[:, hip_idx] *= -1.0

    if clip.joint_vel is not None:
        new_joint_vel = clip.joint_vel[:, swap].copy()
        new_joint_vel[:, hip_idx] *= -1.0
    else:
        new_joint_vel = None

    if clip.toe_pos_local is not None and clip.toe_pos_local.shape[1] == 12:
        new_toe_pos = clip.toe_pos_local[:, swap].copy()
        new_toe_pos[:, foot_y_idx] *= -1.0
    else:
        new_toe_pos = None

    if clip.toe_vel_local is not None and clip.toe_vel_local.shape[1] == 12:
        new_toe_vel = clip.toe_vel_local[:, swap].copy()
        new_toe_vel[:, foot_y_idx] *= -1.0
    else:
        new_toe_vel = None

    if clip.root_pos is not None and clip.root_pos.shape[1] == 3:
        new_root_pos = clip.root_pos.copy()
        new_root_pos[:, 1] *= -1.0
    else:
        new_root_pos = None

    if clip.root_quat is not None and clip.root_quat.shape[1] == 4:
        new_root_quat = clip.root_quat.copy()
        new_root_quat[:, 0] *= -1.0
        new_root_quat[:, 2] *= -1.0
        new_root_quat = _standardize_quats(new_root_quat)
    else:
        new_root_quat = None

    if clip.lin_vel is not None and clip.lin_vel.shape[1] == 3:
        new_lin_vel = clip.lin_vel.copy()
        new_lin_vel[:, 1] *= -1.0
    else:
        new_lin_vel = None

    if clip.ang_vel is not None and clip.ang_vel.shape[1] == 3:
        new_ang_vel = clip.ang_vel.copy()
        new_ang_vel[:, 0] *= -1.0
        new_ang_vel[:, 2] *= -1.0
    else:
        new_ang_vel = None

    new_metadata = dict(clip.metadata)
    new_metadata["mirrored_from"] = clip.name

    return MotionClip(
        name=clip.name + "_mirror",
        format_id=clip.format_id,
        fps=clip.fps,
        loop_mode=clip.loop_mode,
        motion_weight=clip.motion_weight,
        task_tag=clip.task_tag,
        root_pos=new_root_pos,
        root_quat=new_root_quat,
        joint_pos=new_joint_pos,
        joint_vel=new_joint_vel,
        toe_pos_local=new_toe_pos,
        toe_vel_local=new_toe_vel,
        lin_vel=new_lin_vel,
        ang_vel=new_ang_vel,
        amp_obs_dim=clip.amp_obs_dim,
        metadata=new_metadata,
    )


__all__ = [
    "MotionLoader",
    "NpyLoader",
    "AMPLegGymLoader",
    "get_loader",
    "LoaderError",
    "mirror_motion_clip",
]
