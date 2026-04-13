"""Motion loaders — format_id → MotionClip.

Two formats supported in phase_2:

- ``unitport_npy`` — legacy .npy layout used by the existing tracking
  reward path (``motion_library.py`` + ``unitree_gym_env
  ._init_reference_motion``). Each row is a joint-position frame. This
  loader is intentionally thin; it produces a tracking-only
  ``MotionClip`` with no AMP payload.

- ``amp_legged_gym`` — DeepMimic-style JSON/txt used by
  ``custom_mods/archives/AMP_for_hardware/datasets/mocap_motions/``
  and compatible repos. 61-float frames laid out as

    [root_pos(3) | root_quat(4) | joint_pos(12) | toe_pos_local(12)
     | lin_vel(3) | ang_vel(3) | joint_vel(12) | toe_vel_local(12)]

  This loader is vendored from
  ``custom_mods/archives/AMP_for_hardware/rsl_rl/rsl_rl/datasets/motion_loader.py``
  (BSD-3, NVIDIA + ETH Zurich). It produces a full AMP-capable
  ``MotionClip`` but drops the pytorch/pybullet dependencies — everything
  stays numpy so unit tests don't need GPU.

Per AMP_design.yaml §3.reference_motion.new_modules.loader.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from src.system.training.motion.clip import MotionClip, MotionClipError


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

    #: Stable identifier used by ``get_loader(format_id)`` and
    #: ``ReferenceMotionConfig.format_id``.
    format_id: str = ""

    @abstractmethod
    def load(self, path: Path | str) -> MotionClip:
        """Read *path* and return a populated ``MotionClip``.

        Raises ``LoaderError`` on any parse or structural failure. Does
        not run the phase_2 ``validator.validate_clip`` — that is the
        caller's responsibility (separation of concerns: loaders produce
        data, validator cross-checks against a robot asset).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# unitport_npy — legacy tracking format
# ---------------------------------------------------------------------------


class NpyLoader(MotionLoader):
    """Load a .npy file of shape ``(n_frames, dof)`` joint positions.

    This is the legacy path: same files ``motion_library.py`` already
    manages, same semantics as ``unitree_gym_env._init_reference_motion``.
    The resulting clip has **no AMP payload** (``has_amp_payload() ==
    False``) — it can only feed the tracking reward.

    The caller supplies ``fps`` explicitly (usually from
    ``ReferenceMotionConfig.motion_fps``) because .npy files don't carry
    timing metadata. Defaults to 50 Hz if not given.
    """

    format_id = "unitport_npy"

    def __init__(self, fps: float = 50.0) -> None:
        self._fps = float(fps)

    def load(self, path: Path | str) -> MotionClip:
        p = Path(path)
        if not p.is_file():
            raise LoaderError(f"NpyLoader: file not found: {p}")
        try:
            arr = np.load(str(p))
        except Exception as exc:
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
            joint_pos=arr,
            # No AMP payload — tracking-only clip
            amp_obs_dim=0,
            metadata={"source_path": str(p.resolve())},
        )


# ---------------------------------------------------------------------------
# amp_legged_gym — DeepMimic-style JSON used by AMP_for_hardware etc.
# ---------------------------------------------------------------------------


class AMPLegGymLoader(MotionLoader):
    """Load an amp_legged_gym ``.txt``/``.json`` motion file.

    Frame layout (61 floats, all loaders in the AMP_for_hardware family
    share this schema):

    ==========  =====  =======================================
    slice       size   contents
    ==========  =====  =======================================
    ``[0:3]``   3      root world xyz
    ``[3:7]``   4      root quaternion (pybullet order: x,y,z,w)
    ``[7:19]``  12     joint positions (FR, FL, RR, RL)
    ``[19:31]`` 12     local toe positions (FR, FL, RR, RL) ×3
    ``[31:34]`` 3      root linear velocity (world)
    ``[34:37]`` 3      root angular velocity (world)
    ``[37:49]`` 12     joint velocities (FR, FL, RR, RL)
    ``[49:61]`` 12     local toe velocities ×3
    ==========  =====  =======================================

    Legs are reordered at load time from the source's FR/FL/RR/RL
    convention to UnitPort's canonical FL/FR/RL/RR — this matches what
    the vendored AMPLoader already does for IsaacGym compatibility.
    Downstream consumers that need a different ordering must re-permute.

    ``MotionWeight`` and ``FrameDuration`` are read from the JSON header;
    ``LoopMode`` ``"Wrap"`` maps to ``"wrap"``, everything else to
    ``"clamp"``.
    """

    format_id = "amp_legged_gym"

    # Slice constants — mirror AMPLoader for bit-exact compatibility
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TOE_POS_SIZE = 12          # 4 feet × 3
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TOE_VEL_SIZE = 12          # 4 feet × 3

    FRAME_DIM = (
        POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TOE_POS_SIZE
        + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + TOE_VEL_SIZE
    )  # == 61

    def __init__(self, *, reorder_legs: bool = True) -> None:
        """
        Parameters
        ----------
        reorder_legs:
            Permute FR/FL/RR/RL → FL/FR/RL/RR at load time. Default True,
            which matches AMPLoader's behavior and the joint ordering
            used by a1/go2 in Isaac / legged_gym environments. Set to
            False if your robot asset was exported in the native
            pybullet order.
        """
        self._reorder_legs = reorder_legs

    def load(self, path: Path | str) -> MotionClip:
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
                f"AMPLegGymLoader: {p} has invalid FrameDuration="
                f"{frame_duration}"
            )

        loop_mode = "wrap" if str(j.get("LoopMode", "Wrap")).lower() == "wrap" else "clamp"
        motion_weight = float(j.get("MotionWeight", 1.0))

        # ── Slice into the eight canonical fields ──
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

        # ── Normalize + standardize quaternions ──
        root_quat = _normalize_quats(root_quat)
        root_quat = _standardize_quats(root_quat)

        # ── Leg reorder (FR/FL/RR/RL → FL/FR/RL/RR) ──
        if self._reorder_legs:
            joint_pos = _reorder_quad_legs(joint_pos)
            toe_pos = _reorder_quad_feet(toe_pos)
            joint_vel = _reorder_quad_legs(joint_vel)
            toe_vel = _reorder_quad_feet(toe_vel)

        # ── AMP observation dim ──
        # joint_pos(12) + toe_pos(12) + lin_vel(3) + ang_vel(3) + joint_vel(12) + root_z(1) = 43
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
    """Factory: return a loader instance for *format_id*.

    ``**kwargs`` are forwarded to the concrete loader's constructor —
    e.g. ``get_loader('unitport_npy', fps=30)``.

    Unknown ``format_id`` raises ``LoaderError``.
    """
    cls = _LOADER_CLASSES.get(format_id)
    if cls is None:
        raise LoaderError(
            f"Unknown motion format_id '{format_id}'. "
            f"Known formats: {sorted(_LOADER_CLASSES)}"
        )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    """Normalize a ``(n, 4)`` batch of quaternions."""
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / (norms + 1e-12)


def _standardize_quats(quats: np.ndarray) -> np.ndarray:
    """Flip quaternions to the positive-w hemisphere.

    Matches ``motion_util.standardize_quaternion`` in AMP_for_hardware:
    a quaternion ``q`` and ``-q`` represent the same rotation, so by
    convention we keep the one with ``w >= 0`` to avoid discontinuities
    when interpolating or training a discriminator.
    """
    out = quats.copy()
    # x, y, z, w ordering
    mask = out[:, 3] < 0.0
    out[mask] = -out[mask]
    return out


def _reorder_quad_legs(arr: np.ndarray) -> np.ndarray:
    """Permute a ``(n, 12)`` joint array from FR/FL/RR/RL to FL/FR/RL/RR.

    Source rsl_rl implementation:

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(joints, 4, axis=1)
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])
    """
    if arr.shape[1] != 12:
        raise LoaderError(
            f"_reorder_quad_legs: expected 12 dof, got {arr.shape[1]}"
        )
    fr, fl, rr, rl = np.split(arr, 4, axis=1)
    return np.hstack([fl, fr, rl, rr])


def _reorder_quad_feet(arr: np.ndarray) -> np.ndarray:
    """Same permutation for 4-feet × 3-axis foot arrays (shape ``(n, 12)``)."""
    if arr.shape[1] != 12:
        raise LoaderError(
            f"_reorder_quad_feet: expected 12 (= 4 feet × 3), got {arr.shape[1]}"
        )
    fr, fl, rr, rl = np.split(arr, 4, axis=1)
    return np.hstack([fl, fr, rl, rr])
