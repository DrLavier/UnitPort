"""MotionClip — in-memory representation of a reference motion trajectory.

DIRECT-MIGRATE from DEMO ``src/system/training/motion/clip.py`` with one
behavioural change: ``_sample_amp_obs_batch`` deferred to Stage 8 (the
DEMO version dispatched into ``amp_obs_terms``, a registry that lives
on the SB3-AMP port path Stage 8 will land). Tracking-path callers
(``frame_at``) keep working without that dependency.

Storage is **numpy only** at this layer. Tensor conversion for AMP
discriminator training happens in Stage 8 ``amp.data_provider``; the
clip layer stays torch-free so unit tests don't need a GPU/CUDA stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


class MotionClipError(ValueError):
    """Raised on structural problems within a MotionClip (NaN/Inf,
    mismatched shapes, missing AMP payload). ``MotionValidationError``
    in :mod:`validator` wraps these for user-facing messages."""


@dataclass
class MotionClip:
    """A reference motion trajectory in a single canonical numpy layout.

    All arrays are shaped ``(n_frames, ...)``. Missing fields are
    ``None`` (e.g. a legacy .npy tracking clip has no toe positions).
    """

    name: str
    format_id: str
    fps: float
    loop_mode: str = "wrap"
    motion_weight: float = 1.0
    task_tag: str = ""

    root_pos: Optional[np.ndarray] = None
    root_quat: Optional[np.ndarray] = None
    joint_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    joint_vel: Optional[np.ndarray] = None
    toe_pos_local: Optional[np.ndarray] = None
    toe_vel_local: Optional[np.ndarray] = None
    lin_vel: Optional[np.ndarray] = None
    ang_vel: Optional[np.ndarray] = None

    amp_obs_dim: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.joint_pos.shape[0]) if self.joint_pos.size else 0

    @property
    def dof(self) -> int:
        if self.joint_pos.size == 0:
            return 0
        return int(self.joint_pos.shape[1])

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.frame_dt

    def has_amp_payload(self) -> bool:
        """True iff this clip can produce discriminator transitions."""
        return self.amp_obs_dim > 0 and self.joint_vel is not None

    # ------------------------------------------------------------------
    # Frame access (tracking + preview)
    # ------------------------------------------------------------------

    def frame_at(self, t: float) -> Dict[str, np.ndarray]:
        """Return the frame at time ``t`` seconds with linear interpolation.

        Out-of-range times handled by ``loop_mode``: ``"wrap"`` cycles,
        ``"clamp"`` clamps to ``[0, duration]``.
        """
        if self.n_frames == 0 or self.fps <= 0:
            raise MotionClipError(
                f"MotionClip '{self.name}' is empty or has invalid fps"
            )

        dur = self.duration_s
        if self.loop_mode == "wrap":
            t = t % dur if dur > 0 else 0.0
        else:
            t = max(0.0, min(t, dur - self.frame_dt))

        idx_f = t / self.frame_dt
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, self.n_frames - 1)
        alpha = float(idx_f - i0)

        out: Dict[str, np.ndarray] = {
            "joint_pos": _lerp(self.joint_pos[i0], self.joint_pos[i1], alpha),
        }
        if self.root_pos is not None:
            out["root_pos"] = _lerp(self.root_pos[i0], self.root_pos[i1], alpha)
        if self.root_quat is not None:
            out["root_quat"] = _quat_slerp(self.root_quat[i0], self.root_quat[i1], alpha)
        if self.joint_vel is not None:
            out["joint_vel"] = _lerp(self.joint_vel[i0], self.joint_vel[i1], alpha)
        if self.toe_pos_local is not None:
            out["toe_pos_local"] = _lerp(
                self.toe_pos_local[i0], self.toe_pos_local[i1], alpha
            )
        if self.toe_vel_local is not None:
            out["toe_vel_local"] = _lerp(
                self.toe_vel_local[i0], self.toe_vel_local[i1], alpha
            )
        if self.lin_vel is not None:
            out["lin_vel"] = _lerp(self.lin_vel[i0], self.lin_vel[i1], alpha)
        if self.ang_vel is not None:
            out["ang_vel"] = _lerp(self.ang_vel[i0], self.ang_vel[i1], alpha)
        return out

    # ------------------------------------------------------------------
    # Transition sampling (AMP — Stage 8 deferred)
    # ------------------------------------------------------------------

    def sample_transitions(
        self,
        n: int,
        dt: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` random ``(s_t, s_{t+dt})`` pairs for AMP training.

        Stage 5 contract: validates pre-conditions (payload, dt, duration)
        and computes the time pairs. The actual obs vector layout is
        delegated to :meth:`_sample_amp_obs_batch`, which is implemented
        by Stage 8 once the ``amp_obs_terms`` registry lands. Calling
        this method on a clip without ``has_amp_payload()`` raises.
        """
        if not self.has_amp_payload():
            raise MotionClipError(
                f"MotionClip '{self.name}' does not carry an AMP payload "
                f"(joint_vel/toe_pos/lin_vel/ang_vel missing). Use "
                f"format_id='amp_legged_gym' for AMP consumption."
            )
        if dt <= 0:
            raise MotionClipError(f"sample_transitions: dt must be > 0, got {dt}")
        if self.duration_s <= dt:
            raise MotionClipError(
                f"MotionClip '{self.name}' is too short ({self.duration_s:.3f}s) "
                f"for dt={dt}s transition sampling"
            )
        if rng is None:
            rng = np.random.default_rng()
        max_t = self.duration_s - dt
        times = rng.uniform(0.0, max_t, size=n)
        obs_t = self._sample_amp_obs_batch(times)
        obs_tp1 = self._sample_amp_obs_batch(times + dt)
        return obs_t, obs_tp1

    def _sample_amp_obs_batch(
        self,
        times: np.ndarray,
        term_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Vectorised AMP observation lookup for *N* times.

        Stage 8 wired this to dispatch through the
        :mod:`application.training.amp.obs_terms` registry. Default
        layout = ``DEFAULT_QUADRUPED_TERMS`` (43-dim
        ``joint_pos|toe_pos|lin_vel|ang_vel|joint_vel|root_z`` for
        a 12-DoF quadruped).
        """
        times = np.asarray(times, dtype=np.float64)
        # Loop-mode wrapping is MotionClip's responsibility — the registry
        # assumes the caller has already clamped times to the valid range.
        if self.loop_mode == "wrap":
            times = np.mod(times, max(self.duration_s, 1e-9))
        else:
            times = np.clip(
                times, 0.0, max(self.duration_s - self.frame_dt, 0.0)
            )
        # Lazy import — keeps motion/clip.py importable without the AMP
        # subpackage's torch dependency.
        from application.training.amp.obs_terms import (
            DEFAULT_QUADRUPED_TERMS,
            compute_amp_obs_from_motion,
        )

        names = list(term_names) if term_names is not None else DEFAULT_QUADRUPED_TERMS
        return compute_amp_obs_from_motion(names, self, times)

    def _amp_obs_at(self, t: float) -> np.ndarray:
        """Single-time AMP observation — Stage 8 deferred (see above)."""
        return self._sample_amp_obs_batch(np.asarray([t], dtype=np.float64))[0]


# ---------------------------------------------------------------------------
# Internal numeric helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Minimal SLERP between unit quaternions in ``(x, y, z, w)`` order.

    Falls back to LERP + renormalize when the two quaternions are very
    close. Adequate for preview rendering and AMP interpolation; not a
    general-purpose quaternion library.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_t0 = np.sin(theta_0)
    sin_t = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_t / sin_t0
    s1 = sin_t / sin_t0
    return s0 * q0 + s1 * q1


__all__ = ["MotionClip", "MotionClipError"]
