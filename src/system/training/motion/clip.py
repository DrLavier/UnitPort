"""MotionClip — in-memory representation of a reference motion trajectory.

A ``MotionClip`` is format-agnostic: the concrete ``MotionLoader``
subclasses produce it by parsing their respective file formats
(amp_legged_gym ``.txt/.json``, legacy ``.npy``, …) and filling in
whichever fields they can extract. Downstream consumers (AMP data
provider, tracking reward, preview panel) read the fields they need
and tolerate missing ones.

Storage is **numpy only** at this layer. Tensor conversion for
discriminator training happens in the phase_3 ``amp.data_provider``
bridge — the clip layer stays torch-free so unit tests don't need
a GPU / CUDA toolchain.

Per AMP_design.yaml §3.reference_motion.new_modules.clip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class MotionClipError(ValueError):
    """Raised on structural problems within a MotionClip (NaN/Inf,
    mismatched array shapes, etc.). ``MotionValidationError`` in
    ``validator.py`` wraps these for user-facing messages."""


@dataclass
class MotionClip:
    """A reference motion trajectory in a single canonical numpy layout.

    All arrays are shaped ``(n_frames, ...)``. Missing fields are
    ``None`` (e.g. a legacy .npy tracking clip has no toe positions).

    Attributes
    ----------
    name:
        Human-readable clip name (file stem by default). Used in logs
        and preview panels.
    format_id:
        Which ``MotionLoader`` produced this clip. One of
        ``"amp_legged_gym"`` or ``"unitport_npy"``. Extensible.
    fps:
        Playback rate of the source file. For amp_legged_gym, this is
        ``1 / FrameDuration``; for .npy clips, the user provides it via
        ``ReferenceMotionConfig.motion_fps``.
    frame_dt:
        ``1 / fps`` — cached for convenience.
    loop_mode:
        ``"wrap"`` (cycle at end) or ``"clamp"`` (stop at last frame).
    motion_weight:
        Sampling weight relative to other clips in the same dataset
        (0.5 default). Only used if multiple clips are sampled together.
    root_pos:
        ``(n_frames, 3)`` root xyz. Optional.
    root_quat:
        ``(n_frames, 4)`` root orientation as unit quaternion in
        ``(x, y, z, w)`` order. Optional.
    joint_pos:
        ``(n_frames, dof)`` joint positions in canonical order. Required
        for both tracking and AMP paths.
    joint_vel:
        ``(n_frames, dof)`` joint velocities. Optional (AMP needs it;
        tracking can finite-difference at runtime).
    toe_pos_local:
        ``(n_frames, 3 * n_feet)`` local toe positions, optional
        (amp_legged_gym format only).
    toe_vel_local:
        ``(n_frames, 3 * n_feet)`` local toe velocities, optional.
    lin_vel:
        ``(n_frames, 3)`` root linear velocity (world frame). Optional.
    ang_vel:
        ``(n_frames, 3)`` root angular velocity (world frame). Optional.
    amp_obs_dim:
        Size of the discriminator observation vector this clip supports.
        For amp_legged_gym quadrupeds this is 43
        (joint_pos(12) + toe_pos(12) + lin_vel(3) + ang_vel(3) +
        joint_vel(12) + root_z(1)). For clips that only have joint_pos
        (e.g. a tracking-only .npy), it's 0 — AMP can't consume them.
    metadata:
        Free-form dict the loader can stash extras in (source path,
        original pybullet leg order, etc.). Not contractual.
    """

    name: str
    format_id: str
    fps: float
    loop_mode: str = "wrap"
    motion_weight: float = 1.0

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
    # Frame access
    # ------------------------------------------------------------------

    def frame_at(self, t: float) -> Dict[str, np.ndarray]:
        """Return the frame at time ``t`` seconds with linear interpolation.

        Out-of-range times are handled by the ``loop_mode``:

        - ``"wrap"``: ``t % duration`` (cycles)
        - ``"clamp"``: clamped to ``[0, duration]``

        Returns a dict of only the fields that exist on this clip. A
        tracking-only .npy clip will return ``{"joint_pos": ...}``; an
        amp_legged_gym clip will return all eight.
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
            out["toe_pos_local"] = _lerp(self.toe_pos_local[i0], self.toe_pos_local[i1], alpha)
        if self.toe_vel_local is not None:
            out["toe_vel_local"] = _lerp(self.toe_vel_local[i0], self.toe_vel_local[i1], alpha)
        if self.lin_vel is not None:
            out["lin_vel"] = _lerp(self.lin_vel[i0], self.lin_vel[i1], alpha)
        if self.ang_vel is not None:
            out["ang_vel"] = _lerp(self.ang_vel[i0], self.ang_vel[i1], alpha)
        return out

    # ------------------------------------------------------------------
    # Transition sampling (AMP)
    # ------------------------------------------------------------------

    def sample_transitions(
        self,
        n: int,
        dt: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` random ``(s_t, s_{t+dt})`` pairs for AMP training.

        Requires ``has_amp_payload() == True`` — clips that only carry
        joint positions cannot produce discriminator transitions, and
        this method raises ``MotionClipError`` on them.

        The observation vector layout (amp_legged_gym compatible):

            [joint_pos(dof) | toe_pos(12) | lin_vel(3) | ang_vel(3)
             | joint_vel(dof) | root_z(1)]

        Returns
        -------
        (obs_t, obs_tp1) : tuple of np.ndarray
            Each shape ``(n, amp_obs_dim)``. Float32.
        """
        if not self.has_amp_payload():
            raise MotionClipError(
                f"MotionClip '{self.name}' does not carry an AMP payload "
                f"(joint_vel / toe_pos / lin_vel / ang_vel missing). "
                f"Use format_id='amp_legged_gym' for AMP consumption."
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

        B5 (AMP fix plan): this method is now a thin wrapper around
        ``src.system.training.amp_obs_terms.compute_amp_obs_from_motion``.
        The registry is the single source of truth for *what* an AMP
        obs vector contains — both this producer (motion loader) and
        the env-side extractor in ``il_train_launcher`` dispatch
        through the same term table, so they cannot drift.

        Parameters
        ----------
        times:
            1-D array of shape ``(N,)`` of times in seconds.
        term_names:
            Optional override of the term list. When ``None`` the
            legacy ``DEFAULT_QUADRUPED_TERMS`` layout is used
            (``joint_pos, toe_pos_local, base_lin_vel_local,
            base_ang_vel_local, joint_vel, root_height``), which
            reproduces the pre-B5 43-dim amp_legged_gym layout
            bit-for-bit.

        Returns
        -------
        ``(N, amp_obs_dim)`` float32 array.
        """
        times = np.asarray(times, dtype=np.float64)

        # Loop-mode wrapping is still MotionClip's responsibility —
        # it depends on the clip's duration and loop_mode attribute.
        # The amp_obs_terms dispatch assumes the caller has already
        # clamped times to a valid range.
        if self.loop_mode == "wrap":
            times = np.mod(times, max(self.duration_s, 1e-9))
        else:
            times = np.clip(times, 0.0, max(self.duration_s - self.frame_dt, 0.0))

        # Lazy import keeps motion/clip.py importable without paying
        # the registry's import cost at module load.
        from src.system.training.amp_obs_terms import (
            DEFAULT_QUADRUPED_TERMS,
            compute_amp_obs_from_motion,
        )

        names = list(term_names) if term_names is not None else DEFAULT_QUADRUPED_TERMS
        return compute_amp_obs_from_motion(names, self, times)

    def _amp_obs_at(self, t: float) -> np.ndarray:
        """Single-time AMP observation — kept for the preview button.

        Hot path uses :meth:`_sample_amp_obs_batch` instead. This method
        is just a thin wrapper that builds a 1-element batch and squeezes.
        """
        return self._sample_amp_obs_batch(np.asarray([t], dtype=np.float64))[0]


# ---------------------------------------------------------------------------
# Internal numeric helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Minimal SLERP between two unit quaternions in ``(x, y, z, w)`` order.

    Falls back to LERP + renormalize when the two quaternions are very
    close (avoids the divide-by-sin-near-zero). This is good enough for
    preview rendering and AMP interpolation; it is NOT meant to replace
    a robust quaternion library in downstream math.
    """
    dot = float(np.dot(q0, q1))
    # Shortest-arc: flip one if they point away from each other.
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
