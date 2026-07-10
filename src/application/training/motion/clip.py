# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
    def ir_roles(self) -> list:
        """IR role for each clip joint channel (parallel to ``joint_pos``).

        Loaders fill ``metadata["ir_roles"]`` with the canonical role list
        for their format (e.g. ``QUADRUPED_AMP_IR_ROLES`` for
        ``amp_legged_gym``). Falls back to looking up the format default
        when a legacy clip has no explicit metadata entry; returns ``[]``
        for formats without a portable IR contract (``unitport_npy``).
        """
        roles = self.metadata.get("ir_roles")
        if isinstance(roles, list) and roles:
            return list(roles)
        # Fallback: look up the format's default IR roles (avoids importing
        # motion_ir_mapping at module-load time — kept as a lazy import).
        try:
            from application.training.motion_ir_mapping import get_ir_roles
            return get_ir_roles(self.format_id)
        except Exception:
            return []

    def to_robot_joint_order(
        self,
        joints_role_map: Dict[str, str],
        *,
        what: str = "joint_pos",
    ) -> np.ndarray:
        """Reorder the clip's joint channels to match a robot's joint order.

        Parameters
        ----------
        joints_role_map
            ``{joint_name: ir_role}`` for the target robot in the format
            it will be loaded under (``RobotSpecRef.joints_role_map_for(
            active_format)``).
        what
            Which clip array to reorder. Either ``"joint_pos"`` (default)
            or ``"joint_vel"`` (when present).

        Returns
        -------
        np.ndarray
            ``(n_frames, n_joints)`` reordered so column *i* corresponds
            to the robot's *i*-th joint in ``joints_role_map`` insertion
            order. Joints whose IR role doesn't appear in this clip get
            ``0.0``.
        """
        src = self.joint_pos if what == "joint_pos" else self.joint_vel
        if src is None or src.size == 0:
            return np.empty((0, 0), dtype=np.float64)

        clip_roles = self.ir_roles
        if not clip_roles:
            raise MotionClipError(
                f"MotionClip '{self.name}' format {self.format_id!r} has no "
                f"IR-role metadata — cannot reorder to robot joint layout. "
                f"Make sure the loader fills metadata['ir_roles']."
            )
        if len(clip_roles) != src.shape[1]:
            raise MotionClipError(
                f"MotionClip '{self.name}' joint_pos has {src.shape[1]} "
                f"channels but ir_roles declares {len(clip_roles)} — "
                f"clip metadata is inconsistent."
            )

        role_to_col = {r: i for i, r in enumerate(clip_roles)}
        out = np.zeros((src.shape[0], len(joints_role_map)), dtype=src.dtype)
        for j_idx, (_jname, ir_role) in enumerate(joints_role_map.items()):
            col = role_to_col.get(ir_role)
            if col is not None:
                out[:, j_idx] = src[:, col]
        return out

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.frame_dt

    def has_amp_payload(self) -> bool:
        """True iff this clip can produce discriminator transitions."""
        return self.amp_obs_dim > 0 and self.joint_vel is not None

    def subrange(self, start: int, end: int) -> "MotionClip":
        """Return a new clip covering the **inclusive** frame range ``[start, end]``.

        Slices every populated ``(n_frames, …)`` array to ``[start : end+1]``
        (a copy — the parent is untouched) and preserves everything else:
        ``format_id`` / ``fps`` / ``loop_mode`` / ``amp_obs_dim`` and the full
        ``metadata`` dict (so ``ir_roles`` survive → the sub-clip still reorders
        to a robot joint layout). This is the single frame-slicing primitive
        behind the Clip Motion Editor's "segment" feature: a segment is stored
        as ``(clip_ref, start, end)`` and materialised here at load time, never
        as a separate file.

        Raises :exc:`MotionClipError` (fail-loud, §8) on an out-of-bounds range
        — never clamps silently, which would train on frames the user did not
        select.
        """
        n = self.n_frames
        if n == 0:
            raise MotionClipError(
                f"MotionClip '{self.name}' is empty — cannot take subrange "
                f"[{start}, {end}]"
            )
        s, e = int(start), int(end)
        if s < 0 or e < s or e >= n:
            raise MotionClipError(
                f"MotionClip '{self.name}' subrange [{s}, {e}] out of bounds "
                f"(clip has {n} frames; need 0 <= start <= end < {n})"
            )
        sl = slice(s, e + 1)

        def _slice(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return a[sl].copy() if a is not None else None

        new_meta = dict(self.metadata)
        new_meta["subrange"] = {
            "start": s,
            "end": e,
            "parent_name": self.name,
            "parent_n_frames": n,
        }
        return MotionClip(
            name=f"{self.name}[{s}-{e}]",
            format_id=self.format_id,
            fps=self.fps,
            loop_mode=self.loop_mode,
            motion_weight=self.motion_weight,
            task_tag=self.task_tag,
            root_pos=_slice(self.root_pos),
            root_quat=_slice(self.root_quat),
            joint_pos=self.joint_pos[sl].copy(),
            joint_vel=_slice(self.joint_vel),
            toe_pos_local=_slice(self.toe_pos_local),
            toe_vel_local=_slice(self.toe_vel_local),
            lin_vel=_slice(self.lin_vel),
            ang_vel=_slice(self.ang_vel),
            amp_obs_dim=self.amp_obs_dim,
            metadata=new_meta,
        )

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
        term_names: Sequence[str],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` random ``(s_t, s_{t+dt})`` pairs for AMP training.

        Validates pre-conditions (payload, dt, duration) and computes the
        time pairs. The obs vector layout is delegated to
        :meth:`_sample_amp_obs_batch` which dispatches through the
        :mod:`application.training.amp.obs_terms` registry. Calling this
        method on a clip without ``has_amp_payload()`` raises.

        ``term_names`` is keyword-only required — there is no default
        fallback. The caller (launcher / SB3 entry) is expected to inject
        the family-keyed AMP obs term list resolved via
        :func:`registers.amp_obs_templates.get_obs_template`. Refusing to
        substitute a hardcoded quadruped default catches family/term-list
        drift at the call boundary instead of deep inside the
        discriminator loss.
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
        obs_t = self._sample_amp_obs_batch(times, term_names=term_names)
        obs_tp1 = self._sample_amp_obs_batch(times + dt, term_names=term_names)
        return obs_t, obs_tp1

    def _sample_amp_obs_batch(
        self,
        times: np.ndarray,
        *,
        term_names: Sequence[str],
    ) -> np.ndarray:
        """Vectorised AMP observation lookup for *N* times.

        Dispatches through the :mod:`application.training.amp.obs_terms`
        registry. ``term_names`` is the ordered list of obs term names
        the consumer requires; it is keyword-only required (no hardcoded
        ``DEFAULT_QUADRUPED_TERMS`` fallback). The launcher resolves it
        once via :func:`registers.amp_obs_templates.get_obs_template`
        keyed on the robot's primary family and threads the same list
        through both the motion-side path here and the env-side
        :func:`application.training.amp.obs_terms.compute_amp_obs_from_env`.
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
            compute_amp_obs_from_motion,
        )

        return compute_amp_obs_from_motion(list(term_names), self, times)

    def _amp_obs_at(
        self,
        t: float,
        *,
        term_names: Sequence[str],
    ) -> np.ndarray:
        """Single-time AMP observation. ``term_names`` is keyword-only
        required and forwarded to :meth:`_sample_amp_obs_batch`."""
        return self._sample_amp_obs_batch(
            np.asarray([t], dtype=np.float64), term_names=term_names,
        )[0]


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
