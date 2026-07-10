# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``lafan_unitree_g1`` — LAFAN1 Unitree-G1 retargeting CSV loader.

The LAFAN1 Retargeting Dataset (lvhaidong/LAFAN1_Retargeting_Dataset on
HuggingFace) ships Unitree humanoid mocap as headerless CSV. Each row is
one frame; the G1 split (``g1/*.csv``) carries **36 columns** at
**30 FPS**::

    [ root_x root_y root_z | quat_x quat_y quat_z quat_w | 29 joint angles ]
      └──── 0:3 ────┘        └──────── 3:7 (xyzw) ───────┘  └─── 7:36 ───┘

The 29 joint angles are laid out in the dataset README's documented
order, which is **bit-identical** to UnitPort G1's canonical MJCF joint
order — the source→IR permutation is the identity (asserted at load via
``motion_ir_mapping.LAFAN_UNITREE_G1_SOURCE_JOINTS`` /
``LAFAN_UNITREE_G1_IR_ROLES`` being declared in parallel).

Velocity derivation (this loader, not the MotionNormalizer)
-----------------------------------------------------------
Unlike ``amp_legged_gym`` (which stores velocities in the file) the
LAFAN1 CSV carries **pose only** — no joint / root velocities. AMP needs
them (the biped obs template uses ``joint_vel`` /
``base_lin_vel_local`` / ``base_ang_vel_local``), so the loader derives
them by finite differencing adjacent frames:

* ``joint_vel[t] = (joint_pos[t+1] - joint_pos[t]) * fps``.
* root **world**-frame linear velocity ``vw[t] = (root_pos[t+1] -
  root_pos[t]) * fps``, then rotated into the **body** frame
  (``lin_vel``) so it matches the env-side AMP reader ``root_lin_vel_b``.
* **body**-frame angular velocity ``ang_vel`` from the incremental
  rotation ``conj(q[t]) ⊗ q[t+1]`` (also matching the env reader
  ``root_ang_vel_b``). Both producers therefore emit velocities in the
  base frame — a frame mismatch here is the RC-2 class of AMP bug (disc
  separates expert vs policy on the frame artifact alone).

The last frame replicates the previous step's velocity so the velocity
arrays stay aligned to ``n_frames``.

Quaternion convention: the CSV is already scalar-last ``(x, y, z, w)``,
matching ``MotionClip.root_quat``; the loader only unit-normalises and
flips to the positive-w hemisphere (parallel to ``amp_legged_gym``).

AMP payload: ``amp_obs_dim`` is the sum of the biped AMP obs template
applied to 29 DoF — ``joint_pos(29) + base_lin_vel_local(3) +
base_ang_vel_local(3) + joint_vel(29) + root_height(1) = 65``. This
matches ``registers/data/amp_obs_templates.json`` ``biped`` /
``humanoid`` and the env-side ``compute_amp_obs_dim`` for a 29-DoF G1.
"""
from __future__ import annotations

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
from application.training.motion_ir_mapping import (
    LAFAN_UNITREE_G1_IR_ROLES,
    LAFAN_UNITREE_G1_SOURCE_JOINTS,
)


# ---------------------------------------------------------------------------
# Layout pins
# ---------------------------------------------------------------------------

#: Root free joint: 3 translation + 4 quaternion (xyzw) per frame.
_ROOT_POS_DIM = 3
_ROOT_QUAT_DIM = 4
_FREE_ROOT_DIM = _ROOT_POS_DIM + _ROOT_QUAT_DIM  # 7

#: G1 source revision pinned: 29 joint angles in the LAFAN1 README order.
_G1_N_JOINTS = 29

#: Total CSV columns per frame.
_FRAME_DIM = _FREE_ROOT_DIM + _G1_N_JOINTS  # 36

#: Dataset sampling rate. The LAFAN1 retargeting CSVs are headerless, so
#: FPS is not recoverable from the bytes — it is a documented dataset
#: constant (README: "G1: (30 FPS)"). Exposed as a constructor argument
#: so a re-sampled derivative can override it explicitly rather than
#: silently inheriting 30.
_LAFAN_G1_FPS = 30.0

#: AMP obs dim for a 29-DoF biped: joint_pos(29) + base_lin_vel(3) +
#: base_ang_vel(3) + joint_vel(29) + root_height(1). Mirrors the biped
#: amp_obs_templates layout; must equal the env-side compute_amp_obs_dim.
_AMP_OBS_DIM = _G1_N_JOINTS + 3 + 3 + _G1_N_JOINTS + 1  # 65


def _assert_source_ir_parallel() -> None:
    """Pin the source-name / IR-role tables to the same length.

    The two tables are declared in parallel in ``motion_ir_mapping``; a
    future edit that grows one without the other would silently emit a
    mis-mapped clip. Fail loud at import-of-loader time instead.
    """
    if len(LAFAN_UNITREE_G1_SOURCE_JOINTS) != _G1_N_JOINTS:
        raise RuntimeError(
            f"LAFAN_UNITREE_G1_SOURCE_JOINTS has "
            f"{len(LAFAN_UNITREE_G1_SOURCE_JOINTS)} entries, expected "
            f"{_G1_N_JOINTS}"
        )
    if len(LAFAN_UNITREE_G1_IR_ROLES) != _G1_N_JOINTS:
        raise RuntimeError(
            f"LAFAN_UNITREE_G1_IR_ROLES has "
            f"{len(LAFAN_UNITREE_G1_IR_ROLES)} entries, expected "
            f"{_G1_N_JOINTS}"
        )


_assert_source_ir_parallel()


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw, numpy-only — main-venv / Kit-venv safe)
# ---------------------------------------------------------------------------


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return quats / (norms + 1e-12)


def _standardize_quats(quats: np.ndarray) -> np.ndarray:
    """Flip quaternions to the positive-w hemisphere (xyzw)."""
    out = quats.copy()
    mask = out[:, 3] < 0.0
    out[mask] = -out[mask]
    return out


def _quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express world-frame vectors *v* in the body frame of *q*.

    ``q``: ``(T, 4)`` unit quaternions (xyzw, body→world). ``v``:
    ``(T, 3)`` world-frame vectors. Returns ``(T, 3)`` = ``R(q)^T · v``,
    computed via the conjugate-rotation identity
    ``v_body = v - 2w(q_vec×v) + 2 q_vec×(q_vec×v)``.
    """
    q_vec = q[:, 0:3]
    w = q[:, 3:4]
    c1 = np.cross(q_vec, v)
    c2 = np.cross(q_vec, c1)
    return v - 2.0 * w * c1 + 2.0 * c2


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` for batched xyzw quaternions."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty_like(a)
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _body_angular_velocity(quats: np.ndarray, fps: float) -> np.ndarray:
    """Body-frame angular velocity by finite-differencing a quat sequence.

    ``quats``: ``(T, 4)`` unit quaternions (xyzw, body→world). Returns
    ``(T, 3)`` body-frame angular velocity. For each step the incremental
    body rotation ``q_rel = conj(q[t]) ⊗ q[t+1]`` is converted to
    axis·angle (double-cover corrected via ``w_rel ≥ 0``) and scaled by
    ``fps``. The final frame replicates the previous step.
    """
    n = quats.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return out
    q0 = quats[:-1]
    q1 = quats[1:]
    q0_conj = q0.copy()
    q0_conj[:, 0:3] *= -1.0
    q_rel = _quat_mul_xyzw(q0_conj, q1)
    # Double-cover: keep the shortest-arc representation (w_rel ≥ 0).
    flip = q_rel[:, 3] < 0.0
    q_rel[flip] *= -1.0
    vec = q_rel[:, 0:3]
    w = np.clip(q_rel[:, 3], -1.0, 1.0)
    vnorm = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(vnorm, w)  # [0, pi]
    # axis = vec / |vec|; guard the near-zero-rotation case (axis irrelevant
    # because angle → 0). scale = angle / |vec| · fps applied to vec.
    safe = vnorm > 1e-9
    scale = np.zeros_like(vnorm)
    scale[safe] = angle[safe] / vnorm[safe] * float(fps)
    out[:-1] = vec * scale[:, None]
    out[-1] = out[-2]
    return out


def _finite_diff(values: np.ndarray, fps: float) -> np.ndarray:
    """Forward finite difference along axis 0, last row replicated."""
    out = np.zeros_like(values)
    if values.shape[0] < 2:
        return out
    out[:-1] = (values[1:] - values[:-1]) * float(fps)
    out[-1] = out[-2]
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class LAFANUnitreeG1Loader(MotionLoader):
    """Load a LAFAN1 Unitree-G1 retargeting ``.csv`` into a contract.

    The skeleton is fixed — 29-DoF G1 (3 waist + 3-DoF wrists). Files
    recorded for H1 / H1_2 (different DoF / joint order) raise on the
    column-count check; load them with their own loader.
    """

    format_id = "lafan_unitree_g1"

    def __init__(self, *, fps: float = _LAFAN_G1_FPS) -> None:
        if not (fps > 0):
            raise LoaderError(
                f"LAFANUnitreeG1Loader: fps must be > 0, got {fps!r}"
            )
        self._fps = float(fps)

    def load(
        self,
        path: Union[Path, str],
        *,
        target_sku: str,
        target_family: str,
    ) -> ReferenceMotionContract:
        p = Path(path)
        if not p.is_file():
            raise LoaderError(f"LAFANUnitreeG1Loader: file not found: {p}")

        try:
            arr = np.loadtxt(str(p), delimiter=",", dtype=np.float32, ndmin=2)
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(
                f"LAFANUnitreeG1Loader: cannot parse {p} as a comma-"
                f"separated float matrix: {exc}"
            ) from exc

        if arr.ndim != 2 or arr.shape[1] != _FRAME_DIM:
            raise LoaderError(
                f"LAFANUnitreeG1Loader: {p} frame shape is {tuple(arr.shape)}, "
                f"expected (n_frames, {_FRAME_DIM}) for free root "
                f"(3 pos + 4 quat) + {_G1_N_JOINTS} G1 joints. A "
                f"{arr.shape[1] if arr.ndim == 2 else '?'}-column file is "
                f"likely an H1 / H1_2 retargeting CSV — load it with the "
                f"matching skeleton loader."
            )
        if arr.shape[0] < 2:
            raise LoaderError(
                f"LAFANUnitreeG1Loader: {p} has {arr.shape[0]} frame(s); "
                f"at least 2 are required to finite-difference velocities."
            )

        root_pos = arr[:, 0:_ROOT_POS_DIM].astype(np.float32, copy=True)
        root_quat = arr[:, _ROOT_POS_DIM:_FREE_ROOT_DIM].astype(np.float64, copy=True)
        joint_pos = arr[:, _FREE_ROOT_DIM:].astype(np.float32, copy=True)

        root_quat = _normalize_quats(root_quat)
        root_quat = _standardize_quats(root_quat)

        # ── Derive velocities (CSV carries pose only). ──
        joint_vel = _finite_diff(joint_pos.astype(np.float64), self._fps)
        vw = _finite_diff(root_pos.astype(np.float64), self._fps)  # world frame
        lin_vel = _quat_rotate_inverse_xyzw(root_quat, vw)         # body frame
        ang_vel = _body_angular_velocity(root_quat, self._fps)     # body frame

        clip = MotionClip(
            name=p.stem,
            format_id=self.format_id,
            fps=self._fps,
            loop_mode="wrap",
            motion_weight=1.0,
            task_tag=infer_task_tag(p.name),
            root_pos=root_pos,
            root_quat=root_quat.astype(np.float32, copy=False),
            joint_pos=joint_pos,
            joint_vel=joint_vel.astype(np.float32, copy=False),
            lin_vel=lin_vel.astype(np.float32, copy=False),
            ang_vel=ang_vel.astype(np.float32, copy=False),
            amp_obs_dim=_AMP_OBS_DIM,
            metadata={
                "source_path": str(p.resolve()),
                "source_format": self.format_id,
                "source_skeleton": "unitree_g1",
                "source_fps": self._fps,
                # joint_pos / joint_vel columns follow this canonical IR-role
                # order (identity to the source CSV column order — see
                # motion_ir_mapping.LAFAN_UNITREE_G1_SOURCE_JOINTS). Mirrors
                # the ir_roles metadata key amp_legged_gym writes so the env
                # side routes by IR role, not positional index.
                "ir_roles": list(LAFAN_UNITREE_G1_IR_ROLES),
                # Velocities are derived (finite difference), not measured —
                # recorded so post-mortem tooling can distinguish them from
                # file-stored velocities (amp_legged_gym / loco_mujoco).
                "velocities_derived": "finite_difference",
            },
        )
        # Identity-only normaliser — see motion/normalizer.py. With no
        # alignment override on disk the call returns the same MotionClip.
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


__all__ = ["LAFANUnitreeG1Loader"]
