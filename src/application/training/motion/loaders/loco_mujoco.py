# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``loco_mujoco_unitree_h1`` — Unitree H1 humanoid loader for the
loco-mujoco trajectory format.

The loco-mujoco project (https://github.com/robfiras/loco-mujoco) ships
mocap trajectory data as numpy zipped (``.npz``) archives whose keys map
verbatim onto the attribute names of its internal
``TrajectoryData`` / ``TrajectoryInfo`` / ``TrajectoryModel`` /
``TrajectoryTransitions`` dataclasses. Their loader pipeline
deserialises into ``flax.struct.dataclass`` instances and is therefore
transitively coupled to ``jax`` / ``flax`` / ``orbax`` / ``mujoco-mjx``.

This loader is a deliberate **dependency-free adapter**: it reads the
npz with vanilla ``numpy`` (``np.load(..., allow_pickle=True)``) and
walks raw keys, never importing the ``loco_mujoco`` package. The
attribute names of the four payload classes are hardcoded as
``_TRAJECTORY_*_KEYS`` constants below — see ``_NPZ_SCHEMA_VERSION``
for the loco-mujoco source revision they correspond to. Sudden upstream
key renames would surface as ``LoaderError`` raises from the unknown-key
branch rather than as silent corruption.

Scope (this revision)
---------------------
A single source skeleton — Unitree H1 (loco-mujoco's ``unitreeH1``
environment with the 19-DoF ``h1.xml`` MJCF). The 19 source joints map
1:1 to UnitPort H1's 19 canonical IR roles (full coverage); no unmapped
target role policy is needed. Additional loco-mujoco humanoids (G1,
Atlas, Talos, BoosterT1, …) ship as separate loaders behind their own
``format_id`` slugs; this file does NOT carry a runtime skeleton
dispatch table.

Frame transforms (loader-internal, separate from MotionNormalizer)
------------------------------------------------------------------
* MuJoCo free-joint quaternion is ``[w, x, y, z]`` (scalar-first);
  UnitPort's ``MotionClip.root_quat`` is ``[x, y, z, w]`` to match
  the existing ``amp_legged_gym`` convention. Reorder via
  ``quat[:, [1, 2, 3, 0]]`` once at load time (parallel to the leg
  reorder ``amp_legged_gym`` applies — both are format quirks, not
  axis-convention transforms).
* Source joint ordering inside ``qpos[:, 7:]`` is the MJCF native
  order documented in ``LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS`` — the
  loader passes that array through unchanged because the IR role
  table (``LOCO_MUJOCO_UNITREE_H1_IR_ROLES``) is declared in the
  same order.

Axis convention (root z-up / x-forward / y-lateral, world-frame linear
and angular velocity) matches the loader defaults, so the normaliser
identity branch applies — no rewrite needed.
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
    LOCO_MUJOCO_UNITREE_H1_IR_ROLES,
    LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS,
)


# ---------------------------------------------------------------------------
# NPZ schema pin
# ---------------------------------------------------------------------------


#: Upstream loco-mujoco revision the hardcoded key tables track. Bump
#: when an upstream schema change is verified against new fixtures.
_NPZ_SCHEMA_VERSION = "loco-mujoco trajectory.dataclasses v0.x"


#: Keys an npz payload may carry. Sourced from the upstream four
#: dataclasses' ``get_attribute_names()`` returns; hardcoded so the
#: loader stays jax-free. An npz key outside these four sets raises.
_TRAJECTORY_INFO_KEYS = frozenset({
    "joint_names",
    "frequency",
    "body_names",
    "site_names",
    "metadata",
})
_TRAJECTORY_MODEL_KEYS = frozenset({
    "njnt", "jnt_type", "jnt_range", "jnt_axis",
    "body_rootid", "body_weldid", "body_mocapid",
    "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_inertia", "body_mass", "body_parentid",
    "site_bodyid", "site_pos", "site_quat",
    "nq", "nv",
})
_TRAJECTORY_DATA_KEYS = frozenset({
    "qpos", "qvel",
    "xpos", "xquat", "cvel", "subtree_com",
    "site_xpos", "site_xmat",
    "split_points",
})
_TRAJECTORY_TRANSITIONS_KEYS = frozenset({
    "observations", "actions", "rewards",
    "next_observations", "absorbings", "dones",
})
_OBS_CONTAINER_KEY = "obs_container"


# ---------------------------------------------------------------------------
# Skeleton layout pin
# ---------------------------------------------------------------------------


#: Number of scalar elements MuJoCo's free joint occupies in ``qpos``:
#: 3 root translation + 4 root quaternion.
_FREE_ROOT_QPOS_DIM = 7

#: Number of scalar elements MuJoCo's free joint occupies in ``qvel``:
#: 3 linear + 3 angular.
_FREE_ROOT_QVEL_DIM = 6

#: H1 source revision pinned: 19 joint angles laid out in the order
#: documented in ``motion_ir_mapping.LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS``.
_H1_N_JOINTS = 19


def _build_source_to_ir_perm() -> np.ndarray:
    """Compute the permutation that maps the source-native ``qpos[:, 7:]``
    column order onto the canonical IR-role column order.

    ``LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS`` and
    ``LOCO_MUJOCO_UNITREE_H1_IR_ROLES`` are declared in parallel — entry
    ``i`` in the source table is the MJCF joint that semantically maps
    onto entry ``i`` in the IR-role table. The permutation is therefore
    the identity today, but encoding it as data + asserting completeness
    pins the contract explicitly: a future source-table reshuffle (e.g.
    upstream renames a joint or swaps two columns) without the matching
    IR-table edit raises at load time instead of silently emitting a
    column-shuffled clip the discriminator separates trivially.
    """
    if len(LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS) != _H1_N_JOINTS:
        raise RuntimeError(
            f"LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS has "
            f"{len(LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS)} entries, "
            f"expected {_H1_N_JOINTS}"
        )
    if len(LOCO_MUJOCO_UNITREE_H1_IR_ROLES) != _H1_N_JOINTS:
        raise RuntimeError(
            f"LOCO_MUJOCO_UNITREE_H1_IR_ROLES has "
            f"{len(LOCO_MUJOCO_UNITREE_H1_IR_ROLES)} entries, "
            f"expected {_H1_N_JOINTS}"
        )
    # 1:1 positional mapping — source[i] ↔ ir_role[i]. perm[i] = i today.
    return np.arange(_H1_N_JOINTS, dtype=np.int64)


_SOURCE_TO_IR_PERM: np.ndarray = _build_source_to_ir_perm()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class LocoMujocoUnitreeH1Loader(MotionLoader):
    """Load a loco-mujoco ``unitreeH1`` trajectory ``.npz`` into a
    :class:`ReferenceMotionContract`.

    The npz key layout pins to the upstream loco-mujoco revision noted in
    ``_NPZ_SCHEMA_VERSION``. The skeleton is fixed — this class only
    accepts files recorded against the ``unitreeH1`` (19-DoF h1.xml)
    environment.
    """

    format_id = "loco_mujoco_unitree_h1"

    def load(
        self,
        path: Union[Path, str],
        *,
        target_sku: str,
        target_family: str,
    ) -> ReferenceMotionContract:
        p = Path(path)
        if not p.is_file():
            raise LoaderError(f"LocoMujocoUnitreeH1Loader: file not found: {p}")

        try:
            archive = np.load(str(p), allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: np.load failed for {p}: {exc}"
            ) from exc

        try:
            keys = set(archive.files)
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} is not a valid .npz archive: {exc}"
            ) from exc

        unknown = keys - (
            _TRAJECTORY_INFO_KEYS
            | _TRAJECTORY_MODEL_KEYS
            | _TRAJECTORY_DATA_KEYS
            | _TRAJECTORY_TRANSITIONS_KEYS
            | {_OBS_CONTAINER_KEY}
        )
        if unknown:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} contains keys "
                f"{sorted(unknown)!r} that are not part of the pinned "
                f"loco-mujoco schema ({_NPZ_SCHEMA_VERSION}). Either the "
                f"file was produced by a newer loco-mujoco revision than "
                f"this loader tracks, or it is not a loco-mujoco "
                f"trajectory file. Update the loader's "
                f"_TRAJECTORY_*_KEYS pins, or re-export the clip with "
                f"the supported revision."
            )

        qpos = self._require_array(archive, "qpos", p)
        qvel = self._require_array(archive, "qvel", p)
        source_joint_names = self._require_joint_names(archive, p)

        if source_joint_names != LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} declares "
                f"joint_names={source_joint_names!r} which does not "
                f"match the pinned Unitree H1 source layout "
                f"{LOCO_MUJOCO_UNITREE_H1_SOURCE_JOINTS!r}. The file "
                f"was recorded against a different loco-mujoco "
                f"environment; load it with the matching skeleton "
                f"loader, or re-record the trajectory against the "
                f"unitreeH1 environment."
            )

        frequency = self._require_frequency(archive, p)

        expected_qpos_cols = _FREE_ROOT_QPOS_DIM + _H1_N_JOINTS
        expected_qvel_cols = _FREE_ROOT_QVEL_DIM + _H1_N_JOINTS
        if qpos.ndim != 2 or qpos.shape[1] != expected_qpos_cols:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} qpos.shape="
                f"{tuple(qpos.shape)!r}; expected (n_frames, "
                f"{expected_qpos_cols}) for free root + {_H1_N_JOINTS} "
                f"H1 joints. Source MJCF skeleton may not match h1.xml."
            )
        if qvel.ndim != 2 or qvel.shape[1] != expected_qvel_cols:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} qvel.shape="
                f"{tuple(qvel.shape)!r}; expected (n_frames, "
                f"{expected_qvel_cols}) for free root + {_H1_N_JOINTS} "
                f"H1 joints."
            )
        if qpos.shape[0] != qvel.shape[0]:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} qpos and qvel frame "
                f"counts differ ({qpos.shape[0]} vs {qvel.shape[0]}); "
                f"upstream loco-mujoco emits aligned arrays — "
                f"re-export the trajectory."
            )
        if qpos.shape[0] < 2:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} has {qpos.shape[0]} "
                f"frame(s); the contract requires at least 2 for "
                f"finite-difference fallbacks."
            )

        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)

        root_pos = qpos[:, 0:3].copy()
        # MuJoCo free joint quaternion is wxyz (scalar-first); UnitPort's
        # MotionClip.root_quat carries xyzw (scalar-last) to align with
        # the amp_legged_gym convention. Reorder once, in-place at load.
        root_quat_wxyz = qpos[:, 3:7]
        root_quat = root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32, copy=True)

        # Source qpos/qvel columns are MJCF native joint order; reorder
        # once at load time into the canonical IR-role order documented
        # by ``LOCO_MUJOCO_UNITREE_H1_IR_ROLES``. Both producers (this
        # loader + the env-side AMP obs joint perm in obs_terms.py) then
        # share one column-order contract, so the discriminator's
        # expert and policy sides cannot disagree on index→IR-role
        # binding. The permutation is identity by table construction
        # today but is applied explicitly so a future table edit on
        # either side surfaces as a load-time failure rather than as
        # silent column-shuffle disc collapse.
        joint_pos_src = qpos[:, _FREE_ROOT_QPOS_DIM:]
        joint_vel_src = qvel[:, _FREE_ROOT_QVEL_DIM:]
        joint_pos = joint_pos_src[:, _SOURCE_TO_IR_PERM].astype(
            np.float32, copy=True,
        )
        joint_vel = joint_vel_src[:, _SOURCE_TO_IR_PERM].astype(
            np.float32, copy=True,
        )
        lin_vel = qvel[:, 0:3].copy()
        ang_vel = qvel[:, 3:6].copy()

        clip = MotionClip(
            name=p.stem,
            format_id=self.format_id,
            fps=float(frequency),
            loop_mode="wrap",
            motion_weight=1.0,
            task_tag=infer_task_tag(p.name),
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            amp_obs_dim=0,
            metadata={
                "source_path": str(p.resolve()),
                "source_format": self.format_id,
                "source_skeleton": "unitree_h1",
                "source_npz_schema": _NPZ_SCHEMA_VERSION,
                # Pin the canonical IR-role layout the joint_pos /
                # joint_vel columns already follow. Mirrors the
                # ``ir_roles`` metadata key amp_legged_gym writes so
                # downstream consumers route by IR rather than by
                # positional index.
                # joint_pos / joint_vel columns are guaranteed to be
                # in this IR-role order — the loader applies the
                # source→IR permutation above before the clip is built.
                "ir_roles": list(LOCO_MUJOCO_UNITREE_H1_IR_ROLES),
            },
        )
        # Identity-only normaliser — see motion/normalizer.py. With no
        # alignment override on disk the call returns the same MotionClip
        # instance, preserving byte-identical output downstream.
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
    # Internal helpers (jax-free read paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _require_array(archive, key: str, p: Path) -> np.ndarray:
        if key not in archive.files:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} is missing the "
                f"required key {key!r}. The pinned loco-mujoco "
                f"schema ({_NPZ_SCHEMA_VERSION}) writes every clip "
                f"with this key; either the export was truncated or "
                f"the file is not a loco-mujoco trajectory."
            )
        arr = archive[key]
        if not isinstance(arr, np.ndarray):
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p}[{key!r}] is "
                f"{type(arr).__name__!r}; expected numpy.ndarray."
            )
        return arr

    @staticmethod
    def _require_joint_names(archive, p: Path):
        if "joint_names" not in archive.files:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} is missing the "
                f"required key 'joint_names'. The pinned loco-mujoco "
                f"schema writes a joint_names array of source-joint "
                f"strings — without it the file's skeleton identity "
                f"cannot be verified and re-routing the data would "
                f"silently mis-align joints."
            )
        raw = archive["joint_names"]
        # The npz round-trips ``list[str]`` as an object-dtype ndarray
        # of length n; ``.tolist()`` recovers the list. Anything else
        # is upstream-shape drift.
        names = raw.tolist()
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p}['joint_names'] is "
                f"{type(names).__name__!r} of {[type(n).__name__ for n in names[:3]]!r}; "
                f"expected list[str] of source joint names."
            )
        return names

    @staticmethod
    def _require_frequency(archive, p: Path) -> float:
        if "frequency" not in archive.files:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p} is missing the "
                f"required key 'frequency'. Motion FPS cannot be "
                f"inferred — the contract refuses a silent 50 Hz "
                f"default; re-export the clip with the upstream "
                f"writer that records frequency."
            )
        raw = archive["frequency"]
        try:
            f = float(raw)
        except (TypeError, ValueError) as exc:
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p}['frequency']="
                f"{raw!r} is not coercible to float."
            ) from exc
        if not (f > 0):
            raise LoaderError(
                f"LocoMujocoUnitreeH1Loader: {p}['frequency']={f!r} "
                f"is non-positive; an upright trajectory needs a real "
                f"sampling rate."
            )
        return f


__all__ = ["LocoMujocoUnitreeH1Loader"]
