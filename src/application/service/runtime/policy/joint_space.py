"""Named, ordered joint reference frames — verbatim port from DEMO.

This module is the **single source of truth** for joint ordering in the
runtime stack. Every layer that touches joint-aligned vectors carries an
explicit ``JointSpace`` describing the ordering its slots use, and any
operation crossing layers must call ``space_a.permute(vec, space_b)`` to
reorder the vector — there is no implicit "this list and that list happen
to be in the same order" anywhere.

Why this exists
---------------
A robot has at least four different joint orderings in its life-cycle, and
they almost never agree across layers:

* **Bundle space** — the joint order the policy was trained against.
  Encoded in ``bundle.joint_names`` (from ``manifest.yaml``).
* **MuJoCo qpos space** — the joints in ``mj_data.qpos[7:]`` slot order.
  Determined by the order ``<joint>`` tags appear in the MJCF, after the
  free joint of the floating base. Read with ``mj_id2name(mjOBJ_JOINT, k)``.
* **MuJoCo ctrl space** — the joints driven by ``mj_data.ctrl[i]``, where
  ``actuator[i]`` may target a *different* joint slot than slot ``i`` in
  qpos. Read by walking ``model.actuator_trnid``.
* **Real SDK space** — the joint enum the SDK uses on the wire.
  ``unitree_legged_msgs.LowState.motor_state[i]`` for Unitree, the
  ``JointName`` proto for Spot, the cyberdog motor map for Cyberdog.

Adding a new robot family is declarative: define a ``JointSpace`` for each
layer that has a stable order; the framework's permutations work as long
as joint names canonicalize to the same value via
``joint_name_utils.canonicalize_joint_name``.

Limitations
-----------
This module currently assumes **single-DOF joints** (hinge / slider). For
multi-DOF joints (ball, free) you need a slot-table abstraction; that's a
future extension. Quadrupeds, most bipeds, and all standard manipulators
are 1-DOF throughout, so the current limitation is not blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .joint_name_utils import canonicalize_joint_name

if TYPE_CHECKING:
    from .deploy_contract import DeployContract


@dataclass(frozen=True)
class JointSpace:
    """An ordered, named reference frame for joint-aligned vectors."""

    label: str
    joints: Tuple[str, ...]

    # ──────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_iterable(cls, label: str, joints: Iterable[str]) -> "JointSpace":
        return cls(label=label, joints=tuple(joints))

    # ──────────────────────────────────────────────────────────────────
    # Basic queries
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.joints)

    def __iter__(self):
        return iter(self.joints)

    def _index_table(self) -> Dict[str, int]:
        cached = self.__dict__.get("_canon_index")
        if cached is None:
            cached = {
                canonicalize_joint_name(j): i for i, j in enumerate(self.joints)
            }
            object.__setattr__(self, "_canon_index", cached)
        return cached

    def index_of(self, joint_name: str) -> int:
        key = canonicalize_joint_name(joint_name)
        try:
            return self._index_table()[key]
        except KeyError as exc:
            raise KeyError(
                f"JointSpace {self.label!r}: no joint '{joint_name}'. "
                f"Available joints: {list(self.joints)}"
            ) from exc

    def index_of_safe(self, joint_name: str) -> Optional[int]:
        return self._index_table().get(canonicalize_joint_name(joint_name))

    def has(self, joint_name: str) -> bool:
        return canonicalize_joint_name(joint_name) in self._index_table()

    def equivalent_to(self, other: "JointSpace") -> bool:
        """True iff the two spaces have the same canonical joint sequence."""
        if len(self) != len(other):
            return False
        return all(
            canonicalize_joint_name(a) == canonicalize_joint_name(b)
            for a, b in zip(self.joints, other.joints)
        )

    # ──────────────────────────────────────────────────────────────────
    # Permutation
    # ──────────────────────────────────────────────────────────────────

    def permutation_to(self, target: "JointSpace") -> np.ndarray:
        """Return ``idx`` such that ``vec_self[idx]`` aligns to *target*."""
        cache = self.__dict__.setdefault("_perm_cache", {})
        key = id(target)
        cached = cache.get(key)
        if cached is not None:
            return cached
        out = np.empty(len(target), dtype=np.int64)
        for k, jname in enumerate(target.joints):
            idx = self.index_of_safe(jname)
            if idx is None:
                raise KeyError(
                    f"JointSpace permutation {self.label!r} → {target.label!r}: "
                    f"joint '{jname}' present in target but not source. "
                    f"source joints: {list(self.joints)}"
                )
            out[k] = idx
        cache[key] = out
        return out

    def permute(self, vec: np.ndarray, target: "JointSpace") -> np.ndarray:
        """Reorder *vec* (aligned to *self*) into *target*'s order."""
        arr = np.asarray(vec)
        if self is target or self.equivalent_to(target):
            if arr.shape[0] != len(target):
                raise ValueError(
                    f"JointSpace.permute: vector length {arr.shape[0]} != "
                    f"target length {len(target)} for {self.label!r}→{target.label!r}"
                )
            return arr
        if arr.shape[0] != len(self):
            raise ValueError(
                f"JointSpace.permute: vector length {arr.shape[0]} != "
                f"source length {len(self)} for {self.label!r}→{target.label!r}"
            )
        idx = self.permutation_to(target)
        return arr[idx]


# ---------------------------------------------------------------------------
# MuJoCo model adapters
# ---------------------------------------------------------------------------

def joint_spaces_from_mj_model(model) -> Tuple[JointSpace, JointSpace]:
    """Extract the two MuJoCo joint orderings from an mj_model.

    Returns ``(qpos_space, ctrl_space)`` with the same semantics as DEMO —
    walks ``mj_id2name(mjOBJ_JOINT, jid)`` skipping free + ball joints for
    qpos_space, and walks ``actuator_trnid[aid, 0]`` for ctrl_space.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "joint_spaces_from_mj_model: mujoco package not importable"
        ) from exc

    qpos_names: List[str] = []
    for jid in range(int(getattr(model, "njnt", 0) or 0)):
        try:
            jtype = int(model.jnt_type[jid])
        except Exception:
            continue
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            qpos_names.append(str(name))

    ctrl_names: List[str] = []
    for aid in range(int(getattr(model, "nu", 0) or 0)):
        try:
            tid = int(model.actuator_trnid[aid, 0])
        except Exception:
            continue
        if tid < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, tid)
        if name:
            ctrl_names.append(str(name))

    return (
        JointSpace(label="mj_qpos", joints=tuple(qpos_names)),
        JointSpace(label="mj_ctrl", joints=tuple(ctrl_names)),
    )


def joint_spaces_from_deploy_contract(
    model,
    contract: "DeployContract",
    robot_sku: str,
) -> Tuple[JointSpace, JointSpace, JointSpace]:
    """Build (bundle_space, qpos_space, ctrl_space) from a deploy_contract.

    Phase 5 IR-only contract: ``contract.joint_sdk_names`` carries **IR role
    names** (e.g. ``"hip_FL"``).  We resolve each IR role to the bound
    robot's physical joint name via the caller-supplied ``robot_sku``
    first, then run the existing strict MJCF lookup against the *physical*
    names.

    The returned ``bundle_space`` keeps the IR labels (so action vectors
    coming out of the policy stay tagged in the bundle's vocabulary), while
    ``qpos_space`` / ``ctrl_space`` carry physical names (so MJCF binding
    works).  ActionApplier uses ``JointSpaceMapper`` between the two to
    translate at the substrate boundary.

    ``robot_sku`` is plumbed in by the caller (PolicyRunner /
    CompatibilityChecker) — typically ``bundle.robot_sku``. The contract
    no longer carries its own SKU copy (legacy ``contract.robot_sku``
    was a redundant second source of truth that could silently diverge
    from manifest.robot.sku; it has been removed).

    Backwards compat: when ``robot_sku`` is empty AND every name in
    ``joint_sdk_names`` already resolves in the MJCF, fall back to the
    legacy direct-physical-name treatment so very old bundles still load.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "joint_spaces_from_deploy_contract: mujoco package not importable"
        ) from exc

    # MuJoCo-deploy opt-out gate (CLAUDE.md §1.10). Bundle finalize sets
    # this field when the registered MJCF couldn't cover the trained joint
    # set; the bundle still ships for IsaacSim deploy, but binding it to
    # a MuJoCo model is contractually unsupported. PolicyRunner.load
    # already checks this earlier with a richer error, but defending here
    # too means any future caller (custom test rig, third-party MuJoCo
    # binder) gets the loud refusal instead of the downstream
    # ir_roles_to_physical_names KeyError.
    unsupported = getattr(contract, "mujoco_deploy_unsupported", None)
    if unsupported:
        raise ValueError(
            f"joint_spaces_from_deploy_contract: bundle is marked "
            f"MuJoCo-deploy-unsupported by the finalizer "
            f"(reason: {unsupported}). The trained joint set cannot be "
            f"bound to a MuJoCo articulation; use IsaacSim / cloud deploy "
            f"target or re-export with a matching MJCF asset."
        )

    contract_names = list(contract.joint_sdk_names)
    if not contract_names:
        raise ValueError(
            "joint_spaces_from_deploy_contract: contract.joint_sdk_names is empty"
        )

    robot_sku = (robot_sku or "").strip()
    bundle_labels = list(contract_names)  # preserved for bundle_space label

    # ── Phase 5: IR → physical translation ────────────────────────────────
    if robot_sku:
        from application.service.runtime.policy.joint_name_utils import (
            ir_roles_to_physical_names,
        )
        try:
            physical_names = ir_roles_to_physical_names(contract_names, robot_sku)
        except ValueError as exc:
            # Fall back to direct treatment ONLY if every contract name
            # already resolves in MJCF — handles the rare case of a Phase 5
            # bundle that mistakenly persisted physical names while declaring
            # robot_sku. Otherwise re-raise the IR error.
            if all(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0
                for n in contract_names
            ):
                physical_names = list(contract_names)
            else:
                raise
    else:
        # Legacy path — contract was built before Phase 5; treat names as
        # physical directly. The strict MJCF check below still applies.
        physical_names = list(contract_names)

    # ── Pass 1: validate every physical name resolves in the MJCF ────────
    unresolved_joints: List[str] = []
    name_to_jid: Dict[str, int] = {}
    for name in physical_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            unresolved_joints.append(name)
        else:
            name_to_jid[name] = int(jid)

    if unresolved_joints:
        raise ValueError(
            f"joint_spaces_from_deploy_contract: physical joint names not "
            f"present in MJCF — missing: {unresolved_joints} "
            f"(translated from IR roles: {bundle_labels} via robot_sku="
            f"{robot_sku!r}). Available joints in MJCF: "
            f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(int(model.njnt))]}"
        )

    # ── Pass 2: each physical joint must have an actuator ────────────────
    name_to_aid: Dict[str, int] = {}
    unresolved_actuators: List[str] = []
    for name in physical_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            target_jid = name_to_jid[name]
            for try_aid in range(int(model.nu)):
                # ``actuator_trnid`` is a numpy int array shaped
                # (model.nu, 2). Indexing is a deterministic read; if
                # it ever raises, that's a mujoco binding incompatibility
                # we want to surface, not swallow into "actuator not
                # found" (which is a different failure mode with very
                # different remediation).
                if int(model.actuator_trnid[try_aid, 0]) == target_jid:
                    aid = try_aid
                    break
        if aid < 0:
            unresolved_actuators.append(name)
        else:
            name_to_aid[name] = int(aid)

    if unresolved_actuators:
        raise ValueError(
            f"joint_spaces_from_deploy_contract: no actuator targets the "
            f"following physical joints: {unresolved_actuators}. The MJCF "
            f"must declare an actuator for every joint the policy controls."
        )

    # ── Pass 3: contiguity check (current ActionApplier limitation) ──────
    qpos_addrs = sorted(int(model.jnt_qposadr[name_to_jid[n]]) for n in physical_names)
    expected = list(range(qpos_addrs[0], qpos_addrs[0] + len(physical_names)))
    if qpos_addrs != expected:
        raise NotImplementedError(
            f"joint_spaces_from_deploy_contract: contract joints occupy "
            f"non-contiguous qpos addresses {qpos_addrs}. ActionApplier "
            f"currently reads qpos[7:7+n] as a contiguous block; "
            f"partial-actuation contracts are not yet supported. Either "
            f"export a contract that covers all hinge joints, or extend "
            f"ActionApplier to use per-joint qpos addresses."
        )
    if qpos_addrs[0] != 7:
        raise NotImplementedError(
            f"joint_spaces_from_deploy_contract: first contract joint "
            f"qpos address is {qpos_addrs[0]}, expected 7 (free joint + "
            f"hinges). Non-standard MJCF root structure not yet supported."
        )

    # ── Build the three spaces ───────────────────────────────────────────
    # bundle_space carries PHYSICAL names in policy execution order (=
    # IR-list order, since the IR↔physical map is order-preserving by
    # construction in RobotSpec). This lets the existing
    # JointSpaceMapper.permute string-match work unchanged.  The IR
    # labels remain authoritative on the contract via
    # contract.joint_sdk_names — callers needing the IR vocabulary look
    # there, not at bundle_space.joints.
    bundle_space = JointSpace(label="bundle_sdk", joints=tuple(physical_names))
    # Stash the original IR labels for downstream introspection (logs,
    # compatibility checks). JointSpace is frozen — set via __dict__.
    bundle_space.__dict__["ir_labels"] = tuple(bundle_labels)

    qpos_sorted = sorted(
        physical_names, key=lambda n: int(model.jnt_qposadr[name_to_jid[n]])
    )
    qpos_space = JointSpace(label="mj_qpos", joints=tuple(qpos_sorted))

    ctrl_sorted = sorted(physical_names, key=lambda n: name_to_aid[n])
    ctrl_space = JointSpace(label="mj_ctrl", joints=tuple(ctrl_sorted))

    return bundle_space, qpos_space, ctrl_space
