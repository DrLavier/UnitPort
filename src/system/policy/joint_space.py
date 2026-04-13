"""Named, ordered joint reference frames.

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

These four orderings are independent. Past code conflated them via a
hand-maintained ``env.joint_names`` attribute that was actually built
from actuator names but read as if it were qpos names — fine for
``unitree_go2`` from MuJoCo Menagerie (where they coincidentally match),
broken for Unitree's own ``go2.xml``, broken for any biped, broken for
any manipulator. The architectural fix is to never assume coincidence.

Adding a new robot
------------------
Adding a new robot family is now declarative:

1. Define a ``JointSpace`` constant for each layer that has a stable
   "official" joint order (typically: the SDK side, since the sim and
   bundle sides are derived from the MJCF and manifest at runtime).
2. The framework's permutations work as long as joint names canonicalize
   to the same value via ``joint_name_utils.canonicalize_joint_name``.

That's it. There is no per-robot lookup table to maintain anywhere else.

Limitations
-----------
This module currently assumes **single-DOF joints** (hinge / slider). The
permutation API uses one slot per joint name. For multi-DOF joints (ball,
free) you need a slot-table abstraction; that's a future extension.
Quadrupeds, most bipeds, and all standard manipulators are 1-DOF
throughout, so the current limitation is not blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .joint_name_utils import canonicalize_joint_name

if TYPE_CHECKING:
    from src.system.policy.deploy_contract import DeployContract


@dataclass(frozen=True)
class JointSpace:
    """An ordered, named reference frame for joint-aligned vectors.

    Two JointSpaces with the same canonical joint sequence are
    equivalent for permutation purposes regardless of label or surface
    spelling — e.g. ``"FL_hip"`` and ``"FL_hip_joint"`` canonicalize to
    the same key, so a vector aligned to either space round-trips
    cleanly through the other.

    Instances are frozen / hashable so they can be cached and compared
    cheaply. ``permute`` is the only API callers usually need.
    """

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
        # Computed lazily and cached on the instance via object.__setattr__
        # to keep the dataclass frozen but still memoise the lookup.
        cached = self.__dict__.get("_canon_index")
        if cached is None:
            cached = {
                canonicalize_joint_name(j): i for i, j in enumerate(self.joints)
            }
            object.__setattr__(self, "_canon_index", cached)
        return cached

    def index_of(self, joint_name: str) -> int:
        """Return the slot for *joint_name* in this space.

        Raises ``KeyError`` if absent. Comparison is via
        ``canonicalize_joint_name`` so trailing ``_joint`` suffixes,
        case differences, etc. all collapse to the same key.
        """
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
        """Return ``idx`` such that ``vec_self[idx]`` aligns to *target*.

        ``idx[k]`` is the slot in *self* holding the same joint as
        ``target.joints[k]``. Raises ``KeyError`` if any joint in
        *target* is missing from *self*. The result is cached per
        ``(self, target)`` pair so the lookup is paid exactly once per
        runtime session.
        """
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
        """Reorder *vec* (aligned to *self*) into *target*'s order.

        Returns a new array of length ``len(target)``. The fast path
        when the two spaces are equivalent is the identity (no copy
        beyond ``np.asarray``).
        """
        arr = np.asarray(vec)
        if self is target or self.equivalent_to(target):
            if arr.shape[0] != len(target):
                # Equivalent labels but mismatched length — caller bug.
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

    Returns ``(qpos_space, ctrl_space)``:

    * ``qpos_space`` — joints in ``mj_data.qpos[7:]`` slot order. Built
      by walking ``mj_id2name(mjOBJ_JOINT, jid)`` for every joint whose
      type is *not* ``mjJNT_FREE``. The free joint owns the first 7
      qpos slots (xyz + quaternion); every subsequent joint contributes
      one qpos slot (assumes 1-DOF joints).

    * ``ctrl_space`` — joints in ``mj_data.ctrl[i]`` slot order. Built
      by walking ``actuator_trnid[aid, 0]`` for every actuator. Each
      ``ctrl[i]`` drives ``actuator[i]`` which targets the joint at
      ``actuator_trnid[i, 0]``; the joint name comes from the same
      ``mj_id2name`` call.

    Both spaces are independent and may have different orderings even
    on the same robot — see ``unitree_legged_robots/go2/go2.xml`` where
    qpos joints are FL/FR/RL/RR but actuators are FR/FL/RR/RL.

    Raises ``RuntimeError`` if MuJoCo is not importable or the model
    handle is invalid.
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
        # mjJNT_FREE is the floating base — eats 7 qpos slots and is not
        # part of the per-joint vector the obs/action stack reorders.
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        # Future TODO: ball joints (mjJNT_BALL) take 4 qpos slots and
        # need a multi-DOF slot table; for now skip them with a warning
        # so 1-DOF assumptions stay sound.
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
) -> Tuple[JointSpace, JointSpace, JointSpace]:
    """Build (bundle_space, qpos_space, ctrl_space) from a deploy_contract.

    Strict counterpart to ``joint_spaces_from_mj_model``: every joint name in
    ``contract.joint_sdk_names`` MUST resolve via
    ``mujoco.mj_name2id(model, mjOBJ_JOINT, name)`` and must have a backing
    actuator. There is **no** ``canonicalize_joint_name`` fallback in this
    path — silent reorder is the second-leading sim2sim killer per
    SIM2SIM/report.yaml rule_2_joint_ordering, so we fail loud instead.

    Returns
    -------
    bundle_space : JointSpace
        Labeled ``"bundle_sdk"``. Joints are exactly ``contract.joint_sdk_names``
        in their declared order — this is the order the policy was trained
        against and the order ObsBuilder/ActionApplier should treat as the
        canonical bundle.
    qpos_space : JointSpace
        Labeled ``"mj_qpos"``. Same joint names but reordered by ascending
        ``model.jnt_qposadr``, i.e. the order they appear in
        ``mj_data.qpos[7:]``.
    ctrl_space : JointSpace
        Labeled ``"mj_ctrl"``. Same joint names but reordered by ascending
        actuator id (the order they appear in ``mj_data.ctrl``).

    Limitations
    -----------
    The current ActionApplier reads ``qpos[7:7+n_contract]`` as a contiguous
    block, so the contract joints MUST occupy the first ``n_contract`` qpos
    slots after the free joint, in some permutation. We validate this here
    and raise ``NotImplementedError`` if the contract is a non-contiguous
    subset of the MJCF joints (the multi-DOF / partial-actuation case).
    """
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "joint_spaces_from_deploy_contract: mujoco package not importable"
        ) from exc

    names = list(contract.joint_sdk_names)
    if not names:
        raise ValueError(
            "joint_spaces_from_deploy_contract: contract.joint_sdk_names is empty"
        )

    # ── Pass 1: validate every contract name resolves in the MJCF ────────
    unresolved_joints: List[str] = []
    name_to_jid: Dict[str, int] = {}
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            unresolved_joints.append(name)
        else:
            name_to_jid[name] = int(jid)

    if unresolved_joints:
        raise ValueError(
            f"joint_spaces_from_deploy_contract: contract joint names not "
            f"present in MJCF — missing: {unresolved_joints}. Available "
            f"joints in MJCF: "
            f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(int(model.njnt))]}"
        )

    # ── Pass 2: each contract joint must have an actuator. The actuator's
    # mjOBJ_ACTUATOR name often matches the joint name, but in some MJCFs
    # actuators are named separately and identified only via actuator_trnid.
    # Try the name first, fall through to walking actuator_trnid.
    name_to_aid: Dict[str, int] = {}
    unresolved_actuators: List[str] = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            target_jid = name_to_jid[name]
            for try_aid in range(int(model.nu)):
                try:
                    if int(model.actuator_trnid[try_aid, 0]) == target_jid:
                        aid = try_aid
                        break
                except Exception:
                    continue
        if aid < 0:
            unresolved_actuators.append(name)
        else:
            name_to_aid[name] = int(aid)

    if unresolved_actuators:
        raise ValueError(
            f"joint_spaces_from_deploy_contract: no actuator targets the "
            f"following contract joints: {unresolved_actuators}. The MJCF "
            f"must declare an actuator for every joint the policy controls."
        )

    # ── Pass 3: contiguity check (current ActionApplier limitation) ──────
    # The contract joints must occupy a contiguous range of qpos addresses
    # immediately after the floating-base free joint (slots 7..7+n-1). If
    # they do not, the [:n_qpos] slice in apply_with_pd would silently read
    # the wrong joints. Fail loud rather than fail wrong.
    qpos_addrs = sorted(int(model.jnt_qposadr[name_to_jid[n]]) for n in names)
    expected = list(range(qpos_addrs[0], qpos_addrs[0] + len(names)))
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
        # Free joint should land at qpos[0..6], so the first hinge joint
        # is normally at qpos[7]. If it isn't, the MJCF has something
        # unusual — refuse to guess.
        raise NotImplementedError(
            f"joint_spaces_from_deploy_contract: first contract joint "
            f"qpos address is {qpos_addrs[0]}, expected 7 (free joint + "
            f"hinges). Non-standard MJCF root structure not yet supported."
        )

    # ── Build the three spaces ───────────────────────────────────────────
    bundle_space = JointSpace(label="bundle_sdk", joints=tuple(names))

    qpos_sorted = sorted(names, key=lambda n: int(model.jnt_qposadr[name_to_jid[n]]))
    qpos_space = JointSpace(label="mj_qpos", joints=tuple(qpos_sorted))

    ctrl_sorted = sorted(names, key=lambda n: name_to_aid[n])
    ctrl_space = JointSpace(label="mj_ctrl", joints=tuple(ctrl_sorted))

    return bundle_space, qpos_space, ctrl_space
