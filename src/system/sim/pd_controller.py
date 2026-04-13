"""PD controller matching Isaac Lab's ImplicitActuator model.

Isaac Lab's ImplicitActuator computes joint torques as:
    torque = Kp * (target_pos - current_pos) + Kd * (0 - current_vel)

with per-joint Kp (stiffness), Kd (damping), effort_limit, and velocity_limit.

For sim-to-sim transfer (Isaac Lab → MuJoCo), UnitPort must replicate this
exactly so that policies trained in Isaac Lab behave the same in MuJoCo.

Usage::

    pd = PDController(kp=kp_array, kd=kd_array, effort_limit=effort_array,
                       vel_limit=vel_array, default_pos=default_array)
    torques = pd.compute(target_pos, current_pos, current_vel)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from src.system.policy.deploy_contract import DeployContract

log = logging.getLogger(__name__)


class PDController:
    """Per-joint PD controller matching Isaac Lab's ImplicitActuator."""

    def __init__(
        self,
        kp: Union[np.ndarray, Sequence[float], float],
        kd: Union[np.ndarray, Sequence[float], float],
        effort_limit: Union[np.ndarray, Sequence[float], float] = 33.5,
        vel_limit: Union[np.ndarray, Sequence[float], float] = 21.0,
        default_pos: Optional[Union[np.ndarray, Sequence[float]]] = None,
        num_joints: int = 12,
    ) -> None:
        self._n = num_joints
        self._kp = self._to_array(kp, num_joints)
        self._kd = self._to_array(kd, num_joints)
        self._effort_limit = self._to_array(effort_limit, num_joints)
        self._vel_limit = self._to_array(vel_limit, num_joints)
        self._default_pos = (
            np.asarray(default_pos, dtype=np.float32)
            if default_pos is not None
            else np.zeros(num_joints, dtype=np.float32)
        )

    @property
    def num_joints(self) -> int:
        return self._n

    @property
    def default_pos(self) -> np.ndarray:
        return self._default_pos.copy()

    def compute(
        self,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        *,
        use_default_offset: bool = True,
    ) -> np.ndarray:
        """Compute joint torques from PD law.

        Parameters
        ----------
        target_pos:
            Policy output — joint position targets (action_dim,).
            If ``use_default_offset`` is True, these are offsets from
            the default standing pose.
        current_pos:
            Current joint positions from simulation.
        current_vel:
            Current joint velocities from simulation.
        use_default_offset:
            When True (Isaac Lab default), adds default_pos to target
            before computing the error.

        Returns
        -------
        np.ndarray
            Joint torques clipped to effort limits.
        """
        target = np.asarray(target_pos, dtype=np.float32).flatten()
        pos = np.asarray(current_pos, dtype=np.float32).flatten()
        vel = np.asarray(current_vel, dtype=np.float32).flatten()

        if use_default_offset:
            target = target + self._default_pos

        # PD law: torque = Kp * (target - current) + Kd * (0 - current_vel)
        torques = self._kp * (target - pos) + self._kd * (0.0 - vel)

        # Clip to effort limits
        torques = np.clip(torques, -self._effort_limit, self._effort_limit)

        return torques.astype(np.float32)

    @staticmethod
    def _to_array(val: Union[np.ndarray, Sequence[float], float], n: int) -> np.ndarray:
        """Convert scalar or sequence to float32 array of length n."""
        if isinstance(val, (int, float)):
            return np.full(n, float(val), dtype=np.float32)
        arr = np.asarray(val, dtype=np.float32).flatten()
        if arr.shape[0] == 1:
            return np.full(n, arr[0], dtype=np.float32)
        if arr.shape[0] != n:
            log.warning("PD param length %d != num_joints %d; padding/truncating", arr.shape[0], n)
            if arr.shape[0] < n:
                arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=arr[-1])
            else:
                arr = arr[:n]
        return arr

    # ------------------------------------------------------------------
    # Factory: from deploy_contract (preferred sim2sim path)
    # ------------------------------------------------------------------

    @classmethod
    def from_deploy_contract(
        cls,
        contract: "DeployContract",
        joint_names: Sequence[str],
    ) -> "PDController":
        """Build a PDController from a deploy_contract, in BUNDLE order.

        ``contract.stiffness`` / ``damping`` / ``effort_limit`` are stored in
        SDK order (the order ``contract.joint_sdk_names`` declares).
        ``contract.default_joint_pos`` is stored in **isaac/bundle order**
        (the order the policy's obs/action layout uses) — see the schema
        comment in ``deploy_contract.py``.

        ``joint_names`` is the BUNDLE-order joint name list (typically
        ``bundle.joint_names`` from the loaded CheckpointBundle). When
        bundle order ≠ sdk order we use ``contract.joint_ids_map`` to
        permute stiffness/damping/effort_limit from sdk → bundle.

        Returns a PDController whose internal arrays are all in BUNDLE
        order, ready to be consumed by ``ActionApplier.apply_with_pd``
        (which calls ``compute(target_bundle, current_pos_bundle,
        current_vel_bundle)`` with no further reorder).

        Parameters
        ----------
        contract:
            Loaded DeployContract instance.
        joint_names:
            Bundle-order joint names (length must equal contract.n_joints).

        Raises
        ------
        ValueError:
            On any length mismatch between contract fields and joint_names,
            or if joint_ids_map is malformed.
        """
        n = len(joint_names)
        length_check = {
            "stiffness": len(contract.stiffness),
            "damping": len(contract.damping),
            "effort_limit": len(contract.effort_limit),
            "joint_ids_map": len(contract.joint_ids_map),
            "default_joint_pos": len(contract.default_joint_pos),
        }
        mismatched = {k: v for k, v in length_check.items() if v != n}
        if mismatched:
            raise ValueError(
                f"PDController.from_deploy_contract: contract field lengths "
                f"do not match joint_names length {n}: {mismatched}"
            )

        # joint_ids_map[isaac_idx] = sdk_idx, where the contract's stiffness/
        # damping/effort_limit are stored in SDK order. We want the resulting
        # PDController's _kp[i] (bundle order, == isaac order for IL) to hold
        # the stiffness for the i-th bundle joint. That stiffness lives at
        # sdk index joint_ids_map[i], so:
        jmap = list(contract.joint_ids_map)

        if contract.is_identity_joint_map():
            # Fast path: no permutation needed.
            kp = np.asarray(contract.stiffness, dtype=np.float32)
            kd = np.asarray(contract.damping, dtype=np.float32)
            effort = np.asarray(contract.effort_limit, dtype=np.float32)
        else:
            kp = np.array(
                [contract.stiffness[jmap[i]] for i in range(n)], dtype=np.float32
            )
            kd = np.array(
                [contract.damping[jmap[i]] for i in range(n)], dtype=np.float32
            )
            effort = np.array(
                [contract.effort_limit[jmap[i]] for i in range(n)], dtype=np.float32
            )

        # default_joint_pos is ALREADY in isaac/bundle order — no permute.
        default_bundle = np.asarray(contract.default_joint_pos, dtype=np.float32)

        log.info(
            "PDController: built from deploy_contract — "
            "n=%d Kp[%.1f..%.1f] Kd[%.2f..%.2f] effort[%.1f..%.1f] "
            "identity_map=%s",
            n,
            float(kp.min()),
            float(kp.max()),
            float(kd.min()),
            float(kd.max()),
            float(effort.min()),
            float(effort.max()),
            contract.is_identity_joint_map(),
        )

        return cls(
            kp=kp,
            kd=kd,
            effort_limit=effort,
            default_pos=default_bundle,
            num_joints=n,
        )

    # ------------------------------------------------------------------
    # Factory: from env.yaml actuator config (LEGACY fallback)
    # ------------------------------------------------------------------

    @classmethod
    def from_env_yaml(
        cls,
        raw_env: Dict[str, Any],
        num_joints: int = 12,
        joint_names: Optional[Sequence[str]] = None,
    ) -> "PDController":
        """Create a PDController from parsed Isaac Lab env.yaml.

        Looks for actuator config under ``scene.robot.actuators``.
        Each actuator group specifies ``stiffness``, ``damping``,
        ``effort_limit``, ``velocity_limit``.

        Also reads ``scene.robot.init_state.joint_pos`` for default positions.

        ``joint_names`` is REQUIRED for correct default-pose expansion when
        Isaac Lab's env.yaml stores ``init_state.joint_pos`` as a dict of
        regex patterns (the typical case for the stock Go2 task — one
        pattern like ``.*L_hip_joint`` covers two real joints). Without
        the joint name list, default positions get written into wrong
        slots and the robot starts in a flat-leg pose the policy was
        never trained on, producing the "flailing" motion the user sees.
        Falls back to the legacy zero-fill if not provided so existing
        callers stay functional.
        """
        scene = raw_env.get("scene", {})
        robot = scene.get("robot", scene.get("articulation", {}))
        actuators = robot.get("actuators", {}) if isinstance(robot, dict) else {}

        kp_list: List[float] = []
        kd_list: List[float] = []
        effort_list: List[float] = []
        vel_list: List[float] = []

        import re as _re_grp
        for _group_name, act_cfg in actuators.items():
            if not isinstance(act_cfg, dict):
                continue
            patterns = act_cfg.get("joint_names_expr", []) or []
            # Each entry is a regex pattern that may match multiple real joints
            # (e.g. ".*_hip_joint" matches FL/FR/RL/RR_hip_joint). Expand the
            # patterns against the actual joint name list to get the true
            # per-group joint count. Falls back to len(patterns) when joint
            # names are unavailable (legacy callers / unit tests).
            if joint_names:
                matched = 0
                for pat in patterns:
                    try:
                        rx = _re_grp.compile(str(pat))
                    except _re_grp.error:
                        continue
                    for jname in joint_names:
                        if rx.fullmatch(jname) or rx.match(jname):
                            matched += 1
                n_joints_in_group = matched
            else:
                n_joints_in_group = len(patterns)
            if n_joints_in_group == 0:
                n_joints_in_group = 1

            stiffness = float(act_cfg.get("stiffness", 20.0))
            damping = float(act_cfg.get("damping", 0.5))
            effort = float(act_cfg.get("effort_limit", 33.5))
            vel = float(act_cfg.get("velocity_limit", 21.0))

            kp_list.extend([stiffness] * n_joints_in_group)
            kd_list.extend([damping] * n_joints_in_group)
            effort_list.extend([effort] * n_joints_in_group)
            vel_list.extend([vel] * n_joints_in_group)

        # Fallback if no actuators found.
        # WARNING: this path is a known sim2sim killer — generic Kp=20/Kd=0.5
        # almost never matches the values the policy was trained with, and
        # the resulting torques cause the robot to flail or topple. Log loudly
        # so users notice the env.yaml is missing the actuator block instead
        # of silently shipping a broken bundle.
        if not kp_list:
            log.error(
                "PDController.from_env_yaml: env.yaml has no usable "
                "scene.robot.actuators block — falling back to generic "
                "Kp=20.0 / Kd=0.5 / effort=33.5 / vel=21.0 for all %d "
                "joints. This will almost certainly NOT match training "
                "and is the leading cause of sim2sim divergence. Verify "
                "the bundle's env.yaml contains the actuator config.",
                num_joints,
            )
            kp_list = [20.0] * num_joints
            kd_list = [0.5] * num_joints
            effort_list = [33.5] * num_joints
            vel_list = [21.0] * num_joints

        # ── Default joint positions ───────────────────────────────────
        # Isaac Lab's env.yaml writes ``init_state.joint_pos`` as a dict of
        # *regex patterns* keyed by joint-name regex (e.g.
        # ``.*L_hip_joint: 0.1``, ``F[L,R]_thigh_joint: 0.8``). Each pattern
        # may match multiple actual joints. The previous implementation
        # iterated ``items()`` and wrote one value per slot in dict order,
        # which means a 5-pattern dict only filled 5 of the 12 joints — the
        # other 7 stayed at zero, producing a flat-leg pose with no
        # crouching at all. Expand the patterns properly against the real
        # joint name list when one is provided.
        init_state = robot.get("init_state", {}) if isinstance(robot, dict) else {}
        joint_pos_dict = init_state.get("joint_pos", {})
        default_pos = np.zeros(num_joints, dtype=np.float32)
        unmatched_joints: List[str] = []
        if isinstance(joint_pos_dict, dict) and joint_names:
            import re as _re
            filled = np.zeros(num_joints, dtype=bool)
            for pattern, value in joint_pos_dict.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                try:
                    rx = _re.compile(str(pattern))
                except _re.error:
                    continue
                for i, jname in enumerate(joint_names[:num_joints]):
                    if rx.fullmatch(jname) or rx.match(jname):
                        default_pos[i] = v
                        filled[i] = True
            unmatched_joints = [
                str(joint_names[i]) for i in range(min(num_joints, len(joint_names)))
                if not filled[i]
            ]
            if unmatched_joints:
                log.error(
                    "PDController.from_env_yaml: %d joint(s) had no matching "
                    "init_state.joint_pos pattern in env.yaml and will start "
                    "at qpos=0 — this typically yields a flat-leg/standing-OOD "
                    "pose the policy was never trained on, causing immediate "
                    "flailing. Unmatched joints: %s",
                    len(unmatched_joints),
                    ", ".join(unmatched_joints),
                )
        elif isinstance(joint_pos_dict, dict):
            # No joint name list — last-ditch positional fill so unit tests
            # without bundle context still get *something* non-zero.
            idx = 0
            for _pattern, val in joint_pos_dict.items():
                if idx < num_joints:
                    try:
                        default_pos[idx] = float(val)
                    except (TypeError, ValueError):
                        pass
                    idx += 1

        if not isinstance(joint_pos_dict, dict) or not joint_pos_dict:
            log.error(
                "PDController.from_env_yaml: env.yaml has no "
                "scene.robot.init_state.joint_pos block — all %d joints "
                "default to qpos=0. The policy was almost certainly trained "
                "from a non-zero standing pose, so reset() will land in an "
                "out-of-distribution configuration.",
                num_joints,
            )

        actual_n = max(len(kp_list), num_joints)
        return cls(
            kp=kp_list[:actual_n],
            kd=kd_list[:actual_n],
            effort_limit=effort_list[:actual_n],
            vel_limit=vel_list[:actual_n],
            default_pos=default_pos,
            num_joints=actual_n,
        )
