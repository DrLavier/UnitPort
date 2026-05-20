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

Ported verbatim from DEMO/src/system/runtime/simulation/pd_controller.py
(Phase 1, TRAINING_BACKEND_PLAN.yaml R4): the ImplicitActuator formula and
DCMotor envelope are sim2sim invariants — DO NOT refactor or simplify.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    # Forward reference — application/service/runtime/policy/deploy_contract.py
    # lands in Phase 3. The type hint is unused at runtime.
    from application.service.runtime.policy.deploy_contract import DeployContract

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
        *,
        velocity_limit: Optional[Union[np.ndarray, Sequence[float]]] = None,
        saturation_effort: Optional[Union[np.ndarray, Sequence[float]]] = None,
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
        # DCMotor torque-speed envelope. Both arrays must be present together
        # for the envelope to engage; either being None keeps PDController in
        # its legacy IdealPDActuator behaviour.
        self._dc_velocity_limit: Optional[np.ndarray] = (
            self._to_array(velocity_limit, num_joints)
            if velocity_limit is not None
            else None
        )
        self._dc_saturation_effort: Optional[np.ndarray] = (
            self._to_array(saturation_effort, num_joints)
            if saturation_effort is not None
            else None
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

        # Isaac Lab DCMotor torque-speed envelope. When the bundle's actuator
        # is a DCMotor, available torque drops linearly with |qdot| from
        # ``saturation_effort`` at zero speed to 0 at ``velocity_limit``. The
        # PD result is squashed against this envelope BEFORE the final hard
        # effort_limit clip — without it MuJoCo's PD ships up to 2× the
        # torque Isaac Lab would have at typical walking joint speeds, which
        # is the difference between "decelerates cleanly" and "front-flips".
        if (
            self._dc_velocity_limit is not None
            and self._dc_saturation_effort is not None
        ):
            avail = self._dc_saturation_effort * np.maximum(
                0.0,
                1.0 - np.abs(vel) / np.maximum(self._dc_velocity_limit, 1e-9),
            )
            torques = np.clip(torques, -avail, avail)

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
        # Bundle finalize sets ``mujoco_deploy_unsupported`` (carrying a
        # reason string) when the registered MJCF couldn't cover the
        # trained joint set — the bundle still ships for IsaacSim / cloud
        # deploy, but the MuJoCo PD path can't honor it. Refuse at load
        # time with the carried reason rather than crashing later inside
        # the scalar fallback path with a less informative error.
        unsupported = getattr(contract, "mujoco_deploy_unsupported", None)
        if unsupported:
            raise ValueError(
                f"PDController.from_deploy_contract: bundle is marked as "
                f"MuJoCo-deploy-unsupported by the finalizer. Reason: "
                f"{unsupported}. Use IsaacSim / cloud deploy target, or "
                f"re-export this bundle against a robot whose MJCF "
                f"covers the trained joint set."
            )

        n = len(joint_names)
        length_check = {
            "stiffness": len(contract.stiffness),
            "damping": len(contract.damping),
            "effort_limit": len(contract.effort_limit),
            "joint_ids_map": len(contract.joint_ids_map),
            "default_joint_pos": len(contract.default_joint_pos),
        }
        # Belt-and-suspenders: DeployContract.from_dict already enforces that
        # mujoco_pd_gains / mujoco_pd_damping are length-n_joints when present
        # (schema v2 with pd_param). We re-check here so any caller that
        # constructs DeployContract directly (bypassing from_dict) still gets
        # caught before we silently mis-broadcast a wrong-length kp array.
        if contract.mujoco_pd_gains is not None:
            length_check["mujoco_pd_gains"] = len(contract.mujoco_pd_gains)
            length_check["mujoco_pd_damping"] = len(contract.mujoco_pd_damping or [])
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

        # ----- Gain source selection (sim2sim PD framework, CLAUDE.md §1.10).
        #
        # MuJoCo runtime MUST use the mass-weighted gains the bundle finalizer
        # pre-derived via ``mujoco_gain_solver`` (kp = M_diag(q₀) · ωn²). The
        # legacy ``contract.stiffness`` / ``contract.damping`` are PhysX's
        # unit-mass form (kp = ωn²) and only happen to be in the right
        # ballpark for very light robots — heavier robots (Spot ~32 kg) get
        # under-damped 10–50×, leading to the "robot flies away" failure mode
        # in MuJoCo review. See the DeployContract docstring at
        # deploy_contract.py L327–335 for the cross-engine contract.
        if contract.mujoco_pd_gains is not None:
            # Schema v2 with ActuatorPDNode wired: prefer derived MuJoCo
            # gains. ``from_dict`` guarantees length(mujoco_pd_gains) == n.
            kp_sdk = np.asarray(contract.mujoco_pd_gains, dtype=np.float32)
            kd_sdk = np.asarray(contract.mujoco_pd_damping, dtype=np.float32)
            gain_source = "mujoco_derived"
        else:
            # WHY KEPT: legacy v1 bundle compat (CLAUDE.md §1.8 c) +
            # schema v2 passthrough for bundles compiled before an
            # ActuatorPDNode was wired (pd_param=None). ``from_dict``
            # already ensures mujoco_pd_gains / pd_param are consistent.
            log.warning(
                "PDController: bundle has no mujoco_pd_gains — falling back "
                "to legacy stiffness/damping (PhysX unit-mass form, kp=ωn²). "
                "This diverges from training dynamics for heavier robots. "
                "Re-export the bundle through the Stage F pipeline (wire an "
                "ActuatorPDNode in the canvas) to populate mujoco_pd_gains."
            )
            kp_sdk = np.asarray(contract.stiffness, dtype=np.float32)
            kd_sdk = np.asarray(contract.damping, dtype=np.float32)
            gain_source = "scalar_fallback"

        effort_sdk = np.asarray(contract.effort_limit, dtype=np.float32)

        if contract.is_identity_joint_map():
            # Fast path: no permutation needed.
            kp = kp_sdk
            kd = kd_sdk
            effort = effort_sdk
        else:
            kp = np.array(
                [kp_sdk[jmap[i]] for i in range(n)], dtype=np.float32
            )
            kd = np.array(
                [kd_sdk[jmap[i]] for i in range(n)], dtype=np.float32
            )
            effort = np.array(
                [effort_sdk[jmap[i]] for i in range(n)], dtype=np.float32
            )

        # default_joint_pos is ALREADY in isaac/bundle order — no permute.
        default_bundle = np.asarray(contract.default_joint_pos, dtype=np.float32)

        # DCMotor envelope params. Stored in SDK order on the contract, so
        # the same permutation applies as for kp/kd/effort.
        velocity_limit_bundle: Optional[np.ndarray] = None
        saturation_effort_bundle: Optional[np.ndarray] = None
        if contract.velocity_limit is not None and contract.saturation_effort is not None:
            if contract.is_identity_joint_map():
                velocity_limit_bundle = np.asarray(
                    contract.velocity_limit, dtype=np.float32
                )
                saturation_effort_bundle = np.asarray(
                    contract.saturation_effort, dtype=np.float32
                )
            else:
                velocity_limit_bundle = np.array(
                    [contract.velocity_limit[jmap[i]] for i in range(n)],
                    dtype=np.float32,
                )
                saturation_effort_bundle = np.array(
                    [contract.saturation_effort[jmap[i]] for i in range(n)],
                    dtype=np.float32,
                )

        dcmotor_marker = (
            f" DCMotor(vel_lim[{float(velocity_limit_bundle.min()):.1f}.."
            f"{float(velocity_limit_bundle.max()):.1f}],sat_eff[{float(saturation_effort_bundle.min()):.1f}.."
            f"{float(saturation_effort_bundle.max()):.1f}])"
            if velocity_limit_bundle is not None and saturation_effort_bundle is not None
            else ""
        )
        log.info(
            "PDController: built from deploy_contract (source=%s) — "
            "n=%d Kp[%.1f..%.1f] Kd[%.2f..%.2f] effort[%.1f..%.1f] "
            "identity_map=%s%s",
            gain_source,
            n,
            float(kp.min()),
            float(kp.max()),
            float(kd.min()),
            float(kd.max()),
            float(effort.min()),
            float(effort.max()),
            contract.is_identity_joint_map(),
            dcmotor_marker,
        )

        if not getattr(cls, "_logged_pd_per_joint_dump", False):
            try:
                # Label by contract.joint_sdk_names: after PolicyRunner's
                # _normalize_il_contract_joint_order, those names match the
                # by-type ordering that contract.stiffness/damping/
                # default_joint_pos and the policy's action/obs vectors use.
                # `joint_names` (= bundle.joint_names, from manifest.robot.
                # joint_names) is the un-normalized by-leg list and would
                # mis-label every row past index 0.
                sdk_names = list(
                    getattr(contract, "joint_sdk_names", None) or []
                )
                rows = []
                for i in range(n):
                    if i < len(sdk_names):
                        name = sdk_names[i]
                    elif i < len(joint_names):
                        name = joint_names[i]
                    else:
                        name = "?"
                    rows.append(
                        f"  [{i:2d}] {name:24s} Kp={float(kp[i]):7.2f} "
                        f"Kd={float(kd[i]):5.2f} eff={float(effort[i]):6.2f} "
                        f"default_pos={float(default_bundle[i]):+.3f}"
                    )
                log.info(
                    "PDController per-joint dump (source=%s, "
                    "labels=contract.joint_sdk_names after normalize, "
                    "joint_ids_map=%s, identity=%s):\n%s",
                    gain_source,
                    jmap,
                    contract.is_identity_joint_map(),
                    "\n".join(rows),
                )
                cls._logged_pd_per_joint_dump = True
            except Exception:
                log.exception("PDController per-joint dump failed")

        return cls(
            kp=kp,
            kd=kd,
            effort_limit=effort,
            default_pos=default_bundle,
            num_joints=n,
            velocity_limit=velocity_limit_bundle,
            saturation_effort=saturation_effort_bundle,
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
        # DCMotor envelope: collected only when at least one group declares
        # both velocity_limit and saturation_effort. Either being absent on
        # any contributing group disables the envelope (None → legacy path).
        dc_vel_list: List[float] = []
        dc_sat_list: List[float] = []
        any_dc_group = False
        all_groups_have_dc = True

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

            dc_vel_raw = act_cfg.get("velocity_limit")
            dc_sat_raw = act_cfg.get("saturation_effort")
            if dc_vel_raw is not None and dc_sat_raw is not None:
                any_dc_group = True
                try:
                    dc_vel_list.extend([float(dc_vel_raw)] * n_joints_in_group)
                    dc_sat_list.extend([float(dc_sat_raw)] * n_joints_in_group)
                except (TypeError, ValueError):
                    all_groups_have_dc = False
            else:
                all_groups_have_dc = False

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

        # Engage the DCMotor envelope only when every actuator group in the
        # env.yaml declared both velocity_limit and saturation_effort. Mixed
        # configs (some groups DCMotor, some IdealPDActuator) fall back to
        # the legacy clamp-to-effort_limit path because partial envelopes
        # would silently mis-model the joints without the field.
        dc_velocity_limit: Optional[List[float]] = None
        dc_saturation_effort: Optional[List[float]] = None
        if any_dc_group and all_groups_have_dc and len(dc_vel_list) == len(kp_list):
            dc_velocity_limit = dc_vel_list[:actual_n]
            dc_saturation_effort = dc_sat_list[:actual_n]
            log.info(
                "PDController.from_env_yaml: DCMotor envelope engaged — "
                "vel_limit=%.1f sat_effort=%.1f",
                float(dc_velocity_limit[0]),
                float(dc_saturation_effort[0]),
            )

        return cls(
            kp=kp_list[:actual_n],
            kd=kd_list[:actual_n],
            effort_limit=effort_list[:actual_n],
            vel_limit=vel_list[:actual_n],
            default_pos=default_pos,
            num_joints=actual_n,
            velocity_limit=dc_velocity_limit,
            saturation_effort=dc_saturation_effort,
        )
