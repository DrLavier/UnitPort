from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .joint_name_utils import canonicalize_joint_name, canonicalize_joint_names
from .manifest_schema import CheckpointBundle
from .sim_env_context import SimEnvContext

_SUPPORTED_ACTION_TYPES = {"joint_position", "joint_velocity", "torque"}

log = logging.getLogger(__name__)


@dataclass
class ActionSpec:
    action_type: str
    dim: int
    joint_names: List[str]
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    scale: float = 1.0


class ActionApplier:
    """Converts raw policy outputs into control-space action vectors."""

    def __init__(
        self,
        bundle: CheckpointBundle,
        *,
        pd_controller: Optional[Any] = None,
        bundle_space: Optional["JointSpace"] = None,
        ctrl_space: Optional["JointSpace"] = None,
    ):
        if bundle.action_type not in _SUPPORTED_ACTION_TYPES:
            raise ValueError(
                f"Unsupported action type '{bundle.action_type}'. "
                f"Runtime supports: {sorted(_SUPPORTED_ACTION_TYPES)}"
            )
        raw_action_space = dict((bundle.raw_manifest or {}).get("action_space") or {})
        clip = raw_action_space.get("clip")
        clip_val = None
        try:
            clip_val = float(clip) if clip is not None else None
        except (TypeError, ValueError):
            clip_val = None
        self._spec = ActionSpec(
            action_type=bundle.action_type,
            dim=bundle.action_dim,
            joint_names=list(bundle.joint_names),
            clip_min=(-clip_val if clip_val is not None else None),
            clip_max=clip_val,
            scale=float(raw_action_space.get("scale", 1.0) or 1.0),
        )
        self._pd_controller = pd_controller
        # Joint reference frames. ``bundle_space`` is the order the
        # policy was trained against (and the order ``raw_action`` arrives
        # in); ``ctrl_space`` is the order ``mj_data.ctrl[i]`` consumes
        # actuator commands. The two are *independent*: in MuJoCo's stock
        # ``unitree_go2`` they happen to coincide; in Unitree's own
        # ``go2.xml`` and most bipeds/manipulators they don't. When the
        # spaces are configured, every join read/write goes through
        # explicit ``permute(source → target)`` instead of any implicit
        # "the lists happen to be the same" assumption.
        self._bundle_space = bundle_space
        self._ctrl_space = ctrl_space

    @property
    def pd_controller(self) -> Optional[Any]:
        """Public accessor — used by PolicyRunner.step to decide whether
        to route Isaac Lab bundles through the PD path."""
        return self._pd_controller

    def expected_dim(self) -> int:
        return self._spec.dim

    def remap_to_env(self, action: np.ndarray, env: SimEnvContext) -> np.ndarray:
        """Legacy bundle→env joint remap, kept for backward compat with
        callers that don't yet pass JointSpace objects.

        New code paths should rely on the JointSpace permutation inside
        ``apply``/``apply_with_pd`` instead — the spaces handle the
        bundle→qpos and bundle→ctrl direction explicitly.
        """
        bundle_joints = self._spec.joint_names
        env_joints = list(env.joint_names)

        if bundle_joints == env_joints:
            return action.astype(np.float32)

        if canonicalize_joint_names(bundle_joints) == canonicalize_joint_names(env_joints):
            return action.astype(np.float32)

        bundle_index_by_canonical = {
            canonicalize_joint_name(jname): idx
            for idx, jname in enumerate(bundle_joints)
        }

        env_to_bundle: list[int] = []
        missing: list[str] = []
        for jname in env_joints:
            idx = bundle_index_by_canonical.get(canonicalize_joint_name(jname))
            if idx is None:
                missing.append(jname)
            else:
                env_to_bundle.append(idx)

        if missing:
            raise ValueError(
                f"Cannot remap action: env joints not in bundle: {missing}. "
                f"Bundle joints: {bundle_joints}"
            )

        return action.astype(np.float32)[env_to_bundle]

    def _bundle_to_ctrl(self, vec_bundle: np.ndarray) -> np.ndarray:
        """Permute a bundle-space vector into ctrl-space.

        When the JointSpace pair is configured this is a single
        ``permute`` call (cached internally). When it isn't configured
        (legacy SB3 callers without spaces) we fall through to the
        ``remap_to_env`` shim using ``env.joint_names`` — which is what
        the old code used to do — to keep backward compatibility.
        """
        if self._bundle_space is not None and self._ctrl_space is not None:
            return self._bundle_space.permute(vec_bundle, self._ctrl_space).astype(np.float32)
        return vec_bundle.astype(np.float32)

    def apply(self, raw_action: np.ndarray, env: SimEnvContext) -> np.ndarray:
        """Apply scale, clip, joint-space conversion, then adapter ctrl hook."""
        if raw_action.shape[0] != self._spec.dim:
            raise ValueError(
                f"ActionApplier.apply(): input has dimension {raw_action.shape[0]}, "
                f"expected {self._spec.dim}."
            )

        out = raw_action.flatten().astype(np.float32)

        if self._spec.scale != 1.0:
            out = out * np.float32(self._spec.scale)

        if self._spec.clip_min is not None or self._spec.clip_max is not None:
            out = np.clip(out, self._spec.clip_min, self._spec.clip_max).astype(np.float32)

        # Bundle → ctrl. Modern callers go through JointSpace permutation;
        # legacy callers get the env-name based remap_to_env.
        if self._bundle_space is not None and self._ctrl_space is not None:
            in_ctrl = self._bundle_to_ctrl(out)
        else:
            in_ctrl = self.remap_to_env(out, env)

        adapter = getattr(env, "adapter", None)
        action_to_ctrl = getattr(adapter, "_action_to_ctrl", None)
        if callable(action_to_ctrl):
            try:
                converted = np.asarray(action_to_ctrl(in_ctrl), dtype=np.float32)
                if converted.shape == in_ctrl.shape:
                    return converted
                if converted.ndim == 1 and converted.size == in_ctrl.size:
                    return converted.reshape(in_ctrl.shape)
            except Exception:
                pass
        return in_ctrl

    def apply_with_pd(
        self,
        raw_action: np.ndarray,
        env: SimEnvContext,
    ) -> np.ndarray:
        """Apply action through PD controller (Isaac Lab-style).

        Pipeline (when bundle_space and ctrl_space are configured —
        the supported path for IL bundles):

            raw_action  [bundle order]
                ↓ * scale
            target      [bundle order]
                ↓ + default_pos (in bundle order, from PDController)
                ↓ - current_pos_in_bundle_order (qpos→bundle permutation)
                ↓ - current_vel_in_bundle_order
            torques     [bundle order]   (Kp/Kd computation)
                ↓ permute bundle → ctrl
            torques     [ctrl order, ready for mj_data.ctrl]

        Without explicit spaces this falls back to ``apply``.
        """
        if self._pd_controller is None:
            # Surfaced once per applier instance: an IL bundle reaching this
            # branch means PolicyRunner._maybe_build_il_pd_controller silently
            # returned None (env.yaml missing or unparseable). The bundle
            # will then be applied as raw position targets — see the long
            # rationale in the except branch below. Fix #2's loader-side
            # error already fires once at load time; this is a defence-in-
            # depth warning at the first call site.
            if not getattr(self, "_warned_no_pd_controller", False):
                log.error(
                    "ActionApplier.apply_with_pd: no PDController attached. "
                    "If this is an Isaac Lab bundle, the env.yaml could not "
                    "be loaded and PD gains/default-pose were lost — replay "
                    "will not match training."
                )
                self._warned_no_pd_controller = True
            return self.apply(raw_action, env)
        if self._bundle_space is None or self._ctrl_space is None:
            # Cannot run the convention-correct PD path without explicit
            # joint spaces — the old "trust env.joint_names" approach
            # produced silently misaligned torques. Fail loud rather
            # than fail wrong.
            log.warning(
                "ActionApplier.apply_with_pd: bundle_space/ctrl_space not "
                "configured; falling back to apply() — this will produce "
                "miscomputed torques for non-trivial joint orderings."
            )
            return self.apply(raw_action, env)

        try:
            from src.system.policy.joint_space import JointSpace  # noqa: F401

            # 1. raw_action is in bundle order.
            target_bundle = raw_action.flatten().astype(np.float32)
            if self._spec.scale != 1.0:
                target_bundle = target_bundle * np.float32(self._spec.scale)

            # 2. Read MuJoCo state in qpos order (the order the joint
            #    block in qpos[7:] / qvel[6:] uses), permute to bundle.
            qpos_space = getattr(self, "_qpos_space_for_pd", None)
            qpos_full = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
            qvel_full = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()

            joint_block_pos = qpos_full[7:]
            joint_block_vel = qvel_full[6:]

            if qpos_space is not None:
                # Length-match to qpos_space then permute to bundle.
                n_qpos = len(qpos_space)
                if joint_block_pos.shape[0] < n_qpos:
                    joint_block_pos = np.pad(joint_block_pos, (0, n_qpos - joint_block_pos.shape[0]))
                if joint_block_vel.shape[0] < n_qpos:
                    joint_block_vel = np.pad(joint_block_vel, (0, n_qpos - joint_block_vel.shape[0]))
                current_pos_bundle = qpos_space.permute(joint_block_pos[:n_qpos], self._bundle_space).astype(np.float32)
                current_vel_bundle = qpos_space.permute(joint_block_vel[:n_qpos], self._bundle_space).astype(np.float32)
            else:
                # No qpos_space → assume the joint block is already in
                # bundle order. PolicyRunner.load() always sets
                # _qpos_space_for_pd, so this branch is for unit tests
                # / direct ActionApplier construction only.
                n = len(self._bundle_space)
                current_pos_bundle = joint_block_pos[:n].astype(np.float32)
                current_vel_bundle = joint_block_vel[:n].astype(np.float32)

            # 3. PD compute (everything in bundle order).
            torques_bundle = self._pd_controller.compute(
                target_bundle,
                current_pos_bundle,
                current_vel_bundle,
                use_default_offset=True,
            )

            # 4. Permute bundle → ctrl on the way out.
            ctrl_torques = self._bundle_to_ctrl(np.asarray(torques_bundle, dtype=np.float32))

            # 5. Gear correction: MuJoCo motor actuators apply
            #    ``force = ctrl * gear``. The PD computed the desired
            #    JOINT torque, so we divide by gear to get the correct
            #    ctrl value. For menagerie models (gear=1) this is a
            #    no-op; for gym-style models (gear=33.5) it prevents
            #    the gear from amplifying the torque.
            mj_model = getattr(env, "mj_model", None)
            if mj_model is not None:
                n = min(ctrl_torques.shape[0], int(mj_model.nu))
                gear = np.asarray(mj_model.actuator_gear[:n, 0], dtype=np.float32)
                gear = np.where(gear > 0, gear, 1.0)
                ctrl_torques[:n] = ctrl_torques[:n] / gear

            return ctrl_torques
        except Exception as exc:
            # IL bundles MUST be applied through the PD path — falling back
            # to apply() writes raw position targets into mj_data.ctrl, which
            # changes the actuator semantics from "torque via Kp(target-pos)
            # + Kd(0-vel)" to "position target", a completely different
            # control regime from training. This silent degradation was the
            # leading cause of sim2sim "flailing" reports. Promote to error
            # so the divergence is visible in CMD logs the first time it
            # happens, instead of being buried at debug level.
            log.error(
                "ActionApplier.apply_with_pd: PD path raised %s: %s — "
                "falling back to position-target apply(). The bundle was "
                "trained with torque-style PD; this fallback will produce "
                "fundamentally different joint dynamics and is the leading "
                "cause of Isaac Lab → MuJoCo sim2sim divergence. Fix the "
                "exception above instead of relying on this fallback.",
                type(exc).__name__,
                exc,
            )
            return self.apply(raw_action, env)

    # ──────────────────────────────────────────────────────────────────
    # Setup hook used by PolicyRunner.load to attach the qpos space
    # without changing the constructor signature for legacy callers.
    # ──────────────────────────────────────────────────────────────────

    def attach_qpos_space(self, qpos_space: "JointSpace") -> None:
        """Bind the MuJoCo qpos joint order so apply_with_pd can read
        ``mj_data.qpos[7:]`` and reorder it into bundle order. Called by
        ``PolicyRunner.load`` once per session."""
        self._qpos_space_for_pd = qpos_space

    def dispatch_to_backend(
        self,
        raw_action: np.ndarray,
        env: SimEnvContext,
        *,
        skill_manifest: Any = None,
        context: Optional[Dict[str, Any]] = None,
        backend: str = "mujoco",
    ) -> Dict[str, Any]:
        """Apply action then dispatch through the unified ActionDispatcher.

        This method extends the standard :meth:`apply` pipeline with a
        final dispatch step through the unified ActionDispatcher, routing
        the processed action to the correct sim/real executor.

        Parameters
        ----------
        raw_action:
            Raw policy output action vector.
        env:
            Simulation environment context.
        skill_manifest:
            Active SkillManifest (if available).  Used for action_range
            clamping and backend routing in the dispatcher.
        context:
            Runtime context dict (must include ``"robot_model"`` for SDK path).
        backend:
            Target backend (``"mujoco"`` or ``"sdk"``).

        Returns
        -------
        dict
            ``{"action": np.ndarray, "dispatch_result": ExecutorResult.to_dict()}``
            where ``action`` is the processed action vector and
            ``dispatch_result`` is the executor outcome.
        """
        action = self.apply(raw_action, env)

        dispatch_result = None
        if skill_manifest is not None:
            try:
                from src.system.runtime.action_dispatcher import ActionDispatcher

                dispatcher = ActionDispatcher()
                ctx = dict(context or {})
                ctx.setdefault("robot_model", getattr(env, "adapter", None))

                result = dispatcher.dispatch_policy_action(
                    action,
                    skill_manifest=skill_manifest,
                    context=ctx,
                    backend=backend,
                )
                dispatch_result = result.to_dict()
            except Exception as exc:
                log.debug("ActionDispatcher delegation failed: %s", exc)
                dispatch_result = {
                    "success": False,
                    "operation": "policy_action",
                    "reason": str(exc),
                    "diag_code": "dispatcher.unavailable",
                }

        return {
            "action": action,
            "dispatch_result": dispatch_result,
        }
