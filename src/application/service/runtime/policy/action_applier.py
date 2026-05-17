"""ActionApplier — raw policy output → control vector (verbatim port from DEMO).

Phase 3 RELEASE port. Adjustments:

* ``manifest_schema`` import → ``application.training.manifest_schema``.
* ``joint_space`` import stays relative.
* ``ActionDispatcher`` (DEMO ``runtime/action_dispatcher``) is gated on
  RELEASE having that module — at port time it does not, so
  ``dispatch_to_backend`` returns ``dispatch_result=None``. The MVP
  Phase 3 round-trip uses ``apply`` / ``apply_with_pd`` directly via
  PolicyRunner, never the dispatcher path.

The scale → clip → bundle→ctrl permute order, the gear correction
on the ctrl-space output, the no-PD-controller error path — all
byte-identical with DEMO.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from application.training.manifest_schema import CheckpointBundle

from .joint_name_utils import canonicalize_joint_name, canonicalize_joint_names
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
    # ``scale`` may be a scalar (the typical IL case, e.g. 0.25) or a
    # per-joint array (when the bundle's deploy_contract.action.scale is
    # a list). Both broadcast cleanly against a 12-dim action.
    scale: Any = 1.0


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

        # Prefer the deploy_contract's action.scale — it is the
        # authoritative copy for IL bundles. Legacy ``BundleExporter``
        # versions wrote only ``action_space.dim`` and ``action_space.type``
        # at the top level (no ``scale``), so reading just from
        # ``raw_manifest['action_space']`` silently defaulted scale to 1.0
        # and dropped the trained Isaac Lab 0.25× — the policy's targets
        # ended up 4× too large, producing PD wind-up and "twitching"
        # sim2sim runs. Falls back to ``action_space.scale`` if present
        # (newer exporters mirror it for defense in depth) and finally 1.0.
        contract_action = getattr(
            getattr(bundle, "deploy_contract", None), "action", None
        )
        scale_raw: Any
        scale_source: str
        if contract_action is not None and getattr(contract_action, "scale", None) is not None:
            scale_raw = contract_action.scale
            scale_source = "deploy_contract.action.scale"
        elif "scale" in raw_action_space and raw_action_space["scale"] is not None:
            scale_raw = raw_action_space["scale"]
            scale_source = "action_space.scale"
        else:
            scale_raw = 1.0
            scale_source = "default(1.0)"

        scale_final: Any
        if isinstance(scale_raw, (list, tuple)):
            arr = np.asarray(scale_raw, dtype=np.float32)
            scale_final = float(arr[0]) if arr.size == 1 else arr
        else:
            try:
                scale_final = float(scale_raw)
            except (TypeError, ValueError):
                scale_final = 1.0
        if not getattr(ActionApplier, "_logged_action_scale", False):
            log.info(
                "ActionApplier: action scale=%s (source=%s)",
                scale_final
                if not isinstance(scale_final, np.ndarray)
                else f"array[{scale_final.size}] first={float(scale_final[0])}",
                scale_source,
            )
            ActionApplier._logged_action_scale = True

        self._spec = ActionSpec(
            action_type=bundle.action_type,
            dim=bundle.action_dim,
            joint_names=list(bundle.joint_names),
            clip_min=(-clip_val if clip_val is not None else None),
            clip_max=clip_val,
            scale=scale_final,
        )
        self._pd_controller = pd_controller
        self._bundle_space = bundle_space
        self._ctrl_space = ctrl_space

        # offset_mode: PD target = scaled_action + default_pos when
        # ``"default_joint_pos"`` (the Isaac Lab convention the deploy
        # stack historically hardcoded), or = scaled_action when
        # ``"zero"`` (SB3 canvas users can opt out via
        # ``actor.action_use_default_offset=False``). Reading from
        # ``deploy_contract.action.offset_mode`` keeps the deploy
        # behaviour aligned with whatever the trainer produced.
        self._use_default_offset = True
        try:
            if contract_action is not None:
                mode = str(
                    getattr(contract_action, "offset_mode", "default_joint_pos") or ""
                ).strip().lower()
                if mode == "zero":
                    self._use_default_offset = False
        except Exception:                                   # pragma: no cover
            self._use_default_offset = True

    @property
    def pd_controller(self) -> Optional[Any]:
        """Public accessor — used by PolicyRunner.step to decide whether
        to route Isaac Lab bundles through the PD path."""
        return self._pd_controller

    def expected_dim(self) -> int:
        return self._spec.dim

    def remap_to_env(self, action: np.ndarray, env: SimEnvContext) -> np.ndarray:
        """Legacy bundle→env joint remap (kept for callers without JointSpaces)."""
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
        """Permute a bundle-space vector into ctrl-space."""
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

        # Always broadcast against np.asarray — works for scalar 1.0
        # (no-op), scalar 0.25 (IL Go2 typical), or per-joint array scale.
        out = out * np.asarray(self._spec.scale, dtype=np.float32)

        if self._spec.clip_min is not None or self._spec.clip_max is not None:
            out = np.clip(out, self._spec.clip_min, self._spec.clip_max).astype(np.float32)

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

        Pipeline (when bundle_space and ctrl_space are configured):

            raw_action  [bundle order]
                ↓ * scale
            target      [bundle order]
                ↓ + default_pos (in bundle order, from PDController)
                ↓ - current_pos_in_bundle_order (qpos→bundle permutation)
                ↓ - current_vel_in_bundle_order
            torques     [bundle order]   (Kp/Kd computation)
                ↓ permute bundle → ctrl
            torques     [ctrl order, ready for mj_data.ctrl]
        """
        if self._pd_controller is None:
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
            log.warning(
                "ActionApplier.apply_with_pd: bundle_space/ctrl_space not "
                "configured; falling back to apply() — this will produce "
                "miscomputed torques for non-trivial joint orderings."
            )
            return self.apply(raw_action, env)

        try:
            from .joint_space import JointSpace  # noqa: F401

            target_bundle = raw_action.flatten().astype(np.float32)
            target_bundle = target_bundle * np.asarray(
                self._spec.scale, dtype=np.float32
            )

            qpos_space = getattr(self, "_qpos_space_for_pd", None)
            qpos_full = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
            qvel_full = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()

            joint_block_pos = qpos_full[7:]
            joint_block_vel = qvel_full[6:]

            if qpos_space is not None:
                n_qpos = len(qpos_space)
                if joint_block_pos.shape[0] < n_qpos:
                    joint_block_pos = np.pad(
                        joint_block_pos, (0, n_qpos - joint_block_pos.shape[0])
                    )
                if joint_block_vel.shape[0] < n_qpos:
                    joint_block_vel = np.pad(
                        joint_block_vel, (0, n_qpos - joint_block_vel.shape[0])
                    )
                current_pos_bundle = qpos_space.permute(
                    joint_block_pos[:n_qpos], self._bundle_space
                ).astype(np.float32)
                current_vel_bundle = qpos_space.permute(
                    joint_block_vel[:n_qpos], self._bundle_space
                ).astype(np.float32)
            else:
                n = len(self._bundle_space)
                current_pos_bundle = joint_block_pos[:n].astype(np.float32)
                current_vel_bundle = joint_block_vel[:n].astype(np.float32)

            if not getattr(type(self), "_logged_first_apply_pd", False):
                try:
                    names = (
                        list(self._bundle_space.joints)
                        if self._bundle_space is not None
                        else [f"j{i}" for i in range(target_bundle.shape[0])]
                    )
                    scale_arr = np.asarray(self._spec.scale, dtype=np.float32)
                    if scale_arr.ndim == 0 or scale_arr.size == 1:
                        scale_arr = np.full(
                            target_bundle.shape,
                            float(scale_arr.reshape(()).item()) if scale_arr.size == 1
                            else float(scale_arr),
                            dtype=np.float32,
                        )
                    default_pos_arr = np.asarray(
                        self._pd_controller._default_pos, dtype=np.float32
                    )
                    raw_flat = raw_action.flatten()
                    rows = []
                    for i in range(target_bundle.shape[0]):
                        name = names[i] if i < len(names) else "?"
                        rows.append(
                            f"  [{i:2d}] {name:24s} "
                            f"raw_action={float(raw_flat[i]):+.4f} "
                            f"scale={float(scale_arr[i]):.3f} "
                            f"target(scaled)={float(target_bundle[i]):+.4f} "
                            f"default_pos={float(default_pos_arr[i]):+.3f} "
                            f"current_pos={float(current_pos_bundle[i]):+.3f} "
                            f"current_vel={float(current_vel_bundle[i]):+.3f}"
                        )
                    log.info(
                        "ActionApplier first apply_with_pd dump (bundle order, "
                        "use_default_offset=%s; PD target = scaled%s):\n%s",
                        self._use_default_offset,
                        " + default_pos" if self._use_default_offset else " (no offset)",
                        "\n".join(rows),
                    )
                    type(self)._logged_first_apply_pd = True
                except Exception:
                    log.exception("first apply_with_pd dump failed")

            torques_bundle = self._pd_controller.compute(
                target_bundle,
                current_pos_bundle,
                current_vel_bundle,
                use_default_offset=self._use_default_offset,
            )

            ctrl_torques = self._bundle_to_ctrl(np.asarray(torques_bundle, dtype=np.float32))

            # Gear correction: ctrl * gear == joint torque, so divide.
            mj_model = getattr(env, "mj_model", None)
            if mj_model is not None:
                n = min(ctrl_torques.shape[0], int(mj_model.nu))
                gear = np.asarray(mj_model.actuator_gear[:n, 0], dtype=np.float32)
                gear = np.where(gear > 0, gear, 1.0)
                ctrl_torques[:n] = ctrl_torques[:n] / gear

            return ctrl_torques
        except Exception as exc:
            log.error(
                "ActionApplier.apply_with_pd: PD path raised %s: %s — "
                "falling back to position-target apply(). The bundle was "
                "trained with torque-style PD; this fallback will produce "
                "fundamentally different joint dynamics and is the leading "
                "cause of Isaac Lab → MuJoCo sim2sim divergence.",
                type(exc).__name__,
                exc,
            )
            return self.apply(raw_action, env)

    # ──────────────────────────────────────────────────────────────────
    # Setup hook used by PolicyRunner.load to attach the qpos space.
    # ──────────────────────────────────────────────────────────────────

    def attach_qpos_space(self, qpos_space: "JointSpace") -> None:
        """Bind the MuJoCo qpos joint order so apply_with_pd can read
        ``mj_data.qpos[7:]`` and reorder it into bundle order."""
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
        """Apply action and (if available) dispatch through ActionDispatcher.

        RELEASE Phase 3: ``application/service/runtime/action_dispatcher.py``
        is not yet ported. Until then, the dispatch step is skipped and
        ``dispatch_result`` is returned as None. Callers that need the
        dispatcher path should wait for the runtime/dispatcher port.
        """
        action = self.apply(raw_action, env)

        dispatch_result = None
        if skill_manifest is not None:
            try:
                from application.service.runtime.action_dispatcher import (
                    ActionDispatcher,
                )

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
            except ImportError:
                # ActionDispatcher not ported yet — skip silently. Phase 3
                # round-trip never sets skill_manifest, so this branch is
                # only hit by callers explicitly opting into the dispatcher.
                dispatch_result = None
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
