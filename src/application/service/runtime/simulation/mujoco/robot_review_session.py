# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Robot-only review session — preview a robot asset in MuJoCo without a policy.

Pure MJCF viewer: load the asset, write either the home keyframe pose or
the user-supplied :class:`InitPoseOverride` into ``mj_data.qpos`` (under
``viewer.lock()`` so it lands after ``simulate.load``'s internal
``mj_resetData``), then pump ``viewer.sync()``. No PolicyRunner, no
bundle default-pose overlay, no ``il_init_base_pos`` lift, no contact-
based ground-clearance fallback — the user explicitly asked for the
"just load MJCF and show it" path, so this module does not depend on
:class:`MjSimEnv` at all.

Two live-physics modes:

* ``live_physics=False`` (default) — the viewer is pumped at ~60 Hz but
  ``mj_step`` is **not** called, so the robot stays frozen at the
  keyframe (or override) pose. Useful for inspecting the design-time
  stance.
* ``live_physics=True`` — physics runs at the model timestep with
  ``ctrl=0`` (gravity-only, no actuators). The robot will typically
  settle or collapse onto the ground. Wall-clock pacing is done locally
  from ``model.opt.timestep`` instead of borrowing ``MjSimEnv.render``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from unitport_sdk import Task, log_info, log_warning

from application.service.robot_init_poses import InitPoseOverride


# GLFW key code for Backspace — the MuJoCo viewer's built-in "Reset"
# shortcut. We intercept it so the user gets back to our pose (keyframe
# or override) instead of the MJCF's design rest pose that the viewer's
# default Reset would produce via ``mj_resetData`` → ``qpos = m.qpos0``.
_GLFW_KEY_BACKSPACE = 259

# Belt-and-suspenders count for the post-launch pose re-apply. In the
# current mujoco bindings ``simulate.load`` runs ``mj_resetData`` exactly
# once during ``_reload``, and ``viewer.lock()`` synchronises with that
# load, so step 1 (the locked write) should be sufficient. Three extra
# frames is cheap insurance in case a future bindings revision moves the
# internal reset into the first one or two render-loop iterations.
_POST_LAUNCH_FALLBACK_FRAMES = 3
_POST_LAUNCH_FALLBACK_DT = 0.02


class RobotReviewTask(Task):
    """SDK Task: launch a passive MuJoCo viewer on an MJCF asset.

    Cancellable between viewer-sync iterations. Returns a result dict
    via ``task_finished`` for telemetry.
    """

    def __init__(
        self,
        sku: str,
        scene_id: str,
        *,
        live_physics: bool = False,
        init_pose_override: Optional[InitPoseOverride] = None,
    ) -> None:
        suffix = "physics" if live_physics else "static"
        super().__init__(f"Review {sku} ({suffix})")
        self._sku = str(sku)
        self._scene_id = str(scene_id)
        self._live_physics = bool(live_physics)
        self._init_pose_override = init_pose_override

    def run(self) -> Dict[str, Any]:
        # Lazy imports — mujoco is heavy and absent on headless CI.
        import mujoco
        import mujoco.viewer

        from .mj_actor import MjActor

        self.check_cancelled()

        # ── 1. Resolve MJCF path + joint name list via MjActor ────────
        # MjActor.from_sku builds its own (model, data) pair internally;
        # we discard those and load a fresh pair below so this task does
        # not share state with anything else.
        actor = MjActor.from_sku(self._sku)
        mjcf_path = actor.mjcf_path
        joint_names: List[str] = list(actor.joint_names)
        self.check_cancelled()

        # ── 2. Load fresh (model, data) directly from the MJCF ────────
        try:
            model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        except Exception as exc:
            log_warning(
                f"[review/robot] MJCF load failed for {self._sku!r} "
                f"({mjcf_path}): {exc}"
            )
            return {
                "viewer_used": False,
                "sku": self._sku,
                "scene_id": self._scene_id,
                "live_physics": self._live_physics,
                "aborted": "mjcf_load_failed",
            }
        data = mujoco.MjData(model)

        # ── 3. Locate the "home" keyframe (or fall back to keyframe 0) ─
        kid = self._resolve_home_keyframe_id(mujoco, model)

        # ── 4. Locate the free-joint root (if any) ────────────────────
        free_qposadr = -1
        has_freejoint_root = False
        for jid in range(int(model.njnt)):
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                free_qposadr = int(model.jnt_qposadr[jid])
                has_freejoint_root = True
                break

        # ── 5. Build the pose-applier closure ─────────────────────────
        # Collect (jid, qposadr) for actuated joints (skip free + ball)
        # in MuJoCo joint order, plus their parallel IR-role list. The
        # IR-role list is what InitPoseOverride.to_bundle_order maps to
        # joint angles. Building it from the FILTERED joint list avoids
        # the off-by-one bug where the free joint's empty IR role would
        # eat slot 0 and shift every leg by one.
        actuated_jids: List[int] = []
        actuated_qposadrs: List[int] = []
        actuated_ir_roles: List[str] = []
        unmapped_joint_names: List[str] = []
        if self._init_pose_override is not None:
            name_to_role = self._build_name_to_role()
            for jid in range(int(model.njnt)):
                jtype = int(model.jnt_type[jid])
                if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                    continue
                if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
                    continue
                actuated_jids.append(jid)
                actuated_qposadrs.append(int(model.jnt_qposadr[jid]))
                jname = str(joint_names[jid] if jid < len(joint_names) else "")
                ir_role = name_to_role.get(jname, "")
                if not ir_role:
                    unmapped_joint_names.append(jname)
                actuated_ir_roles.append(ir_role)
            # CLAUDE.md §1.8: any actuated joint without an IR-role mapping
            # is a registry vs. MJCF mismatch that used to silently zero-fill
            # the entire pose and produce the "四脚伸直" T-pose bug. Fail loud
            # so the user sees the exact joint names missing from the registry
            # MJCF table for this SKU.
            if unmapped_joint_names:
                raise RuntimeError(
                    f"[review/robot] {len(unmapped_joint_names)} MuJoCo joint(s) "
                    f"have no IR-role mapping in registry "
                    f"joints_per_format['MJCF'] for sku {self._sku!r}: "
                    f"{unmapped_joint_names}. The MJCF asset declares "
                    f"these joints but the canonical/overlay registry does "
                    f"not — fix registers/data/robots_canonical.json (or the "
                    f"user overlay) by adding entries with the correct "
                    f"ir_role for each. Silent zero-fill (the previous "
                    f"behavior) is forbidden by CLAUDE.md §1.8."
                )
        override = self._init_pose_override

        def _apply_initial_pose() -> None:
            """Write override / keyframe / qpos0 into ``data.qpos``.

            Priority: override (if given) > home keyframe (if found) >
            ``mj_resetData`` (qpos0). Always finishes with ``mj_forward``
            so ``xpos`` reflects the new ``qpos``.
            """
            if override is not None:
                qpos_bundle = override.to_bundle_order(
                    bundle_ir_roles=actuated_ir_roles,
                    default_joint_pos=[0.0] * len(actuated_ir_roles),
                )
                if has_freejoint_root:
                    bp = override.base_pos
                    # USD↔MJCF anchor offset: the user's init_pos_z is
                    # authored against USD/IsaacLab semantics; in MJCF the
                    # same value drops the trunk too low (穿地). The
                    # one-shot calibration overlay holds the per-SKU lift
                    # needed at standing pose. See plan
                    # curious-nibbling-plum.md and
                    # application.training.validation.mjcf_base_calibration.
                    from application.service.robot_assets.runtime import (
                        read_mjcf_base_offset,
                    )
                    off_z, _ = read_mjcf_base_offset(self._sku)
                    data.qpos[free_qposadr + 0] = float(bp[0])
                    data.qpos[free_qposadr + 1] = float(bp[1])
                    data.qpos[free_qposadr + 2] = float(bp[2]) + off_z
                    data.qpos[free_qposadr + 3] = 1.0
                    data.qpos[free_qposadr + 4] = 0.0
                    data.qpos[free_qposadr + 5] = 0.0
                    data.qpos[free_qposadr + 6] = 0.0
                for idx, qposadr in enumerate(actuated_qposadrs):
                    if idx >= len(qpos_bundle):
                        break
                    data.qpos[qposadr] = float(qpos_bundle[idx])
            elif kid >= 0:
                mujoco.mj_resetDataKeyframe(model, data, kid)
            else:
                mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

        # ── 6. First write (pre-launch) ───────────────────────────────
        try:
            _apply_initial_pose()
        except Exception as exc:
            log_warning(
                f"[review/robot] pre-launch pose apply failed: {exc}"
            )
        self._log_pose_finalized(model, data, stage="pre-launch")

        # ── 7. launch_passive with Backspace key callback ─────────────
        viewer = None
        try:
            def _on_key(keycode: int) -> None:
                if keycode != _GLFW_KEY_BACKSPACE:
                    return
                try:
                    _apply_initial_pose()
                    if viewer is not None:
                        viewer.sync()
                    log_info("[review/robot] keyframe re-apply on Backspace ok")
                except Exception as exc:  # noqa: BLE001
                    log_warning(
                        f"[review/robot] keyframe re-apply on Backspace "
                        f"failed: {exc}"
                    )

            viewer = mujoco.viewer.launch_passive(
                model, data, key_callback=_on_key,
            )
        except Exception as exc:
            log_warning(
                f"[review/robot] viewer unavailable — review aborted: {exc}"
            )
            return {
                "viewer_used": False,
                "sku": self._sku,
                "scene_id": self._scene_id,
                "live_physics": self._live_physics,
                "aborted": "no viewer",
            }

        # ── 8. Atomic post-launch write under viewer.lock() ──────────
        # The real fix: simulate.load (run by the viewer's side thread
        # during _reload) calls mj_resetData internally, which writes
        # qpos = m.qpos0 and clobbers our pre-launch write. Grabbing
        # viewer.lock() serialises against that side thread so this
        # write lands strictly after simulate.load returns.
        try:
            with viewer.lock():
                _apply_initial_pose()
            viewer.sync()
        except Exception as exc:
            log_warning(
                f"[review/robot] locked post-launch pose apply failed: {exc}"
            )

        # ── 9. Belt-and-suspenders: 3 extra re-applies over ~60 ms ───
        for _ in range(_POST_LAUNCH_FALLBACK_FRAMES):
            try:
                time.sleep(_POST_LAUNCH_FALLBACK_DT)
                _apply_initial_pose()
                viewer.sync()
            except Exception:
                break

        self._log_pose_finalized(model, data, stage="post-launch")
        log_info(
            f"[review/robot] viewer launched for {self._sku} "
            f"(live_physics={self._live_physics})"
        )

        # ── 10/11. Run the chosen loop until the viewer is closed ────
        try:
            if self._live_physics:
                self._run_with_physics(mujoco, model, data, viewer)
            else:
                self._wait_for_viewer_close(viewer)
            return {
                "viewer_used": True,
                "sku": self._sku,
                "scene_id": self._scene_id,
                "live_physics": self._live_physics,
            }
        finally:
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_home_keyframe_id(mujoco_mod: Any, model: Any) -> int:
        """Return the keyframe id for ``home`` / ``stand`` / etc., or 0.

        Returns ``-1`` if the model ships no keyframes at all — caller
        falls back to ``mj_resetData`` (qpos0).
        """
        # WHY KEPT (Rule 1.c — MuJoCo C API boundary): model.nkey is a
        # ctypes accessor that may raise AttributeError on stripped builds
        # of mujoco; mj_name2id may raise when the keyframe table is
        # absent. Both are environmental rather than logical failures —
        # this helper is invoked only when no InitPoseOverride is given,
        # so falling back to qpos0 (caller's ``mj_resetData`` branch) is
        # the documented MuJoCo convention. The InitPoseOverride path
        # (the one the user actually triggers via Review Pose) does NOT
        # depend on this helper.
        try:
            nkey = int(model.nkey)
        except Exception:
            return -1
        if nkey <= 0:
            return -1
        for name in ("home", "stand", "stand_up", "initial", "default"):
            try:
                cand = int(
                    mujoco_mod.mj_name2id(
                        model, int(mujoco_mod.mjtObj.mjOBJ_KEY), name
                    )
                )
            except Exception:
                cand = -1
            if cand >= 0:
                return cand
        return 0

    def _build_name_to_role(self) -> Dict[str, str]:
        """Build a physical-joint-name → IR-role lookup for the MJCF format.

        Canvas IR-only rule (memory: canvas-ir-only) carves out exactly one
        IR→physical mapping site: the sim-preview path. This function is
        that site for MuJoCo review — it translates IR roles in the user's
        InitPoseOverride to the MJCF physical joint names MjData.qpos is
        keyed by.

        Reads ``registers.robots.get_robot(sku)['joints_per_format']['MJCF']``.
        Previous implementation read ``entry['joints']`` which never existed
        on the canonical schema (Phase-5 unified registry uses per-format
        sub-tables); the silent empty-dict return then propagated to
        ``actuated_ir_roles = ['', '', ...]``, causing every IR-role lookup
        in ``InitPoseOverride.to_bundle_order`` to miss and the viewer to
        render the default qpos=0 pose ("四脚伸直"). Fixed 2026-05-19.
        """
        from registers import robots as _robots_registry

        entry = _robots_registry.get_robot(self._sku) or {}
        if not entry:
            raise RuntimeError(
                f"[review/robot] sku {self._sku!r} is not registered — "
                f"cannot build IR→MJCF joint mapping for the review viewer."
            )
        joints_dict = (
            entry.get("joints_per_format", {}).get("MJCF", {}) or {}
        )
        if not isinstance(joints_dict, dict) or not joints_dict:
            raise RuntimeError(
                f"[review/robot] robot {self._sku!r} has no "
                f"joints_per_format['MJCF'] entries in the registry — "
                f"the MuJoCo review viewer cannot map IR roles to MJCF "
                f"physical joint names. Add MJCF joints to "
                f"registers/data/robots_canonical.json (or the user "
                f"overlay) for this SKU."
            )
        name_to_role: Dict[str, str] = {}
        for jspec in joints_dict.values():
            if not isinstance(jspec, dict):
                continue
            n = str(jspec.get("name", ""))
            r = str(jspec.get("ir_role", ""))
            if n:
                name_to_role[n] = r
        return name_to_role

    def _log_pose_finalized(self, model: Any, data: Any, *, stage: str) -> None:
        """One-shot diagnostic so the actual qpos hitting the viewer is
        visible in logs without spamming every frame.
        """
        try:
            nq = int(model.nq)
            base_z = float(data.qpos[2]) if nq >= 3 else float("nan")
            joints_slice = (
                data.qpos[7 : min(nq, 19)].tolist() if nq >= 8 else []
            )
            log_info(
                f"[review/robot] pose finalized ({stage}) base_z={base_z:.4f} "
                f"joints[0:3]={['%.3f' % v for v in joints_slice[:3]]} "
                f"override={'yes' if self._init_pose_override is not None else 'no'}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------
    def _wait_for_viewer_close(self, viewer: Any) -> None:
        """Pump ``viewer.sync()`` at ~60 Hz until the user closes it.

        Physics is NOT advanced, so the robot stays at its keyframe (or
        override) pose for inspection.
        """
        period = 1.0 / 60.0
        while True:
            try:
                if not viewer.is_running():
                    return
            except Exception:
                return
            try:
                viewer.sync()
            except Exception:
                return
            try:
                self.check_cancelled()
            except Exception:
                return
            time.sleep(period)

    def _run_with_physics(
        self, mujoco_mod: Any, model: Any, data: Any, viewer: Any,
    ) -> None:
        """Step physics at the model timestep with ctrl=0 until close.

        ``ctrl=0`` means no actuator torques — the robot is at the mercy
        of gravity from its initial pose. Wall-clock pacing is done from
        ``model.opt.timestep`` directly (no ``MjSimEnv.render`` borrow).
        """
        try:
            data.ctrl[:] = 0.0
        except Exception:
            pass

        try:
            dt = float(model.opt.timestep)
        except Exception:
            dt = 1.0 / 500.0
        if dt <= 0.0:
            dt = 1.0 / 500.0

        next_frame_time: Optional[float] = None
        while True:
            try:
                if not viewer.is_running():
                    return
            except Exception:
                return
            try:
                mujoco_mod.mj_step(model, data)
            except Exception as exc:
                log_warning(f"[review/robot] mj_step failed: {exc}")
                return
            try:
                viewer.sync()
            except Exception:
                return
            try:
                self.check_cancelled()
            except Exception:
                return

            now = time.perf_counter()
            if next_frame_time is None or now - next_frame_time > 0.25:
                next_frame_time = now + dt
                continue
            remaining = next_frame_time - now
            if remaining > 0.0:
                time.sleep(remaining)
            next_frame_time = next_frame_time + dt


__all__ = ["RobotReviewTask"]
