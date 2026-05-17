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
                jname = joint_names[jid] if jid < len(joint_names) else ""
                actuated_ir_roles.append(name_to_role.get(str(jname), ""))
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
                    data.qpos[free_qposadr + 0] = float(bp[0])
                    data.qpos[free_qposadr + 1] = float(bp[1])
                    data.qpos[free_qposadr + 2] = float(bp[2])
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
        """Build a physical-joint-name → IR-role lookup for this SKU.

        Reads ``registers.robots.get_robot(sku)["joints"]``. Joints not
        in the registry simply don't appear in the returned dict —
        callers default to ``""`` (which makes the override fall back to
        the per-slot default).
        """
        from registers import robots as _robots_registry

        entry = _robots_registry.get_robot(self._sku) or {}
        joints_dict = entry.get("joints", {}) or {}
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
