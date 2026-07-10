# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Motion-clip review session — kinematically replay a reference clip in MuJoCo.

Sibling of :mod:`robot_review_session` (which previews a *static* robot
pose). This module previews a *reference motion clip*: it loads the clip
the user picked in the Training Motion Editor, loads the bound robot's
MJCF, and drives ``mj_data.qpos`` frame-by-frame so the skeleton plays
the trajectory back **without physics** (no ``mj_step``, no actuators —
pure kinematic playback). This is the "Review Motion" button on the
Training Motion Editor's Clip row, which opens an INTERACTIVE MuJoCo
viewer.

(The Clip Motion Editor in the Resources panel renders the same clip
*offscreen* into an embedded view via
:class:`~application.service.runtime.simulation.mujoco.clip_render_session.ClipRenderSession`;
both paths share the clip-loading + frame-apply code in
:mod:`clip_loading` / :mod:`clip_frame_apply` so the viewer and the
embedded render can never drift — CLAUDE.md §11.)

Why kinematic (no physics):
  The clip is a *reference* trajectory — the joint angles and root pose
  are authored data, not the output of a controller. Stepping physics
  would let gravity collapse the robot because there is no policy
  holding it up. Writing the clip frame straight into ``qpos`` +
  ``mj_forward`` shows exactly what the discriminator / tracking target
  sees, which is the whole point of a review.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from unitport_sdk import Task, log_info, log_warning

from .clip_frame_apply import FrameApplyPlan, apply_frame, build_frame_apply_plan
from .clip_loading import load_clip_ref, resolve_clip_path, resolve_target_family


class MotionReviewTask(Task):
    """SDK Task: kinematically replay a reference motion clip in MuJoCo.

    Construction args:
        sku
            Canonical robot SKU whose MJCF supplies the skeleton to drive.
        clip_ref
            Clip reference string as stored on the training item — either
            an absolute path to a motion file, or a
            ``pack:<package>:<subdir>/<file>`` reference resolved through
            :func:`scripts.training_motion.library.resolve_pack_ref`.
        speed
            Playback rate multiplier (1.0 = real time at the clip's own
            fps). Kept for parity with the design-time review controls;
            defaults to real time.

    Cancellable between viewer-sync iterations. Returns a result dict for
    telemetry. Failures raise with a user-facing message — the caller
    (Training Motion Editor) surfaces it via ``QMessageBox`` so the user
    learns *why* the viewer never opened (CLAUDE.md §8 forbids the silent
    "log and return" path).
    """

    def __init__(
        self,
        sku: str,
        clip_ref: str,
        *,
        speed: float = 1.0,
    ) -> None:
        super().__init__(f"Review Motion {Path(str(clip_ref)).stem or clip_ref}")
        self._sku = str(sku)
        self._clip_ref = str(clip_ref)
        self._speed = float(speed) if speed and speed > 0 else 1.0

    # ------------------------------------------------------------------
    # Task entrypoint
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        # Lazy imports — mujoco is heavy and absent on headless CI.
        import mujoco
        import mujoco.viewer

        from .mj_actor import MjActor

        self.check_cancelled()

        # ── 1. Resolve the clip ref to an on-disk file (base of any segment) ─
        clip_path = resolve_clip_path(self._clip_ref)

        # ── 2. Resolve the robot's primary family (loader requires it) ─
        family = resolve_target_family(self._sku)

        # ── 3. Load the clip, honouring a #seg=lo-hi sub-range so a segment
        #       plays back exactly its selected frames (not the whole file) ─
        clip = load_clip_ref(self._clip_ref, sku=self._sku, family=family)
        if clip.n_frames < 1 or clip.fps <= 0:
            raise RuntimeError(
                f"[review/motion] clip {clip_path.name!r} has no frames "
                f"(n_frames={clip.n_frames}, fps={clip.fps}) — nothing to "
                f"play back."
            )

        # ── 4. Load a fresh (model, data) from the robot's MJCF ───────
        actor = MjActor.from_sku(self._sku)
        mjcf_path = actor.mjcf_path
        self.check_cancelled()
        try:
            model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        except Exception as exc:
            raise RuntimeError(
                f"[review/motion] MJCF load failed for {self._sku!r} "
                f"({mjcf_path}): {exc}"
            ) from exc
        data = mujoco.MjData(model)

        # ── 5+6. Free root + per-joint IR-role routing (shared, §11) ──
        # Fails loud when the clip's IR roles don't intersect the robot.
        plan = build_frame_apply_plan(
            mujoco, model, clip, self._sku, clip_label=clip_path.name
        )
        log_info(
            f"[review/motion] {self._sku} ← {clip_path.name} "
            f"(format={clip.format_id}, frames={clip.n_frames}, "
            f"fps={clip.fps:.1f}, joints matched={plan.matched}/"
            f"{len(plan.joint_plan)}, "
            f"root={'free' if plan.has_freejoint_root else 'fixed'})"
        )

        # ── 7. Apply frame 0 before launch so the viewer opens posed ──
        apply_frame(mujoco, model, data, clip, 0.0, plan)

        # ── 8. Launch the passive viewer ──────────────────────────────
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as exc:
            raise RuntimeError(
                f"[review/motion] MuJoCo viewer unavailable — review "
                f"aborted: {exc}"
            ) from exc

        # ``simulate.load`` runs ``mj_resetData`` on the viewer side
        # thread; grab the lock so our first posed frame lands after it.
        try:
            with viewer.lock():
                apply_frame(mujoco, model, data, clip, 0.0, plan)
            viewer.sync()
        except Exception as exc:
            log_warning(
                f"[review/motion] locked post-launch frame apply failed: {exc}"
            )

        # ── 9. Play frames until the user closes the viewer ───────────
        try:
            self._play_loop(mujoco, model, data, viewer, clip, plan)
            return {
                "viewer_used": True,
                "sku": self._sku,
                "clip": str(clip_path),
                "frames": clip.n_frames,
                "fps": clip.fps,
            }
        finally:
            try:
                viewer.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Playback loop
    # ------------------------------------------------------------------
    def _play_loop(
        self,
        mujoco_mod: Any,
        model: Any,
        data: Any,
        viewer: Any,
        clip: Any,
        plan: FrameApplyPlan,
    ) -> None:
        """Drive the clip in wall-clock time until the viewer is closed.

        Playback time advances from ``time.perf_counter`` so the clip runs
        at its authored fps (scaled by ``self._speed``).
        ``MotionClip.frame_at`` applies the clip's ``loop_mode`` (``wrap``
        loops), so the trajectory cycles for as long as the viewer stays
        open. Physics is never stepped — every frame is set directly.
        """
        render_period = 1.0 / 60.0
        start = time.perf_counter()
        while True:
            try:
                if not viewer.is_running():
                    return
            except Exception:
                return

            t = (time.perf_counter() - start) * self._speed
            try:
                apply_frame(mujoco_mod, model, data, clip, t, plan)
            except Exception as exc:
                log_warning(f"[review/motion] frame apply failed: {exc}")
                return

            try:
                viewer.sync()
            except Exception:
                return
            try:
                self.check_cancelled()
            except Exception:
                return
            time.sleep(render_period)


__all__ = ["MotionReviewTask"]
