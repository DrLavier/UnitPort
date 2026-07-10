# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ClipRenderSession — offscreen MuJoCo render for the Clip Motion Editor.

Loads a robot MJCF + a reference motion clip ONCE, then renders any frame
to an RGB numpy array via ``mujoco.Renderer`` (offscreen — no interactive
viewer, no MuJoCo UI). A base-tracking free camera keeps a translating
clip (e.g. a LAFAN1 walk that travels metres across the floor) inside the
frame.

Frame loading + qpos application are the SHARED implementation in
:mod:`clip_loading` / :mod:`clip_frame_apply` — the same code the
interactive ``MotionReviewTask`` viewer uses — so the embedded editor view
and the external viewer can never disagree about what a frame looks like
(CLAUDE.md §11). ``clip.frame_at(i * frame_dt)`` returns the exact frame
``i`` (no interpolation at integer multiples), so frame 0 of the editor is
byte-for-byte the clip's first frame ("起始帧与动画绝对一致").

THREAD AFFINITY (read before touching):
  ``mujoco.Renderer`` owns a GL context bound to the thread that
  constructed it. This object must be **created and used on ONE thread**
  for its whole lifetime — construct it, call :meth:`render_frame`, and
  call :meth:`close` all on the same thread, and never touch it from
  another. The Clip Motion Editor drives it from a dedicated
  :class:`~application.ui.dialogs.clip_render_worker.ClipRenderWorker`
  ``QThread`` precisely to honour this. Calling across threads is a hard
  crash, not a soft failure.
"""

from __future__ import annotations

from typing import Any, List


class ClipRenderSession:
    """Offscreen renderer for one (robot SKU, clip ref) pair."""

    def __init__(
        self,
        sku: str,
        clip_ref: str,
        *,
        width: int = 1200,
        height: int = 900,
    ) -> None:
        import mujoco

        from .clip_frame_apply import build_frame_apply_plan
        from .clip_loading import (
            load_clip,
            resolve_clip_path,
            resolve_target_family,
        )
        from .mj_actor import MjActor

        self._mujoco = mujoco
        self._sku = str(sku)
        self._clip_ref = str(clip_ref)
        self._width = int(width)
        self._height = int(height)

        # ── 1. Load the clip (fails loud on missing / unparsable) ─────
        clip_path = resolve_clip_path(self._clip_ref)
        family = resolve_target_family(self._sku)
        self._clip = load_clip(clip_path, sku=self._sku, family=family)
        if self._clip.n_frames < 1 or self._clip.fps <= 0:
            raise RuntimeError(
                f"[clip_render] clip {clip_path.name!r} has no playable "
                f"frames (n_frames={self._clip.n_frames}, fps={self._clip.fps})."
            )

        # ── 2. Load the robot MJCF (raw, matching MotionReviewTask) ───
        actor = MjActor.from_sku(self._sku)
        try:
            self._model = mujoco.MjModel.from_xml_path(str(actor.mjcf_path))
        except Exception as exc:
            raise RuntimeError(
                f"[clip_render] MJCF load failed for {self._sku!r} "
                f"({actor.mjcf_path}): {exc}"
            ) from exc
        self._data = mujoco.MjData(self._model)

        # ── 3. Build the free-root + IR-role joint plan (shared, §11) ─
        self._plan = build_frame_apply_plan(
            mujoco, self._model, self._clip, self._sku,
            clip_label=clip_path.name,
        )

        # ── 4. Create the offscreen renderer (GL context born here) ───
        # MuJoCo's offscreen framebuffer defaults to 640x480 (model.vis.global_
        # offwidth/offheight); a larger Renderer raises "Image width > framebuffer
        # width". Enlarge the framebuffer to cover the requested size first.
        try:
            g = self._model.vis.global_
            g.offwidth = max(int(g.offwidth), self._width)
            g.offheight = max(int(g.offheight), self._height)
        except Exception as exc:
            raise RuntimeError(
                f"[clip_render] could not size the offscreen framebuffer to "
                f"{self._width}x{self._height}: {exc}"
            ) from exc
        try:
            self._renderer = mujoco.Renderer(self._model, self._height, self._width)
        except Exception as exc:
            raise RuntimeError(
                f"[clip_render] offscreen renderer unavailable "
                f"({self._width}x{self._height}): {exc}"
            ) from exc

        # ── 5. Base-tracking free camera (orbit/zoom under user control) ─
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, self._cam)
        extent = float(self._model.stat.extent) or 1.0
        self._azimuth = 90.0
        self._elevation = -15.0
        # stat.extent is MuJoCo's model size estimate; ×2 frames the robot
        # with headroom for limbs / a stride of travel.
        self._distance = extent * 2.0
        self._dist_min = extent * 0.25
        self._dist_max = extent * 15.0

    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self._clip.n_frames)

    @property
    def fps(self) -> float:
        return float(self._clip.fps)

    def render_frame(self, frame_index: int):
        """Render frame ``frame_index`` to an ``(H, W, 3)`` uint8 RGB array.

        Index is clamped to ``[0, n_frames-1]``. The returned array is a
        copy detached from the renderer's internal buffer (safe to hand
        across threads / hold past the next render).
        """
        from .clip_frame_apply import apply_frame

        n = self.n_frames
        if frame_index < 0:
            i = 0
        elif frame_index >= n:
            i = n - 1
        else:
            i = int(frame_index)
        t = i * self._clip.frame_dt
        apply_frame(self._mujoco, self._model, self._data, self._clip, t, self._plan)
        self._update_tracking_camera()
        self._renderer.update_scene(self._data, camera=self._cam)
        return self._renderer.render().copy()

    def orbit(self, d_azimuth_deg: float, d_elevation_deg: float) -> None:
        """Rotate the view around the base by mouse-drag deltas (degrees)."""
        self._azimuth = (self._azimuth + float(d_azimuth_deg)) % 360.0
        self._elevation = max(-89.0, min(89.0, self._elevation + float(d_elevation_deg)))

    def zoom(self, factor: float) -> None:
        """Dolly the camera in/out (``factor`` < 1 = closer), clamped to bounds."""
        f = float(factor)
        if f > 0:
            self._distance = max(self._dist_min, min(self._dist_max, self._distance * f))

    def _update_tracking_camera(self) -> None:
        """Point the camera lookat at the robot base for the current frame."""
        target: List[float]
        if self._plan.has_freejoint_root and self._plan.free_qposadr >= 0:
            a = self._plan.free_qposadr
            target = [float(self._data.qpos[a + k]) for k in range(3)]
        else:
            # Fixed-base robot: track the first non-world body. xpos[1] is
            # always valid here — build_frame_apply_plan raised unless the model
            # has ≥1 actuated joint, so it has ≥2 bodies (world + one). No silent
            # origin fallback (§8): a failure here is a real model error.
            target = [float(v) for v in self._data.xpos[1][:3]]
        self._cam.lookat[0] = target[0]
        self._cam.lookat[1] = target[1]
        self._cam.lookat[2] = target[2]
        self._cam.azimuth = self._azimuth
        self._cam.elevation = self._elevation
        self._cam.distance = self._distance

    def close(self) -> None:
        """Release the renderer / GL context. Call on the creating thread."""
        r = getattr(self, "_renderer", None)
        self._renderer = None
        if r is not None:
            try:
                r.close()
            except Exception:
                pass


__all__ = ["ClipRenderSession"]
