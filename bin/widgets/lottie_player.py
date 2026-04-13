"""Thread-safe Lottie animation player with pre-rendered frame caching.

Renders all frames in a background QThread, then drives playback on the main
thread via QTimer.  Falls back silently if ``rlottie_python`` is unavailable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel


class _FrameRenderer(QThread):
    """Worker that pre-renders every Lottie frame into a list of QPixmap."""

    frames_ready = Signal(list, float)  # List[QPixmap], fps

    def __init__(
        self,
        json_path: str,
        render_size: QSize,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._json_path = json_path
        self._render_w = render_size.width()
        self._render_h = render_size.height()

    # ── run (executes in worker thread) ───────────���──────────────────────
    def run(self) -> None:  # noqa: D401
        try:
            from rlottie_python import LottieAnimation
        except ImportError:
            self.frames_ready.emit([], 0.0)
            return

        try:
            raw = Path(self._json_path).read_text(encoding="utf-8")
            data = json.loads(raw)

            # Re-centre the canvas around the spinner so we don't render
            # 1920x1080 of empty space.  Original spinner centre ≈ (960, 540).
            orig_cx = data["w"] / 2
            orig_cy = data["h"] / 2
            crop = 300  # keep 300×300 Lottie-units around the centre
            data["w"] = crop
            data["h"] = crop
            new_cx, new_cy = crop / 2, crop / 2
            for layer in data.get("layers", []):
                ks = layer.get("ks", {})
                pos = ks.get("p", {})
                k = pos.get("k")
                if isinstance(k, list) and len(k) >= 2 and isinstance(k[0], (int, float)):
                    k[0] = k[0] - orig_cx + new_cx
                    k[1] = k[1] - orig_cy + new_cy

            json_str = json.dumps(data)
            anim = LottieAnimation.from_data(json_str)
            total = anim.lottie_animation_get_totalframe()
            fps = anim.lottie_animation_get_framerate()
            w, h = self._render_w, self._render_h

            frames: List[QPixmap] = []
            bpl = w * 4
            buf_size = w * h * 4
            for i in range(total):
                buf = anim.lottie_animation_render(
                    i, buffer_size=buf_size, width=w, height=h,
                    bytes_per_line=bpl,
                )
                img = QImage(
                    buf, w, h, bpl,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                # .copy() detaches from the transient buffer before next render
                frames.append(QPixmap.fromImage(img.copy()))

            self.frames_ready.emit(frames, fps)
        except Exception:
            self.frames_ready.emit([], 0.0)


class LottiePlayer(QLabel):
    """A QLabel that plays a Lottie animation from pre-cached frames.

    * Frame rendering runs in ``_FrameRenderer`` (QThread) — main thread
      is never blocked.
    * Once the frames arrive, a ``QTimer`` drives frame-by-frame updates
      at the animation's native FPS.
    * When the main event loop is blocked (e.g. during a synchronous
      startup sequence), call :meth:`tick` from any ``processEvents()``
      site to keep the animation advancing.
    * Call :meth:`stop` to halt playback and release frames.
    * If ``rlottie_python`` is missing the widget stays blank (transparent).
    """

    def __init__(
        self,
        json_path: str,
        size: QSize,
        *,
        parent: Optional[QLabel] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("startupLottiePlayer")
        self.setFixedSize(size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(
            "#startupLottiePlayer { background: transparent; border: none; padding: 0; border-radius: 0; }"
        )

        self._frames: List[QPixmap] = []
        self._idx: int = 0
        self._interval_s: float = 0.04  # default 25 fps
        self._last_tick: float = 0.0
        self._mutex = QMutex()
        self._playing = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        self._renderer = _FrameRenderer(json_path, size, self)
        self._renderer.frames_ready.connect(self._on_frames_ready)
        self._renderer.start()

    # ── slots ────────────────────────────────────────────────────────────
    def _on_frames_ready(self, frames: List[QPixmap], fps: float) -> None:
        with QMutexLocker(self._mutex):
            self._frames = frames
            self._idx = 0
            if fps > 0:
                self._interval_s = 1.0 / fps
        if frames:
            self.setPixmap(frames[0])
            self._last_tick = time.monotonic()
            self._playing = True
            interval_ms = max(1, int(self._interval_s * 1000))
            self._timer.start(interval_ms)

    def _advance_frame(self) -> None:
        with QMutexLocker(self._mutex):
            if not self._frames:
                return
            self._idx = (self._idx + 1) % len(self._frames)
            pm = self._frames[self._idx]
        self.setPixmap(pm)
        self.repaint()
        self._last_tick = time.monotonic()

    # ── public API ──────────────��────────────────────────────────────────
    def tick(self) -> None:
        """Manually advance the animation when the event loop is blocked.

        Call this from any ``QApplication.processEvents()`` site during
        synchronous startup phases.  It only advances the frame when
        enough time has elapsed since the last update.
        """
        if not self._playing:
            return
        now = time.monotonic()
        with QMutexLocker(self._mutex):
            if not self._frames:
                return
            elapsed = now - self._last_tick
            if elapsed < self._interval_s:
                return
            self._idx = (self._idx + 1) % len(self._frames)
            pm = self._frames[self._idx]
        self.setPixmap(pm)
        self.repaint()
        self._last_tick = now

    def stop(self) -> None:
        """Stop playback and release cached frames."""
        self._playing = False
        self._timer.stop()
        if self._renderer.isRunning():
            self._renderer.quit()
            self._renderer.wait(2000)
        with QMutexLocker(self._mutex):
            self._frames.clear()
