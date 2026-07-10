# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ClipRenderWorker — owns a :class:`ClipRenderSession` on a dedicated thread.

``mujoco.Renderer``'s GL context is thread-affine: it must be created and
used on ONE persistent thread for the whole lifetime of the editor. That
rules out the SDK ``Task`` pattern — ``TasksManager.submit`` spins a fresh
worker thread per submit, which would rebuild (or, worse, cross-thread
touch) the GL context on every frame. So this is a **deliberate deviation**
from ``Task``: a plain ``QObject`` moved onto a long-lived ``QThread`` and
driven by queued slot invocations. Frames are returned to the GUI as
``QImage`` (safe to marshal across a queued cross-thread signal once
``.copy()``-detached).

Lifecycle (driven by :class:`ClipCropLabelDialog`):

    worker = ClipRenderWorker(sku, clip_ref, w, h)
    worker.moveToThread(thread)
    thread.started.connect(worker.open)      # build session on the thread
    thread.start()
    # ... emit a queued render(i, dAz, dEl, zoom); receive frameReady(i, QImage) ...
    # teardown: queued shutdown() → thread.quit() → thread.wait()

All three signals are the worker's only way to talk back to the GUI;
errors surface via :pyattr:`failed` (fail-loud, CLAUDE.md §8) rather than a
silent black frame.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage


class ClipRenderWorker(QObject):
    """Renders clip frames offscreen on the thread it is moved onto."""

    #: Emitted once the session is built: ``(n_frames, fps)``.
    ready = pyqtSignal(int, float)
    #: Emitted per rendered frame: ``(frame_index, image)``.
    frameReady = pyqtSignal(int, QImage)
    #: Emitted on any failure (session build or render): a user-facing string.
    failed = pyqtSignal(str)

    def __init__(
        self,
        sku: str,
        clip_ref: str,
        width: int,
        height: int,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._sku = str(sku)
        self._clip_ref = str(clip_ref)
        self._width = int(width)
        self._height = int(height)
        self._session = None

    @pyqtSlot()
    def open(self) -> None:
        """Build the render session on this thread; emit ``ready`` / ``failed``."""
        try:
            from application.service.runtime.simulation.mujoco.clip_render_session import (
                ClipRenderSession,
            )
            self._session = ClipRenderSession(
                self._sku, self._clip_ref, width=self._width, height=self._height
            )
            self.ready.emit(int(self._session.n_frames), float(self._session.fps))
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI (§8)
            self._session = None
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    @pyqtSlot(int, float, float, float)
    def render(self, frame_index: int, d_azimuth: float, d_elevation: float, zoom: float) -> None:
        """Apply any camera delta (orbit/zoom) then render ``frame_index``.

        Camera mutation must happen on this (GL-owning) thread, so it is
        folded into the render request rather than touched from the GUI.
        Emits ``frameReady`` (or ``failed`` — never a silent swallow, §8).
        """
        if self._session is None:
            self.failed.emit("render session not initialised")
            return
        try:
            import numpy as np

            if d_azimuth or d_elevation:
                self._session.orbit(d_azimuth, d_elevation)
            if zoom and zoom != 1.0:
                self._session.zoom(zoom)
            rgb = self._session.render_frame(int(frame_index))
            rgb = np.ascontiguousarray(rgb[:, :, :3])
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            # .copy() detaches the QImage from the temporary numpy bytes so
            # it survives the queued cross-thread hop to the GUI.
            img = QImage(
                rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888
            ).copy()
            self.frameReady.emit(int(frame_index), img)
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI (§8)
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    @pyqtSlot()
    def shutdown(self) -> None:
        """Release the GL context. Runs on this thread before ``thread.quit``."""
        s = self._session
        self._session = None
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


__all__ = ["ClipRenderWorker"]
