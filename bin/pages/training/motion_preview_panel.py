#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotionPreviewPanel — 3D playback of a MotionPreviewBundle.

Per AMP_design.yaml §5.1.motion_preview_panel: the panel is a **pure
view** — it consumes a ``MotionPreviewBundle`` and renders whatever is
inside. No file I/O happens here; all parsing and preview prep lives in
``src/system/training/motion/preview_renderer.py``.

Layout:

    ┌─────────────────────────────────────────────────────────────┐
    │  Header: clip name · format · fps · duration · robot_name   │
    ├────────────────────────────────────┬────────────────────────┤
    │                                    │                        │
    │         3D body trajectory         │   Metadata sidebar     │
    │         (GLViewWidget)             │   · joint names        │
    │                                    │   · foot links         │
    │                                    │   · warnings           │
    ├────────────────────────────────────┴────────────────────────┤
    │         Joint angles over time (2D plot + playhead)         │
    ├─────────────────────────────────────────────────────────────┤
    │  [<<] [play/pause] [>>]  speed [0.25|0.5|1|2]  time ◄──●──► │
    └─────────────────────────────────────────────────────────────┘

3D view:
    Renders the clip's root_pos as a moving world-space trail and the
    current root pose as an oriented box. This works for any format —
    tracking-only .npy clips (which have no root pose) just show the
    box at origin while joint angles change.

2D view:
    Line plot of the ``dof`` joint angles vs time. A vertical playhead
    slides across as playback advances. For tracking-only clips this is
    the primary visual (since the 3D view is static).

Implementation note: pyqtgraph.opengl is imported lazily so the panel
module can still be imported in headless CI environments (the tests
only exercise the backend bundle builder, not the Qt layer).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.system.training.motion.preview_renderer import (
    FramePose,
    MotionPreviewBundle,
)


# ---------------------------------------------------------------------------
# Rendering speeds (UI → multiplier)
# ---------------------------------------------------------------------------

_SPEEDS = [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)]


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class MotionPreviewPanel(QWidget):
    """Interactive preview of a ``MotionPreviewBundle``.

    Wire it up by calling :meth:`set_bundle`. The panel can be reused —
    calling ``set_bundle`` again replaces the current motion.
    """

    #: Emitted when the user scrubs the time slider. Arg is seconds.
    scrubbed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bundle: Optional[MotionPreviewBundle] = None
        self._playing = False
        self._speed = 1.0
        self._sim_time_s = 0.0

        # ── Playback tick timer ──
        # Fires at 60 Hz regardless of source fps; we advance _sim_time_s
        # by dt * speed on each tick. This decouples visual smoothness
        # from the clip's native frame rate.
        self._tick_interval_ms = 16
        self._timer = QTimer(self)
        self._timer.setInterval(self._tick_interval_ms)
        self._timer.timeout.connect(self._on_tick)

        # ── Build UI ──
        self._build_ui()
        self._render_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_bundle(self, bundle: Optional[MotionPreviewBundle]) -> None:
        """Render a bundle. Passing ``None`` clears the panel."""
        self._stop()
        self._bundle = bundle
        self._sim_time_s = 0.0

        if bundle is None:
            self._render_empty()
            return

        self._header_label.setText(
            f"{bundle.name}  ·  format={bundle.format_id}  ·  "
            f"fps={bundle.fps:.1f}  ·  duration={bundle.duration_s:.2f}s  ·  "
            f"dof={bundle.dof}  ·  robot={bundle.robot_name or '(none)'}"
        )

        # ── Metadata sidebar ──
        lines: List[str] = []
        if bundle.joint_names:
            lines.append("Joint order:")
            for i, jn in enumerate(bundle.joint_names):
                lines.append(f"  [{i}] {jn}")
            lines.append("")
        if bundle.foot_link_names:
            lines.append("Foot links:")
            for fl in bundle.foot_link_names:
                lines.append(f"  · {fl}")
            lines.append("")
        if bundle.warnings:
            lines.append("Warnings:")
            for w in bundle.warnings:
                lines.append(f"  ⚠ {w}")
        self._meta_text.setPlainText("\n".join(lines) or "(no metadata)")

        # ── 3D view: render trail + initial body pose ──
        self._rebuild_3d_scene(bundle)

        # ── 2D joint plot: one line per dof + vertical playhead ──
        self._rebuild_2d_plot(bundle)

        # ── Slider bounds ──
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, bundle.n_frames - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        self._update_current_frame()

    def play(self) -> None:
        if self._bundle is None or self._bundle.n_frames == 0:
            return
        self._playing = True
        self._play_btn.setText("⏸")
        self._timer.start()

    def pause(self) -> None:
        self._stop()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header ──
        self._header_label = QLabel("(no motion loaded)")
        self._header_label.setStyleSheet(
            "font-weight: 600; padding: 4px; "
            "background: rgba(0, 0, 0, 0.04); border-radius: 4px;"
        )
        root.addWidget(self._header_label)

        # ── Upper split: 3D | metadata ──
        upper_split = QSplitter(Qt.Horizontal)
        upper_split.addWidget(self._build_3d_area())
        upper_split.addWidget(self._build_metadata_sidebar())
        upper_split.setStretchFactor(0, 3)
        upper_split.setStretchFactor(1, 1)
        root.addWidget(upper_split, 3)

        # ── Lower: 2D joint plot ──
        plot_area = self._build_2d_plot_area()
        root.addWidget(plot_area, 2)

        # ── Playback controls ──
        root.addLayout(self._build_controls())

    def _build_3d_area(self) -> QWidget:
        """Wrap pyqtgraph.opengl.GLViewWidget with a graceful fallback
        when OpenGL is unavailable (headless CI, stripped Qt installs)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            import pyqtgraph.opengl as gl  # type: ignore
            self._gl_view: Optional[object] = gl.GLViewWidget()
            self._gl_view.setCameraPosition(distance=4.0, elevation=20, azimuth=45)  # type: ignore
            self._gl_view.setBackgroundColor((30, 30, 36))  # type: ignore

            # Static grid for spatial reference
            grid = gl.GLGridItem()
            grid.setSize(4, 4, 1)
            grid.setSpacing(0.25, 0.25, 0.25)
            self._gl_view.addItem(grid)  # type: ignore

            layout.addWidget(self._gl_view)  # type: ignore
            self._gl = gl
        except Exception as exc:
            self._gl_view = None
            self._gl = None
            fallback = QLabel(
                f"(3D view unavailable: {exc})\n"
                "See the joint angles plot below for motion playback."
            )
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setStyleSheet("color: #888;")
            layout.addWidget(fallback)

        # Handles to scene items we add per-bundle; cleaned up in
        # _rebuild_3d_scene so we can swap bundles without leaks.
        self._gl_items: List[object] = []
        return container

    def _build_metadata_sidebar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Metadata")
        label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(label)
        self._meta_text = QTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._meta_text, 1)
        return container

    def _build_2d_plot_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            import pyqtgraph as pg  # type: ignore
            self._plot = pg.PlotWidget()
            self._plot.setBackground((30, 30, 36))
            self._plot.setLabel("bottom", "Time (s)")
            self._plot.setLabel("left", "Joint angle")
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._playhead = pg.InfiniteLine(
                pos=0.0, angle=90, pen=pg.mkPen((255, 160, 40), width=2)
            )
            self._plot.addItem(self._playhead)
            layout.addWidget(self._plot)
            self._pg = pg
            self._plot_curves: List[object] = []
        except Exception as exc:
            self._plot = None
            self._pg = None
            self._playhead = None
            self._plot_curves = []
            fallback = QLabel(f"(2D plot unavailable: {exc})")
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setStyleSheet("color: #888;")
            layout.addWidget(fallback)

        return container

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._rewind_btn = QPushButton("⏮")
        self._rewind_btn.setFixedWidth(36)
        self._rewind_btn.clicked.connect(self._on_rewind)
        row.addWidget(self._rewind_btn)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self._on_play_pause)
        row.addWidget(self._play_btn)

        self._forward_btn = QPushButton("⏭")
        self._forward_btn.setFixedWidth(36)
        self._forward_btn.clicked.connect(self._on_forward)
        row.addWidget(self._forward_btn)

        row.addSpacing(12)
        row.addWidget(QLabel("Speed"))

        self._speed_group = QButtonGroup(self)
        self._speed_group.setExclusive(True)
        for label, mult in _SPEEDS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(44)
            if abs(mult - 1.0) < 1e-6:
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked=False, m=mult: self._on_speed(m))
            self._speed_group.addButton(btn)
            row.addWidget(btn)

        row.addSpacing(12)
        row.addWidget(QLabel("Time"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_change)
        row.addWidget(self._slider, 1)

        return row

    # ------------------------------------------------------------------
    # Per-bundle rendering
    # ------------------------------------------------------------------

    def _rebuild_3d_scene(self, bundle: MotionPreviewBundle) -> None:
        if self._gl_view is None or self._gl is None:
            return
        gl = self._gl

        # Clear previous per-bundle items
        for item in self._gl_items:
            try:
                self._gl_view.removeItem(item)  # type: ignore
            except Exception:
                pass
        self._gl_items = []

        # Root trail: line through every frame's root_pos
        trail_pts = np.stack([f.root_pos for f in bundle.frames], axis=0).astype(np.float32)
        if trail_pts.ndim == 2 and trail_pts.shape[0] >= 2:
            trail = gl.GLLinePlotItem(
                pos=trail_pts, color=(0.4, 0.8, 1.0, 0.8), width=2.0, antialias=True
            )
            self._gl_view.addItem(trail)  # type: ignore
            self._gl_items.append(trail)

        # Body marker: updated per frame in _update_current_frame
        self._body_marker = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            size=np.array([18.0]),
            color=np.array([[1.0, 0.5, 0.0, 1.0]]),
        )
        self._gl_view.addItem(self._body_marker)  # type: ignore
        self._gl_items.append(self._body_marker)

    def _rebuild_2d_plot(self, bundle: MotionPreviewBundle) -> None:
        if self._plot is None or self._pg is None:
            return
        pg = self._pg

        # Drop old curves
        for curve in self._plot_curves:
            try:
                self._plot.removeItem(curve)
            except Exception:
                pass
        self._plot_curves = []

        dof = bundle.dof
        if dof == 0 or bundle.n_frames < 2:
            return

        times = np.arange(bundle.n_frames, dtype=np.float32) / float(bundle.fps)
        joint_mat = np.stack([f.joint_pos for f in bundle.frames], axis=0)

        colors = _palette(dof)
        for j in range(dof):
            pen = pg.mkPen(color=colors[j], width=1.5)
            curve = self._plot.plot(times, joint_mat[:, j], pen=pen)
            name = bundle.joint_names[j] if bundle.joint_names else f"joint_{j}"
            curve.setProperty("jointName", name)
            self._plot_curves.append(curve)

        self._plot.setXRange(0, float(bundle.duration_s), padding=0.02)
        self._playhead.setValue(0.0)  # type: ignore

    def _render_empty(self) -> None:
        self._header_label.setText("(no motion loaded)")
        self._meta_text.clear()
        self._slider.blockSignals(True)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        if self._gl_view is not None:
            for item in self._gl_items:
                try:
                    self._gl_view.removeItem(item)  # type: ignore
                except Exception:
                    pass
            self._gl_items = []
        if self._plot is not None:
            for curve in self._plot_curves:
                try:
                    self._plot.removeItem(curve)
                except Exception:
                    pass
            self._plot_curves = []
            if self._playhead is not None:
                self._playhead.setValue(0.0)

    def _update_current_frame(self) -> None:
        if self._bundle is None or self._bundle.n_frames == 0:
            return

        # Resolve current frame index from _sim_time_s
        idx = int(self._sim_time_s * self._bundle.fps)
        idx = max(0, min(idx, self._bundle.n_frames - 1))

        frame: FramePose = self._bundle.frames[idx]

        # 3D: move the body marker to current root
        if getattr(self, "_body_marker", None) is not None:
            self._body_marker.setData(  # type: ignore
                pos=np.array([frame.root_pos], dtype=np.float32)
            )

        # 2D: move the playhead
        if self._playhead is not None:
            self._playhead.setValue(self._sim_time_s)  # type: ignore

        # Slider (without re-triggering our own handler)
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        if self._bundle is None or not self._playing:
            return
        dt_real = self._tick_interval_ms / 1000.0
        self._sim_time_s += dt_real * self._speed
        if self._sim_time_s >= self._bundle.duration_s:
            # Loop
            self._sim_time_s = 0.0
        self._update_current_frame()

    def _on_play_pause(self) -> None:
        if self._playing:
            self._stop()
        else:
            self.play()

    def _stop(self) -> None:
        self._playing = False
        self._play_btn.setText("▶")
        self._timer.stop()

    def _on_rewind(self) -> None:
        self._sim_time_s = 0.0
        self._update_current_frame()

    def _on_forward(self) -> None:
        if self._bundle is None:
            return
        self._sim_time_s = max(0.0, self._bundle.duration_s - 1e-3)
        self._update_current_frame()

    def _on_speed(self, mult: float) -> None:
        self._speed = float(mult)

    def _on_slider_change(self, value: int) -> None:
        if self._bundle is None:
            return
        self._sim_time_s = float(value) / float(self._bundle.fps)
        self._update_current_frame()
        self.scrubbed.emit(self._sim_time_s)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _palette(n: int) -> List[tuple]:
    """Return *n* distinct RGB tuples by stepping through HSV."""
    out: List[tuple] = []
    for i in range(n):
        h = (i / max(1, n)) * 360.0
        out.append(_hsv_to_rgb_int(h, 0.75, 0.95))
    return out


def _hsv_to_rgb_int(h: float, s: float, v: float) -> tuple:
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c
    if h < 60:
        rp, gp, bp = c, x, 0
    elif h < 120:
        rp, gp, bp = x, c, 0
    elif h < 180:
        rp, gp, bp = 0, c, x
    elif h < 240:
        rp, gp, bp = 0, x, c
    elif h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return (int((rp + m) * 255), int((gp + m) * 255), int((bp + m) * 255))


# ---------------------------------------------------------------------------
# Standalone entry — python motion_preview_panel.py <clip.txt>
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path
    from PySide6.QtWidgets import QApplication

    from src.system.training.motion import (
        AMPLegGymLoader,
        NpyLoader,
        build_preview_bundle,
        validate_clip,
    )

    if len(sys.argv) < 2:
        print("Usage: python motion_preview_panel.py <path/to/clip.txt|.npy>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.suffix.lower() in (".npy",):
        clip = NpyLoader(fps=50.0).load(path)
    else:
        clip = AMPLegGymLoader().load(path)
    validate_clip(clip)
    bundle = build_preview_bundle(clip)

    app = QApplication(sys.argv)
    panel = MotionPreviewPanel()
    panel.set_bundle(bundle)
    panel.resize(1100, 780)
    panel.show()
    sys.exit(app.exec())
