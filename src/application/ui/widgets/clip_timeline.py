# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ClipTimeline — a reusable, video-editor-style timeline widget.

A self-contained custom-painted control modelled on a non-linear video
editor's timeline:

* a **top ruler** scaled in **frames** and a **bottom ruler** scaled in
  **seconds** (drawn only when an fps is known);
* a draggable **playhead** ("操纵杆") that scrubs / drives playback;
* a draggable **in/out selection box** whose edges define a segment;
* already-saved **segments** painted as coloured bands on the track;
* **Ctrl + mouse-wheel** zooms the tick scale (centred on the cursor),
  plain wheel pans when zoomed in.

It is deliberately decoupled from the Clip Motion Editor so other panels
can reuse it: it speaks only frames (ints) and emits three signals. All
colours are read at paint time via ``Config.get_color`` (CLAUDE.md §5 — no
hex / QColor literals defined here); all label text via ``tr``.

The value↔pixel maths live in module-level pure functions (:func:`x_of`,
:func:`frame_of`, :func:`nice_step`) so they unit-test without Qt or a GPU.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
    QPolygonF,
)
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from unitport_sdk import Config, tr

# ---------------------------------------------------------------------------
# Pure value↔pixel maths (Qt-free, GPU-free — unit-tested directly)
# ---------------------------------------------------------------------------


def x_of(frame: float, fpp: float, offset: float, track_left: float) -> float:
    """Pixel x for a frame, given frames-per-pixel ``fpp`` and pan ``offset``."""
    return track_left + (frame - offset) / fpp


def frame_of(x: float, fpp: float, offset: float, track_left: float) -> float:
    """Inverse of :func:`x_of` — the frame under pixel ``x``."""
    return offset + (x - track_left) * fpp


def nice_step(span: float, target_ticks: float) -> float:
    """A human-friendly tick step (1 / 2 / 5 × 10ⁿ) for ``span`` over
    roughly ``target_ticks`` divisions. Always ``> 0``."""
    if span <= 0 or target_ticks <= 0:
        return 1.0
    raw = span / target_ticks
    if raw <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def minor_step(step: float, px_per_unit: float, *, integral: bool) -> float:
    """Sub-division of a major ``step`` into minor ticks.

    Prefers a /10 split (≥ 5 minor ticks *between* two major ticks, as the
    editor requires); degrades to /5 then /2 when the finer split would be
    sub-pixel-cramped. Returns ``0.0`` when no minor tick fits.

    ``integral=True`` (the frame ruler) additionally requires the minor step
    to be a whole frame ≥ 1 — fractional-frame ticks are meaningless there.
    ``px_per_unit`` is pixels per frame (frame ruler) or per second (seconds
    ruler); a minor tick is only drawn when ticks stay ≥ 5 px apart.
    """
    if step <= 0 or px_per_unit <= 0:
        return 0.0
    for div in (10.0, 5.0, 2.0):
        m = step / div
        if integral and (m < 1.0 or abs(m - round(m)) > 1e-9):
            continue
        if m * px_per_unit >= 5.0:
            return m
    return 0.0


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

_PAD = 8                 # left/right padding (track_left)
_RULER_TOP_H = 22        # frame ruler height
_TRACK_H = 46            # selection / playhead / segment band track
_RULER_BOT_H = 18        # seconds ruler height
_GRAB_PX = 6             # mouse grab tolerance for handles
_FLAG_H = 11             # playhead flag height at top of track
_MAX_PX_PER_FRAME = 40.0  # zoom-in limit → min fpp = 1/40


def _mini() -> int:
    return int(Config.get_font_size("size_mini"))


class ClipTimeline(QWidget):
    """Frames/seconds dual-ruler timeline with playhead + in/out selection."""

    #: Playhead dragged live (scrubbing). Argument: frame index.
    playheadMoved = pyqtSignal(int)
    #: Playhead jumped by a ruler/track click. Argument: frame index.
    seeked = pyqtSignal(int)
    #: In/out selection edited. Arguments: (lo, hi) frame indices.
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._n_frames = 1
        self._fps = 0.0
        self._known = True
        self._playhead = 0
        self._sel_lo = 0
        self._sel_hi = 0
        self._segments: List[Tuple[int, int, int]] = []

        self._fpp = 1.0          # frames per pixel
        self._offset = 0.0       # left-edge frame (pan)
        self._user_zoomed = False
        self._drag_mode: Optional[str] = None
        self._drag_anchor_frame = 0.0   # for sel_move

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(_RULER_TOP_H + _TRACK_H + _RULER_BOT_H + 4)
        self.setMouseTracking(True)
        self._refit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame_count(self, n: int) -> None:
        self._n_frames = max(1, int(n))
        self._playhead = min(self._playhead, self._n_frames - 1)
        # Default selection = whole clip; the host may override.
        self._sel_lo = 0
        self._sel_hi = self._n_frames - 1
        self._user_zoomed = False
        self._refit()
        self.update()

    def set_fps(self, fps: float) -> None:
        self._fps = float(fps) if fps and fps > 0 else 0.0
        self.update()

    def set_known(self, known: bool) -> None:
        self._known = bool(known)
        self.update()

    def set_playhead(self, frame: int) -> None:
        self._playhead = self._clamp_frame(frame)
        self.update()

    def playhead(self) -> int:
        return int(self._playhead)

    def set_selection(self, lo: int, hi: int) -> None:
        lo = self._clamp_frame(lo)
        hi = self._clamp_frame(hi)
        self._sel_lo, self._sel_hi = min(lo, hi), max(lo, hi)
        self.update()

    def selection(self) -> Tuple[int, int]:
        return int(self._sel_lo), int(self._sel_hi)

    def set_segments(self, segments: List[Tuple[int, int, int]]) -> None:
        self._segments = [
            (int(lo), int(hi), int(idx)) for (lo, hi, idx) in segments
        ]
        self.update()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _track_left(self) -> float:
        return float(_PAD)

    def _track_width(self) -> float:
        return max(1.0, float(self.width()) - 2.0 * _PAD)

    def _max_fpp(self) -> float:
        return max(1, self._n_frames - 1) / self._track_width()

    def _refit(self) -> None:
        self._fpp = self._max_fpp()
        if self._fpp <= 0:
            self._fpp = 1.0
        self._offset = 0.0

    def _clamp_zoom(self) -> None:
        self._fpp = min(max(self._fpp, 1.0 / _MAX_PX_PER_FRAME), self._max_fpp())

    def _clamp_offset(self) -> None:
        visible = self._track_width() * self._fpp
        max_off = max(0.0, (self._n_frames - 1) - visible)
        self._offset = min(max(0.0, self._offset), max_off)

    def _clamp_frame(self, f) -> int:
        return int(min(max(0, int(round(f))), self._n_frames - 1))

    def _x_of(self, frame: float) -> float:
        return x_of(frame, self._fpp, self._offset, self._track_left())

    def _frame_at_x(self, x: float) -> int:
        return self._clamp_frame(
            frame_of(x, self._fpp, self._offset, self._track_left())
        )

    def _track_rect(self) -> QRectF:
        return QRectF(
            self._track_left(), float(_RULER_TOP_H),
            self._track_width(), float(_TRACK_H),
        )

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            self._paint_background(p)
            self._paint_segment_bands(p)
            self._paint_selection(p)
            self._paint_frame_ruler(p)
            if self._fps > 0:
                self._paint_seconds_ruler(p)
            self._paint_playhead(p)
            if not self._known:
                self._paint_unknown_watermark(p)
        finally:
            p.end()

    def _paint_background(self, p: QPainter) -> None:
        w = float(self.width())
        p.fillRect(QRectF(0, 0, w, float(_RULER_TOP_H)),
                   QColor(Config.get_color("bg_3")))
        p.fillRect(self._track_rect(), QColor(Config.get_color("bg_2")))
        p.fillRect(
            QRectF(0, float(_RULER_TOP_H + _TRACK_H), w, float(_RULER_BOT_H)),
            QColor(Config.get_color("bg_3")),
        )
        # Track border.
        p.setPen(QPen(QColor(Config.get_color("border_1")), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self._track_rect())

    def _paint_segment_bands(self, p: QPainter) -> None:
        tr_rect = self._track_rect()
        for lo, hi, idx in self._segments:
            x0 = self._x_of(lo)
            x1 = self._x_of(hi)
            band = QColor(Config.get_color(f"pie_item_color_{(idx % 12) + 1}"))
            band.setAlpha(90)
            p.fillRect(
                QRectF(x0, tr_rect.top() + 2, max(1.0, x1 - x0), tr_rect.height() - 4),
                band,
            )

    def _paint_selection(self, p: QPainter) -> None:
        if not self._known:
            return
        tr_rect = self._track_rect()
        x0 = self._x_of(self._sel_lo)
        x1 = self._x_of(self._sel_hi)
        fill = QColor(Config.get_color("highlight"))
        fill.setAlpha(55)
        p.fillRect(
            QRectF(x0, tr_rect.top(), max(1.0, x1 - x0), tr_rect.height()), fill
        )
        edge = QColor(Config.get_color("highlight"))
        p.setPen(QPen(edge, 1.5))
        p.drawLine(QPointF(x0, tr_rect.top()), QPointF(x0, tr_rect.bottom()))
        p.drawLine(QPointF(x1, tr_rect.top()), QPointF(x1, tr_rect.bottom()))
        # Flat grab handles.
        p.setBrush(QBrush(edge))
        p.setPen(Qt.PenStyle.NoPen)
        for xx in (x0, x1):
            p.drawRoundedRect(
                QRectF(xx - 2.5, tr_rect.top(), 5.0, tr_rect.height()), 2.0, 2.0
            )

    def _paint_frame_ruler(self, p: QPainter) -> None:
        visible = self._track_width() * self._fpp
        target = max(2.0, self._track_width() / 90.0)
        step = max(1.0, round(nice_step(visible, target)))
        ppf = (1.0 / self._fpp) if self._fpp > 0 else 1.0
        minor = minor_step(step, ppf, integral=True)
        font = QFont(self.font())
        font.setPixelSize(_mini())
        p.setFont(font)
        fm = QFontMetrics(font)
        big_col = QColor(Config.get_color("canvas_grid_big"))
        small_col = QColor(Config.get_color("canvas_grid_small"))
        text_col = QColor(Config.get_color("sub_t1"))
        end = self._offset + visible
        # Minor ticks first (short, faint) so the taller major ticks overlay them.
        if minor > 0:
            p.setPen(QPen(small_col, 1.0))
            f = math.ceil(self._offset / minor) * minor
            while f <= end + 0.5:
                x = self._x_of(f)
                p.drawLine(QPointF(x, float(_RULER_TOP_H - 4)), QPointF(x, float(_RULER_TOP_H)))
                f += minor
        # Major ticks (tall) + frame label.
        first = math.ceil(self._offset / step) * step
        f = first
        while f <= end + 0.5:
            x = self._x_of(f)
            p.setPen(QPen(big_col, 1.0))
            p.drawLine(QPointF(x, float(_RULER_TOP_H - 9)), QPointF(x, float(_RULER_TOP_H)))
            label = str(int(round(f)))
            p.setPen(QPen(text_col, 1.0))
            p.drawText(QPointF(x + 2, fm.ascent() + 1), label)
            f += step

    def _paint_seconds_ruler(self, p: QPainter) -> None:
        top = float(_RULER_TOP_H + _TRACK_H)
        visible_frames = self._track_width() * self._fpp
        visible_s = visible_frames / self._fps
        target = max(2.0, self._track_width() / 90.0)
        step_s = nice_step(visible_s, target)
        pps = (self._fps / self._fpp) if self._fpp > 0 else 1.0  # pixels per second
        minor_s = minor_step(step_s, pps, integral=False)
        font = QFont(self.font())
        font.setPixelSize(_mini())
        p.setFont(font)
        fm = QFontMetrics(font)
        big_col = QColor(Config.get_color("canvas_grid_big"))
        small_col = QColor(Config.get_color("canvas_grid_small"))
        text_col = QColor(Config.get_color("sub_t2"))
        off_s = self._offset / self._fps
        end = off_s + visible_s
        # Minor ticks first (short, faint).
        if minor_s > 0:
            p.setPen(QPen(small_col, 1.0))
            s = math.ceil(off_s / minor_s) * minor_s
            while s <= end + 1e-6:
                x = self._x_of(s * self._fps)
                p.drawLine(QPointF(x, top), QPointF(x, top + 3))
                s += minor_s
        # Major ticks (tall) + seconds label. Decimals derived from the step so
        # adjacent labels never print the same value (e.g. step 0.005 → 3 dp).
        decimals = max(0, min(4, -int(math.floor(math.log10(step_s))))) if step_s > 0 else 0
        first = math.ceil(off_s / step_s) * step_s
        s = first
        while s <= end + 1e-6:
            x = self._x_of(s * self._fps)
            p.setPen(QPen(big_col, 1.0))
            p.drawLine(QPointF(x, top), QPointF(x, top + 6))
            label = f"{s:.{decimals}f}s"
            p.setPen(QPen(text_col, 1.0))
            p.drawText(QPointF(x + 2, top + fm.ascent()), label)
            s += step_s

    def _paint_playhead(self, p: QPainter) -> None:
        x = self._x_of(self._playhead)
        col = QColor(Config.get_color("danger_zone"))
        p.setPen(QPen(col, 1.6))
        p.drawLine(QPointF(x, 0.0), QPointF(x, float(_RULER_TOP_H + _TRACK_H)))
        # Flag at the top of the track.
        flag = QPolygonF([
            QPointF(x - 5, float(_RULER_TOP_H)),
            QPointF(x + 5, float(_RULER_TOP_H)),
            QPointF(x, float(_RULER_TOP_H + _FLAG_H)),
        ])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(col))
        p.drawPolygon(flag)

    def _paint_unknown_watermark(self, p: QPainter) -> None:
        font = QFont(self.font())
        font.setPixelSize(_mini())
        p.setFont(font)
        col = QColor(Config.get_color("sub_t2"))
        p.setPen(QPen(col, 1.0))
        p.drawText(
            self._track_rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            tr("clip_editor.length_unknown", "clip length unknown"),
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _hit(self, pos: QPointF) -> str:
        x, y = pos.x(), pos.y()
        track_top = float(_RULER_TOP_H)
        track_bot = float(_RULER_TOP_H + _TRACK_H)
        lo_x = self._x_of(self._sel_lo)
        hi_x = self._x_of(self._sel_hi)
        ph_x = self._x_of(self._playhead)
        in_track = track_top <= y <= track_bot
        # Playhead flag (narrow top strip) wins first, so it stays grabbable
        # even when the playhead sits exactly on a selection edge.
        if track_top <= y <= track_top + _FLAG_H and abs(x - ph_x) <= _GRAB_PX:
            return "playhead"
        # Selection edges next, over the full track height.
        if in_track and self._known:
            if abs(x - lo_x) <= _GRAB_PX:
                return "sel_lo"
            if abs(x - hi_x) <= _GRAB_PX:
                return "sel_hi"
        # Inside a PROPER sub-range selection → move the whole box. When the
        # selection spans the whole clip (the default), fall through to seek so
        # clicking the track body scrubs the playhead instead of being inert.
        if (
            in_track and self._known
            and (self._sel_hi - self._sel_lo) < (self._n_frames - 1)
            and lo_x + _GRAB_PX < x < hi_x - _GRAB_PX
        ):
            return "sel_move"
        # Anywhere else (rulers or empty/whole-selected track) → seek.
        return "seek"

    def mousePressEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(e)
            return
        pos = e.position()
        mode = self._hit(pos)
        self._drag_mode = mode
        if mode == "seek":
            self._playhead = self._frame_at_x(pos.x())
            self._drag_mode = "playhead"
            self.update()
            self.seeked.emit(self._playhead)
        elif mode == "sel_move":
            self._drag_anchor_frame = frame_of(
                pos.x(), self._fpp, self._offset, self._track_left()
            )
        e.accept()

    def mouseMoveEvent(self, e) -> None:
        if self._drag_mode is None:
            super().mouseMoveEvent(e)
            return
        x = e.position().x()
        if self._drag_mode == "playhead":
            self._playhead = self._frame_at_x(x)
            self.update()
            self.playheadMoved.emit(self._playhead)
        elif self._drag_mode == "sel_lo":
            f = self._frame_at_x(x)
            self._sel_lo = min(f, self._sel_hi)
            self.update()
            self.rangeChanged.emit(self._sel_lo, self._sel_hi)
        elif self._drag_mode == "sel_hi":
            f = self._frame_at_x(x)
            self._sel_hi = max(f, self._sel_lo)
            self.update()
            self.rangeChanged.emit(self._sel_lo, self._sel_hi)
        elif self._drag_mode == "sel_move":
            cur = frame_of(x, self._fpp, self._offset, self._track_left())
            delta = int(round(cur - self._drag_anchor_frame))
            if delta != 0:
                width = self._sel_hi - self._sel_lo
                new_lo = self._clamp_frame(self._sel_lo + delta)
                new_lo = min(new_lo, (self._n_frames - 1) - width)
                new_lo = max(0, new_lo)
                self._sel_lo = new_lo
                self._sel_hi = new_lo + width
                self._drag_anchor_frame = cur
                self.update()
                self.rangeChanged.emit(self._sel_lo, self._sel_hi)
        e.accept()

    def mouseReleaseEvent(self, e) -> None:
        self._drag_mode = None
        e.accept()

    def wheelEvent(self, e) -> None:
        dy = e.angleDelta().y()
        if dy == 0:
            super().wheelEvent(e)
            return
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            x = e.position().x()
            f_cur = frame_of(x, self._fpp, self._offset, self._track_left())
            self._fpp *= 0.8 if dy > 0 else 1.25   # wheel-up = zoom in
            self._clamp_zoom()
            self._user_zoomed = True
            # Keep f_cur under the cursor: x = left + (f_cur - offset)/fpp.
            self._offset = f_cur - (x - self._track_left()) * self._fpp
            self._clamp_offset()
            self.update()
            e.accept()
        else:
            # Plain wheel pans horizontally when zoomed in.
            self._offset += (-dy / 120.0) * self._fpp * 60.0
            self._clamp_offset()
            self.update()
            e.accept()

    def resizeEvent(self, e) -> None:
        if self._user_zoomed:
            self._clamp_zoom()
            self._clamp_offset()
        else:
            self._refit()
        super().resizeEvent(e)


__all__ = ["ClipTimeline", "x_of", "frame_of", "nice_step", "minor_step"]
