# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""WeightSetDialog — pie-chart editor for command-item initial sampling weights.

The Training Motion node's command items (forward / backward / stand / turn …)
are sampled at every env-reset from an initial weight distribution. This modal
dialog lets the user shape that distribution directly:

  * a donut pie shows one wedge per **enabled** item, each in a distinct colour;
    when the weights sum to less than 100% the unallocated remainder is drawn as
    a muted gap, so "how much is still unassigned" is visible at a glance;
  * dragging the boundary between two adjacent wedges transfers weight between
    exactly those two items (their combined share — and therefore the running
    total — is preserved); every other wedge is untouched;
  * below the pie, a legend lists ``[swatch] item: NN.N%`` and the percentage is
    click-to-edit for precise entry. Editing one item changes ONLY that item —
    the others are never auto-adjusted, so the user keeps full manual control.
  * because a free-form edit can make the total drift from 100%, the **Apply
    button is disabled until the weights sum to exactly 100%** (a running total
    is shown). This replaces the old auto-renormalise-on-edit behaviour, which
    left no room to type a set of numbers before they all add up.
  * "Reset to uniform" restores ``1/N`` for all items.

On ``exec() == Accepted`` the weights are available via :attr:`result_weights`
(``{item_id: weight}``) **exactly as the user entered them** — values are NEVER
auto-adjusted to hit 100%. The only enforcement is that Apply stays disabled
until the running total is already 100%.

Colours come from the ``pie_item_color_1..12`` categorical palette in
``system.ini[Theme]`` (CLAUDE.md §5); wedge ``i`` uses
``pie_item_color_{(i % 12) + 1}``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviLineEdit,
    i18n_bind,
    setButton,
    tr,
)

# Smallest fraction a single wedge may shrink to under a boundary drag (0.5%).
# Keeps a drag from collapsing a wedge to an un-grabbable zero. Type-entry is
# NOT clamped to this (the user may type any non-negative value).
_MIN_FRAC = 0.005
# Angular grab tolerance (degrees) for picking up a wedge boundary.
_GRAB_TOL_DEG = 7.0
# How close the total must be to 1.0 (100%) for Apply to enable.
_SUM_TOL = 1e-3


def _pie_color(index: int) -> QColor:
    """Categorical wedge colour for item position ``index`` (wraps at 12)."""
    slot = f"pie_item_color_{(index % 12) + 1}"
    return QColor(Config.get_color(slot, "#888888"))


class _WeightPieWidget(QWidget):
    """Donut pie of RAW per-item weights with draggable wedge boundaries.

    The weights are NOT normalised — when they sum to < 1 the remainder is drawn
    as a muted gap; when they sum to > 1 the wedges are scaled down to fit the
    circle (the legend total then reads > 100% and Apply stays disabled). A
    boundary drag transfers weight between two neighbours, preserving the
    running total. Emits :attr:`weights_changed` (the raw list) on every drag.
    """

    weights_changed = pyqtSignal(list)

    _SIZE = 240
    _INNER_RATIO = 0.52  # donut hole radius / outer radius

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setMouseTracking(True)
        self._weights: List[float] = []
        self._drag_boundary: int = -1  # index i = boundary between wedge i / i+1

    # ---- state -----------------------------------------------------------
    def set_weights(self, weights: Sequence[float]) -> None:
        # Store RAW (clamped non-negative) — no normalisation, so a partial
        # total renders as a gap and the legend can show the true running sum.
        self._weights = [max(0.0, float(w)) for w in weights]
        self.update()

    def weights(self) -> List[float]:
        return list(self._weights)

    def _denom(self) -> float:
        """Circle-fill divisor: total when > 1 (scale to fit), else 1 (gap)."""
        return max(1.0, sum(self._weights))

    # ---- geometry helpers ------------------------------------------------
    def _pie_rect(self) -> QRectF:
        m = 6.0
        side = min(self.width(), self.height()) - 2 * m
        return QRectF(m, m, side, side)

    @staticmethod
    def _cumulative(weights: Sequence[float]) -> List[float]:
        out: List[float] = []
        acc = 0.0
        for w in weights:
            acc += w
            out.append(acc)
        return out

    def _angle_to_top_fraction(self, x: float, y: float) -> float:
        """Map a point to the clockwise-from-top fraction in [0, 1)."""
        rect = self._pie_rect()
        cx = rect.center().x()
        cy = rect.center().y()
        # Qt screen y grows downward; use -dy so math angle is standard CCW.
        ang = math.degrees(math.atan2(-(y - cy), x - cx))
        frac = ((90.0 - ang) % 360.0) / 360.0
        return frac

    def _point_radius(self, x: float, y: float) -> float:
        rect = self._pie_rect()
        return math.hypot(x - rect.center().x(), y - rect.center().y())

    # ---- paint -----------------------------------------------------------
    def paintEvent(self, _e: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self._pie_rect()
        n = len(self._weights)
        if n == 0:
            return

        # Muted "unallocated" backdrop — whatever the wedges don't cover reads
        # as the still-to-assign remainder (only visible when total < 1).
        p.setBrush(QColor(Config.get_color("bg_3", "#2A2C33")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(rect)

        denom = self._denom()
        # Wedges, clockwise from 12 o'clock (negative span in Qt's CCW system).
        start_frac = 0.0
        for i, w in enumerate(self._weights):
            frac = w / denom
            start_angle = int(round((90.0 - start_frac * 360.0) * 16))
            span_angle = int(round(-frac * 360.0 * 16))
            p.setBrush(_pie_color(i))
            p.setPen(QPen(QColor(Config.get_color("bg_2", "#1A1C22")), 1))
            p.drawPie(rect, start_angle, span_angle)
            start_frac += frac

        # Donut hole — overpaint the centre with the panel background.
        hole_r = rect.width() * self._INNER_RATIO / 2.0
        cx = rect.center().x()
        cy = rect.center().y()
        hole = QRectF(cx - hole_r, cy - hole_r, hole_r * 2, hole_r * 2)
        p.setBrush(QColor(Config.get_color("bg_2", "#1A1C22")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(hole)

        # Boundary handles (internal boundaries only — the 12-o'clock start is
        # fixed). A short tick on the ring marks each grabbable boundary.
        if n >= 2:
            outer_r = rect.width() / 2.0
            cum = self._cumulative(self._weights)
            handle_pen = QPen(QColor(Config.get_color("main_t2", "#FFFFFF")), 2)
            p.setPen(handle_pen)
            for i in range(n - 1):
                a_deg = 90.0 - (cum[i] / denom) * 360.0
                a_rad = math.radians(a_deg)
                x0 = cx + math.cos(a_rad) * hole_r
                y0 = cy - math.sin(a_rad) * hole_r
                x1 = cx + math.cos(a_rad) * outer_r
                y1 = cy - math.sin(a_rad) * outer_r
                p.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Centre label — item count.
        p.setPen(QPen(QColor(Config.get_color("main_t1", "#E5E7EB"))))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        p.setFont(font)
        p.drawText(hole, int(Qt.AlignmentFlag.AlignCenter), f"{n}")

    # ---- input -----------------------------------------------------------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        n = len(self._weights)
        if n < 2:
            return
        x, y = e.position().x(), e.position().y()
        rect = self._pie_rect()
        r = self._point_radius(x, y)
        hole_r = rect.width() * self._INNER_RATIO / 2.0
        outer_r = rect.width() / 2.0
        if not (hole_r * 0.6 <= r <= outer_r + 4):
            return
        frac = self._angle_to_top_fraction(x, y)
        denom = self._denom()
        cum = self._cumulative(self._weights)
        # Nearest internal boundary within the grab tolerance (compare in the
        # rendered circle-fraction space = raw_cumulative / denom).
        best_i, best_d = -1, _GRAB_TOL_DEG / 360.0
        for i in range(n - 1):
            d = abs(frac - cum[i] / denom)
            d = min(d, 1.0 - d)  # circular distance
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            self._drag_boundary = best_i
            e.accept()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag_boundary < 0:
            return  # hover only
        i = self._drag_boundary
        n = len(self._weights)
        if i < 0 or i >= n - 1:
            return
        # denom is constant across a drag (a drag preserves the running total).
        denom = self._denom()
        # Mouse position -> raw cumulative (un-scale by denom).
        target_raw = self._angle_to_top_fraction(e.position().x(), e.position().y()) * denom
        cum = self._cumulative(self._weights)
        left_edge = cum[i] - self._weights[i]           # start of wedge i (raw)
        combined = self._weights[i] + self._weights[i + 1]
        lo = left_edge + _MIN_FRAC
        hi = left_edge + combined - _MIN_FRAC
        if hi < lo:
            return
        new_cum = min(max(target_raw, lo), hi)
        self._weights[i] = new_cum - left_edge
        self._weights[i + 1] = combined - self._weights[i]
        self.update()
        self.weights_changed.emit(list(self._weights))
        e.accept()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._drag_boundary >= 0:
            self._drag_boundary = -1
            e.accept()


class WeightSetDialog(QDialog):
    """Modal pie editor for per-item initial sampling weights.

    Construct with the enabled items (``[(item_id, display_name), …]`` in the
    same order they appear in the canvas) and their current weights (a partial
    ``{item_id: weight}`` map — missing items default to the uniform share).
    After ``exec()`` returns ``Accepted`` read :attr:`result_weights`. Apply is
    only enabled when the weights sum to 100%.
    """

    def __init__(
        self,
        items: Sequence[Tuple[str, str]],
        current_weights: Optional[Dict[str, float]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ids: List[str] = [str(i) for i, _ in items]
        self._labels: List[str] = [str(name or i) for i, name in items]
        n = len(self._ids)

        cw = current_weights or {}
        uniform = 1.0 / n if n else 0.0
        # Show stored weights EXACTLY as the user last entered them — never
        # auto-normalise on open. Unset items default to the uniform share.
        raw = [float(cw.get(i, uniform)) for i in self._ids]
        self._weights: List[float] = [max(0.0, w) for w in raw] if n else []

        # Result snapshot — written on accept().
        self.result_weights: Dict[str, float] = {
            i: w for i, w in zip(self._ids, self._weights)
        }

        # Per-item legend percentage editors, in item order.
        self._pct_edits: List[LaviLineEdit] = []
        self._pie: Optional[_WeightPieWidget] = None
        self._ok_btn = None
        self._total_label: Optional[QLabel] = None
        self._suppress_edit_sync = False

        self.setObjectName("weightSetDialog")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setWindowTitle(
            tr("canvas.training_motion.weight_set.title", "Sampling Weights")
        )
        self.setMinimumWidth(360)
        self._build_layout()
        self.apply_theme()
        self._update_total_state()

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        hint = QLabel(self)
        i18n_bind(
            hint, "setText", "canvas.training_motion.weight_set.hint",
            "Drag a boundary between two slices to shift weight between them; "
            "or click a percentage to type it (only that item changes). Apply "
            "is enabled once the total is exactly 100%.",
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Pie, centred.
        pie_row = QHBoxLayout()
        pie_row.addStretch(1)
        self._pie = _WeightPieWidget(self)
        self._pie.set_weights(self._weights)
        self._pie.weights_changed.connect(self._on_pie_changed)
        pie_row.addWidget(self._pie)
        pie_row.addStretch(1)
        root.addLayout(pie_row)

        # Legend — one row per item: [swatch] name : NN.N% (editable).
        for idx, (iid, label) in enumerate(zip(self._ids, self._labels)):
            row = QHBoxLayout()
            row.setSpacing(8)

            swatch = QLabel(self)
            swatch.setFixedSize(14, 14)
            col = _pie_color(idx).name()
            swatch.setStyleSheet(
                f"background: {col}; border: 1px solid "
                f"{Config.get_color('border_1', '#3A3D44')}; border-radius: 2px;"
            )
            row.addWidget(swatch)

            name = QLabel(label, self)
            name.setStyleSheet(
                f"color: {Config.get_color('main_t1')}; background: transparent;"
            )
            row.addWidget(name)
            row.addStretch(1)

            pct = LaviLineEdit(parent=self)
            pct.setFixedWidth(64)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight)
            f = QFont(pct.font())
            f.setPixelSize(int(Config.get_font_size("size_mini", 10)))
            pct.setFont(f)
            pct.editingFinished.connect(
                lambda i=idx: self._on_pct_edited(i)
            )
            self._pct_edits.append(pct)
            row.addWidget(pct)

            unit = QLabel("%", self)
            unit.setStyleSheet(
                f"color: {Config.get_color('main_t2')}; background: transparent;"
            )
            row.addWidget(unit)

            root.addLayout(row)

        self._refresh_pct_texts()

        # Running total — turns to the danger colour while != 100%.
        self._total_label = QLabel(self)
        self._total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self._total_label)

        # Buttons.
        btn_row = QHBoxLayout()
        reset_btn = setButton(
            "canvas.training_motion.weight_set.reset", 120, 28,
            kind="border", spec="none",
            default=tr(
                "canvas.training_motion.weight_set.reset", "Reset to uniform"
            ),
        )
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        ok_btn = setButton(
            "canvas.training_motion.weight_set.ok", 96, 28,
            kind="normal", spec="save",
            default=tr("canvas.training_motion.weight_set.ok", "Apply"),
        )
        ok_btn.clicked.connect(self._on_accept)
        self._ok_btn = ok_btn
        btn_row.addWidget(ok_btn)
        cancel_btn = setButton(
            "canvas.training_motion.weight_set.cancel", 88, 28,
            kind="border", spec="none",
            default=tr("canvas.training_motion.weight_set.cancel", "Cancel"),
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    def apply_theme(self) -> None:
        self.setStyleSheet(
            f"QDialog {{ background: {Config.get_color('bg_2', '#1A1C22')}; }}"
        )

    # ------------------------------------------------------------------
    # Sync between pie, legend %, and the weight vector.
    # ------------------------------------------------------------------
    def _refresh_pct_texts(self) -> None:
        self._suppress_edit_sync = True
        try:
            for i, edit in enumerate(self._pct_edits):
                edit.setText(f"{self._weights[i] * 100.0:.1f}")
        finally:
            self._suppress_edit_sync = False

    def _total(self) -> float:
        return float(sum(self._weights))

    def _is_valid(self) -> bool:
        return abs(self._total() - 1.0) <= _SUM_TOL

    def _update_total_state(self) -> None:
        """Refresh the running-total label + enable/disable Apply."""
        total_pct = self._total() * 100.0
        valid = self._is_valid()
        if self._total_label is not None:
            label = tr("canvas.training_motion.weight_set.total", "Total")
            colour = (
                Config.get_color("safe_zone", "#4CAF50") if valid
                else Config.get_color("danger_zone", "#EF4444")
            )
            self._total_label.setText(f"{label}: {total_pct:.1f}%")
            self._total_label.setStyleSheet(
                f"color: {colour}; background: transparent;"
            )
        if self._ok_btn is not None:
            self._ok_btn.setEnabled(valid)

    def _on_pie_changed(self, weights: list) -> None:
        # A boundary drag preserves the running total; keep raw values as-is.
        self._weights = [max(0.0, float(w)) for w in weights]
        self._refresh_pct_texts()
        self._update_total_state()

    def _on_pct_edited(self, index: int) -> None:
        if self._suppress_edit_sync:
            return
        if index < 0 or index >= len(self._weights):
            return
        txt = self._pct_edits[index].text().strip().rstrip("%").strip()
        try:
            new_frac = float(txt) / 100.0
        except ValueError:
            # Reject a non-numeric entry by restoring the displayed value.
            self._refresh_pct_texts()
            return
        if new_frac < 0.0:
            # A negative weight is meaningless — restore the prior value.
            self._refresh_pct_texts()
            return
        # Set ONLY this item — never auto-adjust the others. The user is free to
        # type a set of values; the running total + disabled Apply enforce the
        # sum-to-100% constraint without stealing their numbers.
        self._weights[index] = new_frac
        if self._pie is not None:
            self._pie.set_weights(self._weights)
        self._refresh_pct_texts()
        self._update_total_state()

    def _on_reset(self) -> None:
        n = len(self._weights)
        if n == 0:
            return
        self._weights = [1.0 / n] * n
        if self._pie is not None:
            self._pie.set_weights(self._weights)
        self._refresh_pct_texts()
        self._update_total_state()

    def _on_accept(self) -> None:
        # Apply is only enabled when valid; guard defensively anyway.
        if not self._is_valid():
            return
        # Persist the user's numbers VERBATIM — never re-normalise. The
        # sum-to-100% constraint is enforced solely by gating Apply, so the
        # values written back are exactly what the user typed.
        self.result_weights = {
            i: w for i, w in zip(self._ids, self._weights)
        }
        self.accept()


__all__ = ["WeightSetDialog"]
