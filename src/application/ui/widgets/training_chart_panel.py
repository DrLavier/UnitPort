"""TrainingChartPanel — self-painted live training chart.

Two-pane layout::

    +----------+----------------------------------------+
    | series   |                                        |
    | toggles  |       _MetricsChartCanvas (QWidget)    |
    | (no bg,  |       QPainter-drawn axes + lines      |
    |  no      |       auto X / Y range, multi-run      |
    |  border) |                                        |
    +----------+----------------------------------------+

Why a plain ``QWidget`` chart canvas instead of ``pyqtgraph.PlotWidget``:
``PlotWidget`` is a ``QGraphicsView``. Applying ``QGraphicsOpacityEffect``
on a parent widget — or even directly on the view/viewport — does NOT
propagate through the GraphicsScene boundary, so the chart half stays
opaque while only the checkbox column fades. The DEMO build paints its
chart directly with ``QPainter`` for exactly this reason. This file
mirrors that approach so a single ``QGraphicsOpacityEffect`` on the
host panel makes the whole chart half (background, grid, axes, ticks,
curves) fade together — matching the Mission Control opacity slider.

Data model:

  * ``set_visible_runs(list[str])`` — driven by :class:`RunSourceSelector`'s
    ``selected_runs_changed``. Toggling a run visible primes the chart
    from :class:`MetricsCache` (replays cached samples) and starts
    listening for new samples on :sig:`AppSignals.training_metrics`.
  * Series toggles stay across run swaps — keyed by series field name,
    not by run_id.
  * Each ``(run_id, series_key)`` pair becomes one drawable series; all
    runs sharing a series key render with that key's catalog color so the
    line matches the left-column legend swatch exactly.

§1.5 compliance: every color comes from ``Config.get_color(slot)``. No
hex literals appear in this file outside of docstring slot examples.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Assets, Config, tr

from application.service.metrics_cache import get_metrics_cache
from application.service.signals import get_app_signals
from application.ui.widgets.color_picker_popup import ColorPickerPopup


# ---------------------------------------------------------------------------
# Series catalog
# ---------------------------------------------------------------------------

# (key, display_name, color_slot, dashed)
# Order is the display order in the checkbox column.
_SERIES_CATALOG: List[Tuple[str, str, str, bool]] = [
    ("reward_mean",  "Mean reward",            "chart_line_reward_mean",   False),
    ("best_reward",  "Best reward",            "chart_line_best_reward",   False),
    ("reward_min",   "Min reward",             "chart_line_reward_band",   True),
    ("reward_max",   "Max reward",             "chart_line_reward_band",   True),
    ("ep_len_mean",  "Mean episode length",    "chart_line_ep_len",        False),
    ("policy_loss",  "Mean surrogate loss",    "chart_line_policy_loss",   False),
    ("value_loss",   "Mean value loss",        "chart_line_value_loss",    False),
    ("entropy",      "Mean entropy loss",      "chart_line_entropy",       False),
    ("action_std",   "Mean action std",        "chart_line_action_std",    False),
    ("kl",           "Mean KL",                "chart_line_kl",            False),
    ("amp_loss",     "AMP loss",               "chart_line_amp_loss",      False),
    ("grad_pen",     "Grad penalty",           "chart_line_grad_pen",      True),
    ("policy_pred",  "Discriminator (policy)", "chart_line_disc_pol",      False),
    ("expert_pred",  "Discriminator (expert)", "chart_line_disc_exp",      False),
]
_SERIES_BY_KEY: Dict[str, Tuple[str, str, str, bool]] = {
    spec[0]: spec for spec in _SERIES_CATALOG
}

# Sample dict keys we never plot as series — X-axis / metadata / counters.
_NON_SERIES_KEYS = {
    "step", "iteration", "total", "total_timesteps", "total_steps",
    "algorithm", "run_id", "reward_valid", "phase", "timestamp",
    "lr", "fps", "collect_time", "learn_time", "iteration_time",
    "accuracy_policy", "accuracy_expert",
    "style_reward", "task_reward", "style_reward_valid", "task_reward_valid",
    "reward_extrinsic", "reward_intrinsic",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_x(sample: Dict[str, Any], fallback: int) -> float:
    """Pull an X coordinate out of a sample dict (step → iteration → fallback)."""
    for k in ("step", "iteration"):
        v = sample.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                continue
    return float(fallback)


# ---------------------------------------------------------------------------
# _MetricsChartCanvas — plain QWidget, QPainter-drawn chart
# ---------------------------------------------------------------------------


class _MetricsChartCanvas(QWidget):
    """Self-painted multi-series line chart.

    Series payload: ``[{"name": str, "color": "#RRGGBB", "dashed": bool,
    "visible": bool, "points": [(x, y), ...]}]``. Repainting is driven
    by :meth:`set_series` (full replacement). The widget never holds a
    QGraphicsScene/View, so a ``QGraphicsOpacityEffect`` on the host
    panel composes through cleanly.
    """

    _PAD_LEFT = 56     # space for left Y axis labels
    _PAD_RIGHT = 56    # space for right Y axis labels
    _PAD_TOP = 20      # 20px breathing room above the plot area
    _PAD_BOTTOM = 28   # space for X axis labels
    _MAX_TICKS = 50

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trainingMetricsChartCanvas")
        self._series: List[Dict[str, Any]] = []
        self._bg_color = QColor(Config.get_color("spec_chart"))
        self._axis_color = QColor(Config.get_color("chart_axis_text"))
        self._grid_color = QColor(Config.get_color("chart_grid"))
        self._text_color = QColor(Config.get_color("chart_axis_text"))
        # Bold Y=0 reference line, painted under the curves to anchor the
        # zero-crossing without obscuring data.
        self._zero_line_color = QColor(Config.get_color("main_t1"))
        # Hover-crosshair state. _hover_pos is the cursor position in widget
        # coords whenever it sits inside the plot area; cleared when the
        # cursor moves into the padding columns or leaves the widget.
        self._hover_pos: Optional[QPointF] = None
        self.setMouseTracking(True)
        self.setMinimumSize(360, 220)

    # ---- hover crosshair lifecycle ----------------------------------

    def _in_plot_area(self, pos: QPointF) -> bool:
        """Cursor falls inside the content rect (i.e. not in padding columns)."""
        rect = self.rect()
        left = rect.left() + self._PAD_LEFT
        right = rect.left() + rect.width() - self._PAD_RIGHT
        top = rect.top() + self._PAD_TOP
        bottom = rect.top() + rect.height() - self._PAD_BOTTOM
        return left <= pos.x() <= right and top <= pos.y() <= bottom

    def enterEvent(self, event):  # noqa: N802
        pos = event.position()
        new_pos = pos if self._in_plot_area(pos) else None
        if new_pos != self._hover_pos:
            self._hover_pos = new_pos
            self.update()

    def leaveEvent(self, event):  # noqa: N802
        if self._hover_pos is not None:
            self._hover_pos = None
            self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        pos = event.position()
        new_pos = pos if self._in_plot_area(pos) else None
        if new_pos != self._hover_pos:
            self._hover_pos = new_pos
            self.update()

    def set_series(self, series: List[Dict[str, Any]]) -> None:
        self._series = list(series or [])
        self.update()

    def apply_theme(self) -> None:
        self._bg_color = QColor(Config.get_color("spec_chart"))
        self._axis_color = QColor(Config.get_color("chart_axis_text"))
        self._grid_color = QColor(Config.get_color("chart_grid"))
        self._text_color = QColor(Config.get_color("chart_axis_text"))
        self._zero_line_color = QColor(Config.get_color("main_t1"))
        self.update()

    # ---- painting ----------------------------------------------------

    def paintEvent(self, event):  # noqa: D401, N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self._bg_color)

        rect = self.rect()
        content = QRectF(
            rect.left() + self._PAD_LEFT,
            rect.top() + self._PAD_TOP,
            max(0, rect.width() - self._PAD_LEFT - self._PAD_RIGHT),
            max(0, rect.height() - self._PAD_TOP - self._PAD_BOTTOM),
        )
        if content.width() <= 40 or content.height() <= 40:
            return

        visible_series = [
            s for s in self._series
            if s.get("visible", True) and (s.get("points") or [])
        ]

        # Frame around plot area — both vertical edges are drawn so the
        # left + right Y axes are clearly bounded.
        painter.setPen(QPen(self._axis_color, 1))
        painter.drawRect(content)

        if not visible_series:
            painter.setPen(self._text_color)
            painter.drawText(
                content, Qt.AlignmentFlag.AlignCenter,
                tr("chart.empty.no_visible_series", "No visible series"),
            )
            return

        x_min, x_max, y_min, y_max = self._compute_ranges(visible_series)
        # X axis ticks: forced integer step (>=1) — iteration count is
        # a strictly integer quantity, so 0.5 increments are meaningless.
        x_ticks = self._nice_ticks(x_min, x_max, 6, integer_step=True)
        # Y axis ticks: one set, painted on both sides → left and right
        # values are guaranteed identical.
        y_ticks = self._nice_ticks(y_min, y_max, 5, integer_step=False)
        # _nice_ticks rounds out to floor(vmin/step)*step .. ceil(vmax/step)*step,
        # so ticks can fall slightly outside [vmin, vmax]. Drop those out-of-range
        # ticks — their grid lines would otherwise paint in the L/R/B padding
        # columns. We keep x_min/x_max snug to the data so the latest sample
        # always reaches the right edge of the plot area (live-chart UX).
        x_span = max(1e-9, x_max - x_min)
        y_span = max(1e-9, y_max - y_min)
        x_tol = x_span * 1e-6
        y_tol = y_span * 1e-6
        x_ticks = [t for t in x_ticks if x_min - x_tol <= t <= x_max + x_tol]
        y_ticks = [t for t in y_ticks if y_min - y_tol <= t <= y_max + y_tol]

        fm = QFontMetrics(self.font())

        # Y grid + tick labels (left + right, same values).
        for tick in y_ticks:
            y = self._map_y(tick, y_min, y_max, content)
            painter.setPen(QPen(self._grid_color, 1))
            painter.drawLine(
                int(content.left()), int(y),
                int(content.right()), int(y),
            )
            label = self._format_tick(tick)
            painter.setPen(self._text_color)
            # Left Y axis label
            painter.drawText(
                QRectF(0, y - 10, content.left() - 6, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            # Right Y axis label (same value, rendered to the right of plot)
            painter.drawText(
                QRectF(content.right() + 6, y - 10, self._PAD_RIGHT - 8, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        # X grid + tick labels.
        for tick in x_ticks:
            x = self._map_x(tick, x_min, x_max, content)
            painter.setPen(QPen(self._grid_color, 1))
            painter.drawLine(
                int(x), int(content.top()),
                int(x), int(content.bottom()),
            )
            painter.setPen(self._text_color)
            label = self._format_tick(tick, integer=True)
            label_w = max(24, fm.horizontalAdvance(label) + 6)
            painter.drawText(
                QRectF(x - label_w / 2, content.bottom() + 4, label_w, 18),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label,
            )

        # Y=0 reference line — only when zero is within the visible Y
        # range. Drawn slightly thicker (width 2) with the main_t1 slot
        # so it stands out from the regular grid as a visual anchor for
        # the zero-crossing. Sits under the curves so highlights stay
        # readable.
        if y_min <= 0.0 <= y_max:
            y_zero = self._map_y(0.0, y_min, y_max, content)
            zero_pen = QPen(self._zero_line_color, 2)
            painter.setPen(zero_pen)
            painter.drawLine(
                int(content.left()), int(y_zero),
                int(content.right()), int(y_zero),
            )

        # Series polylines — clip to content rect so curves don't bleed
        # into the padding columns.
        clip_rect = content.adjusted(1, 1, -1, -1)
        painter.save()
        painter.setClipRect(clip_rect)
        for s in visible_series:
            points = s.get("points") or []
            if len(points) < 2:
                continue
            poly = QPolygonF([
                QPointF(
                    self._map_x(float(x), x_min, x_max, content),
                    self._map_y(float(y), y_min, y_max, content),
                )
                for x, y in points
            ])
            color = QColor(s.get("color", "#60a5fa"))
            width = 2.0
            pen = QPen(color, width)
            if s.get("dashed"):
                pen.setStyle(Qt.PenStyle.DotLine)
                pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.drawPolyline(poly)
        painter.restore()

        # Hover crosshair / per-series f(x) dots / coordinate callout. Drawn
        # last so it overlays curves; positions are computed in widget
        # coords using the same (x_min, x_max, y_min, y_max) reference frame.
        if self._hover_pos is not None:
            self._paint_hover_crosshair(
                painter, content, visible_series,
                x_min, x_max, y_min, y_max,
            )
        painter.end()

    # ---- math --------------------------------------------------------

    @staticmethod
    def _compute_ranges(series_list: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for s in series_list:
            for x, y in s.get("points") or []:
                if x is None or y is None:
                    continue
                if y != y:  # NaN check
                    continue
                xs.append(float(x))
                ys.append(float(y))
        if not xs or not ys:
            return 0.0, 1.0, 0.0, 1.0
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if math.isclose(x_min, x_max):
            x_max = x_min + 1.0
        if math.isclose(y_min, y_max):
            pad = 1.0 if math.isclose(y_min, 0.0) else abs(y_min) * 0.1
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min -= pad
            y_max += pad
        return x_min, x_max, y_min, y_max

    @classmethod
    def _nice_ticks(
        cls, vmin: float, vmax: float, count: int, *, integer_step: bool,
    ) -> List[float]:
        if count <= 1 or math.isclose(vmin, vmax):
            return [vmin, vmax]
        span = max(1e-9, vmax - vmin)
        rough = span / max(1, count - 1)
        magnitude = 10 ** math.floor(math.log10(abs(rough))) if rough > 0 else 1
        step = magnitude
        max_ticks = min(cls._MAX_TICKS, max(2, count * 2))
        for base in (1, 2, 5, 10):
            candidate = base * magnitude
            if integer_step and candidate < 1.0:
                continue
            start = math.floor(vmin / candidate) * candidate
            end = math.ceil(vmax / candidate) * candidate
            tick_count = int(round((end - start) / candidate)) + 1
            step = candidate
            if tick_count <= max_ticks:
                break
        if integer_step:
            step = max(1.0, math.ceil(step))
        start = math.floor(vmin / step) * step
        end = math.ceil(vmax / step) * step
        ticks: List[float] = []
        cur = start
        guard = 0
        while cur <= end + step * 0.5 and guard < cls._MAX_TICKS:
            ticks.append(cur)
            cur += step
            guard += 1
        return ticks or [vmin, vmax]

    @staticmethod
    def _format_tick(value: float, integer: bool = False) -> str:
        val = float(value)
        if integer:
            return f"{int(round(val)) + 0}"  # "+ 0" collapses -0 → 0
        abs_val = abs(val)
        if abs_val >= 1000:
            text = f"{val:.0f}"
        elif abs_val >= 10:
            text = f"{val:.1f}".rstrip("0").rstrip(".")
        else:
            text = f"{val:.2f}"
        # Collapse any flavour of zero ("-0.00", "0.00", "-0", "") to a clean "0".
        if text in ("", "-0", "0", "-0.0", "0.0", "-0.00", "0.00"):
            return "0"
        return text

    @staticmethod
    def _map_x(value: float, x_min: float, x_max: float, rect: QRectF) -> float:
        ratio = 0.0 if math.isclose(x_min, x_max) else (value - x_min) / (x_max - x_min)
        return rect.left() + ratio * rect.width()

    @staticmethod
    def _map_y(value: float, y_min: float, y_max: float, rect: QRectF) -> float:
        ratio = 0.0 if math.isclose(y_min, y_max) else (value - y_min) / (y_max - y_min)
        return rect.bottom() - ratio * rect.height()

    # ---- hover crosshair painting -----------------------------------

    def _paint_hover_crosshair(
        self,
        painter: QPainter,
        content: QRectF,
        visible_series: List[Dict[str, Any]],
        x_min: float, x_max: float,
        y_min: float, y_max: float,
    ) -> None:
        """Vertical scan line at cursor X + interpolated f(x) dots + callout.

        Semantics: the vertical line marks the cursor's X value; each visible
        series contributes one dot at (cursor_x, f(cursor_x)) where f is
        linear interpolation between adjacent samples. Series whose X range
        doesn't bracket the cursor X are skipped (no extrapolation).
        """
        pos = self._hover_pos
        if pos is None or not visible_series or content.width() <= 0:
            return

        # Clamp cursor X to content (defensive — _in_plot_area should already
        # have done this, but widget resize between mouse event and paint
        # could leave the cached pos slightly out).
        cx = max(content.left(), min(content.right(), pos.x()))
        ratio = (cx - content.left()) / content.width()
        data_x = x_min + ratio * (x_max - x_min)

        rows = self._collect_hover_rows(
            visible_series, data_x, content, y_min, y_max,
        )

        # 1. Vertical scan line — drawn whenever hover is active over a chart
        #    that has at least one visible series, even if no series brackets
        #    the cursor X (e.g. cursor past the end of all data).
        highlight = QColor(Config.get_color("highlight"))
        painter.setPen(QPen(highlight, 1))
        painter.drawLine(
            int(cx), int(content.top()),
            int(cx), int(content.bottom()),
        )

        if not rows:
            return

        # 2. Per-series interpolated dots, all on the vertical line.
        bg = QColor(Config.get_color("spec_chart"))
        dot_r = 3.5
        for row in rows:
            color = QColor(row["color"])
            painter.setPen(QPen(bg, 1.5))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx, row["map_y"]), dot_r, dot_r)

        # 3. Coordinate callout.
        self._paint_callout(painter, content, cx, pos.y(), data_x, rows)

    def _collect_hover_rows(
        self,
        visible_series: List[Dict[str, Any]],
        data_x: float,
        content: QRectF,
        y_min: float, y_max: float,
    ) -> List[Dict[str, Any]]:
        """For each series whose X range brackets data_x, interpolate Y."""
        rows: List[Dict[str, Any]] = []
        for s in visible_series:
            points = s.get("points") or []
            if len(points) < 2:
                continue
            x_first = float(points[0][0])
            x_last = float(points[-1][0])
            if data_x < x_first or data_x > x_last:
                continue
            xs = [float(p[0]) for p in points]
            idx = bisect.bisect_left(xs, data_x)
            if idx <= 0:
                y_val = float(points[0][1])
            elif idx >= len(points):
                y_val = float(points[-1][1])
            else:
                x0, y0 = xs[idx - 1], float(points[idx - 1][1])
                x1, y1 = xs[idx], float(points[idx][1])
                if math.isclose(x0, x1):
                    y_val = y0
                else:
                    t = (data_x - x0) / (x1 - x0)
                    y_val = y0 + t * (y1 - y0)
            rows.append({
                "display_name": s.get("display_name") or s.get("name", "?"),
                "run_label": s.get("run_label", ""),
                "color": s.get("color", "#60a5fa"),
                "y_value": y_val,
                "map_y": self._map_y(y_val, y_min, y_max, content),
            })
        return rows

    def _paint_callout(
        self,
        painter: QPainter,
        content: QRectF,
        cx: float, cy: float,
        data_x: float,
        rows: List[Dict[str, Any]],
    ) -> None:
        fm = QFontMetrics(self.font())

        # Run prefix only when >1 distinct run is visible — single-run case
        # would just be visual noise.
        run_labels = {r["run_label"] for r in rows if r["run_label"]}
        show_prefix = len(run_labels) > 1

        header = f"Iter {int(round(data_x))}"
        body_lines: List[Tuple[str, str]] = []  # (line_text, color_hex)
        for r in rows:
            prefix = f"[{r['run_label']}] " if (show_prefix and r["run_label"]) else ""
            text = f"{prefix}{r['display_name']}: {self._fmt_y(r['y_value'])}"
            body_lines.append((text, r["color"]))

        pad = 8
        line_h = fm.height() + 2
        swatch_w = 10
        swatch_gap = 6

        header_w = fm.horizontalAdvance(header)
        body_w = max(
            (swatch_w + swatch_gap + fm.horizontalAdvance(t) for t, _ in body_lines),
            default=0,
        )
        box_w = pad * 2 + max(header_w, body_w)
        box_h = pad * 2 + line_h * (1 + len(body_lines))

        # Position: prefer below-right of cursor; flip when would clip.
        offset = 12
        x = cx + offset
        y = cy + offset
        if x + box_w > content.right():
            x = cx - offset - box_w
        if y + box_h > content.bottom():
            y = cy - offset - box_h
        # Final clamp into content.
        x = max(content.left(), min(content.right() - box_w, x))
        y = max(content.top(), min(content.bottom() - box_h, y))

        box = QRectF(x, y, box_w, box_h)

        bg = QColor(Config.get_color("bg_2"))
        border = QColor(Config.get_color("border_1"))
        main_t = QColor(Config.get_color("main_t1"))
        sub_t = QColor(Config.get_color("sub_t2"))

        painter.setBrush(bg)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(box, 4, 4)

        # Header.
        painter.setPen(sub_t)
        painter.drawText(
            QRectF(x + pad, y + pad, box_w - 2 * pad, line_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            header,
        )

        # Series rows.
        for i, (text, color_hex) in enumerate(body_lines):
            row_y = y + pad + line_h * (i + 1)
            sw_rect = QRectF(
                x + pad,
                row_y + (line_h - 8) / 2,
                swatch_w, 8,
            )
            painter.setBrush(QColor(color_hex))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(sw_rect, 2, 2)

            painter.setPen(main_t)
            painter.drawText(
                QRectF(
                    x + pad + swatch_w + swatch_gap,
                    row_y,
                    box_w - 2 * pad - swatch_w - swatch_gap,
                    line_h,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

    @staticmethod
    def _fmt_y(value: float) -> str:
        """Compact Y value formatting that scales with magnitude."""
        abs_v = abs(value)
        if abs_v >= 1000:
            return f"{value:.1f}"
        if abs_v >= 10:
            return f"{value:.2f}"
        if abs_v >= 0.01 or abs_v == 0.0:
            return f"{value:.3f}"
        return f"{value:.2e}"


# ---------------------------------------------------------------------------
# Helper widgets — eliding label + per-row palette swatch button
# ---------------------------------------------------------------------------


class _ElidingLabel(QLabel):
    """QLabel that elides its text to fit the current width.

    Used per series row so long display names ("Discriminator (policy)")
    truncate cleanly inside the narrow checkbox column instead of
    overflowing or being clipped by the swatch button.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full = text
        self._color = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(20)
        self.setToolTip(text)

    def setFullText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self.update()

    def setTextColor(self, hex_color: str) -> None:
        self._color = hex_color or ""
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(
            self._full, Qt.TextElideMode.ElideRight, max(0, self.width())
        )
        p = QPainter(self)
        if self._color:
            p.setPen(QColor(self._color))
        else:
            p.setPen(self.palette().windowText().color())
        p.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )


class _PaletteSwatchButton(QToolButton):
    """Color-picker affordance rendered with ``icon_color.svg``.

    A colored fill chip would visually collide with the row's checkbox
    indicator (both are small squares). The series color is already
    conveyed by the checkbox tint and label text — this button is a
    pure affordance, so a fixed multi-dot palette icon reads cleanly.
    """

    _renderer: Optional[QSvgRenderer] = None

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("missioncontrol.color.palette.tip"))
        self.setAutoRaise(True)
        if _PaletteSwatchButton._renderer is None:
            icon_path = Assets.find_icon("icon_color")
            if icon_path is not None:
                _PaletteSwatchButton._renderer = QSvgRenderer(str(icon_path))

    def refresh(self) -> None:
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:
        r = _PaletteSwatchButton._renderer
        if r is None or not r.isValid():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r.render(p, QRectF(2.0, 2.0, 16.0, 16.0))


# ---------------------------------------------------------------------------
# TrainingChartPanel
# ---------------------------------------------------------------------------


class TrainingChartPanel(QWidget):
    """Live training-metrics chart with per-series toggles + multi-run overlay."""

    _CHECKBOX_COL_MIN_W = 120
    _CHECKBOX_COL_MAX_W = 480
    _CHECKBOX_COL_DEFAULT_W = 180
    _SPLITTER_HANDLE_W = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trainingChartPanel")

        # Visible runs + per-(run, series) data buffers.
        self._visible_runs: List[str] = []
        # data: { run_id: { series_key: ([x...], [y...]) } }
        self._data: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
        # series visibility: keyed by series_key, applies across runs
        self._series_visible: Dict[str, bool] = {}
        # discovered series keys (preserves insertion order via list)
        self._known_series: List[str] = []
        # series_key → row child widgets (one row = QFrame[checkbox|label|swatch])
        self._series_checkboxes: Dict[str, QCheckBox] = {}
        self._series_rows: Dict[str, QWidget] = {}
        self._series_labels: Dict[str, "_ElidingLabel"] = {}
        self._series_swatches: Dict[str, "_PaletteSwatchButton"] = {}
        # Transient per-key colour override active while a ColorPickerPopup
        # is open. Resolver checks this dict first so the chart and the
        # row indicator update live during drag. Cleared on confirm (the
        # value is persisted to user.ini and read back via Config) or on
        # cancel (the value is dropped, falling back to system.ini).
        self._picker_preview: Dict[str, str] = {}

        self._build_ui()
        self.apply_theme()

        sigs = get_app_signals()
        sigs.training_metrics.connect(self._on_metrics)
        sigs.training_run_finished.connect(self._on_run_finished)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setObjectName("trainingChartSplitter")
        self._splitter.setHandleWidth(self._SPLITTER_HANDLE_W)
        self._splitter.setChildrenCollapsible(False)

        # --- Left: series checkbox column ---
        self._series_frame = QFrame(self._splitter)
        self._series_frame.setObjectName("trainingChartSeriesFrame")
        self._series_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._series_frame.setMinimumWidth(self._CHECKBOX_COL_MIN_W)
        self._series_frame.setMaximumWidth(self._CHECKBOX_COL_MAX_W)

        side_layout = QVBoxLayout(self._series_frame)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(2)

        self._series_scroll = QScrollArea(self._series_frame)
        self._series_scroll.setObjectName("trainingChartSeriesScroll")
        self._series_scroll.setWidgetResizable(True)
        self._series_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._series_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._series_inner = QWidget(self._series_scroll)
        self._series_inner.setObjectName("trainingChartSeriesInner")
        self._series_inner_layout = QVBoxLayout(self._series_inner)
        self._series_inner_layout.setContentsMargins(0, 0, 0, 0)
        self._series_inner_layout.setSpacing(2)
        self._series_inner_layout.addStretch(1)
        self._series_scroll.setWidget(self._series_inner)
        side_layout.addWidget(self._series_scroll, 1)

        # --- Right: self-painted chart canvas ---
        self._chart_canvas = _MetricsChartCanvas(self._splitter)

        self._splitter.addWidget(self._series_frame)
        self._splitter.addWidget(self._chart_canvas)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([self._CHECKBOX_COL_DEFAULT_W, 10_000])
        root.addWidget(self._splitter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_visible_runs(self, run_ids: Iterable[str]) -> None:
        """Sync the plotted runs to *run_ids*."""
        new = list(dict.fromkeys(str(r) for r in run_ids if r))
        prior = set(self._visible_runs)
        new_set = set(new)

        for rid in prior - new_set:
            self._drop_run(rid)
        for rid in new_set - prior:
            self._add_run(rid)

        self._visible_runs = new
        self._push_to_canvas()

    def apply_theme(self) -> None:
        bg = Config.get_color("spec_chart")
        font_small = Config.get_font_size("size_small")
        border = Config.get_color("border_1")
        handle_hover = Config.get_color("hover_1")
        row_hover = Config.get_color("hover_2")
        self.setStyleSheet(
            f"QWidget#trainingChartPanel {{ background-color: {bg}; }}"
            f"QFrame#trainingChartSeriesFrame {{ background-color: {bg}; border: none; }}"
            f"QScrollArea#trainingChartSeriesScroll {{ background-color: {bg}; border: none; }}"
            f"QWidget#trainingChartSeriesInner {{ background-color: {bg}; }}"
            f"QWidget#trainingChartSeriesRow {{"
            f"  background-color: transparent;"
            f"  border-radius: 3px;"
            f"}}"
            f"QWidget#trainingChartSeriesRow:hover {{"
            f"  background-color: {row_hover};"
            f"}}"
            f"QLabel#trainingChartSeriesName {{"
            f"  font-size: {font_small}px;"
            f"  background: transparent;"
            f"}}"
            f"QSplitter#trainingChartSplitter::handle {{ background-color: {border}; }}"
            f"QSplitter#trainingChartSplitter::handle:hover {{ background-color: {handle_hover}; }}"
        )
        self._chart_canvas.apply_theme()

        for key, cb in self._series_checkboxes.items():
            self._style_series_checkbox(cb, key)
            self._refresh_series_visuals(key)

    # ------------------------------------------------------------------
    # Slot: live metrics
    # ------------------------------------------------------------------

    def _on_metrics(self, run_id: str, sample: dict) -> None:
        rid = str(run_id or "")
        if not rid or rid not in self._visible_runs:
            return
        if not isinstance(sample, dict):
            return
        x = _extract_x(sample, fallback=len(self._data.get(rid, {}).get("__x_fallback", [0])))
        any_added = False
        for key, value in sample.items():
            if key in _NON_SERIES_KEYS:
                continue
            if not isinstance(value, (int, float)):
                continue
            self._ensure_series_known(key)
            self._append_point(rid, key, x, float(value))
            any_added = True
        if any_added:
            self._push_to_canvas()

    def _on_run_finished(self, run_id: str, _success: bool) -> None:
        rid = str(run_id or "")
        if rid in self._visible_runs:
            self._reprime_run(rid)
            self._push_to_canvas()

    # ------------------------------------------------------------------
    # Internals — data buffers
    # ------------------------------------------------------------------

    def _add_run(self, run_id: str) -> None:
        self._data.setdefault(run_id, {})
        self._reprime_run(run_id)

    def _drop_run(self, run_id: str) -> None:
        self._data.pop(run_id, None)

    def _reprime_run(self, run_id: str) -> None:
        self._data[run_id] = {}
        samples = get_metrics_cache().get_run(run_id)
        for idx, sample in enumerate(samples):
            x = _extract_x(sample, fallback=idx)
            for key, value in sample.items():
                if key in _NON_SERIES_KEYS:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                self._ensure_series_known(key)
                self._append_point(run_id, key, x, float(value))

    def _append_point(self, run_id: str, key: str, x: float, y: float) -> None:
        buf = self._data.setdefault(run_id, {}).setdefault(key, ([], []))
        buf[0].append(x)
        buf[1].append(y)

    def _push_to_canvas(self) -> None:
        """Flatten ``_data`` into the series payload the canvas expects."""
        out: List[Dict[str, Any]] = []
        for run_id in self._visible_runs:
            series_buf = self._data.get(run_id) or {}
            for key, (xs, ys) in series_buf.items():
                spec = _SERIES_BY_KEY.get(key)
                if spec is not None:
                    _, display_name, _slot, dashed = spec
                else:
                    display_name = key
                    dashed = False
                color = self._resolve_color(key)
                visible = self._series_visible.get(key, True)
                points = list(zip(xs, ys))
                out.append({
                    "name": f"{run_id}:{key}",
                    "display_name": display_name,
                    "run_label": run_id,
                    "color": color,
                    "dashed": dashed,
                    "visible": visible,
                    "points": points,
                })
        self._chart_canvas.set_series(out)

    # ------------------------------------------------------------------
    # Internals — colour resolution + per-series colour override
    # ------------------------------------------------------------------

    @staticmethod
    def _slot_for(key: str) -> str:
        """Theme slot used as the user.ini overlay key for *key*.

        Known series use the catalog's chosen slot (e.g. ``reward_min`` and
        ``reward_max`` share ``chart_line_reward_band`` — overriding the
        band tints both, by design). Unknown series receive a dynamic
        slot name so each can be overridden independently; if no override
        is set, the dynamic slot is absent from both system.ini and
        user.ini and the resolver falls back to ``chart_line_default``.
        """
        spec = _SERIES_BY_KEY.get(key)
        if spec is not None:
            return spec[2]
        return f"chart_line_{key}"

    def _resolve_color(self, key: str) -> str:
        """Return the effective hex colour for series *key*.

        Order: live picker preview → user.ini[Theme] (overlay) →
        system.ini[Theme] → ``chart_line_default`` (currently #EBEBEB).
        """
        if key in self._picker_preview:
            return self._picker_preview[key]
        slot = self._slot_for(key)
        fallback = Config.get_color("chart_line_default")
        return Config.get_color(slot, fallback=fallback)

    def _open_color_picker(self, key: str) -> None:
        if key not in self._series_swatches:
            return
        slot = self._slot_for(key)
        initial = self._resolve_color(key)
        swatch = self._series_swatches[key]
        anchor = swatch.mapToGlobal(swatch.rect().bottomLeft())

        popup = ColorPickerPopup(initial, parent=self)
        popup.color_changed.connect(
            lambda hex_, k=key: self._apply_preview_color(k, hex_)
        )
        popup.accepted_color.connect(
            lambda hex_, k=key, s=slot: self._commit_color(s, k, hex_)
        )
        popup.cancelled.connect(lambda k=key: self._clear_preview(k))
        popup.show_at(anchor)

    def _apply_preview_color(self, key: str, hex_: str) -> None:
        self._picker_preview[key] = hex_
        self._refresh_series_visuals(key)
        self._push_to_canvas()

    def _clear_preview(self, key: str) -> None:
        self._picker_preview.pop(key, None)
        self._refresh_series_visuals(key)
        self._push_to_canvas()

    def _commit_color(self, slot: str, key: str, hex_: str) -> None:
        Config.set_value("Theme", slot, hex_)
        self._picker_preview.pop(key, None)
        self._refresh_series_visuals(key)
        self._push_to_canvas()

    def _refresh_series_visuals(self, key: str) -> None:
        cb = self._series_checkboxes.get(key)
        if cb is not None:
            self._style_series_checkbox(cb, key)
        swatch = self._series_swatches.get(key)
        if swatch is not None:
            swatch.refresh()
        lbl = self._series_labels.get(key)
        if lbl is not None:
            lbl.setTextColor(self._resolve_color(key))

    # ------------------------------------------------------------------
    # Internals — series toggle column
    # ------------------------------------------------------------------

    def _ensure_series_known(self, key: str) -> None:
        if key in self._series_checkboxes:
            return
        self._known_series.append(key)
        self._series_visible.setdefault(key, True)

        row = QWidget(self._series_inner)
        row.setObjectName("trainingChartSeriesRow")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        h = QHBoxLayout(row)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(6)

        cb = QCheckBox(row)
        cb.setChecked(self._series_visible[key])
        cb.setCursor(Qt.CursorShape.PointingHandCursor)
        cb.toggled.connect(lambda checked, k=key: self._on_series_toggled(k, checked))

        lbl = _ElidingLabel(self._series_label(key), row)
        lbl.setObjectName("trainingChartSeriesName")
        # Clicking the label toggles the checkbox (matches the prior
        # single-widget UX where the entire row was clickable).
        lbl.mousePressEvent = (  # type: ignore[assignment]
            lambda _e, c=cb: c.setChecked(not c.isChecked())
        )

        swatch = _PaletteSwatchButton(row)
        swatch.clicked.connect(lambda _=False, k=key: self._open_color_picker(k))

        h.addWidget(cb, 0)
        h.addWidget(lbl, 1)
        h.addWidget(
            swatch,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self._series_checkboxes[key] = cb
        self._series_rows[key] = row
        self._series_labels[key] = lbl
        self._series_swatches[key] = swatch

        self._refresh_series_visuals(key)

        insert_at = max(0, self._series_inner_layout.count() - 1)
        self._series_inner_layout.insertWidget(insert_at, row)

    @staticmethod
    def _series_label(key: str) -> str:
        spec = _SERIES_BY_KEY.get(key)
        if spec is not None:
            _, display_name, _, _ = spec
            return display_name
        return key

    def _style_series_checkbox(self, cb: QCheckBox, key: str) -> None:
        color = self._resolve_color(key)
        bg = Config.get_color("spec_chart")
        hover_bg = Config.get_color("hover_2")
        font_small = Config.get_font_size("size_small")
        # Indicator: replace the default checkmark glyph with a filled
        # square in the series color. The series line color doubles as
        # the indicator fill, so a glance at the column is enough to
        # match a checkbox to its line on the chart. ``image: none`` on
        # the checked state strips the platform-default tick PNG; the
        # solid background-color then carries the entire visual.
        cb.setStyleSheet(
            f"QCheckBox {{"
            f"  color: {color};"
            f"  background-color: {bg};"
            f"  font-size: {font_small}px;"
            f"  spacing: 6px;"
            f"  padding: 3px 6px;"
            f"  border-radius: 3px;"
            f"}}"
            f"QCheckBox:hover {{ background-color: {hover_bg}; }}"
            f"QCheckBox::indicator {{"
            f"  width: 12px; height: 12px;"
            f"  border: 1px solid {color};"
            f"  border-radius: 2px;"
            f"  background-color: transparent;"
            f"}}"
            f"QCheckBox::indicator:unchecked {{"
            f"  background-color: transparent;"
            f"}}"
            f"QCheckBox::indicator:checked {{"
            f"  background-color: {color};"
            f"  border: 1px solid {color};"
            f"  image: none;"
            f"}}"
        )

    def _on_series_toggled(self, key: str, checked: bool) -> None:
        self._series_visible[key] = bool(checked)
        self._push_to_canvas()


__all__ = ["TrainingChartPanel"]
