#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Ground Window 鈥?Phase A3 + A4.

TrainingWorkspaceWindow
  A standalone window that opens per base policy_id.
  Layout: Header | Top Toolbar Row | Progress Strip | TrainingCanvasWidget | BottomTab

TrainingGraphScene (A4)
  GraphScene subclass with a training-only _node_type_mapping.
  _create_logic_node resolves all five training node classes directly from
  nodes.sys_nodes.training_nodes 鈥?never touches the Mission Canvas registry.

TrainingPaletteTree (A4)
  QTreeWidget collapsible accordion palette.  Top-level items are non-draggable
  section headers; child items start a drag carrying the node name as text/plain
  MIME so it can be dropped onto TrainingCanvasView.

TrainingCanvasView (A4)
  QGraphicsView subclass that accepts drops from TrainingPaletteTree and
  creates the corresponding node at the drop position.

TrainingCanvasWidget (A4)
  Combines palette + view.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional

from PySide6.QtCore import Qt, QEvent, QMimeData, QPoint, QPointF, QRect, QRectF, QSize, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFontMetrics, QIcon, QPainter, QPen, QPolygonF, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QScrollArea,
    QSlider,
    QSplitter,
    QSizePolicy,
    QStyledItemDelegate,
    QRubberBand,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from bin.pages.canvas.graph_scene import GraphScene
from bin.pages.homepage.homepage import WindowControlButtons
from bin.pages.layout.sidebar_dock import SidebarDock
from src.system.core.config_manager import ConfigManager
from src.system.core.logger import log_error
from src.system.core.theme_manager import get_color_slot
from src.system.training.robot_family import resolve_robot_family
from src.system.training.task_template_resolver import resolve_task_template
from src.system.core.utils.path_helper import get_project_root


def get_color(color_key: str, fallback: str = "#FFFFFF") -> str:
    """Resolve Training Ground colors from the active global theme."""
    return get_color_slot().get_color(color_key, fallback)


def _load_icon(rel_path: str, fallback_text: str = "") -> tuple:
    """
    Return (QIcon, has_icon: bool).  rel_path is relative to project root.
    Falls back gracefully when the SVG file is absent.
    """
    from pathlib import Path
    import os
    root = Path(os.getcwd())
    full = root / rel_path
    if full.exists():
        icon = QIcon(str(full))
        if not icon.isNull():
            return icon, True
    return QIcon(), False


# ---------------------------------------------------------------------------
# System resource monitor widget
# ---------------------------------------------------------------------------

class SysMonitorWidget(QWidget):
    """
    Compact live system monitor: CPU % / RAM / GPU % / VRAM.

    Updates every 2 s via an internal QTimer.
    GPU metrics use torch.cuda when a CUDA device is present,
    falling back to "N/A" gracefully.
    """

    _INTERVAL_MS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingFloatSysMonitor")
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(self._INTERVAL_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def _build_ui(self):
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(6, 0, 6, 0)
        hbox.setSpacing(4)

        # Initialize with widest possible text so adjustSize() measures correctly.
        # GB values use 4-digit max (1000G = ~1 TB) to avoid overflow on high-mem systems.
        self._cpu_lbl  = QLabel("CPU 100%")
        self._ram_lbl  = QLabel("RAM 1000.00/1000G")
        self._gpu_lbl  = QLabel("GPU 100%")
        self._vram_lbl = QLabel("VRAM 1000.00/1000G")

        for lbl in (self._cpu_lbl, self._ram_lbl, self._gpu_lbl, self._vram_lbl):
            lbl.setObjectName("sysMonitorLabel")

        for lbl in (self._cpu_lbl, self._ram_lbl, self._gpu_lbl, self._vram_lbl):
            hbox.addWidget(lbl)
            if lbl is not self._vram_lbl:
                sep = QLabel("|")
                sep.setObjectName("sysMonitorSep")
                hbox.addWidget(sep)

    def _refresh(self):
        # 鈹€鈹€ CPU / RAM (psutil) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        try:
            import psutil
            cpu = int(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            ram_used_gb = mem.used / 1024 ** 3
            ram_total_gb = mem.total / 1024 ** 3
            self._cpu_lbl.setText(f"CPU {cpu}%")
            self._ram_lbl.setText(f"RAM {ram_used_gb:.2f}/{ram_total_gb:.0f}G")
        except Exception:
            self._cpu_lbl.setText("CPU -")
            self._ram_lbl.setText("RAM -")

        # 鈹€鈹€ GPU / VRAM 鈥?cross-vendor (NVIDIA 鈫?AMD 鈫?fallback) 鈹€鈹€鈹€鈹€鈹€鈹€
        gpu_pct: float | None = None
        vram_used_gb: float | None = None
        vram_total_gb: float | None = None

        # 1) pynvml 鈥?NVIDIA (preferred, no subprocess)
        if gpu_pct is None:
            try:
                import pynvml  # nvidia-ml-py3
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_pct      = float(util.gpu)
                vram_used_gb = mem.used / 1024 ** 3
                vram_total_gb = mem.total / 1024 ** 3
            except Exception:
                pass

        # 2) nvidia-smi subprocess 鈥?NVIDIA fallback
        if gpu_pct is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        gpu_pct      = float(parts[0])
                        vram_total_gb = float(parts[2]) / 1024
                        vram_used_gb = float(parts[1]) / 1024  # MiB 鈫?GiB
            except Exception:
                pass

        # 3) rocm-smi 鈥?AMD
        if gpu_pct is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        # header: GPU use (%),VRAM Total (B),VRAM Used (B)
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
                            gpu_pct      = float(parts[0])
                            vram_used_gb = float(parts[2]) / 1024 ** 3
                            vram_total_gb = float(parts[1]) / 1024 ** 3
                            break
            except Exception:
                pass

        if gpu_pct is not None and vram_used_gb is not None and vram_total_gb is not None:
            self._gpu_lbl.setText(f"GPU {int(gpu_pct)}%")
            self._vram_lbl.setText(f"VRAM {vram_used_gb:.2f}/{vram_total_gb:.0f}G")
        else:
            self._gpu_lbl.setText("GPU N/A")
            self._vram_lbl.setText("VRAM N/A")

    def apply_theme(self, text_color: str = "#9ca3af", bg: str = "transparent"):
        sep_color = "#4b5563"
        style = (
            f"#trainingFloatSysMonitor {{ background: transparent; }}"
            f"QLabel#sysMonitorLabel {{ color: {text_color}; font-size: 10px; "
            f"background: transparent; border: none; }}"
            f"QLabel#sysMonitorSep {{ color: {sep_color}; font-size: 10px; "
            f"background: transparent; border: none; }}"
        )
        self.setStyleSheet(style)

    def stop(self):
        self._timer.stop()


# ---------------------------------------------------------------------------
# 4-layer live metrics panel
# ---------------------------------------------------------------------------

# Layer definitions: (layer_key, display_title, [(metric_key, short_label), ...])
_METRIC_LAYERS = [
    ("behavior", "Behavior", [
        ("velocity",            "Vel Score"),
        ("yaw",                 "Yaw Track"),
        ("reference_tracking",  "Ref Track"),
        ("joint_pose_tracking", "Joint Pose"),
        ("joint_vel_tracking",  "Joint Vel"),
        ("foot_pos_tracking",   "Foot Pos"),
        ("foot_air_time",       "Air Time"),
    ]),
    ("stability", "Stability", [
        ("upright",          "Upright"),
        ("base_height",      "Base H (m)"),
        ("base_height_err",  "H Err²"),
        ("angular_rate",     "Ang Rate"),
        ("alive",            "Alive"),
    ]),
    ("action_quality", "Action Quality", [
        ("energy",        "Energy"),
        ("smoothness",    "Smoothness"),
        ("slip",          "Slip"),
        ("foot_clearance","Foot Clear"),
        ("collision",     "Collision"),
    ]),
    ("algorithm", "Algorithm", [
        ("actor_loss",           "Actor Loss"),
        ("policy_gradient_loss", "PG Loss"),
        ("value_loss",           "Value Loss"),
        ("critic_loss",          "Critic Loss"),
        ("entropy_loss",         "Entropy"),
        ("clip_fraction",        "Clip Frac"),
        ("explained_variance",   "Expl Var"),
        ("ent_coef",             "Ent Coef"),
    ]),
]


class MetricsLayersPanel(QWidget):
    """Compact 4-layer live metrics display for Training Ground stats overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_labels: Dict[str, Dict[str, QLabel]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for layer_key, title, metrics in _METRIC_LAYERS:
            section = self._build_section(layer_key, title, metrics)
            layout.addWidget(section)
        layout.addStretch(1)

    def _build_section(self, layer_key: str, title: str, metrics) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("metricsLayerSection")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(3)

        hdr = QLabel(title, frame)
        hdr.setObjectName("metricsLayerTitle")
        vbox.addWidget(hdr)

        grid_widget = QWidget(frame)
        grid = QHBoxLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        self._value_labels[layer_key] = {}
        for metric_key, label_text in metrics:
            cell = QFrame(grid_widget)
            cell.setObjectName("metricsLayerCell")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(4, 2, 4, 2)
            cell_layout.setSpacing(1)
            key_lbl = QLabel(label_text, cell)
            key_lbl.setObjectName("metricsLayerCellKey")
            val_lbl = QLabel("—", cell)
            val_lbl.setObjectName("metricsLayerCellValue")
            cell_layout.addWidget(key_lbl)
            cell_layout.addWidget(val_lbl)
            grid.addWidget(cell)
            self._value_labels[layer_key][metric_key] = val_lbl
        grid.addStretch(1)
        vbox.addWidget(grid_widget)
        return frame

    def update_metrics(self, metrics: dict) -> None:
        """Update displayed values from a 4-layer metrics dict."""
        for layer_key, _title, layer_metrics in _METRIC_LAYERS:
            layer_data = metrics.get(layer_key) or {}
            labels = self._value_labels.get(layer_key, {})
            for metric_key, _label in layer_metrics:
                lbl = labels.get(metric_key)
                if lbl is None:
                    continue
                val = layer_data.get(metric_key)
                if val is None:
                    lbl.setText("—")
                else:
                    lbl.setText(f"{float(val):.3f}")

    def apply_theme(self) -> None:
        text_primary = get_color("text_primary", "#e5e7eb")
        text_muted   = get_color("text_muted",   "#6b7280")
        text_accent  = get_color("training_workspace_asset_text", "#60a5fa")
        border_color = get_color("border", "#374151")
        cell_bg      = get_color("training_toolbar_bg", "#1f2937")
        self.setStyleSheet(f"""
            QFrame#metricsLayerSection {{
                border: 1px solid {border_color};
                border-radius: 4px;
                background: transparent;
            }}
            QLabel#metricsLayerTitle {{
                color: {text_accent};
                font-size: 10px;
                font-weight: bold;
            }}
            QFrame#metricsLayerCell {{
                background: {cell_bg};
                border-radius: 3px;
            }}
            QLabel#metricsLayerCellKey {{
                color: {text_muted};
                font-size: 9px;
            }}
            QLabel#metricsLayerCellValue {{
                color: {text_primary};
                font-size: 10px;
                font-weight: bold;
            }}
        """)


# ---------------------------------------------------------------------------
# Training stats popup + chart
# ---------------------------------------------------------------------------

class TrainingMetricsChart(QWidget):
    """Lightweight line-chart widget for Training Ground run metrics."""

    _MAX_TICKS = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingMetricsChart")
        self._series = []
        self._bg_color = QColor("#0f172a")
        self._axis_color = QColor("#5b5b5b")
        self._grid_color = QColor("#2d2d2d")
        self._text_color = QColor("#cfcfcf")
        self.setMinimumSize(420, 280)

    def set_series(self, series: list) -> None:
        self._series = list(series or [])
        self.update()

    def apply_theme(self, bg: str, axis: str, grid: str, text: str) -> None:
        self._bg_color = QColor(bg)
        self._axis_color = QColor(axis)
        self._grid_color = QColor(grid)
        self._text_color = QColor(text)
        self.update()

    def paintEvent(self, event):  # noqa: D401
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self._bg_color)

        content = self.rect().adjusted(54, 18, -18, -36)
        if content.width() <= 40 or content.height() <= 40:
            return

        visible_series = [
            series for series in self._series
            if series.get("visible", True) and series.get("points")
        ]
        axis_color = self._axis_color
        grid_color = self._grid_color
        text_color = self._text_color

        painter.setPen(QPen(axis_color, 1))
        painter.drawRect(content)

        if not visible_series:
            painter.setPen(text_color)
            painter.drawText(content, Qt.AlignmentFlag.AlignCenter, "No visible series")
            return

        x_min, x_max, y_min, y_max = self._compute_ranges(visible_series)
        y_ticks = self._nice_ticks(y_min, y_max, 5)
        x_ticks = self._nice_ticks(x_min, x_max, 6)

        fm = QFontMetrics(self.font())
        for tick in y_ticks:
            y = self._map_y(tick, y_min, y_max, content)
            painter.setPen(QPen(grid_color, 1))
            painter.drawLine(content.left(), int(y), content.right(), int(y))
            painter.setPen(text_color)
            painter.drawText(
                QRectF(0, y - 10, content.left() - 8, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._format_tick(tick),
            )

        for tick in x_ticks:
            x = self._map_x(tick, x_min, x_max, content)
            painter.setPen(QPen(grid_color, 1))
            painter.drawLine(int(x), content.top(), int(x), content.bottom())
            painter.setPen(text_color)
            label = self._format_tick(tick, compact=True)
            label_w = max(24, fm.horizontalAdvance(label) + 6)
            painter.drawText(
                QRectF(x - label_w / 2, content.bottom() + 6, label_w, 18),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label,
            )

        painter.setPen(text_color)
        painter.drawText(
            QRectF(content.left(), content.bottom() + 18, content.width(), 18),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "step",
        )
        painter.save()
        painter.translate(14, content.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-content.height() / 2, -12, content.height(), 20),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "value",
        )
        painter.restore()

        clip_rect = content.adjusted(1, 1, -1, -1)
        painter.save()
        painter.setClipRect(clip_rect)
        for series in visible_series:
            points = series.get("points") or []
            if len(points) < 2:
                continue
            poly = QPolygonF([
                QPointF(
                    self._map_x(float(step), x_min, x_max, content),
                    self._map_y(float(value), y_min, y_max, content),
                )
                for step, value in points
            ])
            pen = QPen(QColor(series.get("color", "#60a5fa")), 2.0)
            painter.setPen(pen)
            painter.drawPolyline(poly)
        painter.restore()

    @staticmethod
    def _compute_ranges(series_list: list) -> tuple:
        xs = []
        ys = []
        for series in series_list:
            for step, value in series.get("points") or []:
                xs.append(float(step))
                ys.append(float(value))
        if not xs or not ys:
            return 0.0, 1.0, 0.0, 1.0
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
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

    @staticmethod
    def _nice_ticks(vmin: float, vmax: float, count: int) -> list:
        if count <= 1 or math.isclose(vmin, vmax):
            return [vmin, vmax]
        span = max(1e-9, vmax - vmin)
        rough = span / max(1, count - 1)
        magnitude = 10 ** math.floor(math.log10(abs(rough))) if rough > 0 else 1
        step = magnitude
        max_ticks = min(TrainingMetricsChart._MAX_TICKS, max(2, count * 2))
        for base in (1, 2, 5, 10):
            candidate = base * magnitude
            start = math.floor(vmin / candidate) * candidate
            end = math.ceil(vmax / candidate) * candidate
            tick_count = int(round((end - start) / candidate)) + 1
            step = candidate
            if tick_count <= max_ticks:
                break
        start = math.floor(vmin / step) * step
        end = math.ceil(vmax / step) * step
        ticks = []
        cur = start
        guard = 0
        while cur <= end + step * 0.5 and guard < TrainingMetricsChart._MAX_TICKS:
            ticks.append(cur)
            cur += step
            guard += 1
        return ticks or [vmin, vmax]

    @staticmethod
    def _format_tick(value: float, compact: bool = False) -> str:
        val = float(value)
        abs_val = abs(val)
        if compact and abs_val >= 1000:
            if abs_val >= 1_000_000:
                scaled = val / 1_000_000
                return f"{scaled:.1f}".rstrip("0").rstrip(".") + "M"
            scaled = val / 1000
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
        if abs_val >= 1000:
            return f"{val:.0f}"
        if abs_val >= 10:
            return f"{val:.1f}".rstrip("0").rstrip(".")
        return f"{val:.2f}"

    @staticmethod
    def _map_x(value: float, x_min: float, x_max: float, rect: QRectF) -> float:
        ratio = 0.0 if math.isclose(x_min, x_max) else (value - x_min) / (x_max - x_min)
        return rect.left() + ratio * rect.width()

    @staticmethod
    def _map_y(value: float, y_min: float, y_max: float, rect: QRectF) -> float:
        ratio = 0.0 if math.isclose(y_min, y_max) else (value - y_min) / (y_max - y_min)
        return rect.bottom() - ratio * rect.height()


class TrainingStatsPopup(QFrame):
    """Canvas-local overlay that compares cached Training Ground runs."""

    _LINE_COLORS = [
        "#60a5fa",
        "#f97316",
        "#34d399",
        "#facc15",
        "#f472b6",
        "#a78bfa",
        "#22d3ee",
        "#fb7185",
        "#4ade80",
        "#c084fc",
    ]
    def __init__(self, anchor_bar: "TrainingFloatControlBar", host: QWidget):
        super().__init__(host)
        self._anchor_bar = anchor_bar
        self._host = host
        self._run_checkboxes: Dict[str, QCheckBox] = {}
        self._series_checkboxes: Dict[str, QCheckBox] = {}
        self._selected_runs = set()
        self._visible_series = {}
        self._current_series = []
        self._current_run_tables: Dict[str, dict] = {}
        self._overlay_opacity = 0.9
        self.setObjectName("trainingStatsPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(self._overlay_opacity)
        self.setGraphicsEffect(self._opacity_effect)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(900)
        self._timer.timeout.connect(self.refresh_from_cache)
        self.apply_theme()
        self.hide()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._title_bar = QWidget(self)
        self._title_bar.setObjectName("trainingStatsTitleBar")
        self._title_bar.setFixedHeight(32)
        close_btn_extent = max(20, self._title_bar.height() - 6)
        title_bar_layout = QHBoxLayout(self._title_bar)
        title_bar_layout.setContentsMargins(14, 0, 10, 0)
        title_bar_layout.setSpacing(8)
        title_lbl = QLabel("Training Metrics", self._title_bar)
        title_lbl.setObjectName("trainingStatsTitleBarLabel")
        title_bar_layout.addWidget(title_lbl)
        title_bar_layout.addStretch(1)

        opacity_label = QLabel("Opacity", self._title_bar)
        opacity_label.setObjectName("trainingStatsOpacityLabel")
        title_bar_layout.addWidget(opacity_label)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal, self._title_bar)
        self._opacity_slider.setObjectName("trainingStatsOpacitySlider")
        self._opacity_slider.setRange(50, 100)
        self._opacity_slider.setValue(int(self._overlay_opacity * 100))
        self._opacity_slider.setFixedWidth(120)
        self._opacity_slider.setToolTip("Overlay opacity")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        title_bar_layout.addWidget(self._opacity_slider)

        self._close_btn = QPushButton(self._title_bar)
        self._close_btn.setObjectName("trainingStatsCloseBtn")
        self._close_btn.setToolTip("Close metrics overlay")
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setFixedSize(close_btn_extent, close_btn_extent)
        from src.system.core.theme_manager import get_icon
        close_icon = get_icon("ICON_CL")
        if not close_icon.isNull():
            self._close_btn.setIcon(close_icon)
            self._close_btn.setIconSize(QSize(max(12, close_btn_extent - 8), max(12, close_btn_extent - 8)))
            self._close_btn.setText("")
        else:
            self._close_btn.setText("X")
        self._close_btn.clicked.connect(self.hide_overlay)
        title_bar_layout.addWidget(self._close_btn)
        outer.addWidget(self._title_bar)

        # 鈹€鈹€ Content area 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        content = QWidget(self)
        root = QHBoxLayout(content)
        root.setContentsMargins(14, 8, 14, 14)
        root.setSpacing(12)
        outer.addWidget(content, 1)

        left_panel = self._build_panel("Runs")
        left_layout = left_panel.layout()
        self._runs_scroll = QScrollArea()
        self._runs_scroll.setWidgetResizable(True)
        self._runs_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._runs_box = QWidget()
        self._runs_layout = QVBoxLayout(self._runs_box)
        self._runs_layout.setContentsMargins(0, 0, 0, 0)
        self._runs_layout.setSpacing(4)
        self._runs_layout.addStretch(1)
        self._runs_scroll.setWidget(self._runs_box)
        left_layout.addWidget(self._runs_scroll, 1)
        root.addWidget(left_panel, 0)

        center_panel = self._build_panel("Metrics")
        center_layout = center_panel.layout()
        self._summary_row = QHBoxLayout()
        self._summary_row.setSpacing(10)
        self._summary_boxes: Dict[str, QLabel] = {}
        for key, title in (
            ("reward_mean", "Reward Mean"),
            ("best_reward", "Best"),
            ("ep_len_mean", "Ep Len"),
            ("status", "Idle"),
        ):
            box = QFrame(center_panel)
            box.setObjectName("trainingStatsSummaryBox")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(12, 10, 12, 10)
            box_layout.setSpacing(4)
            title_label = QLabel(title if key != "status" else "Status", box)
            title_label.setObjectName("trainingStatsSummaryTitle")
            value_label = QLabel("-" if key != "status" else "Idle", box)
            value_label.setObjectName("trainingStatsSummaryValue")
            box_layout.addWidget(title_label)
            box_layout.addWidget(value_label)
            self._summary_row.addWidget(box, 1)
            self._summary_boxes[key] = value_label
        center_layout.addLayout(self._summary_row)
        self._chart = TrainingMetricsChart(center_panel)
        center_layout.addWidget(self._chart, 1)

        # ── 4-layer live metrics panel — below chart, full width of center panel
        self._metrics_layers = MetricsLayersPanel(center_panel)
        self._metrics_layers.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        center_layout.addWidget(self._metrics_layers, 0)

        root.addWidget(center_panel, 1)

        right_panel = self._build_panel("Lines")
        right_layout = right_panel.layout()
        self._lines_scroll = QScrollArea()
        self._lines_scroll.setWidgetResizable(True)
        self._lines_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._lines_box = QWidget()
        self._lines_layout = QVBoxLayout(self._lines_box)
        self._lines_layout.setContentsMargins(0, 0, 0, 0)
        self._lines_layout.setSpacing(4)
        self._lines_layout.addStretch(1)
        self._lines_scroll.setWidget(self._lines_box)
        right_layout.addWidget(self._lines_scroll, 1)
        root.addWidget(right_panel, 0)

    def _build_panel(self, title: str) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("trainingStatsPanel")
        if title == "Runs" or title == "Lines":
            panel.setFixedWidth(210)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        label = QLabel(title, panel)
        label.setObjectName("trainingStatsPanelTitle")
        layout.addWidget(label)
        return panel

    def show_for_button(self, button: QPushButton) -> None:
        del button
        self.refresh_from_cache(force_default=True)
        self.setGeometry(self._host.rect())
        self.show()
        self.raise_()
        self._timer.start()
        self._sync_anchor_visibility()

    def toggle_for_button(self, button: QPushButton) -> None:
        if self.isVisible():
            self.hide_overlay()
            return
        self.show_for_button(button)

    def hide_overlay(self) -> None:
        self.hide()
        self._timer.stop()
        self._sync_anchor_visibility()

    def hideEvent(self, event):
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.stop()
        super().hideEvent(event)

    def sync_to_host(self) -> None:
        self.setGeometry(self._host.rect())
        self._sync_anchor_visibility()

    def _sync_anchor_visibility(self) -> None:
        self.raise_()
        if hasattr(self._anchor_bar, "raise_"):
            self._anchor_bar.raise_()

    def _on_opacity_changed(self, value: int) -> None:
        self._overlay_opacity = max(0.5, min(1.0, float(value) / 100.0))
        self._opacity_effect.setOpacity(self._overlay_opacity)

    def refresh_from_cache(self, force_default: bool = False) -> None:
        entries = self._build_run_entries()
        current_run_ids = {entry["run_id"] for entry in entries}
        self._selected_runs.intersection_update(current_run_ids)

        if (force_default or not self._selected_runs) and entries:
            preferred = next((entry["run_id"] for entry in entries if entry.get("preferred")), entries[0]["run_id"])
            self._selected_runs.add(preferred)

        self._rebuild_run_checkboxes(entries)
        self._rebuild_series()

    def _find_workspace_window(self):
        """Walk up the parent chain to find the TrainingWorkspaceWindow instance."""
        widget = self._host
        while widget is not None:
            if hasattr(widget, "_workspace_policy_id") and hasattr(widget, "_active_run_id"):
                return widget
            widget = widget.parent() if hasattr(widget, "parent") else None
        # Fallback: try via anchor_bar.window()
        return self._anchor_bar.window() if self._anchor_bar is not None else None

    def _build_run_entries(self) -> list:
        from src.system.training.training_run_cache import get_training_run_cache
        from src.system.core.logger import log_debug

        cache = get_training_run_cache()
        tables = cache.list_runs()
        ws = self._find_workspace_window()
        workspace_policy = str(getattr(ws, "_workspace_policy_id", "") or getattr(ws, "_policy_id", "") or "").strip()
        active_run_id = str(getattr(ws, "_active_run_id", "") or "").strip()

        log_debug(
            f"[chart] _build_run_entries: {len(tables)} runs in cache, "
            f"ws_type={type(ws).__name__}, "
            f"workspace_policy={workspace_policy!r}, active_run_id={active_run_id!r}"
        )

        filtered = []
        for table in tables:
            tbl_pid = table.get("policy_id", "")
            tbl_rid = table.get("run_id", "")
            n_samples = len(table.get("samples", []))
            if workspace_policy and tbl_pid and tbl_pid != workspace_policy:
                log_debug(f"[chart]   SKIP run {tbl_rid}: policy_id={tbl_pid!r} != {workspace_policy!r}")
                continue
            log_debug(f"[chart]   KEEP run {tbl_rid}: policy_id={tbl_pid!r}, {n_samples} samples")
            filtered.append(table)
        filtered.sort(key=lambda item: float(item.get("created_at", 0.0) or 0.0), reverse=True)

        entries = []
        active_table = next((table for table in filtered if table.get("run_id") == active_run_id), None)
        if active_table is not None:
            entries.append({
                "run_id": active_run_id,
                "label": self._format_run_label(active_table, prefix="<ON GOING> "),
                "tooltip": self._format_run_tooltip(active_table),
                "preferred": True,
            })

        for table in filtered:
            run_id = str(table.get("run_id", "") or "").strip()
            if not run_id or run_id == active_run_id:
                continue
            entries.append({
                "run_id": run_id,
                "label": self._format_run_label(table),
                "tooltip": self._format_run_tooltip(table),
                "preferred": False,
            })
        return entries

    @staticmethod
    def _format_run_label(table: dict, prefix: str = "") -> str:
        run_id = str(table.get("run_id", "") or "")[-8:]
        policy_id_out = str(table.get("policy_id_out", "") or "unnamed")
        status = str(table.get("status", "unknown") or "unknown")
        return f"{prefix}{policy_id_out} [{status}] #{run_id}"

    @staticmethod
    def _format_run_tooltip(table: dict) -> str:
        return "\n".join([
            f"Run ID: {table.get('run_id', '-')}",
            f"Policy: {table.get('policy_id', '-')}",
            f"Export: {table.get('policy_id_out', '-')}",
            f"Algorithm: {table.get('algorithm', '-')}",
            f"Status: {table.get('status', '-')}",
            f"Samples: {len(table.get('samples') or [])}",
        ])

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count() > 1:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_run_checkboxes(self, entries: list) -> None:
        self._clear_layout(self._runs_layout)
        self._run_checkboxes.clear()
        stretch = self._runs_layout.takeAt(self._runs_layout.count() - 1)
        if stretch is not None:
            del stretch
        for entry in entries:
            cb = QCheckBox(entry["label"], self._runs_box)
            cb.setChecked(entry["run_id"] in self._selected_runs)
            cb.setToolTip(entry["tooltip"])
            cb.stateChanged.connect(lambda _state, run_id=entry["run_id"]: self._on_run_toggled(run_id))
            self._runs_layout.addWidget(cb)
            self._run_checkboxes[entry["run_id"]] = cb
        self._runs_layout.addStretch(1)

    def _on_run_toggled(self, run_id: str) -> None:
        cb = self._run_checkboxes.get(run_id)
        if cb is None:
            return
        if cb.isChecked():
            self._selected_runs.add(run_id)
        else:
            self._selected_runs.discard(run_id)
        self._rebuild_series()

    def _rebuild_series(self) -> None:
        from src.system.training.training_run_cache import get_training_run_cache
        from src.system.core.logger import log_debug

        cache = get_training_run_cache()
        self._current_series = []
        self._current_run_tables = {}
        color_idx = 0
        log_debug(f"[chart] _rebuild_series: selected_runs={self._selected_runs}")
        for run_id in sorted(self._selected_runs):
            table = cache.get_run(run_id)
            if not table:
                continue
            self._current_run_tables[run_id] = table
            samples = table.get("samples") or []
            if not samples:
                continue
            run_name = self._format_run_label(table)
            metric_defs = [
                ("reward_mean", "Reward Mean"),
                ("best_reward", "Best Reward"),
                ("ep_len_mean", "Ep Len Mean"),
            ]
            for field_name, display_name in metric_defs:
                series_key = f"{run_id}:{field_name}"
                color = self._LINE_COLORS[color_idx % len(self._LINE_COLORS)]
                color_idx += 1
                points = sorted(
                    (
                        (float(s.get("step", 0) or 0), float(s.get(field_name, 0.0) or 0.0))
                        for s in samples
                    ),
                    key=lambda p: p[0],
                )
                if series_key not in self._visible_series:
                    self._visible_series[series_key] = True
                self._current_series.append({
                    "key": series_key,
                    "name": display_name,
                    "tooltip": run_name,
                    "color": color,
                    "points": points,
                    "visible": bool(self._visible_series.get(series_key, True)),
                })
        self._rebuild_series_checkboxes()
        self._refresh_summary_boxes()
        self._chart.set_series(self._current_series)

    def _refresh_summary_boxes(self) -> None:
        table = self._resolve_summary_table()
        if not table:
            self._summary_boxes["reward_mean"].setText("-")
            self._summary_boxes["best_reward"].setText("-")
            self._summary_boxes["status"].setText("Idle")
            return
        samples = list(table.get("samples") or [])
        latest = samples[-1] if samples else {}
        reward_mean = latest.get("reward_mean")
        best_reward = latest.get("best_reward")
        ep_len_mean = latest.get("ep_len_mean")
        status = str(table.get("status", "") or latest.get("status", "") or "Idle")
        self._summary_boxes["reward_mean"].setText(
            "-" if reward_mean is None else f"{float(reward_mean):.3f}"
        )
        self._summary_boxes["best_reward"].setText(
            "-" if best_reward is None else f"{float(best_reward):.3f}"
        )
        if "ep_len_mean" in self._summary_boxes:
            self._summary_boxes["ep_len_mean"].setText(
                "-" if not ep_len_mean else f"{float(ep_len_mean):.1f}"
            )
        self._summary_boxes["status"].setText(status)

    def _resolve_summary_table(self) -> Optional[dict]:
        ws = self._find_workspace_window()
        active_run_id = str(getattr(ws, "_active_run_id", "") or "").strip()
        if active_run_id and active_run_id in self._current_run_tables:
            return self._current_run_tables[active_run_id]
        for run_id in sorted(self._selected_runs):
            table = self._current_run_tables.get(run_id)
            if table is not None:
                return table
        for table in self._current_run_tables.values():
            return table
        return None

    def _rebuild_series_checkboxes(self) -> None:
        self._clear_layout(self._lines_layout)
        self._series_checkboxes.clear()
        stretch = self._lines_layout.takeAt(self._lines_layout.count() - 1)
        if stretch is not None:
            del stretch
        for series in self._current_series:
            cb = QCheckBox(series["name"], self._lines_box)
            cb.setChecked(series.get("visible", True))
            cb.setToolTip(str(series.get("tooltip", "") or ""))
            cb.setStyleSheet(
                f"QCheckBox {{ color: {series['color']}; }}"
                f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
            )
            cb.stateChanged.connect(lambda _state, key=series["key"]: self._on_series_toggled(key))
            self._lines_layout.addWidget(cb)
            self._series_checkboxes[series["key"]] = cb
        self._lines_layout.addStretch(1)

    def _on_series_toggled(self, key: str) -> None:
        cb = self._series_checkboxes.get(key)
        if cb is None:
            return
        self._visible_series[key] = cb.isChecked()
        for series in self._current_series:
            if series["key"] == key:
                series["visible"] = cb.isChecked()
                break
        self._chart.set_series(self._current_series)

    def apply_theme(self) -> None:
        log_bg = get_color("training_monitor_log_bg", get_color("cmd_bg", "#0f172a"))
        log_text = get_color("training_monitor_log_text", get_color("text_primary", "#e5e7eb"))
        border = get_color("training_monitor_border", get_color("border", "#374151"))
        toolbar_bg = get_color("training_toolbar_bg", "#111827")
        muted_text = get_color("training_monitor_meta_text", get_color("text_secondary", "#9ca3af"))
        panel_bg = get_color("training_monitor_frame_bg", get_color("card_bg", "#18212f"))
        self.setStyleSheet(
            f"""
            #trainingStatsPopup {{
                background: {log_bg};
                border: 1px solid {border};
                border-radius: 6px;
            }}
            #trainingStatsTitleBar {{
                background: {toolbar_bg};
                border-bottom: 1px solid {border};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            #trainingStatsTitleBarLabel {{
                color: {log_text};
                font-size: 12px;
                font-weight: 700;
                background: transparent;
            }}
            #trainingStatsOpacityLabel {{
                color: {muted_text};
                font-size: 11px;
                background: transparent;
            }}
            #trainingStatsOpacitySlider {{
                background: transparent;
            }}
            #trainingStatsCloseBtn {{
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 0px;
            }}
            #trainingStatsCloseBtn:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            #trainingStatsPanel {{
                background: {panel_bg};
                border: none;
                border-radius: 8px;
            }}
            #trainingStatsPanelTitle {{
                color: {log_text};
                font-size: 12px;
                font-weight: 700;
                background: transparent;
            }}
            #trainingStatsSummaryBox {{
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid {border};
                border-radius: 8px;
            }}
            #trainingStatsSummaryTitle {{
                color: {muted_text};
                font-size: 11px;
                font-weight: 600;
                background: transparent;
            }}
            #trainingStatsSummaryValue {{
                color: {log_text};
                font-size: 18px;
                font-weight: 700;
                background: transparent;
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QCheckBox {{
                color: {log_text};
                spacing: 8px;
                background: transparent;
                padding: 3px 2px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid {border};
                background: rgba(255, 255, 255, 0.04);
            }}
            QCheckBox::indicator:checked {{
                background: #f6d393;
                border: 1px solid #f6d393;
            }}
            """
        )
        self._chart.apply_theme(
            bg=log_bg,
            axis=border,
            grid=get_color("training_monitor_frame_bg", "#223145"),
            text=log_text,
        )
        if hasattr(self, "_metrics_layers"):
            self._metrics_layers.apply_theme()

    def update_live_metrics(self, metrics: dict) -> None:
        """Push a 4-layer metrics dict to the live metrics panel."""
        if hasattr(self, "_metrics_layers"):
            self._metrics_layers.update_metrics(metrics)


# ---------------------------------------------------------------------------
# Floating training control bar  (mirrors the main-canvas missionControlFloat)
# ---------------------------------------------------------------------------

class TrainingFloatControlBar(QWidget):
    """
    Floating control bar for the Training Ground canvas.

    Mirrors ``MainZonePanel.mission_control_float`` in structure:
      [drag] [鈻?Start] [鈴?Pause] [鈴?Stop] | CPU/RAM/GPU/VRAM

    Placed as a child widget of TrainingCanvasView so it overlays the canvas.
    Draggable within the parent bounds.  Position is clamped on parent resize.

    Signals
    -------
    start_clicked()   鈥?user pressed Start
    stop_clicked()    鈥?user pressed Stop
    """

    start_clicked = Signal()
    stop_clicked  = Signal()
    stats_clicked = Signal()
    panel_clicked = Signal()

    _ICON_SIZE = QSize(20, 20)
    _BTN_SIZE  = 30
    _EXTRA_WIDTH = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingFloatBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._dragging = False
        self._drag_offset = QPoint()
        self._config = ConfigManager()
        self._build_ui()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._apply_measured_size()
        self.raise_()
        # Default position: bottom-right, set when first shown
        self._position_initialized = False

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(10, 6, 10, 6)
        hbox.setSpacing(6)

        # Drag handle
        self._drag_btn = QPushButton()
        self._drag_btn.setObjectName("trainingFloatDragHandle")
        self._drag_btn.setFlat(True)
        self._drag_btn.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        self._drag_btn.setToolTip("Drag")
        self._drag_btn.installEventFilter(self)
        hbox.addWidget(self._drag_btn)

        hbox.addSpacing(4)

        # Start
        self._start_btn = QPushButton()
        self._start_btn.setObjectName("trainingFloatStart")
        self._start_btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        self._start_btn.setIconSize(self._ICON_SIZE)
        self._start_btn.setToolTip("Start Training  (compiles current canvas)")
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self.start_clicked.emit)
        hbox.addWidget(self._start_btn)

        # Stop
        self._stop_btn = QPushButton()
        self._stop_btn.setObjectName("trainingFloatStop")
        self._stop_btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        self._stop_btn.setIconSize(self._ICON_SIZE)
        self._stop_btn.setToolTip("Stop Training")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        hbox.addWidget(self._stop_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("trainingFloatSep")
        hbox.addWidget(sep)

        # Compute device selector
        self._device_combo = QComboBox(self)
        self._device_combo.setObjectName("trainingFloatDevice")
        self._device_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.addItem("GPU", "cuda")
        self._device_combo.setToolTip("Training compute device")
        if not self._cuda_available():
            gpu_index = self._device_combo.findData("cuda")
            if gpu_index >= 0:
                model = self._device_combo.model()
                item = model.item(gpu_index)
                if item is not None:
                    item.setEnabled(False)
            self._device_combo.setToolTip("GPU training unavailable on this machine")
        self._load_device_preference()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        hbox.addWidget(self._device_combo)

        self._stats_btn = QPushButton()
        self._stats_btn.setObjectName("trainingFloatStats")
        self._stats_btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        self._stats_btn.setIconSize(self._ICON_SIZE)
        self._stats_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stats_btn.setToolTip("Open runtime metrics charts")
        self._stats_btn.clicked.connect(self.stats_clicked.emit)
        hbox.addWidget(self._stats_btn)

        # System monitor
        self._sysmon = SysMonitorWidget(self)
        hbox.addWidget(self._sysmon)

        # Separator before Control Panel
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setObjectName("trainingFloatSep")
        hbox.addWidget(sep2)

        # Control Panel toggle
        self._panel_btn = QPushButton("Control Panel")
        self._panel_btn.setObjectName("trainingFloatPanelBtn")
        self._panel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._panel_btn.setToolTip("Open the overview control panel")
        self._panel_btn.clicked.connect(self.panel_clicked.emit)
        hbox.addWidget(self._panel_btn)

        # Load icons (call adjustSize after so bar measures correctly)
        self._apply_icons()

    def _apply_icons(self):
        from src.system.core.theme_manager import get_icon
        _bindings = [
            (self._drag_btn,  "move",    ""),
            (self._start_btn, "play",    ">"),
            (self._stop_btn,  "stop",    "[]"),
            (self._stats_btn, "graphic", ""),
        ]
        for btn, icon_name, fallback in _bindings:
            icon = get_icon(icon_name)
            if not icon.isNull():
                btn.setIcon(icon)
                btn.setText("")
            else:
                btn.setIcon(QIcon())
                btn.setText(fallback)
        self._apply_measured_size()  # re-measure after icons are applied

    def _apply_measured_size(self) -> None:
        """
        Size the bar from its content hint, plus a small right-side buffer so
        the sysmon text (especially VRAM) does not visually clip against the edge.
        """
        hint = self.sizeHint()
        self.setFixedSize(hint.width() + self._EXTRA_WIDTH, hint.height())

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> None:
        """
        Switch button enabled-states.

        state: "idle" | "running" | "done"
        """
        if state == "running":
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
        else:  # idle / done
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    def selected_compute_device(self) -> str:
        """Return the current training device selection."""
        data = self._device_combo.currentData()
        return str(data or "cpu")

    def _fit_device_combo_width(self) -> None:
        self._device_combo.setFixedWidth(self._device_combo.sizeHint().width() + 6)

    def _load_device_preference(self) -> None:
        preferred = str(
            self._config.get("PREFERENCES", "training_compute_device", fallback="cpu", config_type="user") or "cpu"
        ).strip().lower()
        index = self._device_combo.findData(preferred)
        if preferred == "cuda" and not self._cuda_available():
            index = self._device_combo.findData("cpu")
        self._device_combo.setCurrentIndex(index if index >= 0 else 0)
        self._fit_device_combo_width()

    def _on_device_changed(self, _index: int) -> None:
        self._fit_device_combo_width()
        self._config.set(
            "PREFERENCES",
            "training_compute_device",
            self.selected_compute_device(),
            config_type="user",
        )
        self._config.save_user_config()

    # ------------------------------------------------------------------
    # Drag (mirrors MainZonePanel eventFilter pattern)
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if watched is self._drag_btn:
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                self._dragging = True
                self._drag_offset = event.position().toPoint()
                self._drag_btn.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            if event.type() == QEvent.Type.MouseMove and self._dragging:
                parent = self.parent()
                if parent is not None:
                    gpos = event.globalPosition().toPoint()
                    local = parent.mapFromGlobal(gpos)
                    new_pos = local - self._drag_offset
                    self.move(self._clamp(new_pos))
                return True
            if (event.type() == QEvent.Type.MouseButtonRelease
                    and self._dragging):
                self._dragging = False
                self._drag_btn.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(watched, event)

    def _clamp(self, pos: QPoint) -> QPoint:
        parent = self.parent()
        if parent is None:
            return pos
        max_x = max(0, parent.width()  - self.width())
        max_y = max(0, parent.height() - self.height())
        return QPoint(max(0, min(pos.x(), max_x)),
                      max(0, min(pos.y(), max_y)))

    def place_default(self):
        """Position top-right of parent with 12 px margin."""
        parent = self.parent()
        if parent is None:
            return
        self._apply_measured_size()
        margin = 12
        pw = parent.width()
        if pw == 0:
            return  # not yet laid out; showEvent will retry when visible
        x = max(0, pw - self.width() - margin)
        self.move(x, margin)
        self._position_initialized = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._position_initialized:
            self.place_default()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self):
        bg        = get_color("training_float_bar_bg",     "#1e2d3d")
        border    = "#F6D393"
        btn_bg    = get_color("training_float_btn_bg",      "transparent")
        btn_hover = get_color("training_float_btn_hover",   "#374151")
        btn_text  = get_color("training_float_btn_text",    "#d1d5db")
        sep_color = get_color("training_float_sep_color",   "#4b5563")
        combo_bg  = get_color("training_float_device_bg",   bg)
        combo_border = get_color("training_float_device_border", border)
        start_color = get_color("training_ctrl_start_text", "#4ade80")
        stop_color  = get_color("training_ctrl_stop_text",  "#f87171")
        disabled_text = get_color("text_muted",             "#6b7280")

        self.setStyleSheet(f"""
            #trainingFloatBar {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            #trainingFloatDragHandle {{
                background: transparent;
                border: none;
                color: {btn_text};
            }}
            #trainingFloatDragHandle:hover {{ background: {btn_hover}; border-radius: 4px; }}
            #trainingFloatStart, #trainingFloatStop, #trainingFloatStats {{
                background: {btn_bg};
                border: none;
                border-radius: 4px;
                color: {btn_text};
                font-size: 13px;
            }}
            #trainingFloatStart:hover, #trainingFloatStop:hover, #trainingFloatStats:hover {{
                background: {btn_hover};
            }}
            #trainingFloatStart:enabled  {{ color: {start_color}; }}
            #trainingFloatStop:enabled   {{ color: {stop_color};  }}
            #trainingFloatStart:disabled,
            #trainingFloatStop:disabled,
            #trainingFloatStats:disabled  {{ color: {disabled_text}; }}
            #trainingFloatSep {{
                color: {sep_color};
                background: {sep_color};
                max-width: 1px;
                min-width: 1px;
            }}
            #trainingFloatDevice {{
                padding: 1px 18px 1px 8px;
                background: {combo_bg};
                color: {btn_text};
                border: 1px solid {combo_border};
                border-radius: 4px;
            }}
            #trainingFloatDevice:hover {{
                background: {btn_hover};
            }}
            #trainingFloatDevice::drop-down {{
                border: none;
                width: 16px;
            }}
            #trainingFloatDevice QAbstractItemView {{
                background: {combo_bg};
                color: {btn_text};
                border: 1px solid {combo_border};
                selection-background-color: {btn_hover};
            }}
            #trainingFloatPanelBtn {{
                background: {btn_bg};
                border: none;
                border-radius: 4px;
                color: {btn_text};
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
            }}
            #trainingFloatPanelBtn:hover {{ background: {btn_hover}; }}
        """)
        self._sysmon.apply_theme(
            text_color=get_color("training_float_sysmon_text", "#9ca3af"),
        )
        self._fit_device_combo_width()
        self._apply_measured_size()

    @staticmethod
    def _cuda_available() -> bool:
        """Best-effort CUDA availability check for enabling the GPU option."""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def _get_workspace_store():
    """Lazy import to avoid circular/Qt-free layer importing Qt modules."""
    from src.system.training.training_workspace_store import TrainingWorkspaceStore
    return TrainingWorkspaceStore(root=str(get_project_root()))


# ---------------------------------------------------------------------------
# A4 鈥?Training-only GraphScene subclass
# ---------------------------------------------------------------------------

class TrainingGraphScene(GraphScene):
    """
    GraphScene subclass for the Training Ground canvas.

    Key differences from the Mission Canvas GraphScene:
    - _node_type_mapping contains only the training node types (Layers A-D).
    - _create_logic_node resolves all ten training node classes from
      nodes.sys_nodes.training_nodes (never from REGISTERED_NODES).
    - _place_initial_nodes lays out the default template:
        RobotMJCF 鈫?PhysicsConfig
        Rewards / Terminations 鈫?TaskConfig
        PhysicsConfig / TaskConfig / ObsAction 鈫?EnvAssembler 鈫?Train 鈫?Export
        AlgorithmConfig 鈫?Train
      All template nodes are wired with a deferred connection pass.
    """

    _TRAINING_NODE_MAP: Dict[str, str] = {
        # Layer A 鈥?Environment
        "Robot / MJCF":      "robot_mjcf",
        "Physics Config":    "physics_config",
        "Rewards":           "rewards",
        "Terminations":      "terminations",
        "Scene Config":      "scene_config",
        "Task Config":       "task_config",
        "Domain Rand":       "domain_rand",
        "Obs & Action":      "obs_action_config",
        "Reference Motion":   "reference_motion",
        "Init Pose":          "init_pose",
        "MultiGated (Reward)": "multigated_reward",
        # Layer B 鈥?Assembly
        "Env Assembler":    "env_assembler",
        # Layer C.0 鈥?Start Point
        "Start Point":      "base_asset",
        # Layer C 鈥?Learning
        "Algorithm Config": "algo_config",
        "Train":            "train",
        # Layer D 鈥?Validation & Export
        "Eval Config":      "eval_config",
        "Export":           "export",
        "Vis Check":        "vis_check",
    }

    def __init__(self, parent=None):
        # Counter must exist before super().__init__() which calls _place_initial_nodes
        self._training_node_counter: int = 1000
        super().__init__(parent)
        self._node_type_mapping = dict(self._TRAINING_NODE_MAP)

    def _place_initial_nodes(self):
        """
        Place the default training template and auto-wire connections.

        Initial placement follows a strict grid rule:
          - horizontal gap between adjacent columns = 40 px
          - vertical gap between nodes in the same column = 20 px
        Coordinates are generated from actual node heights after creation so
        the configured vertical gap remains correct after content expansion.
        """
        from PySide6.QtCore import QPointF, QTimer

        node_w = 268
        h_gap = 40
        v_gap = 20
        x0 = -916
        col_step = node_w + h_gap

        column_defs = [
            ["Robot / MJCF", "Rewards", "Terminations", "Scene Config"],
            ["Physics Config", "Task Config", "Obs & Action", "Domain Rand"],
            ["Env Assembler", "Start Point"],
            ["Algorithm Config"],
            ["Train"],
            ["Export", "Vis Check"],
        ]

        placed = {}
        for col_idx, names in enumerate(column_defs):
            x = x0 + col_idx * col_step
            y = 0
            for name in names:
                item = self.create_node(name, QPointF(x, y))
                placed[name] = item
                item_h = item.rect().height() if item is not None else 120
                y += float(item_h) + v_gap

        rm = placed["Robot / MJCF"]
        rw = placed["Rewards"]
        tm = placed["Terminations"]
        sc = placed["Scene Config"]
        dr = placed["Domain Rand"]
        pc = placed["Physics Config"]
        tc = placed["Task Config"]
        oac = placed["Obs & Action"]
        ea = placed["Env Assembler"]
        ba = placed["Start Point"]
        alg = placed["Algorithm Config"]
        tn = placed["Train"]
        ex = placed["Export"]
        vc = placed["Vis Check"]

        def _wire():
            pairs = [
                (rm,  "robot_spec",         "out", pc,  "robot_spec",         "in"),
                (rw,  "rewards",            "out", tc,  "rewards",            "in"),
                (tm,  "terminations",       "out", tc,  "terminations",       "in"),
                (sc,  "scene_config",       "out", ea,  "scene_config",       "in"),
                (dr,  "domain_rand_config", "out", ea,  "domain_rand_config", "in"),
                (pc,  "physics_config",     "out", ea,  "physics_config",     "in"),
                (tc,  "task_config",        "out", ea,  "task_config",        "in"),
                (oac, "obs_action_config",  "out", ea,  "obs_action_config",  "in"),
                (ea,  "env_config",         "out", tn,  "env_config",         "in"),
                (ba,  "base_asset",         "out", alg, "base_asset",         "in"),
                (alg, "algo_config",        "out", tn,  "algo_config",        "in"),
                (tn,  "train_result",       "out", ex,  "train_result",       "in"),
                (tn,  "vis_check",          "out", vc,  "vis_check",          "in"),
            ]
            for src, s_slot, s_io, dst, d_slot, d_io in pairs:
                out_p = self._get_node_port(src, s_slot, s_io)
                in_p  = self._get_node_port(dst, d_slot,  d_io)
                if out_p and in_p:
                    self._create_connection(out_p, in_p)

        QTimer.singleShot(50, _wire)

    def _create_logic_node(self, name: str, node_id: int, rect_item):
        """Resolve training node classes without touching the Mission Canvas registry."""
        node_type = None
        for display_name, ntype in self._TRAINING_NODE_MAP.items():
            if display_name in name:
                node_type = ntype
                break
        if not node_type:
            return None

        try:
            from src.system.nodes.sys_nodes.training_nodes import (
                RobotMJCFNode,
                PhysicsConfigNode,
                RewardsNode,
                TerminationsNode,
                TaskConfigNode,
                DomainRandNode,
                ObsActionConfigNode,
                EnvAssemblerNode,
                BaseAssetNode,
                AlgorithmConfigNode,
                TrainNode,
                EvalConfigNode,
                ExportNode,
                VisCheckNode,
                SceneConfigNode,
                ReferenceMotionNode,  # Plan A: motion imitation
                InitPoseNode,
                MultiGatedRewardNode,
            )
            _class_map = {
                "robot_mjcf":           RobotMJCFNode,
                "physics_config":       PhysicsConfigNode,
                "rewards":              RewardsNode,
                "terminations":         TerminationsNode,
                "task_config":          TaskConfigNode,
                "domain_rand":          DomainRandNode,
                "obs_action_config":    ObsActionConfigNode,
                "env_assembler":        EnvAssemblerNode,
                "base_asset":           BaseAssetNode,
                "algo_config":          AlgorithmConfigNode,
                "train":                TrainNode,
                "eval_config":          EvalConfigNode,
                "export":               ExportNode,
                "vis_check":            VisCheckNode,
                "scene_config":         SceneConfigNode,
                "reference_motion":     ReferenceMotionNode,
                "init_pose":            InitPoseNode,
                "multigated_reward":    MultiGatedRewardNode,
            }
            node_class = _class_map.get(node_type)
            if node_class is None:
                return None
            return node_class(str(node_id))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # TrainingNodeItem factory overrides
    # ------------------------------------------------------------------

    def create_node(self, name: str, scene_pos: QPointF, **kwargs):
        """
        Override: produce a TrainingNodeItem (custom card) instead of the
        base QGraphicsRectItem for all known training node types.
        Falls back to the base implementation for unknown names.
        """
        node_type = self._TRAINING_NODE_MAP.get(name)
        if not node_type:
            return super().create_node(name, scene_pos, **kwargs)

        from bin.nodes.training_node_items import TrainingNodeItem

        self._training_node_counter += 1
        node_id = self._training_node_counter

        item = TrainingNodeItem(str(node_id), node_type, name)
        logic = self._create_logic_node(name, node_id, item)
        if logic is not None:
            item.attach_logic_node(logic)

        if node_type == "reference_motion":
            item.set_robot_type_hint(self._get_current_robot_type())

        self.addItem(item)
        item.setPos(scene_pos)
        return item

    def _get_node_port(self, node_item, slot: str, io: str):
        """
        Override: delegate to TrainingNodeItem.get_port() for training cards;
        fall back to base implementation for standard GraphScene nodes.
        """
        from bin.nodes.training_node_items import TrainingNodeItem
        if isinstance(node_item, TrainingNodeItem):
            return node_item.get_port(slot, io)
        return super()._get_node_port(node_item, slot, io)

    def _apply_port_visual(self, port_item, state: str = "normal"):
        """
        Override: use TrainingNodePort._apply_visual() (type-coloured) for
        training ports; delegate to base for standard GraphScene ports.
        """
        from bin.nodes.training_node_items import TrainingNodePort
        if isinstance(port_item, TrainingNodePort):
            port_item._apply_visual(state)
        else:
            super()._apply_port_visual(port_item, state)

    def refresh_style(self):
        """Refresh theme styles for Training Ground scene and custom node cards."""
        super().refresh_style()
        self.setBackgroundBrush(QColor(get_color("training_canva_bg", get_color("canvas_bg", "#1e1e1e"))))
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in self.items():
            if isinstance(item, TrainingNodeItem):
                item.apply_theme()

    def _get_current_robot_type(self) -> str:
        """Return the robot_type from the RobotMJCFNode in the current scene, or ''."""
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in self.items():
            if isinstance(item, TrainingNodeItem) and item._node_type == "robot_mjcf":
                params = item.get_parameters() if hasattr(item, "get_parameters") else {}
                return str(params.get("robot_type", "") or "").lower().strip()
        return ""

    def set_initial_robot_type(self, robot_type: str) -> None:
        """Propagate robot_type from RobotContext into the RobotMJCFNode.

        Called when a fresh Training Ground window opens so the canvas
        defaults to the robot selected in the main Mission Canvas.
        Has no effect if robot_type is empty or no RobotMJCF node exists.
        Saved experiments loaded afterwards will override this value.
        """
        if not robot_type:
            return
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in self.items():
            if isinstance(item, TrainingNodeItem) and item._node_type == "robot_mjcf":
                item.load_parameters({"robot_type": robot_type})
                break
        # Propagate hint to any Reference Motion nodes already on canvas
        for item in self.items():
            if isinstance(item, TrainingNodeItem) and item._node_type == "reference_motion":
                item.set_robot_type_hint(robot_type)

    def set_initial_runtime_scenario(self, scenario_settings: dict) -> None:
        """Seed Scene Config from Mission runtime settings for fresh canvases."""
        if not isinstance(scenario_settings, dict):
            return
        runtime_scene_xml = str(
            scenario_settings.get("resolved_runtime_scene_xml")
            or scenario_settings.get("mujoco_scene_xml", "")
            or ""
        ).strip()
        runtime_gravity_z = float(scenario_settings.get("mujoco_gravity_z", -9.81))
        scene_params = {
            "scene_type": "custom" if runtime_scene_xml else "flat",
            "custom_scene_path": runtime_scene_xml,
            "gravity_z": str(runtime_gravity_z),
        }
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in self.items():
            if isinstance(item, TrainingNodeItem) and item._node_type == "scene_config":
                item.set_runtime_scene_xml(runtime_scene_xml)
                item.load_parameters(scene_params)
                break

    def reset_to_default_template(self) -> None:
        """Clear the scene and recreate the default Training Ground template."""
        self.clear()
        self._training_node_counter = 1000
        self._place_initial_nodes()

    # ------------------------------------------------------------------
    # Step 4 鈥?Canvas persistence
    # ------------------------------------------------------------------

    # Emitted when the user requests training or environment review from TrainNode
    train_requested = Signal(dict)
    review_requested = Signal(dict)
    export_review_requested = Signal(dict)
    scene_preview_requested = Signal(dict)
    init_pose_preview_requested = Signal(dict)
    node_param_changed = Signal(str, str, str)   # node_type, key, value

    def _on_node_param_changed(self, node_type: str, key: str, value: str) -> None:
        """Called by TrainingNodeItem via _notify_scene_param_changed on any param write."""
        self.node_param_changed.emit(node_type, key, value)

    def serialize_training_graph(self, for_compiler: bool = False) -> dict:
        """
        Return a JSON-serializable snapshot of the canvas.

        By default (for_compiler=False) ALL nodes are included so that the
        full canvas layout — including disconnected/work-in-progress nodes —
        survives a save/load round-trip.

        Pass for_compiler=True when feeding the graph to TrainingSpecCompiler:
        only nodes reachable upstream/downstream of the TrainNode are included
        so the compiler never sees stale orphan configurations.
        """
        from bin.nodes.training_node_items import TrainingNodeItem

        # 鈹€鈹€ Step 1: collect all nodes and all edges 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        all_nodes = []
        all_edges = []
        seen_edges: set = set()

        for item in self.items():
            if not isinstance(item, TrainingNodeItem):
                continue
            all_nodes.append(item.serialize())
            for key, port in item._ports.items():
                if not key.endswith(":out"):
                    continue
                for conn in (port.data(2) or []):
                    edge_id = id(conn)
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)
                    try:
                        out_p = conn.out_port
                        in_p  = conn.in_port
                        src_node = out_p.parentItem()
                        dst_node = in_p.parentItem()
                        all_edges.append({
                            "src_id":   src_node.data(10),
                            "src_slot": out_p.data(3),
                            "dst_id":   dst_node.data(10),
                            "dst_slot": in_p.data(3),
                        })
                    except Exception:
                        pass

        # 鈹€鈹€ Step 2: reachability filter 鈥?keep only nodes connected to
        #    the TrainNode's upstream (inputs) or downstream (outputs). 鈹€
        train_node_id: str = ""
        for n in all_nodes:
            if n.get("node_type") == "train":
                train_node_id = n["id"]
                break

        if not for_compiler or not train_node_id:
            # No filtering: return full canvas (save/load path keeps all nodes)
            return {"schema": "training_canvas_v1", "nodes": all_nodes, "edges": all_edges}

        # for_compiler=True: BFS reachability — only nodes reachable from TrainNode
        # Build adjacency: upstream (dst 鈫?set of src) for BFS backwards
        upstream: dict = {}   # node_id 鈫?set of node_ids that feed into it
        downstream: dict = {} # node_id 鈫?set of node_ids it feeds into
        for edge in all_edges:
            src_id = edge["src_id"]
            dst_id = edge["dst_id"]
            upstream.setdefault(dst_id, set()).add(src_id)
            downstream.setdefault(src_id, set()).add(dst_id)

        def _bfs(adj: dict, start: str) -> set:
            visited: set = {start}
            queue = [start]
            while queue:
                cur = queue.pop()
                for nbr in adj.get(cur, set()):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            return visited

        reachable_ids = _bfs(upstream, train_node_id) | _bfs(downstream, train_node_id)

        nodes = [n for n in all_nodes if n["id"] in reachable_ids]
        edges = [
            e for e in all_edges
            if e["src_id"] in reachable_ids and e["dst_id"] in reachable_ids
        ]

        return {"schema": "training_canvas_v1", "nodes": nodes, "edges": edges}

    def load_training_graph(self, data: dict) -> None:
        """Clear the canvas and restore from serialize_training_graph() output."""
        if data.get("schema") != "training_canvas_v1":
            raise ValueError(f"Unsupported schema: {data.get('schema')!r}")

        from bin.nodes.training_node_items import TrainingNodeItem

        # Build a display-name 鈫?internal key lookup for create_node
        _type_to_display = {v: k for k, v in self._TRAINING_NODE_MAP.items()}

        # Clear existing items
        self.clear()
        self._training_node_counter = 1000

        # Recreate nodes and build id 鈫?item map
        id_map: Dict[str, object] = {}
        for node_data in data.get("nodes", []):
            node_type = node_data["node_type"]
            display_name = _type_to_display.get(node_type, node_data.get("display_name", node_type))
            x, y = node_data.get("pos", [0.0, 0.0])
            item = self.create_node(display_name, QPointF(x, y))
            if item is not None:
                item.load_parameters(node_data.get("parameters", {}))
                id_map[node_data["id"]] = item

        # Restore edges
        for edge in data.get("edges", []):
            src = id_map.get(edge.get("src_id"))
            dst = id_map.get(edge.get("dst_id"))
            if src is None or dst is None:
                continue
            out_p = self._get_node_port(src, edge["src_slot"], "out")
            in_p  = self._get_node_port(dst, edge["dst_slot"], "in")
            if out_p and in_p:
                self._create_connection(out_p, in_p)

    # ------------------------------------------------------------------
    # Connection management (training canvas overrides)
    # ------------------------------------------------------------------

    def _finish_connection(self, target_port):
        """Override: when dropping onto an occupied input port, evict the old
        connection first so the new one can take its place — provided it is
        type-compatible with the target port.
        """
        if not self._temp_start_port or not target_port:
            super()._finish_connection(target_port)
            return

        start_io = self._temp_start_port.data(1)
        target_io = target_port.data(1)
        if start_io == target_io:
            # Same direction — let base handle the rejection
            super()._finish_connection(target_port)
            return

        in_port = target_port if start_io == "out" else self._temp_start_port
        out_port = self._temp_start_port if start_io == "out" else target_port

        try:
            from bin.pages.canvas.graph_scene import isValid
            existing_in = [
                c for c in (in_port.data(2) or [])
                if c and isValid(c) and c.scene() is not None
            ]
        except Exception:
            existing_in = []

        if not existing_in:
            super()._finish_connection(target_port)
            return

        # Verify compatibility ignoring the existing connection count so we
        # can decide whether to evict before the base validator sees the port.
        if not self._can_connect_ports(out_port, in_port, ignore_connection=existing_in[0]):
            # Types are incompatible — let base report the failure normally.
            super()._finish_connection(target_port)
            return

        # Evict old connections then let the base create the new one.
        for old_conn in existing_in:
            try:
                if old_conn and isValid(old_conn) and old_conn.scene() is not None:
                    self._detach_connection(old_conn)
                    self.removeItem(old_conn)
            except Exception:
                pass

        super()._finish_connection(target_port)

    def _create_connection(self, out_port, in_port):
        """
        Override: before wiring a new connection to *in_port*, evict any
        existing connection on that port so each input always has at most
        one upstream source.
        """
        try:
            existing_in = list(in_port.data(2) or [])
            for old_conn in existing_in:
                try:
                    from bin.pages.canvas.graph_scene import isValid
                    if old_conn and isValid(old_conn) and old_conn.scene() is not None:
                        self._detach_connection(old_conn)
                        self.removeItem(old_conn)
                except Exception:
                    pass
        except Exception:
            pass
        super()._create_connection(out_port, in_port)
        # Notify the in_port's parent node so port-driven param widgets update
        self._notify_port_driven_params(in_port)

    def _detach_connection(self, connection):
        """
        Override: after detaching, notify the previously-connected in_port's
        parent node so it can re-enable port-driven param widgets.

        Must resolve the port *before* super() which may destroy the connection
        and invalidate the C++ port objects.
        """
        in_port = None
        node_item = None
        try:
            from shiboken6 import isValid
            if connection is not None and isValid(connection):
                in_port = getattr(connection, "in_port", None)
                if in_port is not None and isValid(in_port):
                    node_item = in_port.parentItem()
        except Exception:
            pass
        super()._detach_connection(connection)
        # Notify using the pre-resolved node_item (port may be invalid now)
        if node_item is not None:
            try:
                from shiboken6 import isValid as _iv
                from bin.nodes.training_node_items import TrainingNodeItem
                if _iv(node_item) and isinstance(node_item, TrainingNodeItem):
                    node_item._refresh_port_driven_params()
            except Exception:
                pass

    @staticmethod
    def _notify_port_driven_params(port) -> None:
        """Call ``_refresh_port_driven_params()`` on the port's parent TrainingNodeItem."""
        try:
            from bin.nodes.training_node_items import TrainingNodeItem
            from shiboken6 import isValid
            if port is None or not isValid(port):
                return
            node_item = port.parentItem()
            if node_item is not None and isValid(node_item) and isinstance(node_item, TrainingNodeItem):
                node_item._refresh_port_driven_params()
        except Exception:
            pass  # port or parent already deleted — nothing to notify

    # ------------------------------------------------------------------
    # Selection management
    # ------------------------------------------------------------------

    def delete_selected_nodes(self) -> None:
        """Delete all selected TrainingNodeItems and their connections."""
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in list(self.selectedItems()):
            if not isinstance(item, TrainingNodeItem):
                continue
            try:
                self._delete_node_connections(item)
                if item.scene() is not None:
                    self.removeItem(item)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# A4 鈥?Drag-enabled palette list
# ---------------------------------------------------------------------------

class TrainingPaletteTree(QTreeWidget):
    """
    QTreeWidget-based collapsible accordion palette for Training Ground nodes.

    Top-level items are non-draggable section headers (expand/collapse via the
    built-in tree arrow).  Child items carry the node display name as text and
    are draggable 鈥?text/plain MIME data is passed to TrainingCanvasView.
    """

    _GROUP_ROLE = Qt.ItemDataRole.UserRole + 1  # marks group header items

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setExpandsOnDoubleClick(False)
        self.setIndentation(10)
        self.setUniformRowHeights(False)
        self.setAnimated(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Prevent group headers from being selected / dragged
        self.setItemDelegate(_TrainingGroupDelegate(self))
        self.itemClicked.connect(self._on_item_clicked)

    def add_section(self, section_name: str, node_names: list) -> None:
        """Add a collapsible group with draggable node children."""
        group = QTreeWidgetItem([section_name])
        group.setData(0, self._GROUP_ROLE, True)
        group.setFlags(Qt.ItemFlag.ItemIsEnabled)   # not selectable, not draggable
        group.setExpanded(True)
        self.addTopLevelItem(group)

        for name in node_names:
            child = QTreeWidgetItem([name])
            child.setToolTip(0, f"Drag onto canvas to add {name}")
            group.addChild(child)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        # Skip group headers
        if item.data(0, self._GROUP_ROLE):
            return
        mime = QMimeData()
        mime.setText(item.text(0))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Toggle section expand/collapse on single-click of the header row."""
        if item is None or not item.data(0, self._GROUP_ROLE):
            return
        item.setExpanded(not item.isExpanded())


class TrainingCanvasBrowserPanel(QWidget):
    """
    Tree browser for Training Ground experiments, grouped by workspace.

    Tree hierarchy:
        [Add WorkSpace]
        workspace row  [+ New Experiment button]  [Rename button]  [X button]
          鈹斺攢鈹€ experiment row  [X button]
    """

    # (workspace_id, experiment_id)
    canvas_requested = Signal(str, str)
    delete_requested = Signal(str, str)             # workspace_id, experiment_id
    new_workspace_requested = Signal(str)           # robot_type
    new_experiment_requested = Signal(str)          # workspace_id
    rename_workspace_requested = Signal(str)        # workspace_id
    delete_workspace_requested = Signal(str)        # workspace_id

    _EXPERIMENT_ROLE = Qt.ItemDataRole.UserRole + 11
    _WORKSPACE_ROLE  = Qt.ItemDataRole.UserRole + 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingCanvasBrowserPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        actions_row = QWidget(self)
        actions_layout = QHBoxLayout(actions_row)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        actions_layout.addStretch(1)

        self._btn_add_workspace = QPushButton("Add WorkSpace", actions_row)
        self._btn_add_workspace.setObjectName("trainingBtnAddWorkspace")
        self._btn_add_workspace.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_add_workspace.clicked.connect(
            lambda _checked=False: self.new_workspace_requested.emit("")
        )
        actions_layout.addWidget(self._btn_add_workspace, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(actions_row, 0)

        self._tree = QTreeWidget()
        self._tree.setObjectName("trainingCanvasBrowserTree")
        self._tree.setHeaderHidden(True)
        self._tree.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree, 1)

    def populate(self, entries: list, current_experiment_id: str = "") -> None:
        """
        entries: list of dicts with keys:
            workspace_id, workspace_name, experiment_id, leaf_label, tooltip
        """
        self._tree.clear()
        if not entries:
            placeholder = QTreeWidgetItem(["No saved canvases"])
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            placeholder.setSizeHint(0, QSize(0, 28))
            self._tree.addTopLevelItem(placeholder)
            return

        workspace_items: Dict[str, QTreeWidgetItem] = {}
        current_item = None

        for entry in entries:
            workspace_id = str(entry.get("workspace_id") or "")
            workspace_name = str(entry.get("workspace_name") or workspace_id or "workspace")
            experiment_id = str(entry.get("experiment_id") or "")
            leaf_label = str(entry.get("leaf_label") or experiment_id or "canvas")
            tooltip = str(entry.get("tooltip") or leaf_label)

            ws_item = workspace_items.get(workspace_id)
            if ws_item is None:
                ws_item = QTreeWidgetItem([""])
                ws_item.setFlags(ws_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                ws_item.setData(0, self._WORKSPACE_ROLE, workspace_id)
                ws_row = self._build_workspace_row(workspace_name, workspace_id)
                self._tree.addTopLevelItem(ws_item)
                self._tree.setItemWidget(ws_item, 0, ws_row)
                ws_item.setSizeHint(0, ws_row.sizeHint())
                workspace_items[workspace_id] = ws_item

            if not experiment_id:
                continue

            leaf = QTreeWidgetItem([""])
            leaf.setData(0, self._EXPERIMENT_ROLE, experiment_id)
            leaf.setData(0, self._WORKSPACE_ROLE, workspace_id)
            leaf.setToolTip(0, tooltip)
            leaf_row = self._build_experiment_row(leaf_label, workspace_id, experiment_id)
            ws_item.addChild(leaf)
            self._tree.setItemWidget(leaf, 0, leaf_row)
            leaf.setSizeHint(0, leaf_row.sizeHint())
            if experiment_id and experiment_id == current_experiment_id:
                current_item = leaf

        self._tree.expandAll()
        if current_item is not None:
            self._tree.setCurrentItem(current_item)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        experiment_id = str(item.data(0, self._EXPERIMENT_ROLE) or "").strip()
        workspace_id  = str(item.data(0, self._WORKSPACE_ROLE)  or "").strip()
        if experiment_id and workspace_id:
            self.canvas_requested.emit(workspace_id, experiment_id)
        elif workspace_id and not experiment_id:
            # Workspace row double-clicked — load its first child experiment.
            for ci in range(item.childCount()):
                child = item.child(ci)
                child_exp = str(child.data(0, self._EXPERIMENT_ROLE) or "").strip()
                if child_exp:
                    self.canvas_requested.emit(workspace_id, child_exp)
                    break

    # ------------------------------------------------------------------
    # Row widget builders
    # ------------------------------------------------------------------

    def _build_workspace_row(self, workspace_name: str, workspace_id: str) -> QWidget:
        """Workspace row with add, rename, and delete actions."""
        row = QWidget(self._tree)
        row.setObjectName("trainingWorkspaceRow")
        row.setMinimumHeight(30)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        label = QLabel(workspace_name)
        label.setObjectName("trainingWorkspaceLabel")
        label.setToolTip(workspace_name)
        layout.addWidget(label, 1)

        rename_btn = QPushButton("Rename")
        rename_btn.setObjectName("trainingBtnRenameWorkspace")
        rename_btn.setToolTip(f"Rename workspace '{workspace_name}'")
        rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        rename_btn.setFixedHeight(22)
        rename_btn.clicked.connect(
            lambda _checked=False, ws_id=workspace_id: self.rename_workspace_requested.emit(ws_id)
        )
        layout.addWidget(rename_btn)

        add_btn = QPushButton("+")
        add_btn.setObjectName("trainingBtnAddExperiment")
        add_btn.setToolTip(f"New experiment in '{workspace_name}'")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setFixedSize(22, 22)
        add_btn.clicked.connect(
            lambda _checked=False, ws_id=workspace_id: self.new_experiment_requested.emit(ws_id)
        )
        layout.addWidget(add_btn)

        delete_btn = QPushButton("X")
        delete_btn.setObjectName("trainingBtnDeleteWorkspace")
        delete_btn.setToolTip(f"Delete workspace '{workspace_name}' and all experiments")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.setFixedSize(22, 22)
        delete_btn.clicked.connect(
            lambda _checked=False, ws_id=workspace_id: self.delete_workspace_requested.emit(ws_id)
        )
        layout.addWidget(delete_btn)

        # Double-click workspace name area → load first experiment in this workspace.
        def _ws_dbl_click(event, ws_id=workspace_id):
            for ti_idx in range(self._tree.topLevelItemCount()):
                ti = self._tree.topLevelItem(ti_idx)
                if str(ti.data(0, self._WORKSPACE_ROLE) or "") == ws_id and ti.childCount():
                    first_child = ti.child(0)
                    exp_id = str(first_child.data(0, self._EXPERIMENT_ROLE) or "").strip()
                    if exp_id:
                        self.canvas_requested.emit(ws_id, exp_id)
                    break
        row.mouseDoubleClickEvent = _ws_dbl_click
        return row

    def _build_experiment_row(self, label_text: str, workspace_id: str, experiment_id: str) -> QWidget:
        row = QWidget(self._tree)
        row.setObjectName("trainingCanvasBrowserRow")
        row.setMinimumHeight(26)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        label = QLabel(label_text)
        label.setObjectName("trainingCanvasBrowserRowLabel")
        label.setToolTip(label_text)
        layout.addWidget(label, 1)

        delete_btn = QPushButton("X")
        delete_btn.setObjectName("trainingCanvasBrowserDeleteBtn")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.setFixedSize(18, 18)
        delete_btn.clicked.connect(
            lambda _checked=False, ws_id=workspace_id, exp_id=experiment_id: self.delete_requested.emit(ws_id, exp_id)
        )
        layout.addWidget(delete_btn)

        # setItemWidget() absorbs mouse events, so QTreeWidget.itemDoubleClicked
        # never fires.  Handle double-click on the row widget directly.
        row.mouseDoubleClickEvent = (
            lambda event, ws_id=workspace_id, exp_id=experiment_id:
                self.canvas_requested.emit(ws_id, exp_id)
        )
        return row

class TrainingExportBrowserPanel(QWidget):
    """Tree browser for exported runtime bundles grouped by category and robot model."""

    delete_requested = Signal(str)

    _POLICY_ROLE = Qt.ItemDataRole.UserRole + 21

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingExportPanel")

        frame = QFrame(self)
        frame.setObjectName("trainingPalette")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(frame)

        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 8, 6, 8)
        vbox.setSpacing(4)

        self._tree = QTreeWidget(frame)
        self._tree.setObjectName("trainingExportTree")
        self._tree.setHeaderHidden(True)
        self._tree.setCursor(Qt.CursorShape.PointingHandCursor)
        vbox.addWidget(self._tree, 1)

        self.apply_theme()

    def populate(self, entries: list, current_policy_id: str = "") -> None:
        self._tree.clear()
        if not entries:
            placeholder = QTreeWidgetItem(["No exports yet"])
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            placeholder.setSizeHint(0, QSize(0, 28))
            self._tree.addTopLevelItem(placeholder)
            self.apply_theme()
            return

        category_items: Dict[str, QTreeWidgetItem] = {}
        robot_items: Dict[tuple, QTreeWidgetItem] = {}
        current_item = None

        for entry in entries:
            category = str(entry.get("category") or "Training")
            robot_type = str(entry.get("robot_type") or "unknown")
            policy_id = str(entry.get("policy_id") or "")
            leaf_label = str(entry.get("leaf_label") or policy_id or "export")
            tooltip = str(entry.get("tooltip") or leaf_label)

            category_item = category_items.get(category)
            if category_item is None:
                category_item = QTreeWidgetItem([category])
                category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                category_item.setSizeHint(0, QSize(0, 26))
                self._tree.addTopLevelItem(category_item)
                category_items[category] = category_item

            robot_key = (category, robot_type)
            robot_item = robot_items.get(robot_key)
            if robot_item is None:
                robot_item = QTreeWidgetItem([robot_type])
                robot_item.setFlags(robot_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                robot_item.setSizeHint(0, QSize(0, 24))
                category_item.addChild(robot_item)
                robot_items[robot_key] = robot_item

            leaf = QTreeWidgetItem([""])
            leaf.setData(0, self._POLICY_ROLE, policy_id)
            leaf.setToolTip(0, tooltip)
            robot_item.addChild(leaf)
            leaf_row = self._build_export_row(leaf_label, policy_id)
            self._tree.setItemWidget(leaf, 0, leaf_row)
            leaf.setSizeHint(0, leaf_row.sizeHint())
            if current_policy_id and (policy_id == current_policy_id or leaf_label == current_policy_id):
                current_item = leaf

        self._tree.expandAll()
        if current_item is not None:
            self._tree.setCurrentItem(current_item)
        self.apply_theme()

    def _build_export_row(self, label_text: str, policy_id: str) -> QWidget:
        row_height = 26
        row = QWidget(self._tree)
        row.setObjectName("trainingExportRow")
        row.setMinimumHeight(row_height)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 0, 2, 0)
        layout.setSpacing(6)

        label = QLabel(label_text)
        label.setObjectName("trainingExportRowLabel")
        label.setToolTip(label_text)
        layout.addWidget(label, 1, Qt.AlignmentFlag.AlignVCenter)

        delete_btn = QPushButton("X")
        delete_btn.setObjectName("trainingExportDeleteBtn")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.setFixedSize(row_height, row_height)
        delete_btn.clicked.connect(
            lambda _checked=False, bundle_id=policy_id: self.delete_requested.emit(bundle_id)
        )
        layout.addWidget(delete_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def apply_theme(self) -> None:
        palette_bg = get_color("training_toolbar_list_bg", "#1f2937")
        palette_text = get_color("training_toolbar_list_text", get_color("text_primary", "#d1d5db"))
        palette_hover = get_color("training_toolbar_item_hover_bg", get_color("hover_bg", "#374151"))
        palette_selected = get_color("training_toolbar_item_selected_bg", get_color("tab_bg_checked", "#4b5563"))
        section_text = get_color("training_palette_section_text", get_color("text_muted", "#6b7280"))

        frame = self.findChild(QFrame, "trainingPalette")
        if frame is not None:
            frame.setStyleSheet(f"#trainingPalette {{ background: {palette_bg}; border: none; }}")

        self._tree.setStyleSheet(
            f"QTreeWidget {{ background: transparent; border: none; color: {palette_text}; font-size: 12px; }}"
            f"QTreeWidget::item {{ padding: 0px 6px; border-radius: 4px; margin: 1px 0px; }}"
            f"QTreeWidget::item:hover {{ background: {palette_hover}; }}"
            f"QTreeWidget::item:selected {{ background: {palette_selected}; }}"
        )

        for row in self.findChildren(QWidget, "trainingExportRow"):
            row.setStyleSheet("QWidget { background: transparent; border: none; }")
        for label in self.findChildren(QLabel, "trainingExportRowLabel"):
            label.setStyleSheet(
                f"QLabel {{ color: {palette_text}; background: transparent; border: none; padding: 0px; }}"
            )
        for btn in self.findChildren(QPushButton, "trainingExportDeleteBtn"):
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {palette_text}; border: none; border-radius: 1px; padding: 0px; min-width: 0px; min-height: 0px; max-width: 26px; max-height: 26px; }}"
                f"QPushButton:hover {{ background: {palette_hover}; }}"
            )

        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            if category_item is None:
                continue
            category_item.setForeground(0, QColor(section_text))
            for j in range(category_item.childCount()):
                robot_item = category_item.child(j)
                if robot_item is not None:
                    robot_item.setForeground(0, QColor(section_text))


class _TrainingGroupDelegate(QStyledItemDelegate):
    """Renders group-header rows with a distinct muted style."""

    _GROUP_ROLE = TrainingPaletteTree._GROUP_ROLE

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.data(self._GROUP_ROLE):
            # Remove selection / hover highlight for group rows
            option.state &= ~(
                option.state.__class__.State_Selected |
                option.state.__class__.State_MouseOver
            )
            option.backgroundBrush = QBrush(
                QColor(get_color("training_palette_group_bg", get_color("sidebar_rail_bg", "#1e1e1e")))
            )


# ---------------------------------------------------------------------------
# A4 鈥?Drop-accepting canvas view
# ---------------------------------------------------------------------------

class TrainingCanvasView(QGraphicsView):
    """
    QGraphicsView for the Training Ground canvas.

    Interactions (identical to the main Mission Canvas GraphView):
      - Right mouse button drag  鈫?pan
      - Middle mouse button drag 鈫?zoom (drag up/right = zoom in)
      - Mouse wheel              鈫?zoom (centred on cursor)
      - Drop from palette        鈫?create node at drop position
    """

    _ZOOM_MIN = 0.3
    _ZOOM_MAX = 3.0
    _ZOOM_SENSITIVITY = 1.004   # per-pixel drag factor

    def __init__(self, scene: TrainingGraphScene, parent=None):
        super().__init__(scene, parent)
        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Pan / zoom state
        self._zoom_factor: float = 1.0
        self._is_panning: bool = False
        self._pan_start_pos = None
        self._is_middle_zooming: bool = False
        self._zoom_drag_pos = None

        # Rubber-band selection state
        self._rubber_band: Optional[QRubberBand] = None
        self._rb_origin: Optional[QPoint] = None

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 0.85
        new_zoom = self._zoom_factor * factor
        if new_zoom < self._ZOOM_MIN or new_zoom > self._ZOOM_MAX:
            return
        self._zoom_factor = new_zoom
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)

    # ------------------------------------------------------------------
    # Pan (right button) + zoom drag (middle button)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_middle_zooming = True
            self._zoom_drag_pos = event.pos()
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # Start rubber-band only when clicking on empty canvas (no item under cursor)
            if self.itemAt(event.pos()) is None:
                if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self.scene().clearSelection()
                self._rb_origin = event.pos()
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
                self._rubber_band.setGeometry(QRect(self._rb_origin, QSize()))
                self._rubber_band.show()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Safety: cancel pan/zoom if button released outside window
        if self._is_panning and not (event.buttons() & Qt.MouseButton.RightButton):
            self._is_panning = False
            self._pan_start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._is_panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        if self._is_middle_zooming:
            delta = event.pos() - self._zoom_drag_pos
            self._zoom_drag_pos = event.pos()
            zoom_delta = delta.x() + delta.y()
            if zoom_delta:
                factor = self._ZOOM_SENSITIVITY ** zoom_delta
                target = max(self._ZOOM_MIN, min(self._ZOOM_MAX,
                             self._zoom_factor * factor))
                applied = target / self._zoom_factor if self._zoom_factor else 1.0
                if applied != 1.0:
                    self._zoom_factor = target
                    self.scale(applied, applied)
            event.accept()
            return

        if self._rubber_band is not None and self._rb_origin is not None:
            rb_rect = QRect(self._rb_origin, event.pos()).normalized()
            self._rubber_band.setGeometry(rb_rect)
            # Live-update scene selection to match rubber-band area
            from bin.nodes.training_node_items import TrainingNodeItem
            scene_rect = self.mapToScene(rb_rect).boundingRect()
            add_mode = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            for item in self.scene().items():
                if isinstance(item, TrainingNodeItem):
                    hit = item.sceneBoundingRect().intersects(scene_rect)
                    if hit or not add_mode:
                        item.setSelected(hit or (add_mode and item.isSelected()))
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() == Qt.MouseButton.RightButton:
            self._is_panning = False
            self._pan_start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._is_middle_zooming and event.button() == Qt.MouseButton.MiddleButton:
            self._is_middle_zooming = False
            self._zoom_drag_pos = None
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._rubber_band is not None and event.button() == Qt.MouseButton.LeftButton:
            self._rubber_band.hide()
            self._rubber_band.deleteLater()
            self._rubber_band = None
            self._rb_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            # Guard 1: any proxy widget in the scene holds focus 鈫?user is
            # editing an input inside a node card.  Never delete in that case.
            scene = self.scene()
            if scene is not None:
                from PySide6.QtWidgets import QGraphicsProxyWidget
                if isinstance(scene.focusItem(), QGraphicsProxyWidget):
                    super().keyPressEvent(event)
                    return

            # Guard 2: a text-input widget anywhere in the application has
            # keyboard focus (belt-and-suspenders for unusual focus states).
            focused = QApplication.focusWidget()
            from PySide6.QtWidgets import (
                QAbstractSpinBox, QComboBox, QLineEdit,
                QPlainTextEdit, QTextEdit,
            )
            if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit,
                                    QAbstractSpinBox, QComboBox)):
                super().keyPressEvent(event)
                return

            # Safe to delete the selected nodes.
            if scene is not None:
                scene.delete_selected_nodes()
            event.accept()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event):
        if not self._is_panning and not self._is_middle_zooming:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep float bar inside bounds on resize
        for child in self.children():
            if isinstance(child, TrainingFloatControlBar):
                child.move(child._clamp(child.pos()))
                child.raise_()

    # ------------------------------------------------------------------
    # Drop (palette 鈫?canvas)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            name = event.mimeData().text()
            scene_pos = self.mapToScene(event.position().toPoint())
            self.scene().create_node(name, scene_pos)
            event.acceptProposedAction()
        else:
            event.ignore()


# ---------------------------------------------------------------------------
# A4 鈥?Training Canvas Widget
# ---------------------------------------------------------------------------

class TrainingPalettePanel(QWidget):
    """Sidebar panel that exposes Training Ground config nodes."""

    # Category annotation: maps display_name 鈫?section name.
    # Any node in TrainingGraphScene._TRAINING_NODE_MAP that is NOT listed here
    # automatically falls into _DEFAULT_SECTION, so new nodes appear without
    # touching this file.
    _NODE_CATEGORY: Dict[str, str] = {
        "Robot / MJCF":        "Environment",
        "Scene Config":        "Environment",
        "Physics Config":      "Environment",
        "Rewards":             "Environment",
        "Terminations":        "Environment",
        "Task Config":         "Environment",
        "Domain Rand":         "Environment",
        "Obs & Action":        "Environment",
        "Reference Motion":    "Environment",
        "Init Pose":           "Environment",
        "Env Assembler":       "Assembly",
        "Start Point":         "Learning",
        "Algorithm Config":    "Learning",
        "Train":               "Learning",
        "Eval Config":         "Eval & Export",
        "Export":              "Eval & Export",
        "Vis Check":           "Eval & Export",
    }
    _SECTION_ORDER = ["Environment", "Assembly", "Learning", "Eval & Export"]
    _DEFAULT_SECTION = "Environment"

    @classmethod
    def _build_palette_sections(cls) -> list:
        """
        Derive palette sections from TrainingGraphScene._TRAINING_NODE_MAP.

        Any display name not in _NODE_CATEGORY falls into _DEFAULT_SECTION.
        Preserves insertion order within each section (order of _TRAINING_NODE_MAP).
        """
        buckets: Dict[str, list] = {s: [] for s in cls._SECTION_ORDER}
        for display_name in TrainingGraphScene._TRAINING_NODE_MAP:
            section = cls._NODE_CATEGORY.get(display_name, cls._DEFAULT_SECTION)
            if section not in buckets:
                buckets[section] = []
            buckets[section].append(display_name)
        return [(s, buckets[s]) for s in cls._SECTION_ORDER if buckets.get(s)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingPalettePanel")

        frame = QFrame(self)
        frame.setObjectName("trainingPalette")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(frame)

        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 8, 6, 8)
        vbox.setSpacing(4)

        self._palette_list = TrainingPaletteTree(frame)
        self._palette_list.setObjectName("trainingPaletteList")
        for section_name, node_names in self._build_palette_sections():
            self._palette_list.add_section(section_name, node_names)
        self._palette_list.expandAll()
        QTimer.singleShot(0, self._palette_list.expandAll)
        vbox.addWidget(self._palette_list, 1)

        hint = QLabel("Drag onto canvas\nto add a node")
        hint.setObjectName("trainingPaletteHint")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

    @property
    def tree(self) -> TrainingPaletteTree:
        return self._palette_list

    def apply_theme(self) -> None:
        palette_bg = get_color("training_toolbar_list_bg", "#1f2937")
        palette_text = get_color("training_toolbar_list_text", get_color("text_primary", "#d1d5db"))
        palette_hover = get_color("training_toolbar_item_hover_bg", get_color("hover_bg", "#374151"))
        palette_selected = get_color("training_toolbar_item_selected_bg", get_color("tab_bg_checked", "#4b5563"))
        section_text = get_color("training_palette_section_text", get_color("text_muted", "#6b7280"))
        hint_text = get_color("training_palette_hint_text", get_color("text_muted", "#6b7280"))

        frame = self.findChild(QFrame, "trainingPalette")
        if frame is not None:
            frame.setStyleSheet(f"#trainingPalette {{ background: {palette_bg}; border: none; }}")
        self._palette_list.setStyleSheet(
            f"QTreeWidget {{ background: transparent; border: none; color: {palette_text}; font-size: 12px; }}"
            f"QTreeWidget::item {{ padding: 5px 6px; border-radius: 4px; margin: 1px 0px; }}"
            f"QTreeWidget::item:hover {{ background: {palette_hover}; }}"
            f"QTreeWidget::item:selected {{ background: {palette_selected}; }}"
        )
        for i in range(self._palette_list.topLevelItemCount()):
            grp = self._palette_list.topLevelItem(i)
            if grp is not None:
                grp.setForeground(0, QColor(section_text))
                self._palette_list.expandItem(grp)
        hint = self.findChild(QLabel, "trainingPaletteHint")
        if hint is not None:
            hint.setStyleSheet(
                f"QLabel {{ font-size: 10px; color: {hint_text}; background: transparent; border: none; padding: 4px 2px 0px 2px; }}"
            )


class TrainingCanvasWidget(QWidget):
    """
    Training-only canvas view area.

    Phase A4: parameter edits persist in memory only (no workspace.json).
    """

    # Forwarded from TrainingGraphScene.train_requested (Step 5)
    train_requested = Signal(dict)
    review_requested = Signal(dict)
    export_review_requested = Signal(dict)
    scene_preview_requested = Signal(dict)
    init_pose_preview_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingCanvasWidget")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scene = TrainingGraphScene(self)
        self._scene.train_requested.connect(self.train_requested)
        self._scene.review_requested.connect(self.review_requested)
        self._scene.export_review_requested.connect(self.export_review_requested)
        self._scene.scene_preview_requested.connect(self.scene_preview_requested)
        self._scene.init_pose_preview_requested.connect(self.init_pose_preview_requested)
        self._view = TrainingCanvasView(self._scene, self)
        self._view.setObjectName("trainingCanvasView")
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setBackgroundBrush(QColor(get_color("training_canva_bg", get_color("canvas_bg", "#1e1e1e"))))
        # NoDrag so left-click selection is handled by the scene; right-click
        # pan and middle-click zoom are managed by TrainingCanvasView directly.
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)

        layout.addWidget(self._view, 1)

        self._stats_overlay = TrainingStatsPopup(None, self._view)

        # Floating control bar — child of the view so it overlays the canvas
        self._float_bar = TrainingFloatControlBar(self._view)
        self._stats_overlay._anchor_bar = self._float_bar
        self._float_bar.stats_clicked.connect(self._toggle_stats_overlay)
        self._float_bar.panel_clicked.connect(self._toggle_overview_panel)
        self._float_bar.show()

        # Overview panel — child of the view, hidden by default
        from bin.components.overview_panel import ControlPanel
        from bin.components.training_overview_content import TrainingOverviewContent
        self._overview_panel = ControlPanel(self._view)
        self._overview_content = TrainingOverviewContent()
        self._overview_panel.set_content(self._overview_content)
        self._overview_panel.set_title("Training Overview")
        self._overview_panel.collapse_requested.connect(self._toggle_overview_panel)
        self._overview_panel.hide()

        # RobotPanel is initialized later by TrainingWorkspaceWindow
        # when the robot type is known from the scene's RobotMJCFNode.

        self.apply_theme()

    @property
    def scene(self) -> TrainingGraphScene:
        return self._scene

    @property
    def float_bar(self) -> "TrainingFloatControlBar":
        return self._float_bar

    def _toggle_stats_overlay(self) -> None:
        self._stats_overlay.toggle_for_button(self._float_bar._stats_btn)
        self._float_bar.raise_()

    def _toggle_overview_panel(self) -> None:
        if self._overview_panel.isVisible():
            self._overview_panel.hide()
        else:
            self._overview_panel.place_center_bottom()
            self._overview_panel.show()
            self._overview_panel.raise_()
        self._float_bar.raise_()

    def apply_theme(self) -> None:
        """Apply theme-driven scene colors from ui.ini."""
        self._view.setBackgroundBrush(QColor(get_color("training_canva_bg", get_color("canvas_bg", "#1e1e1e"))))
        self._scene.refresh_style()
        if hasattr(self, "_float_bar"):
            self._float_bar.apply_theme()
        if hasattr(self, "_stats_overlay"):
            self._stats_overlay.apply_theme()
            self._stats_overlay.sync_to_host()
        if hasattr(self, "_overview_panel"):
            self._overview_panel.apply_theme()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_stats_overlay"):
            self._stats_overlay.sync_to_host()
        if hasattr(self, "_float_bar"):
            self._float_bar.raise_()


# ---------------------------------------------------------------------------
# A3 鈥?Training Ground Window
# ---------------------------------------------------------------------------

class TrainingWorkspaceWindow(QWidget):
    """
    Independent window for a single checkpoint's training workspace.

    Opened from:
    - CheckpointNode 鈫?"Open Training Ground" button

    Bound to ``policy_id`` (the checkpoint asset identity), NOT to a specific
    Canvas node instance.  The same policy_id always reuses the same window.
    """

    # Emitted after start_training() creates the thread (carries policy_id_out)
    train_started = Signal(str)
    # Emitted when training finishes and the bundle is exported (carries bundle_path str)
    checkpoint_exported = Signal(str)
    mission_control_requested = Signal()
    exit_requested = Signal()
    restore_failed = Signal(str)

    def __init__(
        self,
        policy_id: str,
        initial_robot_type: str = "",
        initial_runtime_scenario: Optional[dict] = None,
        parent=None,
        embedded: bool = False,
    ):
        super().__init__(parent)
        self._embedded = embedded
        if not self._embedded:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self._policy_id = str(policy_id or "").strip()
        self._selected_policy_id = self._policy_id
        self._workspace_policy_id = self._policy_id
        self._source_experiment_id = ""
        self._source_info = self._load_checkpoint_source_info(self._selected_policy_id)
        parent_policy_id = str(self._source_info.get("parent_policy_id", "") or "").strip()
        source_experiment_id = str(self._source_info.get("experiment_id", "") or "").strip()
        if parent_policy_id:
            self._workspace_policy_id = parent_policy_id
        if source_experiment_id:
            self._source_experiment_id = source_experiment_id
        self._initial_robot_type = str(initial_robot_type or "")
        self._initial_runtime_scenario = dict(initial_runtime_scenario or {})
        self._canvas_source_state: str = "default_template"
        self._active_thread = None
        self._asset_download_thread = None
        self._active_run_id: str = ""
        self._selected_training_asset_id: str = ""
        self._ws_store = None
        self._current_experiment_id: str = ""
        self._last_export_bundle_path: str = ""
        self._startup_restore_error_message: str = ""
        # Single shared registry instance 鈥?call .refresh() before re-discover
        from src.system.training.training_asset_registry import TrainingAssetRegistry
        self._asset_registry = TrainingAssetRegistry()
        self.setWindowTitle(f"Training Ground 鈥?{policy_id}")
        self.setMinimumSize(1100, 700)
        self.setObjectName("trainingWorkspaceWindow")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # _header_widget is always built (nav labels live here); in embedded mode
        # the parent (MainWindow) takes ownership of it and places it in the shared
        # navigation header row — we do NOT add it to our own root layout.
        self._header_widget = self._build_header()
        if not embedded:
            root.addWidget(self._header_widget)

        body_or_content = self._build_body()  # always sets self._workspace_content_splitter
        root.addWidget(body_or_content, 1)
        self.apply_theme()
        self._set_training_state("idle")

        # Apply robot_type from main canvas before loading saved experiments.
        # _init_workspace() may load a saved experiment which overrides this value,
        # so fresh workspaces inherit the Mission Canvas selection automatically.
        if self._initial_robot_type:
            self._canvas.scene.set_initial_robot_type(self._initial_robot_type)

        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self._save_current_experiment)

        self._init_workspace()

    # ------------------------------------------------------------------
    # Header bar
    # ------------------------------------------------------------------

    def _build_header(self) -> QWidget:
        bar = QWidget()
        if self._embedded:
            bar.setObjectName("trainingWorkspaceHeaderContent")
        else:
            bar.setObjectName("trainingWorkspaceHeader")
            bar.setFixedHeight(48)
        hbox = QHBoxLayout(bar)
        if self._embedded:
            hbox.setContentsMargins(0, 0, 0, 0)
        else:
            hbox.setContentsMargins(16, 0, 16, 0)
        hbox.setSpacing(12)

        hbox.addStretch(1)
        self._window_controls = WindowControlButtons(theme="dark", parent=bar)
        self._window_controls.minimize_requested.connect(self._on_minimize_requested)
        self._window_controls.fullscreen_requested.connect(self._toggle_window_fullscreen)
        self._window_controls.close_requested.connect(self._on_exit_clicked)
        hbox.addWidget(self._window_controls)
        return bar


    # ------------------------------------------------------------------
    # Body
    # ------------------------------------------------------------------

    def _build_body(self) -> QWidget:
        """Build the body area.

        Always creates all internal widgets (canvas, sidebar panels, splitter)
        as attributes so they are accessible regardless of mode.

        Returns ``_workspace_content_splitter`` when *embedded* (the parent
        MainWindow owns the sidebar rail and nav header), or a full body widget
        containing the sidebar + workspace area when running standalone.
        """
        self._canvas = TrainingCanvasWidget()
        self._canvas.train_requested.connect(self._on_train_requested)
        self._canvas.review_requested.connect(self._on_review_requested)
        self._canvas.export_review_requested.connect(self._on_export_review_requested)
        self._canvas.scene_preview_requested.connect(self._on_scene_preview_requested)
        self._canvas.init_pose_preview_requested.connect(self._on_init_pose_preview_requested)
        self._canvas.float_bar.start_clicked.connect(self._on_ctrl_start_clicked)
        self._canvas.float_bar.stop_clicked.connect(self._on_ctrl_stop_clicked)
        self._canvas.scene.node_param_changed.connect(self._on_canvas_node_param_changed)

        # Sidebar panels — always created so the parent can use them in either mode.
        self._canvas_browser_panel = TrainingCanvasBrowserPanel(None)
        self._canvas_browser_panel.canvas_requested.connect(self._on_canvas_requested)
        self._canvas_browser_panel.delete_requested.connect(self._delete_experiment_by_id)
        self._canvas_browser_panel.new_workspace_requested.connect(self._new_workspace)
        self._canvas_browser_panel.new_experiment_requested.connect(self._new_experiment_in_workspace)
        self._canvas_browser_panel.rename_workspace_requested.connect(self._rename_workspace_by_id)
        self._canvas_browser_panel.delete_workspace_requested.connect(self._delete_workspace_by_id)
        self._export_browser_panel = TrainingExportBrowserPanel(None)
        self._export_browser_panel.delete_requested.connect(self._delete_export_by_id)
        self._palette_panel = TrainingPalettePanel(None)

        # Own sidebar — only used in standalone (non-embedded) mode.
        # In embedded mode, the parent MainWindow provides the sidebar rail.
        self._sidebar = SidebarDock(
            nav_items=[
                ("canvas", "Experiments", "prj"),
                ("exports", "Exports", "cp"),
                ("nodes", "Config Nodes", "nod"),
            ],
            panel_width=320,
        )
        self._sidebar.theme_button.hide()
        self._sidebar.language_button.hide()
        self._sidebar.set_panel_widget("canvas", self._canvas_browser_panel, "Experiments")
        self._sidebar.set_panel_widget("exports", self._export_browser_panel, "Exports")
        self._sidebar.set_panel_widget("nodes", self._palette_panel, "Config Nodes")
        self._sidebar.panel_changed.connect(self._on_sidebar_panel_changed)

        self._workspace_left_box = QWidget()
        self._workspace_left_box.setObjectName("trainingWorkspaceLeftBox")
        workspace_left_layout = QVBoxLayout(self._workspace_left_box)
        workspace_left_layout.setContentsMargins(0, 0, 0, 0)
        workspace_left_layout.setSpacing(0)

        # Toolbar row is the primary workspace control surface.
        workspace_left_layout.addWidget(self._build_toolbar_row())

        # Progress strip sits directly under the toolbar in the left content column.
        self._progress_strip = self._build_progress_strip()
        workspace_left_layout.addWidget(self._progress_strip)

        workspace_left_layout.addWidget(self._canvas, 1)

        from bin.pages.training.training_panel import TrainingPanel

        self._training_panel = TrainingPanel()
        self._training_panel.setObjectName("trainingLogPanel")
        self._training_panel.setMinimumWidth(320)

        if self._embedded:
            self._workspace_content_splitter = self._workspace_left_box
            return self._workspace_left_box

        self._workspace_body = QWidget()
        self._workspace_body.setObjectName("trainingWorkspaceBody")
        workspace_body_layout = QHBoxLayout(self._workspace_body)
        workspace_body_layout.setContentsMargins(0, 0, 0, 0)
        workspace_body_layout.setSpacing(0)

        self._workspace_content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._workspace_content_splitter.setObjectName("trainingWorkspaceSplitter")
        self._workspace_content_splitter.addWidget(self._workspace_left_box)
        self._workspace_content_splitter.addWidget(self._training_panel)
        self._workspace_content_splitter.setSizes([980, 360])
        self._workspace_content_splitter.setStretchFactor(0, 1)
        self._workspace_content_splitter.setStretchFactor(1, 0)
        workspace_body_layout.addWidget(self._workspace_content_splitter, 1)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._sidebar)
        body_layout.addWidget(self._workspace_body, 1)
        return body

    def _extract_robot_type_from_canvas_data(self, canvas_data: Optional[dict]) -> str:
        if not isinstance(canvas_data, dict):
            return ""
        for node_data in canvas_data.get("nodes", []) or []:
            params = node_data.get("parameters", {}) or {}
            robot_type = str(params.get("robot_type", "")).strip()
            if robot_type:
                return robot_type
        return ""

    def _current_experiment_name(self) -> str:
        if self._current_experiment_id and self._ws_store is not None:
            try:
                meta = self._ws_store.load_workspace(self._workspace_policy_id)
                exp_meta = meta.get_experiment(self._current_experiment_id)
                if exp_meta is not None and exp_meta.name:
                    return exp_meta.name
                return self._current_experiment_id
            except Exception:
                return self._current_experiment_id
        if self._canvas_source_state == "default_template":
            return "Default Template"
        return "Unsaved"

    def _current_workspace_name(self) -> str:
        return str(self._workspace_policy_id or self._selected_policy_id or "WorkSpace").strip() or "WorkSpace"

    def _on_mission_control_clicked(self) -> None:
        self.mission_control_requested.emit()

    def _on_exit_clicked(self) -> None:
        self.exit_requested.emit()

    def _on_minimize_requested(self) -> None:
        window = self.window()
        if window is not None:
            window.showMinimized()

    def _toggle_window_fullscreen(self) -> None:
        window = self.window()
        if window is None:
            return
        if window.isFullScreen():
            window.showNormal()
        else:
            window.showFullScreen()
        self._sync_window_controls()

    def _sync_window_controls(self) -> None:
        controls = getattr(self, "_window_controls", None)
        if controls is None:
            return
        window = self.window()
        controls.set_fullscreen(bool(window is not None and window.isFullScreen()))

    def _get_canvas_export_name(self) -> str:
        item = self._find_training_node_item("export")
        if item is None:
            return "<NEW>"
        params = item.get_parameters()
        return str(params.get("bundle_name", "") or "").strip() or "<NEW>"

    def _load_checkpoint_source_info(self, policy_id: str = "") -> dict:
        try:
            resolved_policy_id = str(policy_id or self._selected_policy_id or self._policy_id or "").strip()
            if not resolved_policy_id:
                return {}
            source_path = get_project_root() / "custom_mods/training/checkpoints" / resolved_policy_id / "source.json"
            if not source_path.exists():
                return {}
            with open(source_path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _bind_canvas_to_workspace_policy(self) -> None:
        export_item = self._find_training_node_item("export")
        if export_item is not None:
            export_params = export_item.get_parameters()
            export_name = str(export_params.get("bundle_name", "") or "").strip()
            if not export_name or export_name == "<NEW>":
                export_item.load_parameters({"bundle_name": self._selected_policy_id})
        algo_item = self._find_training_node_item("algo_config")
        if algo_item is not None:
            algo_params = algo_item.get_parameters()
            policy_id_out = str(algo_params.get("policy_id_out", "") or "").strip()
            if not policy_id_out:
                algo_item.load_parameters({"policy_id_out": self._selected_policy_id})
        self._refresh_current_canvas_label()

    def _on_canvas_node_param_changed(self, node_type: str, key: str, value: str) -> None:
        """Refresh the Export label whenever the export bundle_name or algo policy_id_out changes."""
        if node_type in ("export", "algo_config"):
            self._refresh_current_canvas_label()

    def _refresh_current_canvas_label(self) -> None:
        if hasattr(self, "_title_label") and self._title_label is not None:
            self._title_label.setText(
                f"[{self._current_workspace_name()}]: {self._current_experiment_name()}"
            )
        if hasattr(self, "_export_label") and self._export_label is not None:
            self._export_label.setText(f"Export: {self._get_canvas_export_name()}")
        self._sync_selected_training_asset_from_canvas()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_window_controls()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_window_controls()

    def _refresh_canvas_browser(self, meta=None) -> None:
        if self._ws_store is None or not hasattr(self, "_canvas_browser_panel"):
            return
        if meta is None and self._workspace_policy_id:
            try:
                meta = self._ws_store.load_workspace(self._workspace_policy_id)
            except Exception:
                self._canvas_browser_panel.populate([], self._current_experiment_id)
                self._refresh_current_canvas_label()
                return

        entries = []
        try:
            all_ws_ids = self._ws_store.list_workspaces()
        except Exception:
            all_ws_ids = [self._workspace_policy_id] if self._workspace_policy_id else []

        for ws_id in all_ws_ids:
            try:
                ws_meta = self._ws_store.load_workspace(ws_id)
            except Exception:
                continue

            if not ws_meta.experiments:
                entries.append({
                    "workspace_id": ws_id,
                    "workspace_name": ws_id,
                    "experiment_id": "",
                    "leaf_label": "",
                    "tooltip": f"Workspace: {ws_id}",
                })
                continue

            for exp_meta in ws_meta.experiments:
                try:
                    canvas_data = self._ws_store.load_experiment(ws_id, exp_meta.experiment_id)
                except Exception:
                    canvas_data = {}
                robot_type = self._extract_robot_type_from_canvas_data(canvas_data) or self._initial_robot_type or "unknown"
                family = resolve_robot_family(robot_type)
                file_name = f"{exp_meta.experiment_id}.canvas.json"
                leaf_label = exp_meta.name or exp_meta.experiment_id
                tooltip = "\n".join([
                    f"Workspace: {ws_id}",
                    f"Robot family: {family}",
                    f"Robot type: {robot_type}",
                    f"Canvas name: {exp_meta.name or exp_meta.experiment_id}",
                    f"File: {file_name}",
                ])
                entries.append({
                    "workspace_id": ws_id,
                    "workspace_name": ws_id,
                    "experiment_id": exp_meta.experiment_id,
                    "leaf_label": leaf_label,
                    "tooltip": tooltip,
                })

        self._canvas_browser_panel.populate(entries, self._current_experiment_id)
        self._refresh_current_canvas_label()

    def _refresh_export_browser(self) -> None:
        panel = getattr(self, "_export_browser_panel", None)
        if panel is None:
            return

        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry
            entries = []
            source_labels = {
                "training": "Training",
                "huggingface": "Hugging Face",
                "local": "Local",
            }
            for checkpoint in CheckpointRegistry().discover():
                if not getattr(checkpoint, "is_valid", True):
                    continue
                category = source_labels.get(
                    str(getattr(checkpoint, "source_type", "") or "").strip(),
                    "Other",
                )
                robot_type = (
                    str(getattr(checkpoint, "robot_model", "") or "").strip()
                    or str(getattr(checkpoint, "robot_brand", "") or "").strip()
                    or "unknown"
                )
                policy_id = str(getattr(checkpoint, "policy_id", "") or "").strip()
                display_name = (
                    str(getattr(checkpoint, "display_name", "") or "").strip()
                    or policy_id
                    or "export"
                )
                tooltip_lines = [
                    f"Category: {category}",
                    f"Robot: {robot_type}",
                    f"Export: {display_name}",
                    f"Policy ID: {policy_id or '-'}",
                    f"Path: {getattr(checkpoint, 'bundle_path', '')}",
                ]
                version = str(getattr(checkpoint, "version", "") or "").strip()
                if version:
                    tooltip_lines.insert(3, f"Version: {version}")
                entries.append(
                    {
                        "category": category,
                        "robot_type": robot_type,
                        "policy_id": policy_id,
                        "leaf_label": display_name,
                        "tooltip": "\n".join(tooltip_lines),
                    }
                )
        except Exception:
            entries = []

        panel.populate(entries, self._get_canvas_export_name())

    def warm_cache(self) -> None:
        """Preload hidden UI state so the first shell switch feels instant."""
        try:
            self.apply_theme()
        except Exception:
            pass

        try:
            self._asset_registry.refresh()
        except Exception:
            pass

        try:
            if self._ws_store is not None and self._workspace_policy_id:
                meta = self._ws_store.load_workspace(self._workspace_policy_id)
                self._populate_lists(meta)
            else:
                self._populate_training_assets()
                self._refresh_canvas_browser(None)
                self._refresh_export_browser()
        except Exception:
            pass

        try:
            self._refresh_current_canvas_label()
        except Exception:
            pass

    def _on_sidebar_panel_changed(self, panel_key: str) -> None:
        if panel_key == "nodes" and hasattr(self, "_palette_panel") and self._palette_panel is not None:
            self._palette_panel.tree.expandAll()

    def _prompt_experiment_name(self, default_name: str) -> str:
        name, ok = QInputDialog.getText(
            self,
            "New Experiment",
            "Experiment name:",
            text=default_name,
        )
        if not ok:
            return ""
        return str(name or "").strip() or default_name

    def _delete_experiment_by_id(self, workspace_id: str, experiment_id: str) -> None:
        if self._ws_store is None or not workspace_id or not experiment_id:
            return
        reply = QMessageBox.question(
            self,
            "Delete Experiment",
            "Are you sure you want to delete this experiment?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._ws_store.delete_experiment(workspace_id, experiment_id)
            if self._workspace_policy_id == workspace_id and self._current_experiment_id == experiment_id:
                self._current_experiment_id = ""
                try:
                    meta = self._ws_store.load_workspace(workspace_id)
                    if meta.active_experiment_id:
                        self._workspace_policy_id = workspace_id
                        self._load_experiment_by_id(meta.active_experiment_id)
                    else:
                        self._workspace_policy_id = workspace_id
                        self._ensure_template_canvas()
                        self._populate_lists(meta)
                        self._refresh_canvas_browser(meta)
                except Exception:
                    self._ensure_template_canvas()
            else:
                self._populate_lists()
                self._refresh_canvas_browser()
        except Exception as exc:
            QMessageBox.warning(self, "Delete Experiment Failed", str(exc))

    def _new_workspace(self, _robot_type: str = "") -> None:
        if self._ws_store is None:
            return
        existing = set(self._ws_store.list_workspaces())
        default_name = f"Workspace_{len(existing) + 1}"
        name, ok = QInputDialog.getText(
            self, "New Workspace", "Workspace name:", text=default_name
        )
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return
        if name in existing:
            QMessageBox.warning(self, "Name Conflict", f"Workspace '{name}' already exists.")
            return
        try:
            self._ws_store.ensure_workspace(name)
            self._refresh_canvas_browser()
        except Exception as exc:
            QMessageBox.warning(self, "Create Workspace Failed", str(exc))

    def _rename_workspace_by_id(self, workspace_id: str) -> None:
        if self._ws_store is None or not workspace_id:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Workspace", "New workspace name:", text=workspace_id
        )
        if not ok:
            return
        new_name = str(new_name or "").strip()
        if not new_name or new_name == workspace_id:
            return
        existing = set(self._ws_store.list_workspaces())
        if new_name in existing:
            QMessageBox.warning(self, "Name Conflict", f"Workspace '{new_name}' already exists.")
            return
        try:
            self._ws_store.rename_workspace(workspace_id, new_name)
            if self._workspace_policy_id == workspace_id:
                self._workspace_policy_id = new_name
            self._refresh_canvas_browser()
        except Exception as exc:
            QMessageBox.warning(self, "Rename Workspace Failed", str(exc))

    def _delete_workspace_by_id(self, workspace_id: str) -> None:
        if self._ws_store is None or not workspace_id:
            return
        reply = QMessageBox.question(
            self,
            "Delete Workspace",
            "Delete this workspace and all experiments inside it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self._ws_store.delete_workspace(workspace_id)
            remaining = list(self._ws_store.list_workspaces())

            if self._workspace_policy_id == workspace_id:
                self._current_experiment_id = ""
                if remaining:
                    self._workspace_policy_id = remaining[0]
                    meta = self._ws_store.load_workspace(self._workspace_policy_id)
                    if meta.active_experiment_id:
                        self._load_experiment_by_id(meta.active_experiment_id)
                    else:
                        self._ensure_template_canvas()
                        self._populate_lists(meta)
                        self._refresh_canvas_browser(meta)
                else:
                    self._workspace_policy_id = ""
                    self._ensure_template_canvas()
                    self._populate_lists()
                    self._refresh_canvas_browser()
            else:
                self._populate_lists()
                self._refresh_canvas_browser()
        except Exception as exc:
            QMessageBox.warning(self, "Delete Workspace Failed", str(exc))

    def _new_experiment_in_workspace(self, workspace_id: str) -> None:
        if self._ws_store is None or not workspace_id:
            return
        try:
            meta = self._ws_store.load_workspace(workspace_id)
            default_name = f"Experiment {len(meta.experiments) + 1}"
            name = self._prompt_experiment_name(default_name)
            if not name:
                return
            exp_meta = self._ws_store.create_experiment(workspace_id, name=name)
            self._workspace_policy_id = workspace_id
            self._current_experiment_id = exp_meta.experiment_id
            canvas_data = self._ws_store.load_experiment(workspace_id, exp_meta.experiment_id)
            self._canvas.scene.load_training_graph(canvas_data)
            self._ensure_template_canvas()
            if self._selected_training_asset_id:
                self._load_asset_into_canvas(self._selected_training_asset_id)
            else:
                self._canvas_source_state = "default_template"
            self._bind_canvas_to_workspace_policy()
            self._populate_lists()
            self._refresh_canvas_browser()
        except Exception as exc:
            log_error(f"New experiment creation failed for workspace='{workspace_id}': {exc}")

    def _on_canvas_requested(self, workspace_id: str, experiment_id: str) -> None:
        if not workspace_id or not experiment_id:
            return
        self._workspace_policy_id = workspace_id
        self._refresh_training_assets()
        self._load_experiment_by_id(experiment_id)

    def _delete_export_by_id(self, policy_id: str) -> None:
        if not policy_id:
            return
        reply = QMessageBox.question(
            self,
            "Delete Export",
            "Are you sure you want to delete this export?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry

            CheckpointRegistry().delete(policy_id)
            self._refresh_export_browser()
        except Exception as exc:
            QMessageBox.warning(self, "Delete Export Failed", str(exc))

    def _build_toolbar_row(self) -> QWidget:
        """
        Toolbar with three dropdown buttons.
        This is the current primary workspace header for run history and assets.
        Each button reveals a slim list panel (QMenu + QWidgetAction) on click.
        """
        bar = QWidget()
        bar.setObjectName("trainingToolbarRow")
        bar.setFixedHeight(36)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(8, 0, 8, 0)
        hbox.setSpacing(4)

        self._title_label = QLabel(f"[{self._current_workspace_name()}]: {self._current_experiment_name()}")
        self._title_label.setObjectName("workspaceTitleLabel")
        hbox.addWidget(self._title_label)

        _sep1 = QLabel("|")
        _sep1.setObjectName("workspaceHeaderSep")
        hbox.addWidget(_sep1)

        self._export_label = QLabel("Export: <NEW>")
        self._export_label.setObjectName("workspaceExportLabel")
        hbox.addWidget(self._export_label)

        _sep2 = QLabel("|")
        _sep2.setObjectName("workspaceHeaderSep")
        hbox.addWidget(_sep2)

        def _make_dropdown_btn(label: str, list_attr: str) -> QToolButton:
            btn = QToolButton()
            btn.setText(label + "  v")
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

            menu = QMenu(btn)

            container = QWidget()
            container.setFixedWidth(220)
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(4, 4, 4, 4)
            vbox.setSpacing(0)

            lst = QListWidget()
            lst.setObjectName(f"toolbar_{list_attr}")
            lst.setFixedHeight(130)
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._populate_toolbar_list(lst, [], f"No {label.lower()} yet")
            setattr(self, list_attr, lst)
            vbox.addWidget(lst)

            wa = QWidgetAction(menu)
            wa.setDefaultWidget(container)
            menu.addAction(wa)
            btn.setMenu(menu)
            return btn

        self._exp_list = QListWidget()
        self._exp_list.hide()
        self._run_list: QListWidget
        self._asset_list: QListWidget

        hbox.addWidget(_make_dropdown_btn("Run History", "_run_list"))
        hbox.addWidget(self._make_assets_dropdown_btn())
        self._asset_list.itemClicked.connect(self._on_training_asset_item_clicked)

        self._btn_save_exp = QPushButton("Save")
        self._btn_save_exp.setObjectName("trainingBtnSaveExp")
        self._btn_save_exp.setFixedHeight(26)
        self._btn_save_exp.setToolTip("Save current canvas as experiment")
        self._btn_save_exp.clicked.connect(self._save_current_experiment)
        hbox.addWidget(self._btn_save_exp)

        self._save_status_label = QLabel("")
        self._save_status_label.setObjectName("trainingSaveStatusLabel")
        self._save_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        hbox.addWidget(self._save_status_label)

        hbox.addStretch(1)
        return bar

    def _make_assets_dropdown_btn(self) -> "QToolButton":
        """
        Training Assets dropdown button with list + Refresh footer.
        Builds a custom panel (wider than the generic helper) so the Refresh
        button is always visible at the bottom of the panel.
        """
        btn = QToolButton()
        self._assets_dropdown_btn = btn
        self._update_assets_dropdown_label("")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        menu = QMenu(btn)

        container = QWidget()
        container.setFixedWidth(260)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        lst = QListWidget()
        lst.setObjectName("toolbar__asset_list")
        lst.setFixedHeight(140)
        lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._asset_list = lst
        vbox.addWidget(lst)

        actions_row = QWidget()
        actions_row.setObjectName("trainingAssetsActionsRow")
        actions_layout = QHBoxLayout(actions_row)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)

        self._btn_import_asset = QPushButton("Import Local")
        self._btn_import_asset.setObjectName("trainingBtnImportAsset")
        self._btn_import_asset.setFixedHeight(24)
        self._btn_import_asset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn_import_asset.setToolTip("Import a local SB3 training package into Training Ground")
        self._btn_import_asset.clicked.connect(self._import_training_asset_local)
        actions_layout.addWidget(self._btn_import_asset, 1)

        self._btn_download_asset = QPushButton("Download HF")
        self._btn_download_asset.setObjectName("trainingBtnDownloadAsset")
        self._btn_download_asset.setFixedHeight(24)
        self._btn_download_asset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn_download_asset.setToolTip("Download a community training package from Hugging Face")
        self._btn_download_asset.clicked.connect(self._download_training_asset_hf)
        actions_layout.addWidget(self._btn_download_asset, 1)

        vbox.addWidget(actions_row)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("trainingAssetsRefreshBtn")
        refresh_btn.setFixedHeight(24)
        refresh_btn.setToolTip("Refresh training assets list from disk")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_training_assets)
        vbox.addWidget(refresh_btn)
        self._btn_refresh_assets = refresh_btn

        wa = QWidgetAction(menu)
        wa.setDefaultWidget(container)
        menu.addAction(wa)
        btn.setMenu(menu)
        return btn

    def _refresh_training_assets(self) -> None:
        """Refresh the registry from disk then repopulate the assets list."""
        self._asset_registry.refresh()
        self._populate_training_assets()

    def _update_assets_dropdown_label(self, label: str = "") -> None:
        btn = getattr(self, "_assets_dropdown_btn", None)
        if btn is None:
            return
        display_label = str(label or "").strip() or "<New Asset>"
        btn.setText(f"{display_label}  v")

    def _find_training_node_item(self, node_type: str):
        from bin.nodes.training_node_items import TrainingNodeItem

        if not hasattr(self, "_canvas") or self._canvas is None:
            return None
        for canvas_item in self._canvas.scene.items():
            if isinstance(canvas_item, TrainingNodeItem) and canvas_item._node_type == node_type:
                return canvas_item
        return None

    def _canvas_has_training_nodes(self) -> bool:
        return self._find_training_node_item("robot_mjcf") is not None

    def _ensure_template_canvas(self) -> None:
        """
        Recreate the default template when the current scene is blank.

        This keeps the "new / blank experiment" path usable while avoiding
        overwriting non-empty saved experiments.
        """
        if self._canvas_has_training_nodes():
            return
        self._canvas.scene.reset_to_default_template()
        if self._initial_robot_type:
            self._canvas.scene.set_initial_robot_type(self._initial_robot_type)
        if self._initial_runtime_scenario:
            self._canvas.scene.set_initial_runtime_scenario(self._initial_runtime_scenario)
        self._apply_resolved_task_template()
        self._canvas_source_state = "default_template"
        self._refresh_current_canvas_label()

    @staticmethod
    def _json_param_dict(raw_value) -> Dict[str, float]:
        if isinstance(raw_value, dict):
            return dict(raw_value)
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, dict):
                    return dict(parsed)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _json_param_dump(data: Dict[str, float]) -> str:
        return json.dumps(data, ensure_ascii=False, sort_keys=True)

    def _get_canvas_robot_type(self) -> str:
        item = self._find_training_node_item("robot_mjcf")
        if item is None:
            return str(self._initial_robot_type or "").strip()
        params = item.get_parameters()
        robot_type = str(params.get("robot_type", "") or "").strip()
        return robot_type or str(self._initial_robot_type or "").strip()

    def _get_canvas_reward_terms(self) -> Dict[str, float]:
        item = self._find_training_node_item("rewards")
        if item is None:
            return {}
        return self._json_param_dict(item.get_parameters().get("reward_terms", "{}"))

    def _get_canvas_termination_conditions(self) -> Dict[str, float]:
        item = self._find_training_node_item("terminations")
        if item is None:
            return {}
        return self._json_param_dict(
            item.get_parameters().get("termination_conditions", "{}")
        )

    def _apply_resolved_task_template(
        self,
        *,
        robot_type: str = "",
        asset_id: str = "",
    ) -> None:
        resolved = resolve_task_template(
            robot_type=robot_type or self._get_canvas_robot_type(),
            asset_id=asset_id or None,
        )
        rewards_item = self._find_training_node_item("rewards")
        terms_item = self._find_training_node_item("terminations")
        if rewards_item is not None:
            rewards_item.load_parameters(
                {
                    "reward_terms": self._json_param_dump(
                        resolved.get("reward_terms", {})
                    )
                }
            )
        if terms_item is not None:
            terms_item.load_parameters(
                {
                    "termination_conditions": self._json_param_dump(
                        resolved.get("termination_conditions", {})
                    )
                }
            )

    def _canvas_matches_resolved_task_template(
        self,
        *,
        robot_type: str = "",
        asset_id: str = "",
    ) -> bool:
        resolved = resolve_task_template(
            robot_type=robot_type or self._get_canvas_robot_type(),
            asset_id=asset_id or None,
        )
        return (
            self._get_canvas_reward_terms() == resolved.get("reward_terms", {})
            and self._get_canvas_termination_conditions()
            == resolved.get("termination_conditions", {})
        )

    def _canvas_uses_current_template_defaults(self) -> bool:
        return self._canvas_matches_resolved_task_template(
            robot_type=self._get_canvas_robot_type(),
            asset_id=self._get_canvas_base_asset_id(),
        )

    def _get_canvas_base_asset_id(self) -> str:
        item = self._find_training_node_item("base_asset")
        if item is None:
            return ""
        params = item.get_parameters()
        start_point = str(params.get("start_point", "") or "").strip()
        if start_point.startswith("asset:"):
            return start_point.split(":", 1)[1].strip()
        return str(params.get("asset_id", "") or "").strip()

    def _sync_selected_training_asset_from_canvas(self) -> None:
        """Make the header/list selection reflect the BaseAssetNode on canvas."""
        asset_id = self._get_canvas_base_asset_id()
        if not asset_id:
            self._set_selected_training_asset("")
            return
        try:
            entry = self._asset_registry.get(asset_id)
            self._set_selected_training_asset(entry.asset_id, entry.label())
        except Exception:
            self._set_selected_training_asset(asset_id)

    @staticmethod
    def _match_contract_preset(entry) -> str:
        from src.system.training.obs_contracts import get_obs_contract, list_preset_names

        for preset_name in list_preset_names():
            if preset_name == "custom":
                continue
            contract = get_obs_contract(preset_name)
            if contract is None:
                continue
            if (
                int(contract.get("obs_dim", 0) or 0) == int(getattr(entry, "obs_dim", 0) or 0)
                and int(contract.get("action_dim", 0) or 0) == int(getattr(entry, "action_dim", 0) or 0)
                and str(contract.get("action_type", "") or "") == str(getattr(entry, "action_type", "") or "")
                and str(contract.get("robot_type", "") or "") == str(getattr(entry, "robot_type", "") or "")
            ):
                return preset_name
        return "custom"

    def _asset_node_params(self, entry) -> Dict[str, dict]:
        load_mode = "resume_sb3" if entry.framework == "sb3" else "warm_start_actor"
        preset_name = self._match_contract_preset(entry)

        params: Dict[str, dict] = {
            "base_asset": {
                "start_point":     f"asset:{entry.asset_id}",
                "asset_id":        entry.asset_id,
                "checkpoint_file": entry.primary_checkpoint,
                "load_mode":       load_mode,
            },
        }

        if getattr(entry, "robot_type", ""):
            params["robot_mjcf"] = {"robot_type": entry.robot_type}

        if getattr(entry, "algorithm", ""):
            params["algo_config"] = {"algorithm": entry.algorithm}

        if getattr(entry, "action_type", ""):
            params["physics_config"] = {"action_type": entry.action_type}

        obs_params = {"contract_preset": preset_name}
        if preset_name == "custom" and getattr(entry, "action_type", ""):
            obs_params["action_type"] = entry.action_type
        params["obs_action_config"] = obs_params
        return params

    @staticmethod
    def _populate_toolbar_list(lst: QListWidget, items: list, placeholder: str) -> None:
        lst.clear()
        if items:
            for it in items:
                lst.addItem(str(it))
        else:
            ph = QListWidgetItem(placeholder)
            ph.setFlags(ph.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            ph.setForeground(Qt.GlobalColor.darkGray)
            lst.addItem(ph)

    def _build_progress_strip(self) -> QWidget:
        """
        Slim progress strip 鈥?always visible between the toolbar row and canvas.
        Controls (Start / Pause / Stop) live on the canvas float bar.

        Layout:  鈻堚枅鈻堚枅鈻戔枒鈻戔枒  0%  Step: 鈥? |  Idle
        """
        bar = QWidget()
        bar.setObjectName("trainingProgressStrip")
        bar.setFixedHeight(36)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(12, 4, 12, 4)
        hbox.setSpacing(6)

        # 鈹€鈹€ Progress bar 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self._strip_bar = QProgressBar()
        self._strip_bar.setObjectName("trainingStripBar")
        self._strip_bar.setRange(0, 100)
        self._strip_bar.setValue(0)
        self._strip_bar.setTextVisible(False)
        self._strip_bar.setFixedHeight(10)
        hbox.addWidget(self._strip_bar, 1)

        # 鈹€鈹€ Step / % label 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self._strip_label = QLabel("Step: -")
        self._strip_label.setObjectName("trainingStripLabel")
        hbox.addWidget(self._strip_label)

        # 鈹€鈹€ State label 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self._strip_state = QLabel("Idle")
        self._strip_state.setObjectName("trainingStripState")
        self._strip_state.setFixedWidth(90)
        self._strip_state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hbox.addWidget(self._strip_state)

        return bar

    def _set_training_state(self, state: str) -> None:
        """
        Switch all control surfaces to one of three states:
        "idle" | "running" | "done"

        Updates the top progress strip labels and the canvas float bar.
        """
        if state == "running":
            self._strip_state.setText("Training")
        elif state == "done":
            self._strip_state.setText("Done")
        else:  # idle
            self._strip_bar.setValue(0)
            self._strip_label.setText("Step: -")
            self._strip_state.setText("Idle")
        # Canvas float bar
        if hasattr(self, "_canvas") and hasattr(self._canvas, "float_bar"):
            self._canvas.float_bar.set_state(state)

    def _on_ctrl_start_clicked(self) -> None:
        """鈻?Start button: compile spec and trigger training."""
        self._on_train_requested({})

    def _on_ctrl_stop_clicked(self) -> None:
        """鈴?Stop button: cancel the active thread via TrainingPanel signal."""
        if hasattr(self, "_training_panel"):
            self._training_panel.cancel_requested.emit()

    def _on_progress_strip(self, step: int, total: int,
                           reward_mean: float, best_reward: float,
                           ep_len_mean: float = 0.0,
                           status: str = "") -> None:
        """Update the control bar progress display."""
        pct = int(step / total * 100) if total > 0 else 0
        self._strip_bar.setValue(pct)
        self._strip_label.setText(f"{pct}%  Step {step:,}/{total:,}")

    def _on_metrics_update(self, metrics: dict) -> None:
        """Forward 4-layer metrics to the stats overlay live metrics panel + chart."""
        overlay = getattr(getattr(self, "_canvas", None), "_stats_overlay", None)
        if overlay is not None:
            overlay.update_live_metrics(metrics)
            # Ensure the overlay timer is running when training is active,
            # even if the user didn't manually open it (the auto-show at
            # training start handles the initial case, but this covers
            # edge cases like the user closing and re-opening mid-run).
            if overlay.isVisible() and hasattr(overlay, "_timer"):
                if not overlay._timer.isActive():
                    overlay._timer.start()

    def apply_theme(self) -> None:
        """Apply theme-driven styles from ui.ini to the full workspace window."""
        header_bg = get_color("training_workspace_header_bg", "#1a1a2e")
        header_border = get_color("training_workspace_header_border", "#2d2d4e")
        title_text = get_color("training_workspace_title_text", get_color("text_primary", "#e5e7eb"))
        sep_text = get_color("training_workspace_header_sep_text", get_color("text_muted", "#4b5563"))
        base_text = get_color("training_workspace_base_text", "#d1d5db")
        asset_text = get_color("training_workspace_asset_text", "#60a5fa")
        experiment_text = get_color("training_workspace_experiment_text", "#d1d5db")
        export_text = get_color("training_workspace_export_text", "#86efac")
        badge_text = get_color("training_workspace_badge_text", "#6ee7b7")
        badge_bg = get_color("training_workspace_badge_bg", "#064e3b")
        badge_border = get_color("training_workspace_badge_border", "#065f46")
        sidebar_rail_bg = get_color("sidebar_rail_bg", "#111827")
        sidebar_panel_bg = get_color("sidebar_panel_bg", "#0f172a")
        sidebar_right_border = get_color("sidebar_right_border", get_color("border", "#374151"))
        sidebar_button_hover_bg = get_color("sidebar_button_hover_bg", "#1f2937")
        sidebar_button_checked_bg = get_color("sidebar_button_checked_bg", "#1f2937")
        sidebar_title_text = get_color("text_primary", "#e5e7eb")
        toolbar_bg = get_color("training_toolbar_bg", "#1f2937")
        toolbar_border = get_color("training_toolbar_border", "#374151")
        toolbar_btn_text = get_color("training_toolbar_button_text", get_color("text_primary", "#d1d5db"))
        toolbar_btn_hover = get_color("training_toolbar_button_hover_bg", get_color("hover_bg", "#374151"))
        toolbar_btn_pressed = get_color("training_toolbar_button_pressed_bg", get_color("tab_bg_checked", "#4b5563"))
        toolbar_list_bg = get_color("training_toolbar_list_bg", "#1f2937")
        toolbar_list_text = get_color("training_toolbar_list_text", get_color("text_primary", "#d1d5db"))
        toolbar_list_hover = get_color("training_toolbar_item_hover_bg", get_color("hover_bg", "#374151"))
        toolbar_list_selected = get_color("training_toolbar_item_selected_bg", get_color("tab_bg_checked", "#4b5563"))
        toolbar_menu_bg = get_color("training_toolbar_menu_bg", "#111827")
        toolbar_menu_border = get_color("training_toolbar_menu_border", "#374151")
        progress_bg = get_color("training_progress_strip_bg", "#111827")
        progress_border = get_color("training_progress_strip_border", "#374151")
        progress_bar_bg = get_color("training_progress_bar_bg", "#374151")
        progress_chunk = get_color("training_progress_bar_chunk", "#3b82f6")
        progress_label = get_color("training_progress_label_text", get_color("text_secondary", "#9ca3af"))

        # In embedded mode, _header_widget is reparented into MainWindow's nav
        # stack so findChild/findChildren starting from self won't reach it.
        # We apply header styles directly via the stored reference instead.
        header_widget = getattr(self, '_header_widget', None)
        if header_widget is not None:
            if self._embedded:
                header_widget.setStyleSheet(
                    "#trainingWorkspaceHeaderContent { background: transparent; border: none; }"
                )
            else:
                header_widget.setStyleSheet(
                    f"#trainingWorkspaceHeader {{ background: {header_bg}; border-bottom: 1px solid {header_border}; }}"
                )
        else:
            header = self.findChild(QWidget, "trainingWorkspaceHeader")
            if header is not None:
                header.setStyleSheet(
                    f"#trainingWorkspaceHeader {{ background: {header_bg}; border-bottom: 1px solid {header_border}; }}"
                )

        # Collect labels/buttons from both the body subtree and the (possibly
        # reparented) header widget so all receive correct theme colours.
        all_labels = list(self.findChildren(QLabel))
        all_buttons = list(self.findChildren(QPushButton))
        if self._embedded and header_widget is not None:
            all_labels += list(header_widget.findChildren(QLabel))
            all_buttons += list(header_widget.findChildren(QPushButton))

        for child in all_labels:
            if child.objectName() == "workspaceTitleLabel":
                child.setStyleSheet(f"QLabel {{ font-size: 12px; color: {base_text}; background: transparent; border: none; font-weight: 700; }}")
            elif child.objectName() == "workspaceAssetLabel":
                child.setStyleSheet(f"QLabel {{ font-size: 12px; color: {asset_text}; background: transparent; border: none; font-weight: 600; }}")
            elif child.objectName() == "workspaceExportLabel":
                child.setStyleSheet(f"QLabel {{ font-size: 12px; color: {export_text}; background: transparent; border: none; font-weight: 700; }}")
            elif child.text() == "Training Ground":
                child.setStyleSheet(f"QLabel {{ font-size: 13px; font-weight: 700; color: {title_text}; background: transparent; border: none; }}")
            elif child.objectName() == "workspaceHeaderSep" or child.text() == "|":
                child.setStyleSheet(f"QLabel {{ color: {sep_text}; background: transparent; border: none; }}")
            elif child.text().startswith("馃弸") or child.text().startswith("HF") or child.text().startswith("Local"):
                child.setStyleSheet(
                    f"QLabel {{ font-size: 11px; color: {badge_text}; background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 4px; padding: 1px 6px; }}"
                )
        for child in all_buttons:
            if child.objectName() == "workspaceHeaderButton":
                child.setStyleSheet(
                    f"QPushButton {{ background: {toolbar_bg}; color: {toolbar_btn_text}; border: 1px solid {toolbar_border}; border-radius: 5px; padding: 4px 10px; }}"
                    f"QPushButton:hover {{ background: {toolbar_btn_hover}; }}"
                    f"QPushButton:pressed {{ background: {toolbar_btn_pressed}; }}"
                )
        if hasattr(self, "_window_controls") and self._window_controls is not None:
            self._window_controls.apply_theme()
            self._sync_window_controls()
        if hasattr(self, "_sidebar") and self._sidebar is not None:
            self._sidebar.setStyleSheet(
                f"#sidebarRail {{ background-color: {sidebar_rail_bg}; border-right: none; }}"
                f"#sidebarContentPanel {{ background-color: {sidebar_panel_bg}; border-right: none; }}"
                f"#sidebarContentHeader {{ background-color: {sidebar_panel_bg}; border-bottom: none; border-right: none; }}"
                f"#sidebarContentTitle {{ color: {sidebar_title_text}; font-size: 13px; font-weight: 700; }}"
                f"#sidebarNavButton, #sidebarUtilityButton {{ background-color: {sidebar_rail_bg}; border: none; color: {toolbar_btn_text}; }}"
                f"#sidebarNavButton:hover, #sidebarUtilityButton:hover {{ background-color: {sidebar_button_hover_bg}; }}"
                f"#sidebarNavButton:checked {{ background-color: {sidebar_button_checked_bg}; }}"
                f"#sidebarContentStack, #sidebarPlaceholderPage {{ background-color: {sidebar_panel_bg}; }}"
            )
        browser_style = (
                f"#trainingCanvasBrowserPanel {{ background: transparent; }}"
                f"#trainingCanvasBrowserTree {{ background: {toolbar_list_bg}; border: none; color: {toolbar_list_text}; font-size: 12px; }}"
                f"#trainingCanvasBrowserTree::item {{ padding: 0px; border-radius: 3px; }}"
                f"#trainingCanvasBrowserTree::item:hover {{ background: {toolbar_list_hover}; }}"
                f"#trainingCanvasBrowserTree::item:selected {{ background: {toolbar_list_selected}; }}"
                f"#trainingCanvasBrowserRow {{ background: transparent; }}"
                f"#trainingWorkspaceRow {{ background: transparent; }}"
                f"#trainingCanvasBrowserRowLabel {{ color: {toolbar_list_text}; background: transparent; border: none; }}"
                f"#trainingWorkspaceLabel {{ color: {toolbar_list_text}; background: transparent; border: none; font-weight: 600; }}"
                f"#trainingBtnAddWorkspace, #trainingBtnAddExperiment, #trainingBtnRenameWorkspace, #trainingBtnDeleteWorkspace {{ background: transparent; color: {toolbar_btn_text}; border: 1px solid {toolbar_border}; border-radius: 4px; padding: 0px 8px; min-height: 22px; }}"
                f"#trainingBtnAddWorkspace:hover, #trainingBtnAddExperiment:hover, #trainingBtnRenameWorkspace:hover, #trainingBtnDeleteWorkspace:hover {{ background: {toolbar_list_hover}; }}"
                f"#trainingCanvasBrowserDeleteBtn {{ background: transparent; color: {toolbar_list_text}; border: none; border-radius: 1px; padding: 0px; }}"
                f"#trainingCanvasBrowserDeleteBtn:hover {{ background: {toolbar_list_hover}; }}"
            )
        browser = getattr(self, "_canvas_browser_panel", None)
        if browser is not None:
            browser.setStyleSheet(browser_style)
        export_browser = getattr(self, "_export_browser_panel", None)
        if export_browser is not None:
            export_browser.apply_theme()
        toolbar = self.findChild(QWidget, "trainingToolbarRow")
        if toolbar is not None:
            toolbar.setStyleSheet(
                f"#trainingToolbarRow {{ background: {toolbar_bg}; border-bottom: 1px solid {toolbar_border}; }}"
            )
        tool_btn_style = (
            f"QToolButton {{ background: transparent; color: {toolbar_btn_text}; border: none; font-size: 12px; padding: 4px 10px; border-radius: 4px; }}"
            f"QToolButton:hover {{ background: {toolbar_btn_hover}; }}"
            f"QToolButton:pressed {{ background: {toolbar_btn_pressed}; }}"
            f"QToolButton::menu-indicator {{ image: none; }}"
        )
        list_style = (
            f"QListWidget {{ background: {toolbar_list_bg}; border: none; color: {toolbar_list_text}; font-size: 12px; }}"
            f"QListWidget::item {{ padding: 5px 8px; border-radius: 3px; }}"
            f"QListWidget::item:hover {{ background: {toolbar_list_hover}; }}"
            f"QListWidget::item:selected {{ background: {toolbar_list_selected}; }}"
        )
        for btn in self.findChildren(QToolButton):
            btn.setStyleSheet(tool_btn_style)
            if btn.menu() is not None:
                btn.menu().setStyleSheet(f"QMenu {{ background: {toolbar_menu_bg}; border: 1px solid {toolbar_menu_border}; padding: 4px; }}")
        for lst_name in ("_exp_list", "_run_list", "_asset_list"):
            lst = getattr(self, lst_name, None)
            if lst is None:
                continue
            lst.setStyleSheet(list_style)
            for i in range(lst.count()):
                item = lst.item(i)
                if item is not None and not bool(item.flags() & Qt.ItemFlag.ItemIsSelectable):
                    item.setForeground(QColor(get_color("training_toolbar_placeholder_text", get_color("text_muted", "#6b7280"))))
        strip = self.findChild(QWidget, "trainingProgressStrip")
        if strip is not None:
            strip.setStyleSheet(f"#trainingProgressStrip {{ background: {progress_bg}; border-bottom: 1px solid {progress_border}; }}")
        if hasattr(self, "_strip_bar"):
            self._strip_bar.setStyleSheet(
                f"QProgressBar {{ background: {progress_bar_bg}; border-radius: 5px; border: none; }}"
                f"QProgressBar::chunk {{ background: {progress_chunk}; border-radius: 5px; }}"
            )
        _ctrl_label_style = (
            f"QLabel {{ color: {progress_label}; font-size: 11px; "
            f"background: transparent; border: none; }}"
        )
        for attr in ("_strip_label", "_strip_state"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setStyleSheet(_ctrl_label_style)
        save_status = getattr(self, "_save_status_label", None)
        if save_status is not None:
            save_status.setStyleSheet(
                f"QLabel {{ color: {toolbar_btn_text}; font-size: 11px; font-weight: 700; "
                f"background: transparent; border: none; }}"
            )
        # Phase B toolbar action buttons (New / Save)
        _ctrl_btn_base = (
            f"QPushButton {{ background: {toolbar_bg}; color: {toolbar_btn_text}; "
            f"border: 1px solid {toolbar_border}; border-radius: 4px; "
            f"font-size: 11px; padding: 2px 10px; }}"
            f"QPushButton:hover {{ background: {toolbar_btn_hover}; }}"
            f"QPushButton:pressed {{ background: {toolbar_btn_pressed}; }}"
            f"QPushButton:disabled {{ color: {get_color('text_muted', '#6b7280')}; "
            f"background: {toolbar_bg}; border-color: {toolbar_border}; }}"
        )
        # Phase B toolbar action buttons (New / Save)
        _action_btn_style = _ctrl_btn_base
        for attr in (
            "_btn_new_exp",
            "_btn_import_asset",
            "_btn_download_asset",
            "_btn_save_exp",
            "_btn_refresh_assets",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(_action_btn_style)
        if hasattr(self, "_training_panel"):
            self._training_panel.apply_theme()
        if hasattr(self, "_canvas"):
            self._canvas.apply_theme()
        if hasattr(self, "_palette_panel") and self._palette_panel is not None:
            self._palette_panel.apply_theme()
        # Ensure every QPushButton in this page has a PointingHandCursor unless
        # it already has an intentional custom cursor (drag handles, etc.).
        for _btn in self.findChildren(QPushButton):
            if _btn.cursor().shape() == Qt.CursorShape.ArrowCursor:
                _btn.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------
    # Phase C 鈥?Train trigger + run persistence
    # ------------------------------------------------------------------

    def start_training(self, spec) -> None:
        """
        Create and start a TrainRunThread from a compiled TrainingJobSpec.

        Persists a run record under ``training_workspaces/<policy_id>/runs/``
        and updates it through the full thread lifecycle.

        Parameters
        ----------
        spec:
            A compiled TrainingJobSpec.  This is the authoritative input;
            no fallback dict path is supported in Phase C.
        """
        import time
        import uuid
        from src.system.core.train_run_thread import TrainRunThread

        # Guard: one active thread per window
        if self._active_thread is not None and self._active_thread.isRunning():
            return

        # Runtime device selection comes from the floating control bar.
        float_bar = getattr(self._canvas, "_float_bar", None) if hasattr(self, "_canvas") else None
        if float_bar is not None and hasattr(spec, "algorithm_config"):
            spec.algorithm_config.device = float_bar.selected_compute_device()

        # Build run_id and create persisted record (status=queued)
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        self._active_run_id = run_id

        algo = spec.algorithm_config
        policy_id_out = (
            getattr(spec.export_config, "bundle_name", "") or
            algo.policy_id_out or
            f"{self._selected_policy_id}_trained"
        )

        if self._ws_store is not None:
            try:
                from src.system.training.training_workspace_store import RunMeta
                # Try to resolve experiment name
                exp_name = ""
                try:
                    ws_meta = self._ws_store.load_workspace(self._workspace_policy_id)
                    exp = ws_meta.get_experiment(spec.experiment_id)
                    if exp is not None:
                        exp_name = exp.name
                except Exception:
                    pass

                run_meta = RunMeta(
                    run_id=run_id,
                    policy_id=self._workspace_policy_id,
                    experiment_id=spec.experiment_id,
                    experiment_name=exp_name,
                    status="queued",
                    algorithm=algo.algorithm or "PPO",
                    total_timesteps=algo.total_timesteps,
                    policy_id_out=policy_id_out,
                )
                self._ws_store.create_run(self._workspace_policy_id, run_meta)
            except Exception:
                pass

        # Build thread from spec
        thread = TrainRunThread.from_spec(spec, run_id=run_id)
        self._active_thread = thread
        self._training_panel.connect_thread(thread)

        # Show progress strip and wire lifecycle signals
        self._progress_strip.setVisible(True)
        self._strip_bar.setValue(0)
        self._strip_label.setText("Training: 0%  Step: 0 / 0")
        thread.progress.connect(self._on_progress_strip)
        thread.metrics_update.connect(self._on_metrics_update)
        thread.finished.connect(self._on_training_finished)
        thread.cancelled.connect(self._on_training_cancelled)
        thread.error.connect(self._on_training_error)
        thread.eval_completed.connect(self._on_eval_completed)
        thread.vis_check_started.connect(self._on_vis_check_started)
        thread.vis_check_ended.connect(self._on_vis_check_ended)

        thread.start()

        # Mark run as running now that the thread is live
        self._update_active_run(status="running", started_at=time.time())
        self._set_training_state("running")
        self.train_started.emit(policy_id_out)
        self._populate_lists()

        # Auto-show training stats overlay so the chart starts updating
        overlay = getattr(getattr(self, "_canvas", None), "_stats_overlay", None)
        if overlay is not None and not overlay.isVisible():
            overlay.show_for_button(None)

    def _update_active_run(self, **fields) -> None:
        """Merge *fields* into the active run's persisted JSON. Silent on error."""
        if not self._active_run_id or self._ws_store is None:
            return
        try:
            self._ws_store.update_run(self._workspace_policy_id, self._active_run_id, fields)
        except Exception:
            pass

    def _refresh_checkpoint_registry(self) -> None:
        """Trigger CheckpointRegistry re-discovery so the new bundle appears."""
        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry
            CheckpointRegistry().refresh()
        except Exception:
            pass

    def _on_training_finished(self, bundle_path: str) -> None:
        import time
        self._last_export_bundle_path = str(bundle_path or "")
        self._update_active_run(
            status="finished",
            finished_at=time.time(),
            bundle_path=bundle_path,
        )
        self._set_training_state("done")
        self._strip_state.setText("Complete")
        self._refresh_checkpoint_registry()
        self._populate_lists()
        self.checkpoint_exported.emit(bundle_path)

    def _on_eval_completed(
        self,
        mean_reward: float,
        std_reward: float,
        success_rate: float,
        passed: bool,
    ) -> None:
        """Persist evaluation results into the active run record."""
        self._update_active_run(
            eval_mean_reward=mean_reward,
            eval_std_reward=std_reward,
            eval_success_rate=success_rate,
            eval_passed=passed,
        )

    def _on_vis_check_started(self, check_num: int) -> None:
        """Show a prominent banner when a vis check milestone is reached."""
        self._strip_state.setText(f"Vis Check #{check_num}")
        self._strip_label.setText(
            f"Visualization milestone #{check_num} - MuJoCo viewer open. "
            "Close the viewer window to resume training."
        )
        if hasattr(self, "_training_panel"):
            self._training_panel.show_vis_check_banner(check_num)

    def _on_vis_check_ended(self) -> None:
        """Clear the vis check banner and restore normal strip labels."""
        self._strip_state.setText("Training")
        if hasattr(self, "_training_panel"):
            self._training_panel.hide_vis_check_banner()

    def _on_training_cancelled(self) -> None:
        import time
        self._update_active_run(status="cancelled", cancelled_at=time.time())
        self._set_training_state("done")
        self._strip_state.setText("Cancelled")
        self._populate_lists()

    def _on_training_error(self, msg: str) -> None:
        import time
        self._update_active_run(
            status="error",
            error=msg,
            finished_at=time.time(),
        )
        self._set_training_state("done")
        self._strip_state.setText("Error")
        self._populate_lists()

    def _on_train_requested(self, _job_spec: dict) -> None:
        """
        Slot wired to canvas.train_requested and the float-bar Start button.

        Compiles the current canvas into a typed TrainingJobSpec.
        If compilation fails the run is blocked and an error is shown;
        no run record is created for invalid graphs.
        """
        try:
            from src.system.training.training_spec import TrainingSpecCompiler
            canvas_data = self._canvas.scene.serialize_training_graph(for_compiler=True)
            spec = TrainingSpecCompiler().compile(
                canvas_data,
                policy_id=self._workspace_policy_id,
                experiment_id=self._current_experiment_id,
            )
            # 3-C: resolve BaseAssetNode → absolute checkpoint path
            entry, intended_mode = self._inject_base_asset_path(spec, canvas_data)
        except Exception as exc:
            self._strip_state.setText("Spec Error")
            self._strip_label.setText(str(exc)[:80])
            self._append_training_error(f"Spec compilation failed: {exc}")
            return

        # ── Pre-flight validation: semantic consistency checks + auto-fix ──
        # Robot/MJCF node is authoritative; conflicting nodes are auto-corrected.
        try:
            from src.system.training.training_spec import preflight_check
            pf_warnings = preflight_check(spec)
            for w in pf_warnings:
                self._append_training_log(f"[preflight] {w}")
        except Exception as exc:
            self._strip_state.setText("Preflight Error")
            self._strip_label.setText(str(exc)[:80])
            self._append_training_error(f"Pre-flight check failed: {exc}")
            return

        # 4-B: compatibility check before starting (resume/warm_start only).
        if intended_mode != "scratch":
            if entry is None:
                self._strip_state.setText("Asset Error")
                self._strip_label.setText(
                    "Base asset could not be resolved — cannot resume/warm-start."
                )
                return
            if not self._run_compat_check(entry, spec):
                return
        self.start_training(spec)

    def _on_review_requested(self, _job_spec: dict) -> None:
        try:
            spec, _entry, _intended_mode = self._compile_current_training_spec()
        except Exception as exc:
            self._strip_state.setText("Review Error")
            self._strip_label.setText("Review failed")
            self._append_training_error(f"Review setup failed: {exc}")
            return

        from src.system.training.vis_check_runner import run_environment_review
        self._run_viewer_task(
            "Review",
            "Opening MuJoCo review viewer...",
            "Review Error",
            run_environment_review,
            spec,
        )

    def _on_export_review_requested(self, _job_spec: dict) -> None:
        try:
            spec, _entry, _intended_mode = self._compile_current_training_spec()
            bundle_path = self._resolve_export_review_bundle_path(spec)
        except Exception as exc:
            self._strip_state.setText("Review Error")
            self._strip_label.setText("Export review failed")
            self._append_training_error(f"Export review setup failed: {exc}")
            return

        from src.system.training.vis_check_runner import run_export_bundle_review
        self._run_viewer_task(
            "Export Review",
            "Opening exported-bundle review viewer...",
            "Review Error",
            run_export_bundle_review,
            bundle_path,
            spec,
        )

    def _on_scene_preview_requested(self, _scene_params: dict) -> None:
        try:
            spec, _entry, _intended_mode = self._compile_current_training_spec()
        except Exception as exc:
            self._strip_state.setText("Preview Error")
            self._strip_label.setText("Scene preview failed")
            self._append_training_error(f"Scene preview setup failed: {exc}")
            return

        from src.system.training.vis_check_runner import run_scene_config_preview
        self._run_viewer_task(
            "Preview",
            "Opening MuJoCo scene preview...",
            "Preview Error",
            run_scene_config_preview,
            spec,
        )

    def _on_init_pose_preview_requested(self, _params: dict) -> None:
        try:
            spec, _entry, _intended_mode = self._compile_current_training_spec()
        except Exception as exc:
            self._strip_state.setText("Preview Error")
            self._strip_label.setText("Init Pose preview failed")
            self._append_training_error(f"Init Pose preview setup failed: {exc}")
            return

        from src.system.training.vis_check_runner import run_init_pose_preview
        self._run_viewer_task(
            "Init Pose Preview",
            "Opening init pose preview...",
            "Preview Error",
            run_init_pose_preview,
            spec,
        )

    def _run_viewer_task(self, state: str, opening_label: str, error_state: str, runner, *args) -> None:
        self._strip_state.setText(state)
        self._strip_label.setText(opening_label)
        self._append_training_log(opening_label)
        QApplication.processEvents()
        try:
            runner(*args, log_fn=self._append_review_log)
        except Exception as exc:
            self._strip_state.setText(error_state)
            self._strip_label.setText(f"{state} failed")
            self._append_training_error(f"{state} failed: {exc}")

    def _append_review_log(self, message: str) -> None:
        try:
            msg = str(message or "")
            if not msg:
                return
            self._append_training_log(msg)
            lowered = msg.lower()
            if any(token in lowered for token in ("failed", "error", "skipped", "unavailable")):
                log_error(msg)
        except Exception:
            pass

    def _append_training_log(self, message: str) -> None:
        msg = str(message or "").strip()
        if not msg:
            return
        panel = getattr(self, "_training_panel", None)
        if panel is not None:
            panel.on_log(msg)
            return
        from src.system.core.logger import log_info
        log_info(msg)

    def _append_training_error(self, message: str) -> None:
        msg = str(message or "").strip()
        if not msg:
            return
        panel = getattr(self, "_training_panel", None)
        if panel is not None:
            panel.on_error(msg)
            return
        log_error(msg)

    def _compile_current_training_spec(self):
        from src.system.training.training_spec import TrainingSpecCompiler, preflight_check

        canvas_data = self._canvas.scene.serialize_training_graph(for_compiler=True)
        spec = TrainingSpecCompiler().compile(
            canvas_data,
            policy_id=self._workspace_policy_id,
            experiment_id=self._current_experiment_id,
        )
        # Auto-fix semantic issues (scene_config mismatch etc.)
        preflight_check(spec)
        entry, intended_mode = self._inject_base_asset_path(spec, canvas_data)
        return spec, entry, intended_mode

    def _resolve_export_review_bundle_path(self, spec):
        from pathlib import Path
        from src.system.service.checkpoint_registry import CheckpointRegistry

        export_target = str(getattr(getattr(spec, "export_config", None), "export_target", "runtime_bundle") or "runtime_bundle")
        if export_target not in ("runtime_bundle", "both"):
            raise FileNotFoundError(
                "Current Export target does not produce a runtime bundle. "
                "Set Export target to runtime_bundle or both, then train/export first."
            )

        policy_id_out = (
            getattr(getattr(spec, "export_config", None), "bundle_name", "") or
            getattr(getattr(spec, "algorithm_config", None), "policy_id_out", "") or
            f"{self._selected_policy_id}_trained"
        )

        last_path = Path(str(self._last_export_bundle_path or "")).expanduser()
        if last_path.exists() and last_path.is_dir() and last_path.name == str(policy_id_out):
            return last_path

        try:
            return Path(CheckpointRegistry().get_bundle_path(str(policy_id_out)))
        except Exception:
            pass

        candidate = get_project_root() / "custom_mods/training/checkpoints" / str(policy_id_out)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No exported runtime bundle found for '{policy_id_out}'. "
            "Train/export once before using Export Review."
        )

    def _resolve_latest_export_training_asset(self):
        workspace_policy_id = str(self._workspace_policy_id or self._selected_policy_id or "").strip()
        if not workspace_policy_id:
            return None
        entries = list(self._asset_registry.list_assets() or [])
        candidates = []
        for entry in entries:
            if not getattr(entry, "is_valid", True):
                continue
            source_path = Path(getattr(entry, "asset_path", "")) / "source.json"
            try:
                with source_path.open("r", encoding="utf-8") as fh:
                    source = json.load(fh) or {}
            except Exception:
                continue
            parent_policy_id = str(source.get("parent_policy_id", "") or "").strip()
            if parent_policy_id != workspace_policy_id:
                continue
            try:
                sort_key = source_path.stat().st_mtime
            except Exception:
                sort_key = 0.0
            candidates.append((sort_key, entry))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _inject_base_asset_path(self, spec, canvas_data: dict):
        """
        Resolve the BaseAssetNode parameters into spec.base_asset_path.

        Reads the serialized canvas nodes, finds the first 'base_asset' node,
        looks up the entry in self._asset_registry, and sets
        spec.base_asset_path to the absolute path of the checkpoint file.

        Returns ``(entry, intended_mode)`` where:
          - ``intended_mode`` is the load_mode from the canvas node ("scratch"
            when no BaseAssetNode is present, or when load_mode == "scratch").
          - ``entry`` is the resolved TrainingAssetEntry, or None when
            intended_mode == "scratch" or asset resolution failed.

        The caller must check ``intended_mode != "scratch"`` to decide whether
        a compat gate is required 鈥?not ``spec.resume_mode``, which is only
        set on successful resolution.
        """
        nodes = canvas_data.get("nodes", [])
        ba_node = next(
            (n for n in nodes if n.get("node_type") == "base_asset"), None
        )
        if ba_node is None:
            return None, "scratch"
        params = ba_node.get("parameters", {})
        start_point = str(params.get("start_point", "") or "").strip()
        asset_id = str(params.get("asset_id", "")).strip()
        checkpoint_file = str(params.get("checkpoint_file", "")).strip()
        load_mode = str(params.get("load_mode", "scratch")).strip()
        if start_point.startswith("asset:") and not asset_id:
            asset_id = start_point.split(":", 1)[1].strip()
        if start_point == "__new__":
            return None, "scratch"
        if load_mode == "scratch":
            return None, "scratch"
        # Non-scratch intent: attempt to resolve the asset
        try:
            if start_point == "__latest_export__":
                entry = self._resolve_latest_export_training_asset()
                checkpoint_file = ""
            else:
                entry = self._asset_registry.get(asset_id)
            if entry is None:
                return None, load_mode   # resolution failed 鈥?caller will block
            ckpt = checkpoint_file or entry.primary_checkpoint
            abs_path = str(entry.asset_path / ckpt) if ckpt else ""
            spec.base_asset_path = abs_path
            # Ensure resume_mode propagates on both top-level and algo_config
            spec.resume_mode = load_mode
            spec.algorithm_config.resume_mode = load_mode
            return entry, load_mode
        except Exception:
            return None, load_mode   # resolution failed 鈥?caller will block

    def _run_compat_check(self, entry, spec) -> bool:
        """
        Phase 4-B: Run TrainingCompatibilityChecker against (entry, spec).

        Returns True  鈫?proceed with training.
        Returns False 鈫?user cancelled (or chose not to force-start on FAIL).

        Dialog behaviour:
          - Any FAIL:  blocking dialog with "Force Start" override + "Cancel".
          - WARN only: informational dialog with "Proceed" + "Cancel".
          - All PASS:  silent pass-through (returns True).
        """
        try:
            from src.system.training.training_compatibility import (
                CompatLevel,
                TrainingCompatibilityChecker,
            )
        except ImportError:
            return True  # checker not available 鈥?allow training

        results = TrainingCompatibilityChecker().check(entry, spec)
        fails = [r for r in results if r.level == CompatLevel.FAIL]
        warns = [r for r in results if r.level == CompatLevel.WARN]

        if not fails and not warns:
            return True  # all clear

        # Build human-readable lines
        lines: list = []
        for r in fails:
            lines.append(f"[FAIL]  {r.field}: {r.message}")
        for r in warns:
            lines.append(f"[WARN]  {r.field}: {r.message}")
        detail = "\n".join(lines)

        if fails:
            msg = QMessageBox(self)
            msg.setWindowTitle("Compatibility Check 鈥?Failures Detected")
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText(
                f"<b>{len(fails)} compatibility failure(s)</b> found between the "
                f"base checkpoint and the current canvas.<br><br>"
                f"Starting training may corrupt the run or crash immediately."
            )
            msg.setDetailedText(detail)
            force_btn = msg.addButton("Force Start Anyway", QMessageBox.ButtonRole.DestructiveRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(msg.button(QMessageBox.StandardButton.Cancel) or force_btn)
            msg.exec()
            return msg.clickedButton() is force_btn

        # WARN only
        msg = QMessageBox(self)
        msg.setWindowTitle("Compatibility Check 鈥?Warnings")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            f"<b>{len(warns)} compatibility warning(s)</b> found.<br><br>"
            f"Training can still proceed but results may be suboptimal."
        )
        msg.setDetailedText(detail)
        msg.addButton("Proceed", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        # AcceptRole button text is "Proceed" 鈥?any non-Cancel click 鈫?proceed
        return clicked is not None and clicked.text() == "Proceed"

    # ------------------------------------------------------------------
    # Phase B 鈥?Workspace persistence wiring
    # ------------------------------------------------------------------

    def _init_workspace(self) -> None:
        """
        Initialise workspace store, load or create the workspace for
        self._workspace_policy_id, populate toolbar lists, and restore the last
        active experiment into the canvas.
        """
        try:
            self._ws_store = _get_workspace_store()
            if self._workspace_policy_id:
                meta = self._ws_store.ensure_workspace(self._workspace_policy_id)
                self._populate_lists(meta)
                self._refresh_canvas_browser(meta)
            else:
                # No policy pre-selected 鈥?show the workspace browser only.
                self._refresh_canvas_browser(None)
                self._populate_training_assets()
                self._ensure_template_canvas()
                return

            # Wire experiment list double-click 鈫?load experiment
            if hasattr(self, "_exp_list") and self._exp_list is not None:
                self._exp_list.itemDoubleClicked.connect(self._on_experiment_item_dclicked)

            # Restore last active experiment
            active_id = self._source_experiment_id or meta.active_experiment_id
            if not active_id and meta.experiments:
                active_id = meta.experiments[0].experiment_id
            if active_id and active_id in [e.experiment_id for e in meta.experiments]:
                self._load_experiment_by_id(active_id, show_dialog=False)
            elif self._selected_training_asset_id:
                self._load_asset_into_canvas(self._selected_training_asset_id)
            else:
                self._apply_resolved_task_template()
                self._canvas_source_state = "default_template"
                self._bind_canvas_to_workspace_policy()
            if self._canvas_source_state == "loaded_experiment":
                self._bind_canvas_to_workspace_policy()
            self._refresh_current_canvas_label()
        except Exception as exc:
            log_error(
                f"Training workspace init failed for selected='{self._selected_policy_id}' "
                f"workspace='{self._workspace_policy_id}': {exc}"
            )
            self._ensure_template_canvas()
            self._bind_canvas_to_workspace_policy()
            self._startup_restore_error_message = (
                f"{self._workspace_policy_id or 'Training workspace'} failed to load, error: {exc}"
            )
            self.restore_failed.emit(self._startup_restore_error_message)

    def _populate_lists(self, meta=None) -> None:
        """Refresh toolbar experiment/run lists from the workspace store."""
        if self._ws_store is None:
            return
        try:
            if meta is None:
                meta = self._ws_store.load_workspace(self._workspace_policy_id)

            # Experiments
            exp_names = [e.name for e in meta.experiments]
            self._populate_toolbar_list(
                self._exp_list, exp_names, "No experiments yet"
            )
            # Store experiment_id in UserRole for each item
            for i, exp_meta in enumerate(meta.experiments):
                item = self._exp_list.item(i)
                if item is not None:
                    from PySide6.QtCore import Qt as _Qt
                    item.setData(_Qt.ItemDataRole.UserRole, exp_meta.experiment_id)

            # Runs 鈥?show useful labels sorted newest-first
            runs = self._ws_store.list_runs(self._workspace_policy_id)
            from PySide6.QtWidgets import QListWidgetItem
            from PySide6.QtCore import Qt as _Qt
            self._run_list.clear()
            if not runs:
                placeholder = QListWidgetItem("No runs yet")
                placeholder.setFlags(placeholder.flags() & ~_Qt.ItemFlag.ItemIsEnabled)
                self._run_list.addItem(placeholder)
            for r in runs:
                status = r.get("status", "?")
                algo   = r.get("algorithm", "?")
                ts     = r.get("created_at", 0.0)
                try:
                    from datetime import datetime
                    dt_str = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
                except Exception:
                    dt_str = "?"
                # Append eval summary if available
                mean_r = r.get("eval_mean_reward")
                passed = r.get("eval_passed")
                if mean_r is not None:
                    eval_tag = f"  \u2713eval:{mean_r:.2f}" if passed else f"  \u2717eval:{mean_r:.2f}"
                    label = f"[{status}] {algo}  {dt_str}{eval_tag}"
                else:
                    label = f"[{status}] {algo}  {dt_str}"
                item = QListWidgetItem(label)
                item.setData(_Qt.ItemDataRole.UserRole, r.get("run_id", ""))
                # Tooltip with eval details
                tip_lines = [
                    f"Run ID:    {r.get('run_id', '?')}",
                    f"Status:    {status}",
                    f"Algorithm: {algo}",
                    f"Steps:     {r.get('total_timesteps', '?')}",
                ]
                if mean_r is not None:
                    tip_lines += [
                        f"Eval mean reward:  {mean_r:.3f}",
                        f"Eval std reward:   {r.get('eval_std_reward', 0.0):.3f}",
                        f"Eval success rate: {r.get('eval_success_rate', 0.0):.1%}",
                        f"Eval passed:       {passed}",
                    ]
                item.setToolTip("\n".join(tip_lines))
                self._run_list.addItem(item)

            self._populate_training_assets()
            self._refresh_canvas_browser(meta)
            self._refresh_export_browser()
        except Exception as exc:
            log_error(
                f"Training workspace list refresh failed for selected='{self._selected_policy_id}' "
                f"workspace='{self._workspace_policy_id}': {exc}"
            )

    def _populate_training_assets(self) -> None:
        from PySide6.QtCore import Qt as _Qt

        self._asset_list.clear()
        assets = self._asset_registry.list_assets()
        if not assets:
            placeholder = QListWidgetItem("No training assets yet")
            placeholder.setFlags(placeholder.flags() & ~_Qt.ItemFlag.ItemIsSelectable)
            self._asset_list.addItem(placeholder)
            self._set_selected_training_asset("")
            return

        selected_asset_id = self._get_canvas_base_asset_id() or self._selected_training_asset_id
        found_selected = False
        for entry in assets:
            status = "OK" if entry.is_valid else "X"
            robot_badge = f"  [{entry.robot_type}]" if entry.robot_type else ""
            label = f"{status}  {entry.label()}{robot_badge}"
            item = QListWidgetItem(label)
            item.setData(_Qt.ItemDataRole.UserRole, entry.asset_id)
            tip_lines = [
                f"Asset ID:   {entry.asset_id}",
                f"Framework:  {entry.framework or '-'}",
                f"Algorithm:  {entry.algorithm or '-'}",
                f"Robot:      {entry.robot_type or '-'}",
                f"Obs dim:    {entry.obs_dim or '-'}",
                f"Action dim: {entry.action_dim or '-'}",
                f"Action type:{entry.action_type or '-'}",
                f"Primary:    {entry.primary_checkpoint or '-'}",
                f"Path:       {entry.asset_path}",
            ]
            if entry.error:
                tip_lines.append(f"Error:      {entry.error}")
            item.setToolTip("\n".join(tip_lines))
            if not entry.is_valid:
                item.setForeground(QColor(get_color("text_muted", "#6b7280")))
            self._asset_list.addItem(item)
            if entry.asset_id == selected_asset_id:
                found_selected = True
                self._asset_list.setCurrentItem(item)

        self._set_selected_training_asset(selected_asset_id if found_selected else "")

    def _on_training_asset_item_clicked(self, item) -> None:
        from PySide6.QtCore import Qt as _Qt

        asset_id = str(item.data(_Qt.ItemDataRole.UserRole) or "").strip()
        if not asset_id:
            return
        self._set_selected_training_asset(asset_id)
        self._load_asset_into_canvas(asset_id)

    def _load_asset_into_canvas(self, asset_id: str) -> None:
        """
        Sync the canvas with the selected TrainingAssetEntry.

        On a blank scene, restore the default template first. Then update the
        relevant nodes so the visible canvas matches the selected asset rather
        than only marking the header label as selected.
        """
        try:
            entry = self._asset_registry.get(asset_id)
        except Exception:
            return

        self._ensure_template_canvas()
        can_apply_task_template = (
            self._canvas_source_state != "loaded_experiment"
            and self._canvas_uses_current_template_defaults()
        )
        for node_type, params in self._asset_node_params(entry).items():
            node_item = self._find_training_node_item(node_type)
            if node_item is None and node_type == "base_asset":
                from PySide6.QtCore import QPointF
                node_item = self._canvas.scene.create_node("Start Point", QPointF(-200, -300))
            if node_item is not None:
                node_item.load_parameters(params)

        if can_apply_task_template:
            self._apply_resolved_task_template(
                robot_type=str(getattr(entry, "robot_type", "") or self._get_canvas_robot_type()),
                asset_id=entry.asset_id,
            )
            self._canvas_source_state = "asset_default_template"

        self._set_selected_training_asset(entry.asset_id, entry.label())
        self._refresh_current_canvas_label()

    def _set_selected_training_asset(self, asset_id: str, label: str = "") -> None:
        self._selected_training_asset_id = str(asset_id or "").strip()
        if not self._selected_training_asset_id:
            self._update_assets_dropdown_label("")
            return

        display_label = label
        if not display_label:
            try:
                entry = self._asset_registry.get(self._selected_training_asset_id)
                display_label = entry.label()
            except Exception:
                display_label = self._selected_training_asset_id
        self._update_assets_dropdown_label(display_label)

    def _import_training_asset_local(self) -> None:
        from pathlib import Path
        from src.system.training.training_asset_registry import TrainingAssetImportError

        source_type, ok = QInputDialog.getItem(
            self,
            "Import Local",
            "Source type:",
            ["Folder", "ZIP Archive"],
            0,
            False,
        )
        if not ok or not str(source_type).strip():
            return

        if source_type == "Folder":
            selected_path = QFileDialog.getExistingDirectory(
                self,
                "Import Training Asset Folder",
                "",
            )
        else:
            selected_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Training Asset Archive",
                "",
                "ZIP archives (*.zip)",
            )
        if not selected_path:
            return

        try:
            entry = self._asset_registry.import_local(Path(selected_path))
        except TrainingAssetImportError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Import Failed", f"Unexpected import error: {exc}")
            return

        self._populate_lists()
        self._set_selected_training_asset(entry.asset_id, entry.label())
        QMessageBox.information(self, "Training Asset Imported", f"Imported: {entry.label()}")

    def _download_training_asset_hf(self) -> None:
        from src.system.core.hf_training_asset_download_thread import HFTrainingAssetDownloadThread

        if self._asset_download_thread is not None and self._asset_download_thread.isRunning():
            QMessageBox.information(self, "Download In Progress", "A Hugging Face asset download is already running.")
            return

        url, ok = QInputDialog.getText(
            self,
            "Download Hugging Face Asset",
            "Repository URL or repo_id:",
        )
        if not ok or not str(url).strip():
            return

        thread = HFTrainingAssetDownloadThread(str(url).strip(), self)
        self._asset_download_thread = thread
        thread.progress.connect(self._on_training_asset_download_progress)
        thread.finished.connect(self._on_training_asset_download_finished)
        thread.failed.connect(self._on_training_asset_download_failed)
        self._strip_state.setText("Asset Download")
        self._strip_label.setText("Preparing community asset download...")
        thread.start()

    def _on_training_asset_download_progress(self, msg: str) -> None:
        self._strip_state.setText("Asset Download")
        self._strip_label.setText(str(msg or "")[:120] or "Downloading training asset...")

    def _on_training_asset_download_finished(self, asset_path_str: str) -> None:
        from pathlib import Path
        import shutil

        asset_path = Path(asset_path_str)
        try:
            entry = self._asset_registry.import_hf_snapshot(asset_path)
            self._populate_lists()
            self._set_selected_training_asset(entry.asset_id, entry.label())
            self._strip_state.setText("Idle")
            self._strip_label.setText("Step: -")
            QMessageBox.information(self, "HF Asset Imported", f"Imported: {entry.label()}")
        except Exception as exc:
            QMessageBox.warning(self, "HF Import Failed", str(exc))
        finally:
            shutil.rmtree(asset_path.parent, ignore_errors=True)
            self._asset_download_thread = None

    def _on_training_asset_download_failed(self, error_msg: str) -> None:
        self._asset_download_thread = None
        self._strip_state.setText("Idle")
        self._strip_label.setText("Step: -")
        QMessageBox.warning(self, "HF Download Failed", str(error_msg or "Unknown download error."))

    def _on_experiment_item_dclicked(self, item) -> None:
        """Load the double-clicked experiment into the canvas."""
        from PySide6.QtCore import Qt as _Qt
        experiment_id = item.data(_Qt.ItemDataRole.UserRole)
        if experiment_id:
            self._load_experiment_by_id(experiment_id)

    def _load_experiment_by_id(self, experiment_id: str, show_dialog: bool = True) -> None:
        """Load an experiment canvas from disk and replace the current canvas."""
        if self._ws_store is None:
            return
        try:
            canvas_data = self._ws_store.load_experiment(self._workspace_policy_id, experiment_id)
            self._canvas.scene.load_training_graph(canvas_data)
            if not canvas_data.get("nodes"):
                self._ensure_template_canvas()
                if self._selected_training_asset_id:
                    self._load_asset_into_canvas(self._selected_training_asset_id)
            else:
                self._canvas_source_state = "loaded_experiment"
                self._sync_selected_training_asset_from_canvas()
            self._current_experiment_id = experiment_id
            self._ws_store.set_active_experiment(self._workspace_policy_id, experiment_id)
            self._bind_canvas_to_workspace_policy()
            self._refresh_canvas_browser()
        except Exception as exc:
            self._strip_state.setText("Load Error")
            self._strip_label.setText(str(exc)[:80])
            self._ensure_template_canvas()
            self._current_experiment_id = ""
            self._bind_canvas_to_workspace_policy()
            self._refresh_current_canvas_label()
            self._refresh_canvas_browser()
            message = f"{experiment_id} failed to load, error: {exc}"
            if show_dialog:
                QMessageBox.warning(self, "Load Experiment Failed", message)
            else:
                self._startup_restore_error_message = message
                self.restore_failed.emit(message)

    def _save_current_experiment(self) -> None:
        """
        Save the current canvas as the active experiment.
        If no experiment exists yet, create one first.
        If no workspace is active (Default template), prompt to create one.
        """
        if self._ws_store is None:
            return
        try:
            # No workspace yet (Default template state) — ask user to create one.
            if not self._workspace_policy_id:
                existing = set(self._ws_store.list_workspaces())
                default_name = f"Workspace_{len(existing) + 1}"
                ws_name, ok = QInputDialog.getText(
                    self, "New Workspace",
                    "Enter a workspace name to save into:",
                    text=default_name,
                )
                if not ok:
                    return
                ws_name = str(ws_name or "").strip()
                if not ws_name:
                    return
                if ws_name in existing:
                    QMessageBox.warning(
                        self, "Name Conflict",
                        f"Workspace '{ws_name}' already exists.",
                    )
                    return
                self._ws_store.ensure_workspace(ws_name)
                self._workspace_policy_id = ws_name
                self._selected_policy_id = ws_name

            self._bind_canvas_to_workspace_policy()
            canvas_data = self._canvas.scene.serialize_training_graph()
            meta = self._ws_store.load_workspace(self._workspace_policy_id)
            current_meta = (
                meta.get_experiment(self._current_experiment_id)
                if self._current_experiment_id else None
            )

            if current_meta is None:
                default_name = f"Experiment {len(meta.experiments) + 1}"
                name = self._prompt_experiment_name(default_name)
                if not name:
                    return
                exp_meta = self._ws_store.create_experiment(
                    self._workspace_policy_id,
                    name=name,
                    initial_canvas=canvas_data,
                )
                self._current_experiment_id = exp_meta.experiment_id
            else:
                self._ws_store.save_experiment(
                    self._workspace_policy_id,
                    self._current_experiment_id,
                    canvas_data,
                )

            self._ws_store.set_active_experiment(
                self._workspace_policy_id, self._current_experiment_id
            )
            self._populate_lists()
            self._refresh_canvas_browser()
            if hasattr(self, "_save_status_label") and self._save_status_label is not None:
                self._save_status_label.setText("Saved")
        except Exception as exc:
            if hasattr(self, "_save_status_label") and self._save_status_label is not None:
                self._save_status_label.setText("")
            self._strip_state.setText("Save Error")
            self._strip_label.setText(str(exc)[:80])
            QMessageBox.warning(self, "Save Experiment Failed", str(exc))

    def _new_experiment(self) -> None:
        """Create a new empty experiment and load it into the canvas."""
        if self._ws_store is None:
            return
        try:
            meta = self._ws_store.load_workspace(self._workspace_policy_id)
            default_name = f"Experiment {len(meta.experiments) + 1}"
            name = self._prompt_experiment_name(default_name)
            if not name:
                return
            exp_meta = self._ws_store.create_experiment(self._workspace_policy_id, name=name)
            self._current_experiment_id = exp_meta.experiment_id
            # Load the empty canvas (clears current graph)
            canvas_data = self._ws_store.load_experiment(
                self._workspace_policy_id, exp_meta.experiment_id
            )
            self._canvas.scene.load_training_graph(canvas_data)
            self._ensure_template_canvas()
            if self._selected_training_asset_id:
                self._load_asset_into_canvas(self._selected_training_asset_id)
            else:
                self._canvas_source_state = "default_template"
            self._bind_canvas_to_workspace_policy()
            self._populate_lists()
            self._refresh_canvas_browser()
        except Exception as exc:
            log_error(
                f"New experiment creation failed for selected='{self._selected_policy_id}' "
                f"workspace='{self._workspace_policy_id}': {exc}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_source_badge(self) -> str:
        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry
            for e in CheckpointRegistry().discover():
                if e.policy_id == self._selected_policy_id and hasattr(e, "source_badge"):
                    return e.source_badge()
        except Exception:
            pass
        return ""

    @property
    def policy_id(self) -> str:
        return self._selected_policy_id









