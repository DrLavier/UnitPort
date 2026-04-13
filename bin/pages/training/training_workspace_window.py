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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.system.core.logger import log_error, log_success, log_warning
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
# SB3 metric layers
_METRIC_LAYERS_SB3 = [
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

# Isaac Lab metric layers
_METRIC_LAYERS_IL = [
    ("training", "Training", [
        ("reward_mean",  "Reward"),
        ("ep_len_mean",  "Ep Len"),
        ("fps",          "FPS"),
    ]),
    ("losses", "Losses", [
        ("policy_loss",  "Policy"),
        ("value_loss",   "Value"),
    ]),
    ("algorithm", "Algorithm", [
        ("kl",           "KL"),
        ("entropy",      "Entropy"),
        ("lr",           "LR"),
    ]),
]

# Default — switched at runtime by MetricsLayersPanel.set_backend()
_METRIC_LAYERS = _METRIC_LAYERS_SB3


class MetricsLayersPanel(QWidget):
    """Compact 4-layer live metrics display for Training Ground stats overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_labels: Dict[str, Dict[str, QLabel]] = {}
        self._build_ui()

    def _build_ui(self, layers=None) -> None:
        if layers is None:
            layers = _METRIC_LAYERS
        layout = self.layout()
        if layout is None:
            layout = QVBoxLayout(self)
        else:
            # Clear existing
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._value_labels.clear()
        self._active_layers = layers
        for layer_key, title, metrics in layers:
            section = self._build_section(layer_key, title, metrics)
            layout.addWidget(section)
        layout.addStretch(1)

    def set_backend(self, backend: str) -> None:
        """Rebuild with SB3 or Isaac Lab metric layers."""
        layers = _METRIC_LAYERS_IL if backend == "isaac_lab" else _METRIC_LAYERS_SB3
        self._build_ui(layers)

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
        """Update displayed values from a layered metrics dict."""
        for layer_key, _title, layer_metrics in getattr(self, "_active_layers", _METRIC_LAYERS):
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

        # ``force_default=True`` is used by the train-start path to mean
        # "reset the chart to track only the brand-new run". The previous
        # behaviour merely *added* the new run on top of whatever was
        # already in ``_selected_runs``, so any cached run the user (or
        # a prior session) had ticked stayed ticked — by the third or
        # fourth training run the chart looked like a forest of legacy
        # series. Wipe the selection on a forced reset so only the
        # ON-GOING run is checked; user-driven toggles afterwards still
        # accumulate normally.
        if force_default:
            self._selected_runs.clear()
        else:
            self._selected_runs.intersection_update(current_run_ids)

        # Self-heal: if the train-start path called us BEFORE the first
        # sample landed in the cache, ``ensure_run`` had been called but
        # ``_build_run_entries`` may have come back empty (depending on
        # the order of subprocess startup). Without this, ``_selected_runs``
        # would stay empty forever and the chart would render
        # "No visible series" even though samples are streaming in.
        # Auto-promote whatever entry is marked ``preferred`` (which is
        # always the on-going run) on every refresh, so the chart picks
        # up the active run as soon as the cache learns about it.
        preferred_id = next(
            (entry["run_id"] for entry in entries if entry.get("preferred")),
            None,
        )
        if preferred_id is not None and preferred_id not in self._selected_runs:
            self._selected_runs.add(preferred_id)

        if not self._selected_runs and entries:
            preferred = preferred_id or entries[0]["run_id"]
            self._selected_runs.add(preferred)

        self._rebuild_run_checkboxes(entries)
        self._rebuild_series()

    def _find_workspace_window(self):
        """Walk up the parent chain to find the TrainingWorkspaceWindow instance."""
        widget = self._host
        while widget is not None:
            if hasattr(widget, "_workspace_name") and hasattr(widget, "_active_run_id"):
                return widget
            widget = widget.parent() if hasattr(widget, "parent") else None
        # Fallback: try via anchor_bar.window()
        return self._anchor_bar.window() if self._anchor_bar is not None else None

    def _build_run_entries(self) -> list:
        from src.system.training.training_run_cache import get_training_run_cache

        cache = get_training_run_cache()
        tables = cache.list_runs()
        ws = self._find_workspace_window()
        workspace_policy = str(getattr(ws, "_workspace_name", "") or getattr(ws, "_policy_id", "") or "").strip()
        active_run_id = str(getattr(ws, "_active_run_id", "") or "").strip()

        filtered = []
        for table in tables:
            tbl_pid = table.get("policy_id", "")
            if workspace_policy and tbl_pid and tbl_pid != workspace_policy:
                continue
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

        cache = get_training_run_cache()
        self._current_series = []
        self._current_run_tables = {}
        color_idx = 0
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
    backend_changed = Signal(str)       # "sb3" | "isaac_lab"
    il_preset_requested = Signal(str)  # preset name
    # Phase B: persistent Isaac Review Session lifecycle
    open_viewer_clicked = Signal()
    close_viewer_clicked = Signal()

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

        # Training target selector — Local vs Cloud server(s).
        # Visible only for backends that support cloud (Isaac Lab).
        # Populated dynamically from the unified engine registry.
        self._target_combo = QComboBox(self)
        self._target_combo.setObjectName("trainingFloatTarget")
        self._target_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._target_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._target_combo.setToolTip("Training target: run locally or on a cloud server")
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        self._target_combo.setVisible(False)  # toggled in set_backend()
        hbox.addWidget(self._target_combo)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("trainingFloatSep")
        hbox.addWidget(sep)

        # Backend is now bound to the experiment at creation time and is
        # surfaced via ``set_backend()`` from TrainingWorkspaceWindow — no
        # in-bar combo. The selected backend is held in ``self._current_backend``
        # and exposed via ``selected_backend()`` for downstream consumers.
        self._current_backend: str = "sb3"
        self._isaac_lab_available = False
        try:
            from src.system.training.isaac_lab_backend import detect_isaac_lab
            self._isaac_lab_available = detect_isaac_lab() is not None
        except Exception:
            pass

        # IL Preset selector removed — Isaac Lab presets are surfaced in the
        # toolbar's "Load Asset" dropdown alongside SB3 training assets, with
        # the list filtered by the active backend so users only see entries
        # compatible with the current canvas. The il_preset_requested signal
        # remains so external callers can still drive a preset apply if they
        # know the name; the auto-emitting combo path is gone.

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

    # --- Backend (programmatic only — no UI selector) ---

    def selected_backend(self) -> str:
        """Return 'sb3', 'isaac_lab', or 'cloud'."""
        return str(getattr(self, "_current_backend", "sb3") or "sb3")

    # ------------------------------------------------------------------
    # Viewer button (Phase B)
    # ------------------------------------------------------------------

    # --- Target combo (Local / Cloud) ---

    def _populate_target_combo(self) -> None:
        """Rebuild the target combo items from the engine registry.

        Items:
          - "Local"                     (data = "local")    — if engine has local
          - "Cloud: <server_name>"      (data = "cloud:<name>")  — per server
        """
        combo = self._target_combo
        combo.blockSignals(True)
        combo.clear()

        backend = self._current_backend
        if backend not in ("isaac_lab",):
            # SB3: only local, no combo needed
            combo.blockSignals(False)
            return

        from src.system.engines.registry import get_engine_registry
        reg = get_engine_registry()
        engine_id = "isaac_lab"

        # Local entry
        local = reg.get_local(engine_id)
        has_local = bool(local.get("registered"))
        if has_local:
            combo.addItem("Local", "local")

        # Cloud entries
        for srv in reg.list_servers(engine_id):
            name = srv.get("name", srv.get("host", "?"))
            combo.addItem(f"Cloud: {name}", f"cloud:{name}")

        if combo.count() == 0:
            combo.addItem("No target available", "")
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)

        # Restore user preference
        self._load_target_preference()
        combo.blockSignals(False)

    def _load_target_preference(self) -> None:
        """Load the per-engine training target from user.ini."""
        engine_id = "isaac_lab"
        key = f"training_target_{engine_id}"
        preferred = str(
            self._config.get("PREFERENCES", key, fallback="", config_type="user") or ""
        ).strip()

        if not preferred:
            # Default: local if available, otherwise first cloud
            idx = self._target_combo.findData("local")
            if idx < 0:
                idx = 0
            self._target_combo.setCurrentIndex(idx)
            return

        idx = self._target_combo.findData(preferred)
        if idx >= 0:
            self._target_combo.setCurrentIndex(idx)
        else:
            # Saved preference no longer valid — fall back to first item
            self._target_combo.setCurrentIndex(0)

    def _on_target_changed(self, _index: int) -> None:
        """Persist the selected training target to user.ini per engine."""
        engine_id = "isaac_lab"
        key = f"training_target_{engine_id}"
        data = self._target_combo.currentData() or ""
        self._config.set("PREFERENCES", key, data, config_type="user")
        self._config.save_user_config()

        # Re-emit backend_changed so dispatch logic picks up the new target
        effective = self.selected_backend()
        self.backend_changed.emit(effective)

    def selected_target(self) -> str:
        """Return the raw target data: 'local', 'cloud:<name>', or ''."""
        return str(self._target_combo.currentData() or "")

    def selected_cloud_server_name(self) -> str:
        """Return the cloud server name if target is cloud, else ''."""
        t = self.selected_target()
        if t.startswith("cloud:"):
            return t[6:]
        return ""

    def set_backend(self, backend: str) -> None:
        """Programmatically set the backend bound to this float bar.

        Backend is *experiment-bound*: set at experiment creation/load time.
        For Isaac Lab, the target combo allows the user to choose Local vs
        Cloud execution without changing the underlying backend.
        """
        backend = str(backend or "sb3").strip().lower() or "sb3"
        if backend not in ("sb3", "isaac_lab", "cloud"):
            backend = "sb3"
        # Normalize: "cloud" as a backend is now expressed as
        # backend="isaac_lab" + target="cloud:<name>".
        if backend == "cloud":
            backend = "isaac_lab"
        self._current_backend = backend
        is_il = backend == "isaac_lab"
        # Isaac Lab always uses GPU
        if is_il:
            gpu_idx = self._device_combo.findData("cuda")
            if gpu_idx >= 0:
                self._device_combo.setCurrentIndex(gpu_idx)
            self._device_combo.setEnabled(False)
        else:
            self._device_combo.setEnabled(True)
        # Target combo visible only for engines with local+cloud options.
        self._target_combo.setVisible(is_il)
        if is_il:
            self._populate_target_combo()
        if hasattr(self, "_apply_measured_size"):
            self._apply_measured_size()
        self.backend_changed.emit(backend)

    def selected_backend(self) -> str:
        """Return the effective backend for training dispatch.

        Returns 'sb3', 'isaac_lab', or 'cloud' depending on the engine
        backend and the user's target selection.
        """
        base = str(getattr(self, "_current_backend", "sb3") or "sb3")
        if base == "isaac_lab":
            target = self.selected_target()
            if target.startswith("cloud:"):
                return "cloud"
        return base

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

    def set_page_mode(self, page: str) -> None:
        """Update border color to match page accent."""
        self._page_mode = page
        self.apply_theme()

    def apply_theme(self):
        page = getattr(self, "_page_mode", "training")
        bg        = get_color("training_float_bar_bg",     "#1e2d3d")
        if page == "training":
            border = get_color("tab_bg_training", "#F6D393")
        else:
            border = get_color("tab_bg_checked", "#A390FC")
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
        # Actor block — unified General nodes (6-section standardization).
        # Replaces: Robot/MJCF, IL Robot Asset, IL Contact Sensor, Joints Mapping.
        "Robot":             "robot",
        "Actor Setting":     "actor_setting",
        "Joint Init":        "joint_init",
        # §2 Scene block — unified General node.
        # Replaces: Scene Config (SB3), IL Terrain Config (IL).
        "Play Ground Setting": "play_ground_setting",
        # § Training Commands — unified sim/sim2real command schema.
        # Replaces: IL Velocity Command, IL Gamepad Input.
        "Training Commands":   "training_commands",
        # Layer A 鈥?Environment
        "Physics Config":    "physics_config",
        "Rewards":           "rewards",
        "Terminations":      "terminations",
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
        "SB3 Train":        "train",
        # Layer D 鈥?Validation & Export
        "Eval Config":      "eval_config",
        # Unified Export (replaces SB3 Export + IL Policy Exporter)
        "Export":           "export",
        "Vis Check":        "vis_check",
        # Isaac Lab granular nodes (IL_A – IL_F)
        "IL Actuator Config":     "il_actuator_config",
        "IL Observation":         "il_observation",
        "IL Policy Network":      "il_policy_network",
        # Unified IL trainer — training_mode param selects PPO / AMP_PPO.
        # Display name is "IL Trainer"; the legacy "IL PPO Trainer" /
        # "AMP PPO Trainer" display names also resolve to the same
        # node type so old canvases keep loading.
        "IL Trainer":             "il_ppo_trainer",
        "IL PPO Trainer":         "il_ppo_trainer",
        "AMP PPO Trainer":        "il_ppo_trainer",
    }

    # Node display names that belong to the Isaac Lab pipeline
    _ISAAC_LAB_NODES = {
        "IL Actuator Config",
        "IL Observation",
        "IL Policy Network", "IL Trainer",
    }
    # Node display names shared across all backends (General)
    _GENERAL_NODES = {
        "Start Point", "Eval Config", "Rewards", "Domain Rand", "Terminations",
        "MultiGated (Reward)",
        # AMP_design.yaml phase_2 — ReferenceMotionNode is consumed by both
        # SB3 (tracking + BC) and Isaac Lab (AMP discriminator). It must
        # be visible in the palette regardless of the active backend.
        "Reference Motion",
    }
    # Node display names that belong to the SB3 pipeline only
    _SB3_ONLY_NODES = {
        "Physics Config",
        "Task Config", "Obs & Action",
        "Init Pose",
        "Env Assembler", "Algorithm Config", "SB3 Train",
        "Vis Check",
    }

    def __init__(self, parent=None):
        # Counter must exist before super().__init__() which calls _place_initial_nodes
        self._training_node_counter: int = 1000
        super().__init__(parent)
        self._node_type_mapping = dict(self._TRAINING_NODE_MAP)

    # ------------------------------------------------------------------
    # History / dirty-state hooks
    # ------------------------------------------------------------------
    # Override the base hooks so the Training canvas uses its own
    # serialize_training_graph / load_training_graph pair for undo snapshots.
    # Parameter-edit touches are routed through _on_node_param_changed below
    # (the choke point for every training-node widget write).

    def _snapshot_for_history(self) -> dict:
        try:
            return self.serialize_training_graph(for_compiler=False)
        except Exception:
            return {}

    def _restore_from_history(self, snapshot: dict) -> None:
        try:
            self.load_training_graph(snapshot)
        except Exception:
            pass

    def _place_initial_nodes(self):
        """
        Place the default training template and auto-wire connections.

        Placement is a two-phase deferred process:
          Phase 1 (immediate): create all nodes at origin (0, 0) so their
                  internal widgets build and emit height_changed.
          Phase 2 (deferred 50 ms): read final rect().height() for every
                  node and reposition using the strict grid rule:
                    - horizontal gap = 40 px
                    - vertical gap   = 20 px
                  Then wire edges (ports are stable after reflow).
        """
        from PySide6.QtCore import QPointF, QTimer

        node_w = 268
        h_gap = 40
        v_gap = 20
        x0 = -916
        col_step = node_w + h_gap

        column_defs = [
            ["Robot", "Actor Setting", "Play Ground Setting", "Training Commands"],
            ["Physics Config", "Rewards", "Terminations", "Task Config",
             "Obs & Action", "Domain Rand"],
            ["Env Assembler", "Start Point"],
            ["Algorithm Config"],
            ["SB3 Train"],
            ["Export", "Vis Check"],
        ]

        # Phase 1: create nodes at origin — widget content builds now.
        placed = {}
        col_map: list = []  # [(col_idx, [name, ...])]
        for col_idx, names in enumerate(column_defs):
            col_map.append((col_idx, list(names)))
            for name in names:
                item = self.create_node(name, QPointF(0, 0))
                placed[name] = item

        def _layout_and_wire():
            # Phase 2: positions from settled heights.
            for col_idx, names in col_map:
                x = x0 + col_idx * col_step
                y = 0.0
                for name in names:
                    item = placed.get(name)
                    if item is not None:
                        item.setPos(x, y)
                        y += float(item.rect().height()) + v_gap

            # Wire edges — ports are positioned after reflow.
            robot = placed["Robot"]
            actor = placed["Actor Setting"]
            pg    = placed["Play Ground Setting"]
            cmds  = placed["Training Commands"]
            rw    = placed["Rewards"]
            tm    = placed["Terminations"]
            tc    = placed["Task Config"]
            oac   = placed["Obs & Action"]
            dr    = placed["Domain Rand"]
            pc    = placed["Physics Config"]
            ea    = placed["Env Assembler"]
            ba    = placed["Start Point"]
            alg   = placed["Algorithm Config"]
            tn    = placed["SB3 Train"]
            ex    = placed["Export"]
            vc    = placed["Vis Check"]
            pairs = [
                (robot, "robot_pipe",         "out", actor, "robot_pipe",         "in"),
                (actor, "actor_pipe",         "out", ea,    "actor_pipe",         "in"),
                (pg,    "scene_pipe",         "out", ea,    "scene_pipe",         "in"),
                (cmds,  "command_pipe",       "out", ea,    "command_pipe",       "in"),
                (rw,  "total_reward",         "out", tc,    "total_reward",       "in"),
                (tm,  "termination_config",   "out", tc,    "termination_config", "in"),
                (dr,  "domain_rand_config",   "out", ea,    "domain_rand_config", "in"),
                (pc,  "physics_config",       "out", ea,    "physics_config",     "in"),
                (tc,  "task_config",          "out", ea,    "task_config",        "in"),
                (oac, "obs_action_config",    "out", ea,    "obs_action_config",  "in"),
                (ea,  "env_config",           "out", tn,    "env_config",         "in"),
                (ba,  "checkpoint",           "out", alg,   "checkpoint",         "in"),
                (alg, "algo_config",          "out", tn,    "algo_config",        "in"),
                (tn,  "train_pipe",           "out", ex,    "train_pipe",         "in"),
                (tn,  "vis_check",            "out", vc,    "vis_check",          "in"),
            ]
            for src, s_slot, s_io, dst, d_slot, d_io in pairs:
                if src is None or dst is None:
                    continue
                out_p = self._get_node_port(src, s_slot, s_io)
                in_p  = self._get_node_port(dst, d_slot,  d_io)
                if out_p and in_p:
                    self._create_connection(out_p, in_p)

        QTimer.singleShot(50, _layout_and_wire)

    def _place_isaac_lab_initial_nodes(self):
        """Place the Go2 Flat Terrain Velocity Tracking IL graph template.

        Two-phase deferred layout (same pattern as SB3):
          Phase 1 (immediate): create all nodes at origin, apply IL
                  parameter overrides so widgets build with final content.
          Phase 2 (deferred 50 ms): read settled rect().height() and
                  reposition using grid rule, then wire edges.
        """
        from PySide6.QtCore import QPointF, QTimer

        node_w = 268
        h_gap = 40
        v_gap = 20
        col_step = node_w + h_gap
        x0 = -800

        column_defs = [
            ["Robot", "Actor Setting", "Play Ground Setting", "Training Commands"],
            ["IL Observation"],
            ["Rewards", "Terminations", "Domain Rand"],
            ["IL Policy Network", "IL Trainer", "Start Point"],
            ["Export"],
        ]

        # Phase 1: create all nodes at origin.
        placed = {}
        col_map: list = []
        for col_idx, names in enumerate(column_defs):
            col_map.append((col_idx, list(names)))
            for name in names:
                item = self.create_node(name, QPointF(0, 0))
                placed[name] = item

        # Apply IL parameter overrides immediately so widget content
        # (registry editors) builds with the correct backend/items and
        # emits accurate height_changed before Phase 2 reads heights.
        import json as _json
        from src.system.training.task_module_registry import (
            default_il_reward_terms,
            default_il_termination_conditions,
        )
        _il_node_overrides: Dict[str, Dict[str, str]] = {
            "Rewards": {
                "backend": "isaac_lab",
                "reward_terms": _json.dumps(default_il_reward_terms()),
            },
            "Domain Rand": {"backend": "isaac_lab"},
            "Terminations": {
                "backend": "isaac_lab",
                "termination_conditions": _json.dumps(
                    default_il_termination_conditions()
                ),
            },
        }
        for _name, _overrides in _il_node_overrides.items():
            _item = placed.get(_name)
            if _item is not None:
                _item.load_parameters(_overrides)

        def _layout_and_wire():
            # Phase 2: reposition from settled heights.
            for col_idx, names in col_map:
                x = x0 + col_idx * col_step
                y = 0.0
                for name in names:
                    item = placed.get(name)
                    if item is not None:
                        item.setPos(x, y)
                        y += float(item.rect().height()) + v_gap

            # Wire edges.
            def n(key):
                return placed.get(key)

            robot    = n("Robot")
            actor    = n("Actor Setting")
            pg       = n("Play Ground Setting")
            cmds     = n("Training Commands")
            obs      = n("IL Observation")
            rewards  = n("Rewards")
            term     = n("Terminations")
            dr       = n("Domain Rand")
            network  = n("IL Policy Network")
            trainer  = n("IL Trainer")
            export   = n("Export")
            start_pt = n("Start Point")

            edges = [
                (robot, "robot_pipe", "out", actor, "robot_pipe", "in"),
                (actor, "actor_pipe", "out", obs,     "actor_pipe", "in"),
                (actor, "actor_pipe", "out", rewards, "actor_pipe", "in"),
                (actor, "actor_pipe", "out", dr,      "actor_pipe", "in"),
                (cmds,  "command_pipe", "out", obs,     "command_pipe", "in"),
                (cmds,  "command_pipe", "out", rewards, "command_pipe", "in"),
                (pg,    "scene_pipe",   "out", trainer, "scene_pipe",    "in"),
                (obs,     "obs_vector",         "out", network, "obs_vector",         "in"),
                (network, "policy",             "out", trainer, "policy",             "in"),
                (rewards, "total_reward",       "out", trainer, "total_reward",       "in"),
                (term,    "termination_config", "out", trainer, "termination_config", "in"),
                (dr,      "domain_rand_config", "out", trainer, "domain_rand_config", "in"),
                (obs,     "obs_vector",         "out", trainer, "obs_vector",         "in"),
                (start_pt, "checkpoint",        "out", trainer, "checkpoint",         "in"),
                (trainer, "train_pipe",         "out", export,  "train_pipe",         "in"),
            ]
            for src, s_slot, s_io, dst, d_slot, d_io in edges:
                if src is None or dst is None:
                    continue
                out_p = self._get_node_port(src, s_slot, s_io)
                in_p  = self._get_node_port(dst, d_slot, d_io)
                if out_p and in_p:
                    self._create_connection(out_p, in_p)

        QTimer.singleShot(50, _layout_and_wire)

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
                # Unified 6-section nodes (General / §1–§3)
                RobotNode,
                ActorSettingNode,
                JointInitNode,
                PlayGroundSettingNode,
                TrainingCommandsNode,
                # SB3 core pipeline
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
                ReferenceMotionNode,   # Reference Motion workspace (§3 Aux, future)
                InitPoseNode,          # SB3 keyframe/reference init
                MultiGatedRewardNode,
                # Isaac Lab granular nodes (after cleanup — only active ones)
                ILObservationNode,
                ILPolicyNetworkNode,
                ILPPOTrainerNode,  # unified — training_mode selects PPO/AMP_PPO
            )
            _class_map = {
                # Unified 6-section nodes
                "robot":                RobotNode,
                "actor_setting":        ActorSettingNode,
                "joint_init":           JointInitNode,
                "play_ground_setting":  PlayGroundSettingNode,
                "training_commands":    TrainingCommandsNode,
                # SB3 core pipeline
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
                "reference_motion":     ReferenceMotionNode,
                "init_pose":            InitPoseNode,
                "multigated_reward":    MultiGatedRewardNode,
                # Isaac Lab granular nodes
                "il_observation":         ILObservationNode,
                "il_policy_network":      ILPolicyNetworkNode,
                "il_ppo_trainer":         ILPPOTrainerNode,
                # Legacy amp_ppo_trainer canvases load as the unified IL
                # trainer with training_mode pre-set to AMP_PPO.
                "amp_ppo_trainer":        ILPPOTrainerNode,
                # Back-compat aliases — same class, different canvas label
                "il_rewards":             RewardsNode,
                "il_termination_config":  TerminationsNode,
                "il_domain_rand_config":  DomainRandNode,
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

    def _swap_node_type(self, old_item, new_type: str, new_display: str) -> None:
        """Replace a node with a different type at the same position.

        Preserves connections on output ports with matching slot names
        (e.g. velocity_command output exists on both il_velocity_cmd
        and il_gamepad_input).
        """
        from bin.nodes.training_node_items import TrainingNodeItem
        pos = old_item.pos()

        # Collect outgoing connections from matching output slots
        saved_edges = []  # [(out_slot, in_port)]
        for key, port in list(old_item._ports.items()):
            if not key.endswith(":out"):
                continue
            slot = key.rsplit(":out", 1)[0]
            for conn in list(port.data(2) or []):
                try:
                    in_port = getattr(conn, "in_port", None)
                    if in_port is not None:
                        saved_edges.append((slot, in_port))
                except Exception:
                    pass

        # Remove old node
        self._remove_node_item(old_item)

        # Create new node
        new_item = self.create_node(new_display, pos)
        if new_item is None:
            return

        # Rewire saved output connections
        for out_slot, in_port in saved_edges:
            out_port = new_item.get_port(out_slot, "out") if isinstance(new_item, TrainingNodeItem) else None
            if out_port is not None and in_port is not None:
                try:
                    self._create_connection(out_port, in_port)
                except Exception:
                    pass

    def _remove_node_item(self, item) -> None:
        """Remove a TrainingNodeItem and all its connections from the scene."""
        from bin.pages.canvas.graph_scene import isValid
        # Disconnect all ports
        for key, port in list(getattr(item, "_ports", {}).items()):
            for conn in list(port.data(2) or []):
                try:
                    if conn and isValid(conn) and conn.scene() is not None:
                        self._detach_connection(conn)
                        self.removeItem(conn)
                except Exception:
                    pass
        try:
            self.removeItem(item)
        except Exception:
            pass

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

    def reset_to_default_template(self, backend: str = "sb3") -> None:
        """Clear the scene and recreate the default template for *backend*.

        Wrapped in ``suppress_history`` + (conditional) ``reset_history`` so
        top-level resets (e.g. user switching backend) install the fresh
        template as the clean baseline, while resets called from inside an
        undo/redo frame (``_history_suppress_depth > 0``) do not clobber the
        enclosing history stack.
        """
        seed_baseline = self._history_suppress_depth == 0
        with self.suppress_history():
            self.clear()
            self._training_node_counter = 1000
            if backend == "isaac_lab":
                self._place_isaac_lab_initial_nodes()
            else:
                self._place_initial_nodes()
        if seed_baseline:
            self.reset_history()

    # ------------------------------------------------------------------
    # Step 4 鈥?Canvas persistence
    # ------------------------------------------------------------------

    # Emitted when the user requests training or environment review from TrainNode
    train_requested = Signal(dict)
    review_requested = Signal(dict)
    export_review_requested = Signal(dict)
    il_export_review_requested = Signal(dict)
    scene_preview_requested = Signal(dict)
    init_pose_preview_requested = Signal(dict)
    # Export Node §3 — unified Review launch signal. Payload:
    #   {"backend": "mujoco"|"isaac_sim"|..., "scene_id": str,
    #    "bundle_name": str, "version": str, "overwrite": bool}
    # Workspace slot runs SafetyReview then dispatches to the matching
    # review-session subprocess (isaac_review_session / il_review_session).
    review_launch_requested = Signal(dict)
    node_param_changed = Signal(str, str, str)   # node_type, key, value
    # Emitted on any mutation that could change the compiled TrainingJobSpec:
    # param writes, edge create, edge remove, node delete. Consumers (the
    # TrainingWorkspaceWindow) route this into TrainingContext.refresh() as
    # the single sanctioned refresh trigger for the 6-section backend.
    graph_dirty = Signal()

    def _on_node_param_changed(self, node_type: str, key: str, value: str) -> None:
        """Called by TrainingNodeItem via _notify_scene_param_changed on any param write."""
        self.node_param_changed.emit(node_type, key, value)
        self.graph_dirty.emit()
        # Training-canvas dirty-state hook: every widget write on every
        # training node funnels through here, making it the single
        # parameter-edit touch point for the Training canvas. No-op while
        # ``suppress_history`` is active (load / preset / backend restore).
        self._touch_content()

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
            return {"schema": "training_canvas_v1", "backend": getattr(self, "_current_backend", "sb3"), "nodes": all_nodes, "edges": all_edges}

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

        return {"schema": "training_canvas_v1", "backend": getattr(self, "_current_backend", "sb3"), "nodes": nodes, "edges": edges}

    def load_training_graph(self, data: dict) -> None:
        """Clear the canvas and restore from serialize_training_graph() output.

        The entire body is wrapped in ``suppress_history`` so none of the
        node/edge recreations push undo entries; top-level (user-initiated)
        loads then seed a fresh clean baseline via ``reset_history``. Loads
        issued from undo/redo skip that reset — they already run inside an
        outer suppress_history frame, signalled via ``_history_suppress_depth``.
        """
        if data.get("schema") != "training_canvas_v1":
            raise ValueError(f"Unsupported schema: {data.get('schema')!r}")

        seed_baseline = self._history_suppress_depth == 0
        with self.suppress_history():
            # If this is a layered envelope, extract the active backend's layer
            if "layers" in data and isinstance(data["layers"], dict):
                backend = getattr(self, "_current_backend", "sb3")
                layer = data["layers"].get(backend) or {}
                if layer.get("schema") == "training_canvas_v1":
                    data = layer
                else:
                    # No layer for this backend — start with empty canvas
                    data = {"schema": "training_canvas_v1", "nodes": [], "edges": []}

            # Build a display-name -> internal key lookup for create_node
            _type_to_display = {v: k for k, v in self._TRAINING_NODE_MAP.items()}

            # Clear existing items
            self.clear()
            self._training_node_counter = 1000

            # Phase 1: recreate nodes and let each item finalize its widget
            # layout before we attempt port lookup. Ports resolve by
            # deterministic slot index inside the item, so the node itself
            # has to be fully laid out or _get_node_port returns None and
            # the edge is silently dropped.
            id_map: Dict[str, object] = {}
            for node_data in data.get("nodes", []):
                node_type = node_data["node_type"]
                # Legacy amp_ppo_trainer aliases to the unified IL trainer
                # with training_mode pre-set to AMP_PPO. Rewrite the
                # serialized dict in-place so downstream reads (validators,
                # compiler) only see the canonical node type.
                if node_type == "amp_ppo_trainer":
                    node_type = "il_ppo_trainer"
                    node_data["node_type"] = "il_ppo_trainer"
                    _p = dict(node_data.get("parameters") or {})
                    _p.setdefault("training_mode", "AMP_PPO")
                    node_data["parameters"] = _p
                # Prefer the serialized display_name (exact round-trip) over
                # the reversed _type_to_display map — the map has ambiguous
                # many-to-one entries (e.g. IL Trainer / IL PPO Trainer /
                # AMP PPO Trainer all map to il_ppo_trainer) so the reverse
                # lookup can pick the wrong display name, which changes the
                # node's visual label on reload.
                display_name = (
                    node_data.get("display_name")
                    or _type_to_display.get(node_type)
                    or node_type
                )
                x, y = node_data.get("pos", [0.0, 0.0])
                item = self.create_node(display_name, QPointF(x, y))
                if item is not None:
                    item.load_parameters(node_data.get("parameters", {}))
                    id_map[node_data["id"]] = item

        edges_to_wire = list(data.get("edges", []))

        def _wire_deferred() -> None:
            # Scene may have been torn down between scheduling and firing.
            try:
                if self.views() is None:
                    return
            except RuntimeError:
                return
            with self.suppress_history():
                for edge in edges_to_wire:
                    src = id_map.get(edge.get("src_id"))
                    dst = id_map.get(edge.get("dst_id"))
                    if src is None or dst is None:
                        continue
                    try:
                        out_p = self._get_node_port(src, edge["src_slot"], "out")
                        in_p  = self._get_node_port(dst, edge["dst_slot"], "in")
                    except RuntimeError:
                        return
                    if out_p and in_p:
                        self._create_connection(out_p, in_p)
            if seed_baseline:
                self.reset_history()

        # Defer edge wiring one tick so QGraphicsLayout finishes widget
        # sizing — matches _place_initial_nodes' 50ms pattern. This is the
        # "load protocol": nodes materialize, items settle, then edges wire.
        QTimer.singleShot(50, _wire_deferred)

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

        # Fan-in ports accept unlimited connections — skip eviction
        in_meta = in_port.data(self._port_meta_key) or {}
        max_conn = in_meta.get("max_connections", 1)
        if max_conn == -1:
            # Unlimited — just verify type compatibility, then connect
            if self._can_connect_ports(out_port, in_port, ignore_connection=existing_in[0]):
                super()._finish_connection(target_port)
            else:
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
        one upstream source — unless the port is a fan-in port
        (max_connections == -1), in which case unlimited connections
        are allowed.
        """
        try:
            meta = in_port.data(self._port_meta_key) or {}
            max_conn = meta.get("max_connections", 1)
            # Fan-in ports (-1) accept unlimited connections — skip eviction
            if max_conn != -1:
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
        self.graph_dirty.emit()

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
        self.graph_dirty.emit()

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
        deleted = False
        for item in list(self.selectedItems()):
            if not isinstance(item, TrainingNodeItem):
                continue
            try:
                self._delete_node_connections(item)
                if item.scene() is not None:
                    self.removeItem(item)
                deleted = True
            except Exception:
                pass
        if deleted:
            self.node_param_changed.emit("__node_deleted__", "", "")
            self.graph_dirty.emit()


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
    publish_requested = Signal(str)   # emits policy_id to publish to shared

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
            scope = str(entry.get("scope") or "legacy")
            published = bool(entry.get("published", False))

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
            leaf_row = self._build_export_row(
                leaf_label, policy_id, scope=scope, published=published
            )
            self._tree.setItemWidget(leaf, 0, leaf_row)
            leaf.setSizeHint(0, leaf_row.sizeHint())
            if current_policy_id and (policy_id == current_policy_id or leaf_label == current_policy_id):
                current_item = leaf

        self._tree.expandAll()
        if current_item is not None:
            self._tree.setCurrentItem(current_item)
        self.apply_theme()

    def _build_export_row(
        self,
        label_text: str,
        policy_id: str,
        scope: str = "legacy",
        published: bool = False,
    ) -> QWidget:
        row_height = 26
        row = QWidget(self._tree)
        row.setObjectName("trainingExportRow")
        row.setMinimumHeight(row_height)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 0, 2, 0)
        layout.setSpacing(4)

        # Scope badge
        if scope in ("local", "shared"):
            badge_text = "LOCAL" if scope == "local" else "GLOB"
            if published and scope == "local":
                badge_text = "PUB"
            badge = QLabel(badge_text)
            badge.setObjectName("trainingExportScopeBadge")
            badge.setFixedWidth(34)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)

        label = QLabel(label_text)
        label.setObjectName("trainingExportRowLabel")
        label.setToolTip(label_text)
        layout.addWidget(label, 1, Qt.AlignmentFlag.AlignVCenter)

        # Publish to Global button (only for local, unpublished checkpoints)
        if scope == "local" and not published:
            pub_btn = QPushButton("\u2191")  # ↑ arrow
            pub_btn.setObjectName("trainingExportPublishBtn")
            pub_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            pub_btn.setFixedSize(row_height, row_height)
            pub_btn.setToolTip("Publish to Global")
            pub_btn.clicked.connect(
                lambda _checked=False, pid=policy_id: self.publish_requested.emit(pid)
            )
            layout.addWidget(pub_btn, 0, Qt.AlignmentFlag.AlignVCenter)

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
        for btn in self.findChildren(QPushButton, "trainingExportPublishBtn"):
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {palette_text}; border: none; border-radius: 1px; padding: 0px; min-width: 0px; min-height: 0px; max-width: 26px; max-height: 26px; font-size: 14px; }}"
                f"QPushButton:hover {{ background: {palette_hover}; }}"
            )
        badge_bg = get_color("training_scope_badge_bg", "#374151")
        badge_text_color = get_color("training_scope_badge_text", "#9ca3af")
        for badge in self.findChildren(QLabel, "trainingExportScopeBadge"):
            badge.setStyleSheet(
                f"QLabel {{ background: {badge_bg}; color: {badge_text_color}; border-radius: 3px; font-size: 9px; font-weight: bold; padding: 1px 2px; }}"
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
    # Backend affinity: "general" nodes appear in both SB3 and Isaac Lab,
    # "sb3" nodes only in SB3 mode, "isaac" nodes only in Isaac Lab mode.
    _NODE_BACKEND: Dict[str, str] = {
        # Actor block — shared across all backends (6-section standardization)
        "Robot":               "general",
        "Actor Setting":       "general",
        "Joint Init":          "general",
        # §2 Scene — unified General node
        "Play Ground Setting": "general",
        # § Training Commands — unified General node
        "Training Commands":   "general",
        # General — shared across all backends
        "Start Point":         "general",
        "Eval Config":         "general",
        # SB3-only — training pipeline
        "Algorithm Config":    "sb3",
        "SB3 Train":           "sb3",
        "Export":              "general",
        "Vis Check":           "sb3",
        # SB3-only — environment & assembly
        "Physics Config":      "sb3",
        "Rewards":             "general",
        "Terminations":        "general",
        "Task Config":         "sb3",
        "Domain Rand":         "general",
        "Obs & Action":        "sb3",
        "Reference Motion":    "general",
        "Init Pose":           "sb3",
        "Env Assembler":       "sb3",
        # Isaac Lab granular nodes
        "IL Actuator Config":     "isaac",
        "IL Observation":         "isaac",
        "IL Policy Network":      "isaac",
        "IL Trainer":             "isaac",
    }

    # Functional palette categories — groups nodes by what they
    # describe, not by which training backend they target. The same
    # node can be useful to both SB3 and Isaac Lab users and the
    # backend filter (filter_by_backend) still hides the ones that
    # don't apply to the active canvas.
    _NODE_CATEGORY: Dict[str, str] = {
        # ── Robot Asset — physical robot definition + init pose
        "Robot":               "Robot Asset",
        "Actor Setting":       "Robot Asset",
        "Joint Init":          "Robot Asset",
        "IL Actuator Config":  "Robot Asset",
        "Init Pose":           "Robot Asset",
        # ── Terrains — scene / playground / world physics
        "Play Ground Setting": "Terrains",
        "Physics Config":      "Terrains",
        # ── Policy — everything the learner observes, acts on, or
        # is rewarded for. Observation, action, rewards, terminations,
        # domain randomisation, commands, reference motion, network.
        "IL Observation":      "Policy",
        "Obs & Action":        "Policy",
        "IL Policy Network":   "Policy",
        "Rewards":             "Policy",
        "Terminations":        "Policy",
        "Domain Rand":         "Policy",
        "Task Config":         "Policy",
        "Training Commands":   "Policy",
        "Reference Motion":    "Policy",
        # ── Trainer — training loop config, algorithm, env assembly,
        # warm-start, eval, visualization.
        "Start Point":         "Trainer",
        "Algorithm Config":    "Trainer",
        "SB3 Train":           "Trainer",
        "IL Trainer":          "Trainer",
        "Env Assembler":       "Trainer",
        "Eval Config":         "Trainer",
        "Vis Check":           "Trainer",
        # ── Export — bundle output + review environment
        "Export":              "Export",
    }
    _SECTION_ORDER = [
        "Robot Asset",
        "Terrains",
        "Policy",
        "Trainer",
        "Export",
    ]
    _DEFAULT_SECTION = "Policy"

    # Legacy display aliases that still resolve in _TRAINING_NODE_MAP
    # so saved canvases keep loading, but must NOT be advertised as
    # palette entries — the unified "IL Trainer" is the only canonical
    # display name for the il_ppo_trainer node type.
    _PALETTE_HIDDEN_DISPLAY_NAMES = {
        "IL PPO Trainer",
        "AMP PPO Trainer",
    }

    @classmethod
    def _build_palette_sections(cls) -> list:
        """
        Derive palette sections from TrainingGraphScene._TRAINING_NODE_MAP.

        Legacy display-name aliases (IL PPO Trainer / AMP PPO Trainer)
        are filtered via ``_PALETTE_HIDDEN_DISPLAY_NAMES`` so the palette
        only advertises the canonical unified names even though the
        aliases still resolve at canvas-load time for old saves.

        To avoid advertising the same logical node twice when two
        display names point at the same node_type, we dedupe on
        node_type and keep the first (canonical) display name we see.
        """
        buckets: Dict[str, list] = {s: [] for s in cls._SECTION_ORDER}
        seen_node_types: set = set()
        for display_name, node_type in TrainingGraphScene._TRAINING_NODE_MAP.items():
            if display_name in cls._PALETTE_HIDDEN_DISPLAY_NAMES:
                continue
            if node_type in seen_node_types:
                continue
            seen_node_types.add(node_type)
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

    def filter_by_backend(self, backend: str) -> None:
        """Show/hide palette sections and nodes based on the active backend.

        backend: "sb3" or "isaac_lab"

        Sections are now functional groupings (Robot Asset / Terrains /
        Policy / Trainer / Export) that span both backends, so the
        backend filter fires at the child (node) level via
        ``_NODE_BACKEND``. A section is hidden only when every child is
        filtered out for the active backend.
        """
        tree = self._palette_list
        for i in range(tree.topLevelItemCount()):
            section_item = tree.topLevelItem(i)
            if section_item is None:
                continue
            any_visible = False
            for ci in range(section_item.childCount()):
                child = section_item.child(ci)
                if child is None:
                    continue
                node_name = child.text(0)
                affinity = self._NODE_BACKEND.get(node_name, "general")
                visible = (
                    affinity == "general"
                    or (affinity == "sb3" and backend == "sb3")
                    or (affinity == "isaac" and backend == "isaac_lab")
                )
                child.setHidden(not visible)
                if visible:
                    any_visible = True
            section_item.setHidden(not any_visible)

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
    il_export_review_requested = Signal(dict)
    scene_preview_requested = Signal(dict)
    init_pose_preview_requested = Signal(dict)
    # Export Node §3 — unified Review launch signal forwarded from the scene.
    review_launch_requested = Signal(dict)

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
        self._scene.il_export_review_requested.connect(self.il_export_review_requested)
        self._scene.scene_preview_requested.connect(self.scene_preview_requested)
        self._scene.init_pose_preview_requested.connect(self.init_pose_preview_requested)
        self._scene.review_launch_requested.connect(self.review_launch_requested)
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
        project_name: str = "",
        project_id: str = "",
    ):
        super().__init__(parent)
        self._embedded = embedded
        self._project_name = str(project_name or "").strip()
        self._active_project_id = str(project_id or "").strip()
        if not self._embedded:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self._policy_id = str(policy_id or "").strip()
        self._selected_policy_id = self._policy_id
        self._workspace_name = self._policy_id
        self._source_experiment_id = ""
        self._source_info = self._load_checkpoint_source_info(self._selected_policy_id)
        parent_policy_id = str(self._source_info.get("parent_policy_id", "") or "").strip()
        source_experiment_id = str(self._source_info.get("experiment_id", "") or "").strip()
        if parent_policy_id:
            self._workspace_name = parent_policy_id
        if source_experiment_id:
            self._source_experiment_id = source_experiment_id
        self._initial_robot_type = str(initial_robot_type or "")
        self._initial_runtime_scenario = dict(initial_runtime_scenario or {})
        self._canvas_source_state: str = "default_template"
        self._active_backend: str = "sb3"
        self._canvas_layers: Dict[str, dict] = {}
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
        self._canvas.il_export_review_requested.connect(self._on_il_export_review_requested)
        # Export Node §3 — unified Review launch signal.
        self._canvas.review_launch_requested.connect(self._on_review_launch_requested)
        self._canvas.scene_preview_requested.connect(self._on_scene_preview_requested)
        self._canvas.init_pose_preview_requested.connect(self._on_init_pose_preview_requested)
        self._canvas.float_bar.start_clicked.connect(self._on_ctrl_start_clicked)
        self._canvas.float_bar.stop_clicked.connect(self._on_ctrl_stop_clicked)
        self._canvas.float_bar.backend_changed.connect(self._on_backend_switched)
        self._canvas.float_bar.il_preset_requested.connect(self._on_il_preset_apply)
        # Phase B: Open/Close Viewer toggle wired to IsaacReviewSession singleton
        self._canvas.float_bar.open_viewer_clicked.connect(self._on_open_viewer_clicked)
        self._canvas.float_bar.close_viewer_clicked.connect(self._on_close_viewer_clicked)
        # Bind state_changed signal from the singleton manager → button UI.
        # We do this here (not in __init__) because float_bar is part of
        # _canvas which gets rebuilt on every backend swap.
        try:
            from src.system.training.isaac_review_session import get_review_session
            review_session = get_review_session()
            review_session.state_changed.connect(self._on_review_session_state_changed)
            review_session.error_occurred.connect(self._on_review_session_error)
            review_session.checkpoint_loaded.connect(self._on_review_checkpoint_loaded)
            review_session.stage_changed.connect(self._on_review_stage_changed)
        except Exception:
            pass
        self._canvas.scene.node_param_changed.connect(self._on_canvas_node_param_changed)
        self._canvas.scene.node_param_changed.connect(lambda *_: self._on_il_preset_dirty())
        # Standardized 6-section backend: every param write + edge mutation
        # funnels through TrainingGraphScene.graph_dirty into the single
        # refresh entry point below.
        self._canvas.scene.graph_dirty.connect(self._on_training_graph_dirty)
        # Save button visual driven by the scene's dirty signal so a single
        # source of truth (graph_scene.is_dirty()) controls both the
        # FilesRow tab dot and the Save button colour.
        self._canvas.scene.content_changed.connect(self._on_scene_dirty_for_save_btn)

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
        self._export_browser_panel.publish_requested.connect(self._publish_checkpoint_to_shared)
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
        self._sidebar.set_panel_widget("canvas", self._canvas_browser_panel, "Experiments")
        self._sidebar.set_panel_widget("exports", self._export_browser_panel, "Exports")
        self._sidebar.set_panel_widget("nodes", self._palette_panel, "Config Nodes")
        self._sidebar.panel_changed.connect(self._on_sidebar_panel_changed)

        self._workspace_left_box = QWidget()
        self._workspace_left_box.setObjectName("trainingWorkspaceLeftBox")
        workspace_left_layout = QVBoxLayout(self._workspace_left_box)
        workspace_left_layout.setContentsMargins(0, 0, 0, 0)
        workspace_left_layout.setSpacing(0)

        # Toolbar row is the primary workspace control surface. The
        # progress bar / step label / state label are inlined into this
        # row (right of Save), so there is no longer a separate strip.
        workspace_left_layout.addWidget(self._build_toolbar_row())

        workspace_left_layout.addWidget(self._canvas, 1)

        from bin.pages.training.training_panel import TrainingPanel

        self._training_panel = TrainingPanel()
        self._training_panel.setObjectName("trainingLogPanel")
        self._training_panel.setMinimumWidth(320)
        self._training_panel.review_btn.clicked.connect(self._on_il_review_clicked)

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
                meta = self._ws_store.load_workspace(self._workspace_name)
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
        return str(
            self._workspace_name
            or self._selected_policy_id
            or self._project_name
            or "WorkSpace"
        ).strip() or "WorkSpace"

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
            from pathlib import Path
            from src.system.service.checkpoint_registry import CheckpointRegistry
            reg = CheckpointRegistry()
            entry = reg.get(resolved_policy_id)
            return CheckpointRegistry._read_source_info(Path(entry.bundle_path))
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

    # ------------------------------------------------------------------
    # Standardized 6-section backend — TrainingContext refresh pipeline
    # ------------------------------------------------------------------

    def _get_training_context(self):
        """Return this workspace's TrainingContext (lazily constructed)."""
        ctx = getattr(self, "_training_context", None)
        if ctx is None:
            from src.system.training.training_context import TrainingContext
            ctx = TrainingContext()
            self._training_context = ctx
        return ctx

    def _on_training_graph_dirty(self) -> None:
        """Single sanctioned refresh trigger for the 6-section training backend.

        Fires on every canvas mutation that could change the compiled
        TrainingJobSpec: node param writes, edge create, edge remove,
        node delete. Serializes the current graph and asks
        ``TrainingContext.refresh`` to keep the 6 sections in sync so
        widgets that read from ``ctx`` (Start Point compat panel, output
        file list, etc.) see current data.

        **Does NOT run SafetyReview.** SafetyReview is a one-shot gate
        that fires on Train / Launch Review button clicks — that's the
        ONLY place issues get surfaced, red borders painted, or training
        actually blocked. Canvas editing is silent; compile errors from
        refresh are swallowed internally (the next safety_review call
        at click time will re-compile and surface them then).
        """
        try:
            scene = self._canvas.scene
        except Exception:
            return
        try:
            graph = scene.serialize_training_graph(for_compiler=True)
        except Exception:
            return
        # Empty-graph guard: prewarm / shell cache / workspace with no
        # loaded experiment has nothing worth refreshing.
        if not (graph.get("nodes") or []):
            return
        ctx = self._get_training_context()
        try:
            ctx.refresh(
                graph,
                policy_id=str(getattr(self, "_selected_policy_id", "") or ""),
                experiment_id=str(getattr(self, "_selected_experiment_id", "") or ""),
            )
        except Exception:
            # Refresh is best-effort; the next SafetyReview gate (on
            # Train / Review click) will raise the real blocking error.
            pass

    # ------------------------------------------------------------------
    # SafetyReview — canvas highlight + blocking gate
    # ------------------------------------------------------------------

    def _run_safety_review(self, *, blocking: bool) -> bool:
        """Run a full SafetyReview pass. Returns True iff no errors.

        ``blocking=True`` is used by the Train / Launch Review click
        handlers — a False return means "do NOT proceed, the user
        needs to fix the highlighted nodes first".

        ``blocking=False`` is advisory — callers apply canvas
        highlights but do not refuse to run. Currently nothing uses
        this form directly (graph_dirty goes through
        ``_on_training_graph_dirty`` which always runs a non-blocking
        review), but the parameter is kept so Launch Review can call
        with blocking=True symmetrically to Train.
        """
        try:
            scene = self._canvas.scene
            graph = scene.serialize_training_graph(for_compiler=True)
        except Exception as exc:
            log_error(f"[SafetyReview] graph serialize failed: {exc}")
            return False

        ctx = self._get_training_context()
        result = ctx.safety_review(
            graph,
            policy_id=str(getattr(self, "_selected_policy_id", "") or ""),
            experiment_id=str(getattr(self, "_selected_experiment_id", "") or ""),
        )
        self._apply_safety_result(result, log_errors=True)

        if blocking and not result.ok:
            n_err = len(result.errors)
            log_error(
                f"[SafetyReview] Training blocked — {n_err} error(s) found. "
                f"See highlighted nodes on the canvas and the log above."
            )
            return False
        return True

    def _apply_safety_result(self, result, *, log_errors: bool) -> None:
        """Paint canvas highlights + log issues from a SafetyReview result.

        Always clears any previous highlights first so nodes stop
        glowing red once the user fixes a problem.
        """
        from src.system.training.training_context import (
            issues_by_node, worst_severity,
        )

        self._clear_safety_highlights()

        by_node = issues_by_node(result)
        for node_id, issues in by_node.items():
            if not node_id:
                continue   # unscoped — log only
            severity = worst_severity(issues)
            self._set_node_safety_status(node_id, severity)

        # Log every error + warning. Error logs use log_error so they
        # surface in the top-strip; warnings stay in the background
        # log panel.
        if log_errors:
            for issue in result.errors:
                log_error(f"[SafetyReview] {issue.code}: {issue.message}")
                if issue.fix_hint:
                    log_error(f"[SafetyReview]   → fix: {issue.fix_hint}")
        for issue in result.warnings:
            log_warning(f"[SafetyReview] {issue.code}: {issue.message}")

    def _clear_safety_highlights(self) -> None:
        """Drop the SafetyReview overlay border from every training node."""
        try:
            scene = self._canvas.scene
        except Exception:
            return
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in scene.items():
            if isinstance(item, TrainingNodeItem):
                try:
                    item.set_safety_status(None)
                except Exception:
                    pass

    def _set_node_safety_status(self, node_id: str, severity: str) -> None:
        """Find the TrainingNodeItem with matching logic node id and
        paint the SafetyReview overlay. Silent on lookup failure —
        a missing node means the graph changed mid-review, which the
        next refresh pass will reconcile."""
        try:
            scene = self._canvas.scene
        except Exception:
            return
        from bin.nodes.training_node_items import TrainingNodeItem
        for item in scene.items():
            if not isinstance(item, TrainingNodeItem):
                continue
            logic = getattr(item, "_logic_node", None)
            if logic is None:
                continue
            if str(getattr(logic, "node_id", "")) == str(node_id):
                try:
                    item.set_safety_status(severity)
                except Exception:
                    pass
                return

    # ------------------------------------------------------------------
    # Export Node §3 — Review launch dispatch
    # ------------------------------------------------------------------

    def _on_review_launch_requested(self, payload: dict) -> None:
        """Handle the ``review_launch_requested`` signal emitted by
        the Export node's Launch Review button.

        Pipeline:
          1. Re-run SafetyReview (defence in depth — the widget
             already ran it, but a canvas edit between click and slot
             execution is in principle possible).
          2. Resolve the bundle path from ``bundle_name`` +
             ``version`` + ``overwrite`` flag.
          3. Dispatch to the right review session subprocess based
             on the requested backend id:

                mujoco   → existing SB3 export_review path
                isaac_sim → existing IL export_review path
                newton   → not yet wired — log a clear warning

        The subprocess lifetime is managed by the existing
        isaac_review_session / il_review_session modules; this slot
        is just the dispatcher.
        """
        if not isinstance(payload, dict):
            log_warning("[Review] review_launch_requested: invalid payload type")
            return

        # SafetyReview is SB3-only — the SB3 TrainingSpecCompiler
        # validates against SB3 node types that don't exist on Isaac
        # Lab canvases.  Running it on an IL graph always fails with
        # a spurious "missing required node types" error.
        _active_be = str(getattr(self, "_active_backend", "sb3") or "sb3")
        if _active_be != "isaac_lab":
            if not self._run_safety_review(blocking=True):
                log_error(
                    "[Review] Launch blocked by SafetyReview — fix the "
                    "highlighted nodes and click Launch again."
                )
                return

        backend = str(payload.get("backend", "mujoco") or "mujoco").strip().lower()
        scene_id = str(payload.get("scene_id", "") or "")
        bundle_name = str(payload.get("bundle_name", "") or "")
        version = str(payload.get("version", "v1") or "v1")
        overwrite = bool(payload.get("overwrite", True))

        if not bundle_name or bundle_name == "<NEW>":
            log_error(
                "[Review] Bundle name is unset — open the Export node "
                "and pick a Checkpoint before launching review."
            )
            return

        # Compose the bundle directory the same way ExportNode.execute does.
        bundle_dir_name = bundle_name if overwrite else f"{bundle_name}_{version}"

        log_success(
            f"[Review] Launch requested → backend={backend} "
            f"scene={scene_id} bundle={bundle_dir_name}"
        )

        # Dispatch. The existing SB3 + IL handlers already know how
        # to spawn the right subprocess; we just funnel the unified
        # payload into their pre-existing shape so the subprocess
        # lifetime and viewer wiring stay in one place.
        dispatch_payload = {
            "bundle_name": bundle_dir_name,
            "scene_id": scene_id,
            "review_backend": backend,
        }
        if backend == "mujoco":
            try:
                self._on_export_review_requested(dispatch_payload)
            except Exception as exc:
                log_error(f"[Review] MuJoCo review dispatch failed: {exc}")
        elif backend == "isaac_sim":
            try:
                self._on_il_export_review_requested(dispatch_payload)
            except Exception as exc:
                log_error(f"[Review] Isaac Sim review dispatch failed: {exc}")
        else:
            log_warning(
                f"[Review] Backend {backend!r} is not wired yet — "
                f"placeholder for a future UnitPort release."
            )

    def _refresh_current_canvas_label(self) -> None:
        # The workspace-title and export labels were removed from the
        # toolbar row (see _build_toolbar_row). This function is kept as a
        # no-op so existing call sites stay intact and so future surfaces
        # (e.g. status bar) can hook back into the same refresh point.
        return
        self._sync_selected_training_asset_from_canvas()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_window_controls()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_window_controls()

    def _is_experiment_canvas_missing(self, ws_id: str, exp_id: str) -> bool:
        """Return True iff the experiment's ``.canvas.json`` file is
        missing from the standard project layout.

        Used by :meth:`_refresh_canvas_browser` to reconcile stale
        registry entries against the actual disk state. Project layout
        v2 puts experiment canvases at
        ``projects/<workspace>/canvas/experiments/<exp>.canvas.json``.
        """
        try:
            from pathlib import Path
            path = (
                Path.cwd() / "projects" / str(ws_id)
                / "canvas" / "experiments"
                / f"{exp_id}.canvas.json"
            )
            return not path.exists()
        except Exception:
            return False

    def _refresh_canvas_browser(self, meta=None) -> None:
        if self._ws_store is None or not hasattr(self, "_canvas_browser_panel"):
            return
        if meta is None and self._workspace_name:
            try:
                meta = self._ws_store.load_workspace(self._workspace_name)
            except Exception:
                self._canvas_browser_panel.populate([], self._current_experiment_id)
                self._refresh_current_canvas_label()
                return

        entries = []
        try:
            all_ws_ids = self._ws_store.list_workspaces()
        except Exception:
            all_ws_ids = [self._workspace_name] if self._workspace_name else []

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

            # Reconcile the workspace metadata against the filesystem
            # BEFORE populating the browser — any experiment whose
            # ``.canvas.json`` was deleted out of band (file manager,
            # git clean, etc.) gets pruned from the registry so the
            # next refresh and cross-session restart don't resurrect
            # it. Mutating the list during iteration is unsafe, so we
            # collect stale ids first and prune after.
            stale_exp_ids: List[str] = []
            for exp_meta in ws_meta.experiments:
                if self._is_experiment_canvas_missing(ws_id, exp_meta.experiment_id):
                    stale_exp_ids.append(exp_meta.experiment_id)
            for stale_id in stale_exp_ids:
                try:
                    self._ws_store.delete_experiment(ws_id, stale_id)
                    log_warning(
                        f"[canvas_browser] pruned stale experiment entry "
                        f"{stale_id!r} — canvas file was deleted on disk"
                    )
                except Exception as exc:
                    log_warning(
                        f"[canvas_browser] failed to prune stale "
                        f"experiment {stale_id!r}: {exc}"
                    )

            # Re-load the (possibly mutated) workspace metadata so the
            # browser iteration sees a reconciled view.
            if stale_exp_ids:
                try:
                    ws_meta = self._ws_store.load_workspace(ws_id)
                except Exception:
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
                scope = str(getattr(checkpoint, "scope", "legacy") or "legacy")
                published = False
                tooltip_lines = [
                    f"Category: {category}",
                    f"Robot: {robot_type}",
                    f"Export: {display_name}",
                    f"Policy ID: {policy_id or '-'}",
                    f"Scope: {scope}",
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
                        "scope": scope,
                        "published": published,
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
            if self._ws_store is not None and self._workspace_name:
                meta = self._ws_store.load_workspace(self._workspace_name)
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
            if self._workspace_name == workspace_id and self._current_experiment_id == experiment_id:
                self._current_experiment_id = ""
                try:
                    meta = self._ws_store.load_workspace(workspace_id)
                    if meta.active_experiment_id:
                        self._workspace_name = workspace_id
                        self._load_experiment_by_id(meta.active_experiment_id)
                    else:
                        self._workspace_name = workspace_id
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
            if self._workspace_name == workspace_id:
                self._workspace_name = new_name
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

            if self._workspace_name == workspace_id:
                self._current_experiment_id = ""
                if remaining:
                    self._workspace_name = remaining[0]
                    meta = self._ws_store.load_workspace(self._workspace_name)
                    if meta.active_experiment_id:
                        self._load_experiment_by_id(meta.active_experiment_id)
                    else:
                        self._ensure_template_canvas()
                        self._populate_lists(meta)
                        self._refresh_canvas_browser(meta)
                else:
                    self._workspace_name = ""
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

            # Save current experiment's canvas to disk first, so switching
            # away doesn't lose in-progress edits. Best-effort — a save
            # failure here must not block the new experiment creation.
            if self._workspace_name == workspace_id and self._current_experiment_id:
                try:
                    self._save_current_experiment()
                except Exception as save_exc:
                    log_error(f"Pre-switch save failed: {save_exc}")

            exp_meta = self._ws_store.create_experiment(workspace_id, name=name)
            self._workspace_name = workspace_id
            self._current_experiment_id = exp_meta.experiment_id

            # Hard reset workspace-scoped canvas layer caches. Without this,
            # a stale layer from the previous experiment would bleed into
            # the new one on the next backend switch or save, and the
            # FilesRow backend tag would mis-report the canvas type.
            self._canvas_layers = {}
            if hasattr(self, "_canvas_layer_history"):
                self._canvas_layer_history = {}

            # Clear the scene directly instead of round-tripping through an
            # empty envelope — that used to leave the old edges in place.
            self._canvas.scene.clear()

            # Seed a fresh template for the currently bound backend.
            self._ensure_template_canvas()
            if self._selected_training_asset_id:
                self._load_asset_into_canvas(self._selected_training_asset_id)
            else:
                self._canvas_source_state = "default_template"
            # Suppress history so load_parameters calls on export/algo
            # nodes don't push spurious undo entries while the canvas is
            # still settling (edges are deferred by 50ms).
            with self._canvas.scene.suppress_history():
                self._bind_canvas_to_workspace_policy()
            self._populate_lists()
            self._refresh_canvas_browser()
        except Exception as exc:
            log_error(f"New experiment creation failed for workspace='{workspace_id}': {exc}")

    def _on_canvas_requested(self, workspace_id: str, experiment_id: str) -> None:
        if not workspace_id or not experiment_id:
            return
        self._workspace_name = workspace_id
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

    def _publish_checkpoint_to_shared(self, policy_id: str) -> None:
        """Publish a local checkpoint to the global shared space."""
        if not policy_id:
            return
        try:
            from src.system.core.project_store import ProjectStore
            store = ProjectStore()

            # Check if shared already has this policy_id
            shared_dir = store._shared_checkpoints_dir() / policy_id
            action = "created"
            if shared_dir.exists():
                reply = QMessageBox.question(
                    self,
                    "Publish to Global",
                    f"'{policy_id}' already exists in shared space.\n"
                    "Overwrite or publish as new version?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return
                action = "overwritten" if reply == QMessageBox.StandardButton.Yes else "versioned"

            # Find which project owns this checkpoint
            project_id = ""
            for proj in store.list_projects():
                ckpt = store.get_checkpoint(proj.project_id, policy_id)
                if ckpt is not None:
                    project_id = proj.project_id
                    break

            if not project_id:
                QMessageBox.warning(
                    self,
                    "Publish Failed",
                    f"Checkpoint '{policy_id}' not found in any project.",
                )
                return

            result = store.publish_to_shared(project_id, policy_id, action=action)
            QMessageBox.information(
                self,
                "Published",
                f"'{result.policy_id}' published to shared space.\n"
                f"Action: {result.action}",
            )
            self._refresh_export_browser()
        except Exception as exc:
            QMessageBox.warning(self, "Publish Failed", str(exc))

    def _build_toolbar_row(self) -> QWidget:
        """
        Toolbar with the workspace's run-history / asset dropdowns + Save +
        the inline progress strip (bar / step / state). The pink backend
        badge and the project / export labels were moved out of this row:
          * backend tag → FilesRow tab pill (SB3 / Isaac)
          * project / export → other surfaces
        Each dropdown reveals a slim list panel (QMenu + QWidgetAction).
        """
        bar = QWidget()
        bar.setObjectName("trainingToolbarRow")
        bar.setFixedHeight(36)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(8, 0, 8, 0)
        hbox.setSpacing(6)

        def _make_dropdown_btn(label: str, list_attr: str) -> QToolButton:
            btn = QToolButton()
            btn.setText(label + "  v")
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

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

        self._btn_run_history = _make_dropdown_btn("Run History", "_run_list")
        hbox.addWidget(self._btn_run_history)
        self._btn_assets_dropdown_inst = self._make_assets_dropdown_btn()
        hbox.addWidget(self._btn_assets_dropdown_inst)
        self._asset_list.itemClicked.connect(self._on_training_asset_item_clicked)

        self._btn_save_exp = QPushButton("Save")
        self._btn_save_exp.setObjectName("trainingBtnSaveExp")
        self._btn_save_exp.setFixedHeight(26)
        self._btn_save_exp.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_save_exp.setToolTip("Save current canvas as experiment")
        self._btn_save_exp.clicked.connect(self._save_current_experiment)
        # Custom property drives a dirty-state QSS variant in apply_theme.
        self._btn_save_exp.setProperty("dirty", False)
        hbox.addWidget(self._btn_save_exp)
        # Visual breathing room between Save and the inline progress strip.
        hbox.addSpacing(10)

        # Inline progress strip (replaces the standalone _progress_strip row).
        # Bar grows; step / state labels sit at the right edge.
        self._strip_bar = QProgressBar()
        self._strip_bar.setObjectName("trainingStripBar")
        self._strip_bar.setRange(0, 100)
        self._strip_bar.setValue(0)
        self._strip_bar.setTextVisible(False)
        self._strip_bar.setFixedHeight(10)
        hbox.addWidget(self._strip_bar, 1)

        self._strip_label = QLabel("Step: -")
        self._strip_label.setObjectName("trainingStripLabel")
        hbox.addWidget(self._strip_label)

        self._strip_state = QLabel("Idle")
        self._strip_state.setObjectName("trainingStripState")
        self._strip_state.setFixedWidth(90)
        self._strip_state.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        hbox.addWidget(self._strip_state)

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
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

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
        display_label = str(label or "").strip() or "Load Asset"
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
        Recreate the default template when the current scene is blank AND
        a real workspace is actively loaded.

        Guard: when the workspace is empty (prewarm / homepage shell
        cache), do NOT place any nodes. The default template is a
        user-visible artifact — it belongs to a specific experiment
        and should only materialise when the user enters that canvas
        (either loading an existing ``.canvas.json`` or creating a new
        experiment). Startup prewarm only warms the widget shell.
        """
        # Prewarm guard: no workspace selected → the Training Ground page
        # is being constructed for shell caching purposes only. Placing
        # default nodes here fires graph_dirty → SafetyReview → log spam
        # for something the user never asked for.
        if not getattr(self, "_workspace_name", "") and not getattr(
            self, "_current_experiment_id", ""
        ):
            return
        if self._canvas_has_training_nodes():
            return
        # Backend is now bound to the experiment — seed the matching default
        # template (SB3 vs Isaac Lab) so a brand-new canvas does not silently
        # fall back to the SB3 layout when the user picked Isaac Lab.
        backend = str(getattr(self, "_active_backend", "sb3") or "sb3")
        self._canvas.scene.reset_to_default_template(backend=backend)
        if self._initial_robot_type:
            self._canvas.scene.set_initial_robot_type(self._initial_robot_type)
        if self._initial_runtime_scenario:
            self._canvas.scene.set_initial_runtime_scenario(self._initial_runtime_scenario)
        # Task templates (resolve_task_template) are SB3-format only.
        # Isaac Lab templates bake their own reward/termination defaults
        # via _il_node_overrides in _place_isaac_lab_initial_nodes —
        # applying the SB3 resolver here would overwrite those IL terms
        # with incompatible SB3 keys (velocity_tracking vs track_lin_vel_xy).
        if backend != "isaac_lab":
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
        load_mode = "resume" if entry.framework == "sb3" else "warm_start_actor"
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
        """Start button: compile spec and trigger training.

        Routes to SB3 or Isaac Lab backend based on the float bar selector.
        """
        float_bar = getattr(self._canvas, "_float_bar", None) if hasattr(self, "_canvas") else None
        backend = float_bar.selected_backend() if float_bar and hasattr(float_bar, "selected_backend") else "sb3"
        if backend == "isaac_lab":
            self._on_isaac_lab_train_requested()
            return
        if backend == "cloud":
            self._on_cloud_train_requested()
            return
        self._on_train_requested({})

    # ------------------------------------------------------------------
    # Isaac Review Session integration (Phase B)
    # ------------------------------------------------------------------

    def _compile_canvas_for_review(self) -> Optional[str]:
        """Compile the current canvas to a temporary @configclass file.

        The Review Session needs the same compiled env_cfg the Train
        Launcher uses.  We re-run the compiler each time the user opens
        the viewer so any canvas edits made since the last training
        run are picked up.

        Returns the absolute path to the compiled .py file, or None if
        compilation failed (errors logged to the training log panel).
        """
        try:
            scene = self._canvas.scene
            # Mirror the granular-train graph extraction path so the
            # compiler sees exactly the same input as a training run.
            graph = scene.serialize_training_graph(for_compiler=False)
        except Exception as exc:
            self._append_training_error(f"[Viewer] canvas serialization failed: {exc}")
            return None
        try:
            from src.system.training.isaac_lab_config_compiler import (
                IsaacLabConfigCompiler,
            )
            compiler = IsaacLabConfigCompiler(graph)
            body_errors = compiler.validate_body_mapping()
            if body_errors:
                for e in body_errors:
                    self._append_training_error(f"[Viewer] {e}")
                return None
            return str(compiler.compile_to_file())
        except Exception as exc:
            import traceback
            self._append_training_error(
                f"[Viewer] compile failed: {exc}\n{traceback.format_exc()}"
            )
            return None

    def _on_open_viewer_clicked(self) -> None:
        """Spawn the persistent il_review_session subprocess."""
        try:
            from src.system.core.logger import log_info
            from src.system.training.isaac_review_session import get_review_session
            from src.system.training.isaac_lab_backend import detect_isaac_lab
        except Exception as exc:
            self._append_training_error(f"[Viewer] import failed: {exc}")
            return

        # 1. Compile the current canvas to a temp config file
        config_file = self._compile_canvas_for_review()
        if not config_file:
            return

        # 2. Detect Isaac Lab paths (same way training does)
        detection = detect_isaac_lab()
        if detection is None:
            self._append_training_error(
                "[Viewer] Isaac Lab installation not detected. "
                "Register Isaac Lab in Settings or set ISAAC_LAB_PATH."
            )
            return

        log_info("[Viewer] Spawning Isaac Review Session subprocess...")
        session = get_review_session()
        # detect_isaac_lab() returns Dict[str, str], NOT an object —
        # use dict access (the previous getattr fallback silently
        # returned None and broke the launcher path).
        ok = session.start(
            env_cfg_file=config_file,
            num_envs=1,
            isaac_lab_launcher=detection.get("launcher", "") or None,
            isaac_lab_python=detection.get("python", "") or None,
        )
        if not ok:
            self._append_training_error(
                f"[Viewer] Failed to launch review session: "
                f"{session.last_error()}"
            )

    def _on_close_viewer_clicked(self) -> None:
        """Gracefully shut down the persistent review session."""
        try:
            from src.system.core.logger import log_info
            from src.system.training.isaac_review_session import get_review_session
        except Exception:
            return
        log_info("[Viewer] Closing Isaac Review Session...")
        get_review_session().stop()

    def _on_review_session_state_changed(self, new_state: str) -> None:
        """Log Isaac Review Session state changes."""
        try:
            from src.system.core.logger import log_info
            log_info(f"[Viewer] state → {new_state}")
        except Exception:
            pass

    def _on_review_session_error(self, where: str, detail: str) -> None:
        try:
            self._append_training_error(f"[Viewer] {where}: {detail}")
        except Exception:
            pass

    def _on_review_checkpoint_loaded(self, path: str, ok: bool) -> None:
        try:
            from src.system.core.logger import log_info, log_error
            if ok:
                log_info(f"[Viewer] Loaded checkpoint: {path}")
            else:
                log_error(f"[Viewer] Failed to load checkpoint: {path}")
        except Exception:
            pass

    def _on_review_stage_changed(self, idx: int, name: str,
                                 vx: float, vy: float, wz: float) -> None:
        try:
            from src.system.core.logger import log_info
            log_info(
                f"[Viewer] AUTO stage {idx + 1}: '{name}'  "
                f"vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}"
            )
        except Exception:
            pass

    def _apply_backend_binding(self, backend: str) -> None:
        """Bind the workspace to *backend* for the currently-loading
        experiment. Backend is **experiment-bound** — each tab owns one
        backend, set at creation time and never mutated by cross-tab
        activity. This method updates internal state + UI bits without
        letting ``_on_backend_switched`` fire its legacy layer-dance:
        the load path will replace the scene wholesale right after this
        runs, and there is no "previous backend layer" worth preserving.
        """
        backend = str(backend or "sb3").strip().lower() or "sb3"
        if backend not in ("sb3", "isaac_lab", "cloud"):
            backend = "sb3"
        # Normalize: "cloud" is now expressed as backend="isaac_lab" +
        # the user's target combo selection.
        if backend == "cloud":
            backend = "isaac_lab"
        self._active_backend = backend
        if hasattr(self, "_canvas") and hasattr(self._canvas, "scene"):
            try:
                self._canvas.scene._current_backend = backend
            except Exception:
                pass
        # Push the backend into the float bar **silently** — the bar owns
        # the pink badge text and the IL/SB3 button visibility, but we
        # must NOT let it re-emit ``backend_changed``, which would drag
        # _on_backend_switched into a save-current-layer-restore-other
        # dance that corrupts the scene mid-load.
        float_bar = getattr(self._canvas, "_float_bar", None) if hasattr(self, "_canvas") else None
        if float_bar is not None and hasattr(float_bar, "set_backend"):
            blocked = False
            try:
                blocked = float_bar.blockSignals(True)
                float_bar.set_backend(backend)
            finally:
                if blocked is not True:
                    # blockSignals returns the prior block state; always
                    # restore it to whatever it was.
                    try:
                        float_bar.blockSignals(blocked)
                    except Exception:
                        pass
                else:
                    try:
                        float_bar.blockSignals(False)
                    except Exception:
                        pass

        # UI-only side effects that _on_backend_switched used to handle.
        # These are all idempotent and do NOT touch scene content.
        if hasattr(self, "_asset_list") and self._asset_list is not None:
            try:
                self._populate_training_assets()
            except Exception:
                pass
        if hasattr(self, "_backend_badge"):
            _badge_map = {"isaac_lab": "Isaac Lab", "cloud": "Cloud", "sb3": "SB3"}
            self._backend_badge.setText(_badge_map.get(backend, "SB3"))
        if hasattr(self, "_palette_panel"):
            try:
                # Cloud uses the same palette as Isaac Lab.
                _palette_backend = "isaac_lab" if backend == "cloud" else backend
                self._palette_panel.filter_by_backend(_palette_backend)
            except Exception:
                pass
        if hasattr(self, "_training_panel"):
            try:
                _panel_backend = "isaac_lab" if backend == "cloud" else backend
                self._training_panel.set_backend_mode(_panel_backend)
            except Exception:
                pass
        if hasattr(self, "_canvas") and hasattr(self._canvas, "_stats_overlay"):
            overlay = self._canvas._stats_overlay
            if hasattr(overlay, "_metrics_layers"):
                try:
                    _metrics_backend = "isaac_lab" if backend == "cloud" else backend
                    overlay._metrics_layers.set_backend(_metrics_backend)
                except Exception:
                    pass

    def _on_backend_switched(self, backend: str) -> None:
        """Float-bar backend_changed handler.

        Backend is now experiment-bound, so this handler should only ever
        fire from programmatic paths (float bar ``set_backend`` called
        inside ``_apply_backend_binding``). We keep it for defensive UI
        refresh parity, but it NO LONGER touches scene content — that is
        handled exclusively by the experiment load/create path.
        """
        backend = str(backend or "sb3").strip().lower()
        if backend not in ("sb3", "isaac_lab", "cloud"):
            backend = "sb3"
        self._active_backend = backend
        if hasattr(self, "_canvas") and hasattr(self._canvas, "scene"):
            try:
                self._canvas.scene._current_backend = backend
            except Exception:
                pass
        if hasattr(self, "_asset_list") and self._asset_list is not None:
            try:
                self._populate_training_assets()
            except Exception:
                pass
        if hasattr(self, "_backend_badge"):
            _badge_map2 = {"isaac_lab": "Isaac Lab", "cloud": "Cloud", "sb3": "SB3"}
            self._backend_badge.setText(_badge_map2.get(backend, "SB3"))
        if hasattr(self, "_palette_panel"):
            try:
                _p_backend = "isaac_lab" if backend == "cloud" else backend
                self._palette_panel.filter_by_backend(_p_backend)
            except Exception:
                pass
        if hasattr(self, "_training_panel"):
            try:
                _t_backend = "isaac_lab" if backend == "cloud" else backend
                self._training_panel.set_backend_mode(_t_backend)
            except Exception:
                pass
        if hasattr(self, "_canvas") and hasattr(self._canvas, "_stats_overlay"):
            overlay = self._canvas._stats_overlay
            if hasattr(overlay, "_metrics_layers"):
                try:
                    _m_backend = "isaac_lab" if backend == "cloud" else backend
                    overlay._metrics_layers.set_backend(_m_backend)
                except Exception:
                    pass

    def _on_il_preset_apply(self, preset_name: str) -> None:
        """Load an Isaac Lab preset: clear → place nodes with preset params → wire.

        The sequence is:
          1. Suppress dirty detection for the entire flow
          2. Clear canvas
          3. Place IL default template (nodes created with default params)
          4. Overwrite every node's params from the preset dict (30ms timer)
          5. Wire connections (50ms timer, inside template code)
          6. Re-enable dirty detection (80ms timer)
        """
        from src.system.training.il_presets import get_il_preset
        try:
            preset = get_il_preset(preset_name)
        except KeyError:
            return

        scene = None
        if hasattr(self, "_canvas"):
            scene = getattr(self._canvas, "scene", None)
        if scene is None:
            return
        if getattr(self, "_active_backend", "sb3") != "isaac_lab":
            return

        # Suppress dirty for the entire preset flow (template + param fill + wiring).
        # Two layers of suppression are used:
        #   - self._il_preset_applying: legacy flag that silences the preset
        #     combo reset in _on_il_preset_dirty.
        #   - scene.suppress_history() entered manually here and released in
        #     _finalize so no undo entries are generated across the 80ms
        #     timer chain. Paired with reset_history() at the end so the
        #     preset-loaded state becomes the clean baseline.
        self._il_preset_applying = True
        scene._history_suppress_depth += 1

        # 1. Clear + place fresh template
        scene.reset_to_default_template(backend="isaac_lab")

        # 2. Apply preset params (30ms — after nodes placed, before wiring at 50ms)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(30, lambda: self._apply_il_preset_params(scene, preset))

        # 3. Re-enable dirty detection after everything settles (80ms — after wiring)
        def _finalize():
            self._il_preset_applying = False
            try:
                scene._history_suppress_depth = max(
                    0, scene._history_suppress_depth - 1
                )
                scene.reset_history()
            except Exception:
                pass
            self._active_il_preset = preset_name
            # Reflect the active preset in the toolbar's "Load Asset" label
            # using the same id namespace populated by _populate_training_assets.
            self._set_selected_training_asset(
                f"il_preset:{preset_name}", preset_name
            )

        QTimer.singleShot(80, _finalize)

    def _apply_il_preset_params(self, scene, preset: dict) -> None:
        """Overwrite node params from preset dict (called after nodes are placed).

        Routes every override through ``TrainingNodeItem.load_parameters``
        rather than mutating ``logic_node.parameters`` directly. The direct
        mutation path used to leave embedded widgets (e.g. the terminations
        / rewards registry editors) holding their pre-preset in-memory state
        while the underlying parameter dict had already changed; the next
        ``_rebuild_geometry`` pass — triggered by save→reload, backend
        switch, page refresh, etc. — would resurrect the widget from the
        new value and silently drop whatever the user could still see on
        screen. ``load_parameters`` calls ``_rebuild_geometry(rebuild_ports
        =False)`` immediately, so widget state and logic state can never
        drift out of sync.
        """
        from bin.nodes.training_node_items import TrainingNodeItem

        applied_types: set[str] = set()
        for canvas_item in scene.items():
            if not isinstance(canvas_item, TrainingNodeItem):
                continue
            ntype = canvas_item._node_type
            overrides = preset.get(ntype)
            if not overrides:
                continue
            canvas_item.load_parameters(overrides)
            applied_types.add(ntype)

        try:
            from src.system.core.logger import log_success
            log_success(
                f"IL preset loaded — params applied to {len(applied_types)} node types"
            )
        except Exception:
            pass

    def _on_il_preset_dirty(self) -> None:
        """Clear the active-preset marker when the user modifies the canvas.

        The toolbar "Load Asset" label is reset back to its default
        ("Load Asset") so a stale preset name doesn't keep claiming the
        canvas after manual edits.
        """
        if getattr(self, "_il_preset_applying", False):
            return
        if not getattr(self, "_active_il_preset", None):
            return
        self._active_il_preset = None
        try:
            self._set_selected_training_asset("")
        except Exception:
            pass

    def _on_ctrl_stop_clicked(self) -> None:
        """Stop button: cancel the active training (SB3, Isaac Lab, or Cloud)."""
        # Cloud SSH backend
        if hasattr(self, "_cloud_backend") and self._cloud_backend is not None:
            if self._cloud_backend.is_running:
                self._cloud_backend.cancel()
                self._set_training_state("idle")
                return
        # Isaac Lab backend
        if hasattr(self, "_isaac_lab_backend") and self._isaac_lab_backend is not None:
            if self._isaac_lab_backend.is_running:
                self._isaac_lab_backend.cancel()
                self._set_training_state("idle")
                return
        # SB3 backend
        if hasattr(self, "_training_panel"):
            self._training_panel.cancel_requested.emit()

    def _on_progress_strip(self, step: int, total: int,
                           reward_mean: float, best_reward: float,
                           ep_len_mean: float = 0.0,
                           status: str = "") -> None:
        """Update the control bar progress display."""
        pct = int(step / total * 100) if total > 0 else 0
        self._strip_bar.setValue(pct)
        # Include stage info if available from the backend
        stage_prefix = ""
        backend = getattr(self, "_isaac_lab_backend", None)
        if backend is not None:
            s_idx = getattr(backend, "_current_stage_idx", -1)
            s_total = getattr(backend, "_current_stage_total", 0)
            s_name = getattr(backend, "_current_stage_name", "")
            if s_idx >= 0 and s_total > 0:
                stage_prefix = f"Stage {s_idx}/{s_total - 1}: {s_name}  |  "
        self._strip_label.setText(f"{stage_prefix}{pct}%  Step {step:,}/{total:,}")

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
        # The toolbar row that hosts the inline progress strip now uses
        # ``card_bg`` so it visually reads as part of the card surface.
        # The progress bar trough then takes the *previous* toolbar bg
        # (``training_progress_strip_bg``) so the bar stays distinguishable
        # against the lighter card-coloured row.
        progress_bg = get_color("card_bg", "#272829")
        progress_border = get_color("training_progress_strip_border", "#374151")
        progress_bar_bg = get_color("training_progress_strip_bg", "#111827")
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
            if child.objectName() == "workspaceAssetLabel":
                child.setStyleSheet(f"QLabel {{ font-size: 12px; color: {asset_text}; background: transparent; border: none; font-weight: 600; }}")
            elif child.text() == "Training Ground":
                child.setStyleSheet(f"QLabel {{ font-size: 13px; font-weight: 700; color: {title_text}; background: transparent; border: none; }}")
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
            # The toolbar now adopts the progress strip's background since
            # the strip widgets were inlined into this row.
            toolbar.setStyleSheet(
                f"#trainingToolbarRow {{ background: {progress_bg}; "
                f"border-bottom: 1px solid {progress_border}; }}"
            )
        tool_btn_style = (
            f"QToolButton {{ background: transparent; color: {toolbar_btn_text}; "
            f"border: 1px solid {toolbar_border}; font-size: 12px; "
            f"padding: 3px 10px; border-radius: 4px; }}"
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
        # Inline progress strip styling (now lives inside trainingToolbarRow).
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
        # Toolbar action buttons (Import / Download / Refresh — generic look).
        _ctrl_btn_base = (
            f"QPushButton {{ background: {toolbar_bg}; color: {toolbar_btn_text}; "
            f"border: 1px solid {toolbar_border}; border-radius: 4px; "
            f"font-size: 11px; padding: 2px 10px; }}"
            f"QPushButton:hover {{ background: {toolbar_btn_hover}; }}"
            f"QPushButton:pressed {{ background: {toolbar_btn_pressed}; }}"
            f"QPushButton:disabled {{ color: {get_color('text_muted', '#6b7280')}; "
            f"background: {toolbar_bg}; border-color: {toolbar_border}; }}"
        )
        for attr in (
            "_btn_new_exp",
            "_btn_import_asset",
            "_btn_download_asset",
            "_btn_refresh_assets",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(_ctrl_btn_base)
        # Save button has two visual states keyed off a "dirty" property:
        # clean → transparent ghost button (text + border in text_secondary);
        # dirty → tab_bg_training fill + tab_text_checked.
        if hasattr(self, "_btn_save_exp") and self._btn_save_exp is not None:
            dirty_bg = get_color("tab_bg_training", "#62E3E2")
            dirty_text = get_color("tab_text_checked", "#000000")
            secondary = get_color("text_secondary", "#9ca3af")
            # Border colour matches the Run History / Load Asset dropdowns
            # (toolbar_border) so the three controls read as a single set.
            self._btn_save_exp.setStyleSheet(
                f"QPushButton#trainingBtnSaveExp {{"
                f" background: transparent; color: {secondary};"
                f" border: 1px solid {toolbar_border}; border-radius: 4px;"
                f" font-size: 11px; padding: 2px 12px; }}"
                f"QPushButton#trainingBtnSaveExp:hover {{ background: {toolbar_btn_hover}; }}"
                f"QPushButton#trainingBtnSaveExp:pressed {{ background: {toolbar_btn_pressed}; }}"
                f"QPushButton#trainingBtnSaveExp[dirty=\"true\"] {{"
                f" background: {dirty_bg}; color: {dirty_text};"
                f" border: 1px solid {dirty_bg}; font-weight: 700; }}"
                f"QPushButton#trainingBtnSaveExp[dirty=\"true\"]:hover {{"
                f" background: {dirty_bg}; }}"
            )
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
                    ws_meta = self._ws_store.load_workspace(self._workspace_name)
                    exp = ws_meta.get_experiment(spec.experiment_id)
                    if exp is not None:
                        exp_name = exp.name
                except Exception:
                    pass

                # RunMeta.policy_id is the *parent* trained policy id (the one
                # this run is iterating on). It is NOT the workspace handle.
                # Empty when the workspace was opened without a source policy.
                run_meta = RunMeta(
                    run_id=run_id,
                    policy_id=self._selected_policy_id or "",
                    experiment_id=spec.experiment_id,
                    experiment_name=exp_name,
                    status="queued",
                    algorithm=algo.algorithm or "PPO",
                    total_timesteps=algo.total_timesteps,
                    policy_id_out=policy_id_out,
                )
                self._ws_store.create_run(self._workspace_name, run_meta)
            except Exception:
                pass

        # Build thread from spec
        thread = TrainRunThread.from_spec(spec, run_id=run_id)
        self._active_thread = thread
        self._training_panel.connect_thread(thread)

        # Wire lifecycle signals (the inline progress strip lives in the
        # toolbar row and is always visible).
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

        # Auto-show training stats overlay + force the chart to pin the new
        # active run, mirroring what the Isaac Lab branch does. Without this,
        # stale _selected_runs from a prior Isaac Lab run would suppress the
        # newly-started SB3 run and the chart would draw "No visible series".
        self._force_refresh_stats_overlay()

    def _force_refresh_stats_overlay(self) -> None:
        """Show the stats overlay (if hidden) and reset its run selection to
        the active run. Called by both SB3 and Isaac Lab start paths so the
        chart always follows the most recently launched training run.
        """
        overlay = getattr(getattr(self, "_canvas", None), "_stats_overlay", None)
        if overlay is None:
            return
        if not overlay.isVisible():
            try:
                overlay.show_for_button(None)
            except Exception:
                pass
        if hasattr(overlay, "refresh_from_cache"):
            try:
                overlay.refresh_from_cache(force_default=True)
            except Exception:
                pass
        # Ensure the cache-poll timer is running, even if the overlay was
        # already visible from a previous session and somehow stopped.
        # Without this, AMP/Isaac Lab runs that never trigger
        # ``show_for_button`` (because the overlay was already up) would
        # stop receiving cache refreshes and the chart would freeze on
        # "No visible series".
        timer = getattr(overlay, "_timer", None)
        if timer is not None and not timer.isActive():
            try:
                timer.start()
            except Exception:
                pass

    def _update_active_run(self, **fields) -> None:
        """Merge *fields* into the active run's persisted JSON. Silent on error."""
        if not self._active_run_id or self._ws_store is None:
            return
        try:
            self._ws_store.update_run(self._workspace_name, self._active_run_id, fields)
        except Exception:
            pass

    def _refresh_checkpoint_registry(self) -> None:
        """Trigger CheckpointRegistry re-discovery so the new bundle appears."""
        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry
            CheckpointRegistry().refresh()
        except Exception:
            pass

    def _refresh_training_asset_registry(self) -> None:
        """Rescan ``custom_mods/training/assets/`` so a just-exported training
        artifact becomes visible to ``_resolve_latest_export_training_asset``
        and the Training Assets list. Without this the registry keeps the
        pre-run cache and ``Start = Latest Export`` silently fails to find the
        bundle a training run just produced.
        """
        try:
            self._asset_registry.refresh()
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
        # Export target may be "runtime_bundle", "training_artifact", or
        # "both" — refresh both registries so whatever the exporter wrote is
        # immediately visible to the next Start (Latest Export) resolution.
        self._refresh_checkpoint_registry()
        self._refresh_training_asset_registry()
        self._populate_lists()
        # Rebuild canvas nodes so Export's bundle picker and other dynamic
        # widgets see the newly created checkpoint/bundle immediately.
        self._refresh_canvas_node_widgets()
        self.checkpoint_exported.emit(bundle_path)

    def _refresh_canvas_node_widgets(self) -> None:
        """Rebuild geometry on every TrainingNodeItem in the current scene.

        Called after training completes so that dynamic widgets (Export
        bundle picker, Start Point checkpoint dropdown, etc.) re-query
        the registries and show freshly-created bundles/checkpoints.
        Without this the Export node's picker would still list pre-
        training entries and the Review button would fail to resolve
        the new bundle path.
        """
        from bin.nodes.training_node_items import TrainingNodeItem

        if not hasattr(self, "_canvas") or self._canvas is None:
            return
        scene = self._canvas.scene
        with scene.suppress_history():
            for item in scene.items():
                if isinstance(item, TrainingNodeItem):
                    item._rebuild_geometry(rebuild_ports=False)

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
        # Mandatory SafetyReview gate — forces a full refresh + every
        # cross-section and Start Point invariant check before even
        # touching the compiler. Errors highlight the offending nodes
        # on the canvas and block training; warnings surface in the log
        # but still allow proceed.
        if not self._run_safety_review(blocking=True):
            return

        try:
            from src.system.training.training_spec import TrainingSpecCompiler
            canvas_data = self._canvas.scene.serialize_training_graph(for_compiler=True)
            spec = TrainingSpecCompiler().compile(
                canvas_data,
                policy_id=self._workspace_name,
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
                # Build a precise, actionable error message — strip stays clean,
                # full diagnostics go to the training log + global log_error so
                # the user can actually see WHY resolution failed.
                diag = getattr(self, "_latest_export_diag", None) or {}
                lines = [
                    f"Base asset could not be resolved for {intended_mode!r} — training not started."
                ]
                ws_id = diag.get("workspace_name", "")
                if ws_id:
                    lines.append(f"  workspace_name (search key): {ws_id!r}")
                else:
                    lines.append("  workspace_name is empty — open the workspace by name first.")
                if diag.get("reason"):
                    lines.append(f"  reason: {diag['reason']}")
                if diag.get("primary_checkpoint"):
                    lines.append(f"  asset.primary_checkpoint: {diag['primary_checkpoint']}")
                if diag.get("asset_path"):
                    lines.append(f"  asset_path: {diag['asset_path']}")
                if diag.get("exception"):
                    lines.append(f"  exception: {diag['exception']}")
                scanned = diag.get("scanned")
                if scanned is not None:
                    if scanned:
                        lines.append("  scanned training assets (asset_id → parent_policy_id):")
                        for asset_id, ppid in scanned:
                            lines.append(f"    - {asset_id} → {ppid}")
                    else:
                        lines.append("  scanned training assets: <none — registry is empty>")
                msg = "\n".join(lines)
                self._strip_state.setText("Asset Error")
                self._strip_label.setText("See log for details")
                self._append_training_error(msg)
                log_error(msg)
                self._latest_export_diag = None
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
            _backend = str(getattr(self, "_active_backend", "sb3") or "sb3")
            if _backend == "isaac_lab":
                # Isaac Lab canvases cannot go through the SB3
                # TrainingSpecCompiler — resolve the bundle path
                # directly from the dispatch payload / Export node.
                bundle_path = self._resolve_bundle_path_from_payload(_job_spec)
                spec = None
            else:
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

    def _resolve_bundle_path_from_payload(self, payload: dict):
        """Resolve a bundle directory from a review dispatch payload.

        Used by the Isaac Lab review path where the SB3
        TrainingSpecCompiler is not available. Falls back to the same
        CheckpointRegistry + project-local search as the SB3 path.
        """
        from pathlib import Path
        from src.system.service.checkpoint_registry import CheckpointRegistry

        bundle_name = str(payload.get("bundle_name", "") or "").strip()
        if not bundle_name or bundle_name == "<NEW>":
            # Try the Export node on the canvas
            export_item = self._find_training_node_item("export")
            if export_item is not None:
                p = export_item.get_parameters()
                bundle_name = str(p.get("bundle_name", "") or "").strip()
        if not bundle_name or bundle_name == "<NEW>":
            raise FileNotFoundError(
                "Bundle name is unset — open the Export node and pick "
                "a Checkpoint before launching review."
            )

        try:
            return Path(CheckpointRegistry().get_bundle_path(bundle_name))
        except Exception:
            pass
        # Project-local fallback
        if self._workspace_name:
            try:
                from src.system.core.project_store import ProjectStore
                store = ProjectStore()
                pid = self._ws_store._resolve_project_id(self._workspace_name) if self._ws_store else ""
                if pid:
                    candidate = store._checkpoints_dir(pid) / bundle_name
                    if candidate.exists():
                        return candidate
            except Exception:
                pass
        # Last export cache
        if self._last_export_bundle_path:
            p = Path(self._last_export_bundle_path)
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Bundle '{bundle_name}' not found in checkpoint registry "
            f"or project directory. Train and export first."
        )

    def _resolve_bundle_checkpoint_path(self, bundle_path) -> "Optional[Path]":
        """Find the actor checkpoint inside an Isaac Lab bundle directory.

        Used by the Phase B fast-path Export Review hot-swap.  A
        UnitPort exported bundle layout is::

            bundle_dir/
              manifest.yaml
              policy.onnx           ← exported actor (preferred)
              discriminator.pt      ← AMP disc (NOT the actor — exclude!)
              env.yaml
              source.json
              unitport_run_meta.yaml

        Training-run directories (when reviewing pre-export) look like::

            run_dir/
              params/
                model_*.pt          ← actor checkpoints
              ...

        Resolution priority:
          1. ``policy.onnx`` at bundle root  (the deployable actor)
          2. ``params/model_*.pt`` (highest iter — pre-export training run)
          3. ``checkpoints/model_*.pt``
          4. ``model_*.pt`` at bundle root
          5. Any ``*.pt`` EXCLUDING ``discriminator.pt`` (last resort)

        ``discriminator.pt`` is intentionally excluded from every
        fallback because it contains the AMP discriminator weights,
        not the actor — loading it as if it were the actor would
        either crash on shape mismatch or silently produce garbage.

        Returns ``None`` if no actor checkpoint can be found.
        """
        from pathlib import Path
        if not isinstance(bundle_path, Path):
            try:
                bundle_path = Path(bundle_path)
            except Exception:
                return None
        if not bundle_path.is_dir():
            return None

        def _iter_num(p: Path) -> int:
            try:
                return int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1

        def _is_disc(p: Path) -> bool:
            return "discriminator" in p.name.lower()

        # 1. policy.onnx — the standard UnitPort bundle export
        onnx_candidate = bundle_path / "policy.onnx"
        if onnx_candidate.is_file():
            return onnx_candidate

        # 2. params/model_*.pt
        params_dir = bundle_path / "params"
        if params_dir.is_dir():
            cands = sorted(
                (c for c in params_dir.glob("model_*.pt") if not _is_disc(c)),
                key=_iter_num, reverse=True
            )
            for c in cands:
                if _iter_num(c) >= 0:
                    return c

        # 3. checkpoints/model_*.pt
        chk_dir = bundle_path / "checkpoints"
        if chk_dir.is_dir():
            cands = sorted(
                (c for c in chk_dir.glob("model_*.pt") if not _is_disc(c)),
                key=_iter_num, reverse=True
            )
            for c in cands:
                if _iter_num(c) >= 0:
                    return c

        # 4. Any model_*.pt at bundle root
        cands = sorted(
            (c for c in bundle_path.glob("model_*.pt") if not _is_disc(c)),
            key=_iter_num, reverse=True
        )
        for c in cands:
            if _iter_num(c) >= 0:
                return c

        # 5. Any actor-named .pt at bundle root
        for name in ("actor.pt", "policy.pt", "actor_critic.pt"):
            p = bundle_path / name
            if p.is_file():
                return p

        # 6. Any .pt that is NOT discriminator.pt
        cands = [c for c in bundle_path.glob("*.pt") if not _is_disc(c)]
        if cands:
            return cands[0]

        return None

    def _on_il_export_review_requested(self, exporter_params: dict) -> None:
        """Replay an Isaac-Lab-trained policy in the MuJoCo viewer.

        Per design intent, IL review uses the same lightweight MuJoCo
        path as SB3 review — opens the bundle, builds a UnitreeGymEnv,
        loads the policy via PolicyRunner, runs one episode in the
        passive viewer. The differences from the SB3 handler are:

          1. Bundle path discovery — IL bundles can come from auto-import
             (``self._last_export_bundle_path``) OR from the user picking
             an existing checkpoint in the IL Policy Exporter's NEW+dropdown.
          2. ``spec`` argument — passed as ``None`` because the IL canvas
             can't be compiled by ``TrainingSpecCompiler``.
             ``_resolve_bundle_lineage_spec`` falls through to
             ``_synthesize_default_replay_spec`` which reads the bundle's
             own ``manifest.yaml`` to extract robot_type and seeds the
             rest from dataclass defaults.
        """
        try:
            from pathlib import Path
            from src.system.service.checkpoint_registry import CheckpointRegistry

            def _is_real_bundle(p: Optional[Path]) -> bool:
                # A real bundle dir always carries manifest.yaml. Without
                # this guard, ``Path("").expanduser()`` resolves to
                # ``Path(".")`` (cwd) which IS an existing directory and
                # would silently sneak through to BundleLoader, producing
                # the cryptic "manifest.yaml not found in bundle: ." error.
                return (
                    p is not None
                    and p.exists()
                    and p.is_dir()
                    and (p / "manifest.yaml").is_file()
                )

            bundle_path: Optional[Path] = None

            # Resolution order — the user's explicit pick ALWAYS wins over
            # the cached "latest export" path. The previous code had it
            # backwards (cached first), which meant clicking Review on
            # any old checkpoint silently replayed whichever bundle was
            # auto-imported last. The cache only acts as a fallback for
            # the "<NEW>" / unset state where the user never picked
            # anything explicitly.

            # 1. Whatever the user picked in the IL Policy Exporter's
            #    Checkpoint dropdown. Empty / "<NEW>" entries leave
            #    bundle_name blank.
            bundle_id = str(exporter_params.get("bundle_name", "") or "").strip()
            if bundle_id and bundle_id != "<NEW>":
                try:
                    candidate = Path(
                        CheckpointRegistry().get_bundle_path(bundle_id)
                    )
                    if _is_real_bundle(candidate):
                        bundle_path = candidate
                except Exception:
                    bundle_path = None

            # 2. Fallback: the auto-imported bundle from the most recent
            #    IL training run on this workspace. Only trust the
            #    cached path when the *raw string* is non-empty — an
            #    empty cache normalises to Path('.') (cwd).
            if bundle_path is None:
                cached = str(self._last_export_bundle_path or "").strip()
                if cached:
                    candidate = Path(cached).expanduser()
                    if _is_real_bundle(candidate):
                        bundle_path = candidate

            if bundle_path is None:
                raise FileNotFoundError(
                    "No exported Isaac Lab bundle found. Train and export "
                    "first, or pick an existing checkpoint in the IL Policy "
                    "Exporter."
                )
        except Exception as exc:
            self._strip_state.setText("Review Error")
            self._strip_label.setText("IL review setup failed")
            self._append_training_error(f"IL export review setup failed: {exc}")
            return

        # ── Phase B: route Isaac Lab bundles through the persistent
        # Isaac Review Session.  Three cases:
        #
        #   (a) Session READY  → IPC hot-swap, robot updates in <1 s
        #   (b) Session STOPPED/ERROR → AUTO-START it now and queue
        #       the checkpoint load to fire when state hits READY.
        #       First call costs ~30-60 s (Kit boot); subsequent
        #       Reviews are instant.
        #   (c) Session STARTING → another auto-start is in flight.
        #       Just queue the load on top.
        #
        # The MuJoCo fallback is only used as a last resort when the
        # Isaac Review Session machinery itself is unavailable (no
        # Isaac Lab installation detected, import failures, etc.).
        try:
            from src.system.training.isaac_review_session import (
                get_review_session, STATE_READY, STATE_STARTING,
            )
            from src.system.core.logger import log_info, log_warning

            session = get_review_session()
            ckpt_path = self._resolve_bundle_checkpoint_path(bundle_path)
            if ckpt_path is None:
                raise RuntimeError(
                    f"No .pt checkpoint found inside bundle {bundle_path}"
                )

            cm = str(exporter_params.get("control_mode", "AUTO") or "AUTO").upper()
            auto_mode = cm != "MANUAL"

            # Always queue+send via load_checkpoint — it handles both
            # the immediate-IPC and queued cases internally.
            current_state = session.state()
            if current_state == STATE_READY:
                if session.load_checkpoint(str(ckpt_path), auto_mode=auto_mode):
                    log_info(
                        f"[Viewer] Hot-swap → {ckpt_path.name} ({cm} mode). "
                        f"Live in <1 s."
                    )
                    return
            elif current_state == STATE_STARTING:
                # Boot already in flight from a previous click; just
                # update the queued target so the freshest bundle wins.
                session.queue_load_checkpoint(str(ckpt_path), auto_mode=auto_mode)
                log_info(
                    f"[Viewer] Queued {ckpt_path.name} for the in-flight "
                    f"viewer boot ({cm} mode)."
                )
                return
            else:
                # STOPPED or ERROR → kick off start() and queue the load
                config_file = self._compile_canvas_for_review()
                if config_file is None:
                    raise RuntimeError(
                        "canvas compile for viewer failed (see log above)"
                    )
                from src.system.training.isaac_lab_backend import detect_isaac_lab
                detection = detect_isaac_lab()
                if detection is None:
                    raise RuntimeError(
                        "Isaac Lab installation not detected — configure "
                        "Register Isaac Lab in Settings or set ISAAC_LAB_PATH"
                    )
                # Queue first, THEN start, so there's no race window
                # where READY fires before _pending_load is set.
                session.queue_load_checkpoint(str(ckpt_path), auto_mode=auto_mode)
                # detect_isaac_lab() returns Dict[str, str] — use dict access.
                ok = session.start(
                    env_cfg_file=config_file,
                    num_envs=1,
                    isaac_lab_launcher=detection.get("launcher", "") or None,
                    isaac_lab_python=detection.get("python", "") or None,
                )
                if ok:
                    log_info(
                        f"[Viewer] Booting Isaac Sim for review of "
                        f"{ckpt_path.name} ({cm} mode). First boot ~30-60 s; "
                        f"the checkpoint will load automatically when Kit "
                        f"finishes booting."
                    )
                    return
                else:
                    raise RuntimeError(
                        f"failed to start review session: {session.last_error()}"
                    )
        except Exception as exc:
            try:
                from src.system.core.logger import log_warning
                log_warning(
                    f"[Viewer] Could not route to Isaac Review Session "
                    f"({exc}). Falling back to MuJoCo replay (no live "
                    f"command input — this is a degraded path)."
                )
            except Exception:
                pass

        # ── Last-resort fallback: spawn MuJoCo replay subprocess ────────
        # Only reached when the Isaac Review Session machinery is
        # broken (no Isaac Lab install, compile failure, etc.).  Logs
        # a warning above so the user knows they're on the degraded
        # path.  No live gamepad input here — the MuJoCo replay
        # currently uses bundle-default commands only.
        from src.system.training.vis_check_runner import run_export_bundle_review
        self._run_viewer_task(
            "Export Review",
            "Opening exported-bundle review viewer...",
            "Review Error",
            run_export_bundle_review,
            bundle_path,
            None,  # spec=None — _resolve_bundle_lineage_spec synthesizes from manifest
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
            lowered = msg.lower()
            if any(token in lowered for token in ("failed", "error", "skipped", "unavailable")):
                # Route error-like messages to the error sink only — avoid the
                # info+error duplicate that used to appear in the log.
                self._append_training_error(msg)
            else:
                self._append_training_log(msg)
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
            policy_id=self._workspace_name,
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

        bundle_name = str(
            getattr(getattr(spec, "export_config", None), "bundle_name", "") or ""
        ).strip()
        if bundle_name == "<NEW>":
            bundle_name = ""
        policy_id_out = (
            bundle_name
            or str(getattr(getattr(spec, "algorithm_config", None), "policy_id_out", "") or "").strip()
            or f"{self._selected_policy_id}_trained"
        )

        # The user's explicit pick wins. The cached "latest export" path
        # is only consulted as a last resort, AFTER the registry lookup
        # has had a chance to honour the chosen bundle. The previous
        # implementation tried the cache first (gated by name match),
        # which broke when ``policy_id_out`` happened to coincide with
        # the most recent auto-versioned bundle name and silently sent
        # the wrong directory into BundleLoader.
        try:
            return Path(CheckpointRegistry().get_bundle_path(str(policy_id_out)))
        except Exception:
            pass

        # Project-local checkpoint fallback
        if self._workspace_name:
            try:
                from src.system.core.project_store import ProjectStore
                store = ProjectStore()
                pid = self._ws_store._resolve_project_id(self._workspace_name) if self._ws_store else ""
                if pid:
                    candidate = store._checkpoints_dir(pid) / str(policy_id_out)
                    if candidate.exists():
                        return candidate
            except Exception:
                pass
        # Legacy fallback
        candidate = get_project_root() / "custom_mods" / "training" / "checkpoints" / str(policy_id_out)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No exported runtime bundle found for '{policy_id_out}'. "
            "Train/export once before using Export Review."
        )

    def _resolve_latest_export_training_asset(self, diag: Optional[dict] = None):
        """Find the most recent training artifact whose source.json's
        ``parent_policy_id`` matches the active workspace policy id.

        When ``diag`` is provided, it is populated with the workspace id we
        searched for and a list of ``(asset_id, parent_policy_id)`` pairs we
        observed. The caller can surface this through ``log_error`` to make
        Latest-Export resolution failures debuggable instead of opaque.
        """
        from pathlib import Path
        workspace_key = str(self._workspace_name or self._selected_policy_id or "").strip()
        if diag is not None:
            diag["workspace_name"] = workspace_key
            diag["scanned"] = []
        if not workspace_key:
            return None
        # Always rescan: a fresh export written between workspace open and
        # the Start click would otherwise miss the cached registry view.
        try:
            self._asset_registry.refresh()
        except Exception:
            pass
        entries = list(self._asset_registry.list_assets() or [])
        candidates = []
        for entry in entries:
            asset_id = str(getattr(entry, "asset_id", "") or "")
            if not getattr(entry, "is_valid", True):
                if diag is not None:
                    diag["scanned"].append((asset_id, "<invalid>"))
                continue
            source_path = Path(getattr(entry, "asset_path", "")) / "source.json"
            try:
                with source_path.open("r", encoding="utf-8") as fh:
                    source = json.load(fh) or {}
            except Exception as exc:
                if diag is not None:
                    diag["scanned"].append((asset_id, f"<source.json error: {exc}>"))
                continue
            parent_policy_id = str(source.get("parent_policy_id", "") or "").strip()
            if diag is not None:
                diag["scanned"].append((asset_id, parent_policy_id or "<missing>"))
            # Historical: when a workspace was opened from a derived checkpoint,
            # _workspace_name carries the parent policy id (see __init__).
            # So the workspace handle doubles as the search key here.
            if parent_policy_id != workspace_key:
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
                diag: dict = {}
                entry = self._resolve_latest_export_training_asset(diag=diag)
                checkpoint_file = ""
                if entry is None:
                    self._latest_export_diag = diag
            else:
                entry = self._asset_registry.get(asset_id)
                self._latest_export_diag = None
            if entry is None:
                return None, load_mode   # resolution failed 鈥?caller will block
            ckpt = checkpoint_file or entry.primary_checkpoint
            abs_path = str(entry.asset_path / ckpt) if ckpt else ""
            if not abs_path:
                # Entry was found but no checkpoint file path could be derived.
                # Without this guard sb3_trainer would silently fall through to
                # scratch training because ``base_asset_path`` is falsy.
                self._latest_export_diag = {
                    "workspace_name": str(self._workspace_name or ""),
                    "asset_id": getattr(entry, "asset_id", ""),
                    "asset_path": str(getattr(entry, "asset_path", "")),
                    "primary_checkpoint": getattr(entry, "primary_checkpoint", "") or "<empty>",
                    "reason": "asset has no resolvable checkpoint file",
                }
                return None, load_mode
            spec.base_asset_path = abs_path
            # Ensure resume_mode propagates on both top-level and algo_config
            spec.resume_mode = load_mode
            spec.algorithm_config.resume_mode = load_mode
            return entry, load_mode
        except Exception as exc:
            self._latest_export_diag = {
                "workspace_name": str(self._workspace_name or ""),
                "exception": f"{type(exc).__name__}: {exc}",
            }
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

        # Pre-check diagnostic dump: log the values both sides will compare so
        # any false-positive can be diagnosed straight from the training log,
        # without needing to repro under a debugger.
        try:
            self._append_training_log(
                "[compat] entry: "
                f"framework={getattr(entry, 'framework', '')!r}, "
                f"algorithm={getattr(entry, 'algorithm', '')!r}, "
                f"robot_type={getattr(entry, 'robot_type', '')!r}, "
                f"obs_dim={getattr(entry, 'obs_dim', 0)}, "
                f"action_dim={getattr(entry, 'action_dim', 0)}, "
                f"action_type={getattr(entry, 'action_type', '')!r}, "
                f"asset_id={getattr(entry, 'asset_id', '')!r}"
            )
            self._append_training_log(
                "[compat] canvas: "
                f"algorithm={getattr(spec.algorithm_config, 'algorithm', '')!r}, "
                f"robot_type={getattr(spec.robot_spec, 'robot_type', '')!r}, "
                f"action_dim={getattr(spec.robot_spec, 'action_dim', 0)}, "
                f"physics.action_type={getattr(spec.physics_config, 'action_type', '')!r}, "
                f"obs.action_type={getattr(spec.obs_action_config, 'action_type', '')!r}, "
                f"contract_preset={getattr(spec.obs_action_config, 'contract_preset', '')!r}, "
                f"frame_stack={getattr(spec.obs_action_config, 'frame_stack', 1)}, "
                f"obs_components={list(getattr(spec.obs_action_config, 'obs_components', []) or [])}"
            )
        except Exception:
            pass

        results = TrainingCompatibilityChecker().check(entry, spec)
        fails = [r for r in results if r.level == CompatLevel.FAIL]
        warns = [r for r in results if r.level == CompatLevel.WARN]

        if not fails and not warns:
            return True  # all clear

        # Build human-readable lines (plain text for log) and HTML rows (popup)
        lines: list = []
        html_rows: list = []
        for r in fails:
            lines.append(f"[FAIL]  {r.field}: {r.message}")
            html_rows.append(
                f"<tr><td><b style='color:#c0392b'>FAIL</b></td>"
                f"<td><code>{r.field}</code></td>"
                f"<td>{r.message}</td></tr>"
            )
        for r in warns:
            lines.append(f"[WARN]  {r.field}: {r.message}")
            html_rows.append(
                f"<tr><td><b style='color:#d68910'>WARN</b></td>"
                f"<td><code>{r.field}</code></td>"
                f"<td>{r.message}</td></tr>"
            )
        detail = "\n".join(lines)
        # Always log details so diagnosis doesn't depend on the popup.
        for line in lines:
            self._append_training_log(f"[compat] {line}")

        details_html = (
            "<table cellspacing='0' cellpadding='4' "
            "style='border-collapse:collapse;font-size:11px'>"
            + "".join(html_rows)
            + "</table>"
        )

        if fails:
            msg = QMessageBox(self)
            msg.setWindowTitle("Compatibility Check - Failures Detected")
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText(
                f"<b>{len(fails)} compatibility failure(s)</b> found between the "
                f"base checkpoint and the current canvas.<br><br>"
                f"Starting training may corrupt the run or crash immediately."
                f"<br><br>{details_html}"
            )
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setDetailedText(detail)
            force_btn = msg.addButton("Force Start Anyway", QMessageBox.ButtonRole.DestructiveRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(msg.button(QMessageBox.StandardButton.Cancel) or force_btn)
            msg.exec()
            return msg.clickedButton() is force_btn

        # WARN only
        msg = QMessageBox(self)
        msg.setWindowTitle("Compatibility Check - Warnings")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            f"<b>{len(warns)} compatibility warning(s)</b> found.<br><br>"
            f"Training can still proceed but results may be suboptimal."
            f"<br><br>{details_html}"
        )
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setDetailedText(detail)
        msg.addButton("Proceed", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        # AcceptRole button text is "Proceed"; any non-Cancel click -> proceed
        return clicked is not None and clicked.text() == "Proceed"

    # ------------------------------------------------------------------
    # Phase B 鈥?Workspace persistence wiring
    # ------------------------------------------------------------------

    def _init_workspace(self) -> None:
        """
        Initialise workspace store, load or create the workspace for
        self._workspace_name, populate toolbar lists, and restore the last
        active experiment into the canvas.
        """
        try:
            self._ws_store = _get_workspace_store()

            # If we have an active project from the main shell, bind the
            # store so all training data goes into that project instead of
            # accidentally creating a new one.
            if self._active_project_id and self._project_name:
                self._ws_store.bind_project_id(
                    self._project_name, self._active_project_id
                )
                # Use the project name as workspace identity when no
                # explicit policy was provided.
                if not self._workspace_name:
                    self._workspace_name = self._project_name

            if self._workspace_name:
                # Only open existing workspaces — never auto-create from a
                # cached policy_id.  If the workspace was deleted externally
                # we fall through to the "no policy" branch instead of
                # silently recreating it.
                if not self._ws_store.workspace_exists(self._workspace_name):
                    log_warning(
                        f"Workspace '{self._workspace_name}' no longer exists, "
                        "resetting to empty state."
                    )
                    self._workspace_name = ""
                else:
                    meta = self._ws_store.ensure_workspace(self._workspace_name)
                    self._populate_lists(meta)
                    self._refresh_canvas_browser(meta)
            if not self._workspace_name:
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
                if str(getattr(self, "_active_backend", "sb3") or "sb3") != "isaac_lab":
                    self._apply_resolved_task_template()
                self._canvas_source_state = "default_template"
                with self._canvas.scene.suppress_history():
                    self._bind_canvas_to_workspace_policy()
            # Note: _load_experiment_by_id already calls
            # _bind_canvas_to_workspace_policy inside suppress_history,
            # so no redundant naked call is needed here.
            self._refresh_current_canvas_label()
        except Exception as exc:
            log_error(
                f"Training workspace init failed for selected='{self._selected_policy_id}' "
                f"workspace='{self._workspace_name}': {exc}"
            )
            self._ensure_template_canvas()
            with self._canvas.scene.suppress_history():
                self._bind_canvas_to_workspace_policy()
            self._startup_restore_error_message = (
                f"{self._workspace_name or 'Training workspace'} failed to load, error: {exc}"
            )
            self.restore_failed.emit(self._startup_restore_error_message)

    def _populate_lists(self, meta=None) -> None:
        """Refresh toolbar experiment/run lists from the workspace store."""
        if self._ws_store is None:
            return
        try:
            if meta is None:
                meta = self._ws_store.load_workspace(self._workspace_name)

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
            runs = self._ws_store.list_runs(self._workspace_name)
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
                f"workspace='{self._workspace_name}': {exc}"
            )

    def _populate_training_assets(self) -> None:
        """Populate the toolbar's "Load Asset" dropdown.

        Filtered by the workspace's currently bound backend so users only
        see entries compatible with the active canvas:

          * SB3 mode → registry entries whose ``framework`` is ``"sb3"`` or
            unknown/empty (legacy untagged assets remain accessible).
          * Isaac Lab mode → built-in IL_PRESETS surfaced as virtual
            entries (``asset_id = "il_preset:<name>"``) plus any registry
            entries explicitly tagged as Isaac.

        OK / failed status uses ``✔`` / ``❌`` glyphs.
        """
        from PySide6.QtCore import Qt as _Qt

        self._asset_list.clear()

        backend = str(getattr(self, "_active_backend", "sb3") or "sb3").lower()
        is_il = backend == "isaac_lab"

        # ─── Build the filtered display list ───────────────────────────
        # Each entry is a small dict with the fields the renderer needs.
        display_entries: List[dict] = []

        if is_il:
            try:
                from src.system.training.il_presets import list_il_presets
                for pname in list_il_presets():
                    display_entries.append({
                        "asset_id": f"il_preset:{pname}",
                        "label": pname,
                        "robot_badge": "",
                        "is_valid": True,
                        "is_il_preset": True,
                        "tooltip_lines": [
                            f"Isaac Lab preset: {pname}",
                            "Click to load preset onto the current canvas.",
                        ],
                    })
            except Exception:
                pass
            # Isaac entries from the on-disk registry (if any framework=isaac/isaac_lab).
            for entry in (self._asset_registry.list_assets() or []):
                framework = str(getattr(entry, "framework", "") or "").lower()
                if framework not in ("isaac", "isaac_lab"):
                    continue
                display_entries.append(self._asset_display_dict(entry))
        else:
            # SB3 mode: keep registry entries that are sb3 or untagged.
            for entry in (self._asset_registry.list_assets() or []):
                framework = str(getattr(entry, "framework", "") or "").lower()
                if framework and framework != "sb3":
                    continue
                display_entries.append(self._asset_display_dict(entry))

        if not display_entries:
            placeholder = QListWidgetItem(
                "No Isaac Lab presets" if is_il else "No training assets yet"
            )
            placeholder.setFlags(placeholder.flags() & ~_Qt.ItemFlag.ItemIsSelectable)
            self._asset_list.addItem(placeholder)
            self._set_selected_training_asset("")
            return

        # Selected entry id can be either a registry asset_id or an il_preset:* token.
        selected_asset_id = self._get_canvas_base_asset_id() or self._selected_training_asset_id
        if not selected_asset_id and getattr(self, "_active_il_preset", None):
            selected_asset_id = f"il_preset:{self._active_il_preset}"

        found_selected = False
        for d in display_entries:
            status = "✔" if d["is_valid"] else "❌"
            label = f"{status}  {d['label']}{d['robot_badge']}"
            item = QListWidgetItem(label)
            item.setData(_Qt.ItemDataRole.UserRole, d["asset_id"])
            item.setToolTip("\n".join(d["tooltip_lines"]))
            if not d["is_valid"]:
                item.setForeground(QColor(get_color("text_muted", "#6b7280")))
            self._asset_list.addItem(item)
            if d["asset_id"] == selected_asset_id:
                found_selected = True
                self._asset_list.setCurrentItem(item)

        self._set_selected_training_asset(selected_asset_id if found_selected else "")

    def _asset_display_dict(self, entry) -> dict:
        """Adapt a TrainingAssetEntry into the dict shape used by the
        toolbar dropdown renderer."""
        return {
            "asset_id": entry.asset_id,
            "label": entry.label(),
            "robot_badge": f"  [{entry.robot_type}]" if entry.robot_type else "",
            "is_valid": bool(entry.is_valid),
            "is_il_preset": False,
            "tooltip_lines": [
                f"Asset ID:   {entry.asset_id}",
                f"Framework:  {entry.framework or '-'}",
                f"Algorithm:  {entry.algorithm or '-'}",
                f"Robot:      {entry.robot_type or '-'}",
                f"Obs dim:    {entry.obs_dim or '-'}",
                f"Action dim: {entry.action_dim or '-'}",
                f"Action type:{entry.action_type or '-'}",
                f"Primary:    {entry.primary_checkpoint or '-'}",
                f"Path:       {entry.asset_path}",
            ] + ([f"Error:      {entry.error}"] if entry.error else []),
        }

    def _on_training_asset_item_clicked(self, item) -> None:
        from PySide6.QtCore import Qt as _Qt

        asset_id = str(item.data(_Qt.ItemDataRole.UserRole) or "").strip()
        if not asset_id:
            return
        # IL preset entries use a synthetic id namespace — route them to
        # the existing IL preset application path. The label update is
        # handled inside _on_il_preset_apply's _finalize callback.
        if asset_id.startswith("il_preset:"):
            preset_name = asset_id.split(":", 1)[1].strip()
            if preset_name:
                self._on_il_preset_apply(preset_name)
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
        # Task templates (resolve_task_template) are SB3-only. On Isaac Lab
        # canvases the IL reward/termination registries are authoritative —
        # applying the SB3 resolver would overwrite them with incompatible
        # keys (e.g. velocity_tracking instead of track_lin_vel_xy).
        _backend = str(getattr(self, "_active_backend", "sb3") or "sb3")
        can_apply_task_template = (
            _backend != "isaac_lab"
            and self._canvas_source_state != "loaded_experiment"
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
        """Load an experiment canvas from disk and replace the current canvas.

        Backend is experiment-bound. We derive it from the envelope
        deterministically, bind the workspace UI to it, then load the
        flat (or legacy layered) canvas payload. Per-experiment isolation
        is guaranteed because the in-memory ``_canvas_layers`` dict is
        fully reset here — never inherited from a sibling tab.
        """
        if self._ws_store is None:
            return
        try:
            raw_data = self._ws_store.load_experiment(self._workspace_name, experiment_id)

            # Derive the experiment's backend deterministically.
            # Priority: active_backend > backend > first layer key.
            derived_backend = ""
            if isinstance(raw_data, dict):
                derived_backend = str(raw_data.get("active_backend") or "").strip()
                if not derived_backend:
                    derived_backend = str(raw_data.get("backend") or "").strip()
                if not derived_backend and isinstance(raw_data.get("layers"), dict):
                    layer_keys = list(raw_data["layers"].keys())
                    if layer_keys:
                        derived_backend = str(layer_keys[0])
            if derived_backend not in ("sb3", "isaac_lab"):
                derived_backend = "sb3"
            self._apply_backend_binding(derived_backend)

            # Hard reset workspace-scoped layer caches. Any stale slot
            # here belongs to the PREVIOUS experiment and would otherwise
            # bleed into the next save or tab swap.
            self._canvas_layers = {}
            if hasattr(self, "_canvas_layer_history"):
                self._canvas_layer_history = {}

            # Extract the flat canvas payload. Supports three shapes:
            #   1) NEW flat envelope: {schema, active_backend, backend,
            #      nodes, edges}
            #   2) LEGACY layered envelope: {schema, active_backend,
            #      layers: {"sb3": ..., "isaac_lab": ...}}
            #   3) TRULY legacy flat: {schema, backend, nodes, edges}
            canvas_data: Dict[str, Any] = {}
            if isinstance(raw_data, dict):
                if isinstance(raw_data.get("nodes"), list):
                    # Flat format — use as-is.
                    canvas_data = raw_data
                elif isinstance(raw_data.get("layers"), dict):
                    # Layered format — pull the layer matching derived
                    # backend (which, by construction, is the only layer
                    # the user ever saw on this experiment).
                    canvas_data = raw_data["layers"].get(derived_backend) or {}
                    if not (canvas_data and canvas_data.get("nodes")):
                        # Fall through to whatever layer exists, if any.
                        for _k, _v in raw_data["layers"].items():
                            if isinstance(_v, dict) and _v.get("nodes"):
                                canvas_data = _v
                                break
            self._canvas_layers = {derived_backend: canvas_data} if canvas_data else {}

            # Set the experiment id BEFORE any canvas mutation so that
            # signals fired during load (graph_dirty, content_changed)
            # route through the correct experiment, not the previous tab.
            self._current_experiment_id = experiment_id

            if canvas_data and canvas_data.get("nodes"):
                self._canvas.scene.load_training_graph(canvas_data)
                self._canvas_source_state = "loaded_experiment"
                self._sync_selected_training_asset_from_canvas()
            else:
                # Empty layer — force-seed the default template for the
                # *currently bound* backend, even if the canvas already holds
                # nodes from a previously loaded experiment. We can't call
                # _ensure_template_canvas() here because it short-circuits
                # whenever robot_mjcf is on the scene, which would leave the
                # old SB3-shaped canvas behind for a freshly-created Isaac
                # Lab experiment (and vice versa).
                self._canvas.scene.reset_to_default_template(
                    backend=self._active_backend
                )
                if self._initial_robot_type:
                    self._canvas.scene.set_initial_robot_type(self._initial_robot_type)
                if self._initial_runtime_scenario:
                    self._canvas.scene.set_initial_runtime_scenario(
                        self._initial_runtime_scenario
                    )
                if str(getattr(self, "_active_backend", "sb3") or "sb3") != "isaac_lab":
                    self._apply_resolved_task_template()
                self._canvas_source_state = "default_template"
                self._refresh_current_canvas_label()
                if self._selected_training_asset_id:
                    self._load_asset_into_canvas(self._selected_training_asset_id)

            self._ws_store.set_active_experiment(self._workspace_name, experiment_id)
            # Wrap _bind_canvas_to_workspace_policy inside suppress_history
            # so that load_parameters calls on export/algo nodes don't fire
            # graph_dirty or push spurious undo entries while the canvas is
            # still loading (edges are deferred by 50ms).
            scene = self._canvas.scene
            with scene.suppress_history():
                self._bind_canvas_to_workspace_policy()
            self._refresh_canvas_browser()
        except Exception as exc:
            self._strip_state.setText("Load Error")
            self._strip_label.setText(str(exc)[:80])
            self._ensure_template_canvas()
            self._current_experiment_id = ""
            with self._canvas.scene.suppress_history():
                self._bind_canvas_to_workspace_policy()
            self._refresh_current_canvas_label()
            self._refresh_canvas_browser()
            message = f"{experiment_id} failed to load, error: {exc}"
            if show_dialog:
                QMessageBox.warning(self, "Load Experiment Failed", message)
            else:
                self._startup_restore_error_message = message
                self.restore_failed.emit(message)

    def _on_scene_dirty_for_save_btn(self, dirty: bool) -> None:
        """Toggle the Save button's "dirty" QSS property when the scene's
        clean baseline shifts (edit / undo / save). The QSS in apply_theme
        re-paints the button via the property selector."""
        btn = getattr(self, "_btn_save_exp", None)
        if btn is None:
            return
        was = bool(btn.property("dirty"))
        now = bool(dirty)
        if was == now:
            return
        btn.setProperty("dirty", now)
        # Force a style re-evaluation so the property change actually paints.
        style = btn.style()
        if style is not None:
            style.unpolish(btn)
            style.polish(btn)
        btn.update()

    def _save_current_experiment(self) -> None:
        """
        Save the current canvas as the active experiment in the current
        project workspace.  If no experiment exists yet, prompt to create one.
        """
        if self._ws_store is None:
            return
        try:
            # No workspace (= no active project) — cannot save.
            if not self._workspace_name:
                QMessageBox.warning(
                    self, "No Project",
                    "Please open or create a project first.",
                )
                return

            self._bind_canvas_to_workspace_policy()

            # FLAT per-experiment envelope. Backend is experiment-bound,
            # so each .canvas.json owns exactly ONE backend + ONE scene.
            # The legacy layered envelope ({"layers": {"sb3": ..., ...}})
            # let cross-tab state leak into siblings on every save; the
            # flat format makes each tab's disk file fully independent.
            current_canvas = self._canvas.scene.serialize_training_graph()
            # Keep the in-memory layers dict in sync for any legacy
            # code path that still reads it, but the disk truth is flat.
            self._canvas_layers = {self._active_backend: current_canvas}

            save_data = {
                "schema": "training_canvas_v1",
                "active_backend": self._active_backend,
                "backend": self._active_backend,
                "nodes": current_canvas.get("nodes", []),
                "edges": current_canvas.get("edges", []),
            }

            meta = self._ws_store.load_workspace(self._workspace_name)
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
                    self._workspace_name,
                    name=name,
                    initial_canvas=save_data,
                )
                self._current_experiment_id = exp_meta.experiment_id
            else:
                self._ws_store.save_experiment(
                    self._workspace_name,
                    self._current_experiment_id,
                    save_data,
                )

            self._ws_store.set_active_experiment(
                self._workspace_name, self._current_experiment_id
            )
            self._populate_lists()
            self._refresh_canvas_browser()
            # Pin the post-save state as the new clean baseline. Both the
            # scene's own mark_clean() (for Ctrl+Z's clean-index) and the
            # stashed history for the currently-active backend layer are
            # updated so flipping backends and coming back reflects the
            # save on both sides.
            try:
                scene = self._canvas.scene
                scene.mark_clean()
                if hasattr(self, "_canvas_layer_history"):
                    self._canvas_layer_history[self._active_backend] = (
                        scene.export_history_state()
                    )
            except Exception:
                pass
            log_success(
                f"Saved experiment '{self._current_experiment_name()}' "
                f"to workspace '{self._current_workspace_name()}'"
            )
        except Exception as exc:
            self._strip_state.setText("Save Error")
            self._strip_label.setText(str(exc)[:80])
            QMessageBox.warning(self, "Save Experiment Failed", str(exc))

    def _new_experiment(self) -> None:
        """Create a new empty experiment and load it into the canvas."""
        if self._ws_store is None:
            return
        try:
            meta = self._ws_store.load_workspace(self._workspace_name)
            default_name = f"Experiment {len(meta.experiments) + 1}"
            name = self._prompt_experiment_name(default_name)
            if not name:
                return
            exp_meta = self._ws_store.create_experiment(self._workspace_name, name=name)
            self._current_experiment_id = exp_meta.experiment_id
            # Load the empty canvas (clears current graph)
            canvas_data = self._ws_store.load_experiment(
                self._workspace_name, exp_meta.experiment_id
            )
            self._canvas.scene.load_training_graph(canvas_data)
            self._ensure_template_canvas()
            if self._selected_training_asset_id:
                self._load_asset_into_canvas(self._selected_training_asset_id)
            else:
                self._canvas_source_state = "default_template"
            with self._canvas.scene.suppress_history():
                self._bind_canvas_to_workspace_policy()
            self._populate_lists()
            self._refresh_canvas_browser()
        except Exception as exc:
            log_error(
                f"New experiment creation failed for selected='{self._selected_policy_id}' "
                f"workspace='{self._workspace_name}': {exc}"
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

    # ------------------------------------------------------------------
    # Isaac Lab backend integration
    # ------------------------------------------------------------------

    def _on_isaac_lab_train_requested(self) -> None:
        """Handle Start click when Isaac Lab backend is selected.

        If granular IL nodes (il_ppo_trainer etc.) are present on the canvas,
        compiles the full graph via IsaacLabConfigCompiler.  Otherwise falls
        back to reading legacy IsaacLabTaskNode / IsaacLabTrainNode params.
        """
        scene = None
        if hasattr(self, "_canvas"):
            scene = getattr(self._canvas, "scene", None) or getattr(self._canvas, "_scene", None)

        # Detect whether the canvas uses granular IL nodes
        has_granular = False
        if scene and hasattr(scene, "serialize_training_graph"):
            graph = scene.serialize_training_graph(for_compiler=False)
            node_types = [n.get("node_type", "") for n in graph.get("nodes", [])]
            for nt in node_types:
                if nt.startswith("il_") or nt in ("rewards", "terminations", "domain_rand"):
                    has_granular = True
                    break

        if has_granular:
            self._on_isaac_lab_granular_train(graph)
        else:
            self._on_isaac_lab_legacy_train(scene)

    def _on_isaac_lab_granular_train(self, graph: dict) -> None:
        """Compile the granular IL node graph and launch training.

        task_name is read directly from the il_ppo_trainer node's parameters.
        If it's a registered Isaac Lab task, use it as-is (no compilation).
        If empty or unregistered, compile a custom config file.
        """
        from src.system.core.logger import log_info

        # Unified IL trainer — training_mode param picks PPO / AMP_PPO.
        # Legacy "amp_ppo_trainer" node_type aliases to "il_ppo_trainer"
        # at canvas load time, so the iteration below only needs to look
        # at il_ppo_trainer (with both node_type variants kept as a belt-
        # and-suspenders safety net for canvases that skipped the alias
        # rewrite). ``task_name`` is no longer a user-facing parameter;
        # _infer_il_task_name() derives it from the robot + scene.
        task_name = ""
        algorithm = "PPO"
        il_bundle_name = ""
        for n in graph.get("nodes", []):
            nt = n.get("node_type", "")
            if nt in ("il_ppo_trainer", "amp_ppo_trainer"):
                p = n.get("parameters", {}) or {}
                mode = str(p.get("training_mode", "PPO") or "PPO").strip()
                if nt == "amp_ppo_trainer" or mode == "AMP_PPO":
                    algorithm = "AMP_PPO"
            elif nt == "il_policy_exporter":
                raw = str(
                    n.get("parameters", {}).get("bundle_name", "") or ""
                ).strip()
                if raw and raw != "<NEW>":
                    il_bundle_name = raw

        # Always compile — the compiled config IS the source of truth.
        # For presets, the compiled output matches Isaac Lab's built-in config.
        # For custom edits, it contains the user's modifications.
        from src.system.training.isaac_lab_config_compiler import IsaacLabConfigCompiler
        compiler = IsaacLabConfigCompiler(graph)

        # ── Body-mapping gate ──
        # Validate that all required IR body roles are resolved before
        # allowing training to proceed. Unresolved roles would produce
        # broken body_names in the compiled config (ValueError at runtime).
        body_errors = compiler.validate_body_mapping()
        if body_errors:
            from src.system.core.logger import log_error
            for e in body_errors:
                log_error(e)
            log_error(
                "Training blocked: open the Policy Exporter node and "
                "resolve all body mapping roles before starting training."
            )
            return

        config_path = compiler.compile_to_file()
        config_file = str(config_path)

        # task_name from PPO trainer node (set by preset or user);
        # if empty, infer from robot/terrain; if custom, launcher registers it.
        if not task_name:
            task_name = self._infer_il_task_name(graph)

        # Extract key params from graph
        num_envs = 4096
        max_iterations = 1500
        headless = True
        robot_asset_id = ""
        # AMP-specific (populated only when training_mode == AMP_PPO)
        amp_motion_files: List[str] = []
        amp_reward_coef = 2.0
        amp_num_preload_transitions = 2_000_000
        amp_transition_dt = 0.02
        amp_task_reward_lerp = 0.5   # B3: spec §1.reward_mixing default
        amp_replay_buffer_size = 1_000_000
        amp_disc_grad_penalty = 10.0
        amp_disc_lr = 1e-4
        amp_disc_label_smoothing = 0.9
        amp_lerp_schedule = ""  # JSON string, empty = no schedule
        # Staged training (Phase 1 Stage Pipeline)
        stage_schedule_json = ""   # JSON string, empty = no staging
        stage_checkpoint_strategy = "both"

        for n in graph.get("nodes", []):
            nt = n.get("node_type", "")
            p = n.get("parameters", {})
            if nt == "il_robot_asset":
                try:
                    num_envs = int(p.get("num_envs", num_envs))
                except (TypeError, ValueError):
                    pass
                # Phase_1 of AMP_design.yaml §3: asset_id → registry
                # lookup at launcher CLI build time.
                robot_asset_id = str(p.get("asset_id", "") or "").strip()
            elif nt in ("il_ppo_trainer", "amp_ppo_trainer"):
                try:
                    max_iterations = int(p.get("max_iterations", max_iterations))
                except (TypeError, ValueError):
                    pass
                _hv = str(p.get("headless", "true")).strip().lower()
                headless = _hv not in ("false", "0", "no", "off")
                _mode = str(p.get("training_mode", "PPO") or "PPO").strip()
                if nt == "amp_ppo_trainer" or _mode == "AMP_PPO":
                    try:
                        amp_reward_coef = float(p.get("amp_reward_coef", amp_reward_coef))
                    except (TypeError, ValueError):
                        pass
                    try:
                        amp_num_preload_transitions = int(
                            p.get("num_preload_transitions", amp_num_preload_transitions)
                        )
                    except (TypeError, ValueError):
                        pass
                    try:
                        amp_task_reward_lerp = float(
                            p.get("task_reward_lerp", amp_task_reward_lerp)
                        )
                    except (TypeError, ValueError):
                        pass
                    try:
                        amp_replay_buffer_size = int(
                            p.get("amp_replay_buffer_size", amp_replay_buffer_size)
                        )
                    except (TypeError, ValueError):
                        pass
                    try:
                        amp_disc_grad_penalty = float(
                            p.get("disc_grad_penalty", amp_disc_grad_penalty)
                        )
                    except (TypeError, ValueError):
                        pass
                    try:
                        amp_disc_lr = float(p.get("disc_lr", amp_disc_lr))
                    except (TypeError, ValueError):
                        pass
                    try:
                        amp_disc_label_smoothing = float(
                            p.get("disc_label_smoothing", amp_disc_label_smoothing)
                        )
                    except (TypeError, ValueError):
                        pass
                    _raw_ls = p.get("lerp_schedule", "")
                    if _raw_ls and str(_raw_ls).strip():
                        amp_lerp_schedule = str(_raw_ls).strip()
                # ── Staged training pipeline ──
                _stages_on = str(p.get("stages_enabled", "false")).strip().lower()
                if _stages_on in ("true", "1", "yes"):
                    _raw_stages = p.get("stages_json", "[]") or "[]"
                    try:
                        import json as _json_mod
                        _parsed = _json_mod.loads(_raw_stages) if isinstance(_raw_stages, str) else _raw_stages
                        if isinstance(_parsed, list) and _parsed:
                            _stage_total = sum(int(s.get("iterations", 0)) for s in _parsed if isinstance(s, dict))
                            if _stage_total > 0:
                                max_iterations = _stage_total
                                stage_schedule_json = _raw_stages if isinstance(_raw_stages, str) else _json_mod.dumps(_parsed)
                                stage_checkpoint_strategy = str(p.get("stage_checkpoint_strategy", "both"))
                    except Exception:
                        pass
            elif nt == "reference_motion":
                # Phase_2 of AMP_design.yaml §3: when consumer_mode picks
                # the AMP path, the motion file(s) flow into the launcher
                # via --unitport_amp_motion_files. Three sources possible:
                #
                #   (a) motion_source = "pack:<pkg>:<subdir>" — expand to
                #       the full clip list (the natural AMP granularity)
                #   (b) motion_source = "generate:walk" / "loco:..." —
                #       resolve to a single .npy via the legacy helper
                #       (still works for AMP, just one-clip pool)
                #   (c) motion_file = "<absolute path>" — single explicit
                #       file (legacy behaviour)
                #
                # Multiple ReferenceMotionNode instances on the canvas
                # all contribute to the same amp_motion_files list, so
                # users can mix-and-match (e.g. one node for an AMP pack
                # plus a node with a custom hand-recorded clip).
                mode = str(p.get("consumer_mode", "tracking") or "tracking").strip().lower()
                if mode in ("amp", "both"):
                    src = str(p.get("motion_source", "") or "").strip()
                    mf = str(p.get("motion_file", "") or "").strip()

                    if src:
                        # Single source of truth: delegate to the node
                        # resolver so pack:/loco:/generate: expansion is
                        # identical to the UI preview and node execute()
                        # paths. Hard-error on failure — a silent warning
                        # here used to let empty amp_motion_files reach
                        # il_train_launcher.py, which then sys.exit(4)'d
                        # deep in a subprocess with a cryptic stack.
                        try:
                            from src.system.nodes.sys_nodes.training_nodes import (
                                ReferenceMotionNode,
                            )
                            resolved = ReferenceMotionNode.resolve_motion_source_files(src)
                        except Exception as exc:
                            from src.system.core.logger import log_error
                            log_error(
                                f"[AMP] failed to resolve motion_source "
                                f"{src!r}: {exc}. AMP training needs at "
                                f"least one clip — aborting launch."
                            )
                            return
                        if not resolved:
                            from src.system.core.logger import log_error
                            log_error(
                                f"[AMP] motion_source {src!r} resolved to "
                                f"zero clips. Pick a 📦 pack: source or a "
                                f"concrete .txt/.json file in the Reference "
                                f"Motion node — aborting launch."
                            )
                            return
                        if len(resolved) == 1 and not src.startswith("pack:"):
                            from src.system.core.logger import log_warning
                            log_warning(
                                f"[AMP] motion_source {src!r} produced a "
                                f"single clip — discriminator training with "
                                f"one clip risks mode collapse. Prefer a "
                                f"📦 pack: source for AMP."
                            )
                        amp_motion_files.extend(resolved)
                    elif mf:
                        amp_motion_files.append(mf)

        # Start Point (BaseAssetNode) → --load_run [+ --unitport_warm_start_actor].
        # Both "resume" and "warm_start_actor" are now genuinely supported:
        # full resume loads actor+critic+optimizer+discriminator; warm-start
        # loads only actor-critic weights (strict=False) and resets the
        # optimizer / discriminator / iter counter, which is what lets us
        # seed AMP-PPO from a pure PPO checkpoint.
        resume_checkpoint, warm_start_actor = self._resolve_il_base_asset_checkpoint(graph)

        # Diagnostic dump of the resolved Train config — helps the user
        # catch canvas hygiene contamination (e.g. duplicate trainers,
        # mismatched task / terrain) BEFORE the Isaac Sim subprocess
        # boots and crashes deep in PhysX with an unreadable stack.
        try:
            from src.system.core.logger import log_info
            log_info("[Canvas] Resolved IL training config:")
            log_info(f"[Canvas]   task_name        = {task_name!r}")
            log_info(f"[Canvas]   algorithm        = {algorithm!r}")
            log_info(f"[Canvas]   num_envs         = {num_envs}")
            log_info(f"[Canvas]   max_iterations   = {max_iterations}")
            log_info(f"[Canvas]   headless         = {headless}")
            log_info(f"[Canvas]   robot_asset_id   = {robot_asset_id!r}")
            log_info(f"[Canvas]   bundle_name      = {il_bundle_name!r}")
            log_info(f"[Canvas]   resume           = {resume_checkpoint or '(scratch)'}")
            if algorithm == "AMP_PPO":
                log_info(f"[Canvas]   amp_motion_files = {len(amp_motion_files)} clip(s)")
                for _i, _f in enumerate(amp_motion_files[:5]):
                    log_info(f"[Canvas]     [{_i}] {_f}")
                if len(amp_motion_files) > 5:
                    log_info(f"[Canvas]     ... ({len(amp_motion_files) - 5} more)")
                log_info(f"[Canvas]   amp_reward_coef  = {amp_reward_coef}")
                log_info(f"[Canvas]   amp_preload      = {amp_num_preload_transitions}")
                log_info(f"[Canvas]   task_reward_lerp = {amp_task_reward_lerp}")
        except Exception:
            pass

        self.start_isaac_lab_training(
            task_name=task_name,
            num_envs=num_envs,
            max_iterations=max_iterations,
            config_file=config_file,
            headless=headless,
            bundle_name=il_bundle_name,
            resume_checkpoint=resume_checkpoint,
            warm_start_actor=warm_start_actor,
            # Phase_1 + phase_3 additive kwargs. All keyword-only so
            # legacy callers (e.g. _on_isaac_lab_legacy_train) keep
            # working without change.
            algorithm=algorithm,
            robot_asset_id=robot_asset_id,
            amp_motion_files=amp_motion_files,
            amp_reward_coef=amp_reward_coef,
            amp_num_preload_transitions=amp_num_preload_transitions,
            amp_transition_dt=amp_transition_dt,
            amp_task_reward_lerp=amp_task_reward_lerp,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_disc_grad_penalty=amp_disc_grad_penalty,
            amp_disc_lr=amp_disc_lr,
            amp_disc_label_smoothing=amp_disc_label_smoothing,
            amp_lerp_schedule=amp_lerp_schedule,
            stage_schedule_json=stage_schedule_json,
            stage_checkpoint_strategy=stage_checkpoint_strategy,
        )

    def _resolve_il_base_asset_checkpoint(self, graph: dict) -> Tuple[str, bool]:
        """Look up the first BaseAssetNode in *graph* and resolve it to an
        absolute checkpoint path + a warm-start flag.

        Returns ``(path, warm_start_actor)``. ``path`` is empty when no
        Start Point is present, when ``load_mode`` is ``scratch``, or
        when resolution fails. ``warm_start_actor`` is True when
        ``load_mode == "warm_start_actor"`` — wired through to the
        launcher via ``--unitport_warm_start_actor``.

        Three start_point source families are handled:

        1. ``run:<abs .pt path>`` — direct run-dir checkpoint, the new
           path exposed by the Start Point picker's "From Run
           Checkpoint" option. Bypasses TrainingAssetRegistry entirely.
        2. ``asset:<asset_id>`` or a bare ``asset_id`` param — legacy
           path through TrainingAssetRegistry.
        3. ``__latest_export__`` — resolved against the current
           project's exported bundle; currently falls through to the
           asset_id branch (the caller is expected to fill asset_id
           when this token is selected).
        """
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        ba_node = next(
            (n for n in nodes if n.get("node_type") == "base_asset"), None
        )
        if ba_node is None:
            return "", False

        p = ba_node.get("parameters", {}) or {}
        load_mode = str(p.get("load_mode", "scratch"))
        if load_mode == "scratch":
            return "", False
        warm = (load_mode == "warm_start_actor")

        start_point = str(p.get("start_point", "") or "").strip()

        # --- Path 1: direct run-dir checkpoint (start_point="run:<abs>") ---
        if start_point.startswith("run:"):
            abs_path = start_point[len("run:"):].strip()
            if abs_path and Path(abs_path).is_file():
                return abs_path, warm
            try:
                from src.system.core.logger import log_warning
                log_warning(
                    f"[Isaac Lab] Start Point run checkpoint {abs_path!r} "
                    f"does not exist — starting from scratch."
                )
            except Exception:
                pass
            return "", False

        # --- Path 2: legacy TrainingAssetRegistry lookup ------------------
        asset_id = str(p.get("asset_id", "") or "")
        if start_point.startswith("asset:") and not asset_id:
            asset_id = start_point[len("asset:"):].strip()
        if not asset_id:
            return "", False

        try:
            from src.system.training.training_asset_registry import TrainingAssetRegistry
            entry = TrainingAssetRegistry().get(asset_id)
            ckpt_file = str(p.get("checkpoint_file", "") or "") or entry.primary_checkpoint
            if not ckpt_file:
                return "", False
            return str(entry.asset_path / ckpt_file), warm
        except Exception as exc:
            try:
                from src.system.core.logger import log_warning
                log_warning(
                    f"[Isaac Lab] Could not resolve Start Point asset "
                    f"'{asset_id}': {exc}. Training will start from scratch."
                )
            except Exception:
                pass
            return "", False

    @staticmethod
    def _infer_il_task_name(graph: dict) -> str:
        """Infer the Isaac Lab registered task name from graph content.

        Matches robot USD path + terrain type to known built-in tasks.
        Falls back to Go2 Flat if no match.
        """
        usd_path = ""
        terrain = "flat"
        for n in graph.get("nodes", []):
            nt = n.get("node_type", "")
            p = n.get("parameters", {})
            if nt == "il_robot_asset":
                usd_path = p.get("usd_path", "").lower()
            elif nt == "il_terrain_config":
                terrain = p.get("terrain_type", "flat").lower()

        is_rough = terrain not in ("flat", "plane")

        # Match known robots
        if "go2" in usd_path:
            return "Isaac-Velocity-Rough-Unitree-Go2-v0" if is_rough else "Isaac-Velocity-Flat-Unitree-Go2-v0"
        if "go1" in usd_path:
            return "Isaac-Velocity-Rough-Unitree-Go1-v0" if is_rough else "Isaac-Velocity-Flat-Unitree-Go1-v0"
        if "anymal" in usd_path and ("_d" in usd_path or "anymal_d" in usd_path):
            return "Isaac-Velocity-Rough-Anymal-D-v0" if is_rough else "Isaac-Velocity-Flat-Anymal-D-v0"
        if "anymal" in usd_path:
            return "Isaac-Velocity-Rough-Anymal-C-v0" if is_rough else "Isaac-Velocity-Flat-Anymal-C-v0"
        if "h1" in usd_path:
            return "Isaac-Velocity-Rough-H1-v0" if is_rough else "Isaac-Velocity-Flat-H1-v0"
        if "g1" in usd_path:
            return "Isaac-Velocity-Rough-G1-v0" if is_rough else "Isaac-Velocity-Flat-G1-v0"
        if "spot" in usd_path:
            return "Isaac-Velocity-Rough-Spot-v0" if is_rough else "Isaac-Velocity-Flat-Spot-v0"

        # Default
        return "Isaac-Velocity-Flat-Unitree-Go2-v0"

    def _on_isaac_lab_legacy_train(self, scene) -> None:
        """Fall back to legacy 3-node IsaacLab pipeline."""
        task_name = "Isaac-Velocity-Flat-Go2-v0"
        num_envs = 4096
        max_iterations = 1500
        resume_checkpoint = ""

        if scene and hasattr(scene, "_logic_nodes"):
            for _nid, node in scene._logic_nodes.items():
                ntype = getattr(node, "node_type", "")
                params = getattr(node, "parameters", {})
                if ntype == "isaac_lab_task":
                    task_name = str(params.get("task_name", task_name))
                    try:
                        num_envs = int(params.get("num_envs", num_envs))
                    except (TypeError, ValueError):
                        pass
                elif ntype == "isaac_lab_train":
                    try:
                        max_iterations = int(params.get("max_iterations", max_iterations))
                    except (TypeError, ValueError):
                        pass
                    # Static parameter fallback — overridden below if a
                    # BaseAssetNode is also present on the canvas.
                    resume_checkpoint = str(params.get("resume_checkpoint", "") or "")

            # Serialize the scene to a graph dict and reuse the unified
            # Start Point resolver so the legacy path picks up the same
            # BaseAssetNode wiring the user drew.
            try:
                graph = {
                    "nodes": [
                        {
                            "node_type": getattr(n, "node_type", ""),
                            "parameters": getattr(n, "parameters", {}) or {},
                        }
                        for n in scene._logic_nodes.values()
                    ],
                }
                ba_ckpt, ba_warm = self._resolve_il_base_asset_checkpoint(graph)
                if ba_ckpt:
                    resume_checkpoint = ba_ckpt
            except Exception:
                ba_warm = False

        self.start_isaac_lab_training(
            task_name=task_name,
            num_envs=num_envs,
            max_iterations=max_iterations,
            resume_checkpoint=resume_checkpoint,
            warm_start_actor=bool(locals().get("ba_warm", False)),
        )

    def start_isaac_lab_training(
        self,
        task_name: str = "Isaac-Velocity-Flat-Go2-v0",
        num_envs: int = 4096,
        max_iterations: int = 1500,
        config_file: str = "",
        headless: bool = True,
        bundle_name: str = "",
        resume_checkpoint: str = "",
        warm_start_actor: bool = False,
        *,
        algorithm: str = "PPO",
        robot_asset_id: str = "",
        amp_motion_files: Optional[List[str]] = None,
        amp_reward_coef: float = 2.0,
        amp_num_preload_transitions: int = 2_000_000,
        amp_transition_dt: float = 0.02,
        amp_task_reward_lerp: float = 0.5,
        amp_replay_buffer_size: int = 1_000_000,
        amp_disc_grad_penalty: float = 10.0,
        amp_disc_lr: float = 1e-4,
        amp_disc_label_smoothing: float = 0.9,
        amp_lerp_schedule: str = "",
        stage_schedule_json: str = "",
        stage_checkpoint_strategy: str = "both",
    ) -> None:
        """Launch Isaac Lab training as an external subprocess.

        The progress is displayed through the same training panel and
        progress strip as SB3 training.
        """
        import uuid
        from src.system.training.isaac_lab_config import IsaacLabConfig
        from src.system.training.isaac_lab_backend import IsaacLabBackend, detect_isaac_lab

        # Detect installation
        detection = detect_isaac_lab()
        if detection is None:
            self._set_training_state("idle")
            try:
                from src.system.core.logger import log_error
                log_error(
                    "Isaac Lab not found. "
                    "Register it via the setup wizard or set the "
                    "ISAAC_LAB_PATH environment variable."
                )
            except Exception:
                pass
            return

        # Guard: one active training at a time
        if self._active_thread is not None and self._active_thread.isRunning():
            return

        # Layout v2: <experiment-or-policy>__<UTC-timestamp>
        from datetime import datetime as _dt, timezone as _tz
        _exp = ""
        try:
            _ws_meta = self._ws_store.load_workspace(self._workspace_name) if self._ws_store else None
            if _ws_meta and _ws_meta.active_experiment_id:
                _exp = _ws_meta.active_experiment_id
        except Exception:
            pass
        if not _exp:
            _exp = (bundle_name or self._workspace_name or "run")
        from src.system.core.project_store import slugify as _slug
        _exp_slug = _slug(_exp, fallback="run")
        _ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{_exp_slug}__{_ts}"
        self._active_run_id = run_id
        # Stash the user-chosen export name so ``_on_isaac_lab_finished``
        # can hand it to ``import_isaac_lab_bundle`` as the on-disk
        # bundle directory name. Empty string => fall back to the run id.
        self._desired_il_bundle_name = str(bundle_name or "").strip()

        # Build log directory under the active project:
        #   <root>/projects/<project_id>/training/runs/<run_id>/
        # Not <root>/<policy_id>/runs/<run_id>/ — that flat layout pollutes the
        # repo root and sidesteps ProjectStore entirely.
        log_dir = ""
        if self._ws_store is not None:
            try:
                project_id = self._ws_store._resolve_project_id(self._workspace_name)
                if not project_id:
                    # Auto-create the project entry if this workspace has not
                    # been persisted yet — mirrors ensure_workspace behaviour.
                    self._ws_store.ensure_workspace(self._workspace_name)
                    project_id = self._ws_store._resolve_project_id(self._workspace_name)
                if project_id:
                    runs_dir = self._ws_store._ps._runs_dir(project_id)
                    ws_root = runs_dir / run_id
                    ws_root.mkdir(parents=True, exist_ok=True)
                    log_dir = str(ws_root)
            except Exception:
                pass

        # headless comes from the il_ppo_trainer node param (graph-level,
        # persisted with the canvas). Default True if not specified.
        panel = self._training_panel if hasattr(self, "_training_panel") else None

        config = IsaacLabConfig(
            task_name=task_name,
            num_envs=num_envs,
            max_iterations=max_iterations,
            log_dir=log_dir,
            run_id=run_id,
            headless=headless,
            isaac_lab_path=detection.get("path", ""),
            isaac_lab_python=detection.get("python", ""),
            isaac_lab_launcher=detection.get("launcher", ""),
            config_file=config_file,
            resume_checkpoint=(resume_checkpoint or None),
            warm_start_actor=bool(warm_start_actor),
            # Phase_1 + phase_3: additive fields (see AMP_design.yaml §3/§4).
            algorithm=algorithm,
            robot_asset_id=robot_asset_id,
            amp_motion_files=list(amp_motion_files or []),
            amp_reward_coef=amp_reward_coef,
            amp_num_preload_transitions=amp_num_preload_transitions,
            amp_transition_dt=amp_transition_dt,
            amp_task_reward_lerp=amp_task_reward_lerp,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_disc_grad_penalty=amp_disc_grad_penalty,
            amp_disc_lr=amp_disc_lr,
            amp_disc_label_smoothing=amp_disc_label_smoothing,
            amp_lerp_schedule=amp_lerp_schedule,
            stage_schedule_json=stage_schedule_json,
            stage_checkpoint_strategy=stage_checkpoint_strategy,
        )

        backend = IsaacLabBackend(config)
        self._isaac_lab_backend = backend

        # Wire messages to UI.
        # IMPORTANT: on_message runs in the IsaacLabBackend daemon thread.
        # All Qt widget updates must be marshaled to the main thread via
        # QMetaObject.invokeMethod or QTimer.singleShot(0, ...).
        if panel:
            panel.start_run(run_id)
            # Start checkpoint watcher if log_dir is available
            if log_dir:
                panel.start_checkpoint_watcher(log_dir)

        from src.system.core.logger import log_info, log_error
        log_info("Isaac Lab training started")
        log_info(f"Starting training run {run_id}")
        log_info(f"task={task_name}  envs={num_envs}  iters={max_iterations}")
        log_info(f"config_file={config_file or '(none)'}")
        log_info(f"log_dir={log_dir or '(none)'}")
        log_info(f"resume_checkpoint={resume_checkpoint or '(scratch)'}")

        # Accumulate messages from the daemon thread into a thread-safe queue.
        # A QTimer on the main thread polls the queue and dispatches to UI.
        import queue
        from PySide6.QtCore import QTimer

        _msg_queue: queue.Queue = queue.Queue()

        # Register IL run in the training run cache so the chart widget picks it up
        from src.system.training.training_run_cache import get_training_run_cache
        _run_cache = get_training_run_cache()
        _il_best_reward = [float("-inf")]  # mutable container for closure
        _ws_policy = getattr(self, "_workspace_name", "") or ""
        _run_cache.ensure_run(
            run_id,
            policy_id=_ws_policy,
            algorithm="PPO (Isaac Lab)",
            total_timesteps=max_iterations,
            status="running",
        )

        def on_message(msg):
            """Called from IsaacLabBackend daemon thread — just enqueue."""
            _msg_queue.put(msg)

        def _poll_messages():
            """Called from main thread QTimer — drain queue and update UI."""
            while not _msg_queue.empty():
                try:
                    msg = _msg_queue.get_nowait()
                except queue.Empty:
                    break
                msg_type = msg.get("type", "")
                try:
                    if msg_type == "log":
                        text = str(msg["data"])
                        if panel:
                            panel.on_log(text)
                    elif msg_type == "progress":
                        step, total, reward, best, ep_len, status_str = msg["data"]
                        self._on_progress_strip(step, total, reward, best, ep_len, status_str)
                        if panel:
                            panel.on_progress(step, total, reward, best, ep_len, status_str)
                    elif msg_type == "metrics":
                        d = dict(msg["data"])
                        _cur_reward = d.get("reward_mean", 0.0)
                        if _cur_reward > _il_best_reward[0]:
                            _il_best_reward[0] = _cur_reward
                        # Write to run cache for the chart widget
                        _run_cache.record_progress(
                            run_id,
                            step=d.get("iteration", 0),
                            total=d.get("total", max_iterations),
                            reward_mean=_cur_reward,
                            best_reward=_il_best_reward[0],
                            ep_len_mean=d.get("ep_len_mean", 0.0),
                            status="running",
                            metrics={
                                "policy_loss": d.get("policy_loss", 0.0),
                                "value_loss": d.get("value_loss", 0.0),
                                "kl": d.get("kl", 0.0),
                                "entropy": d.get("entropy", 0.0),
                                "lr": d.get("lr", 0.0),
                            },
                        )
                        if panel:
                            panel.on_metrics(d)
                        # Update live metrics layers panel
                        if hasattr(self, "_canvas") and hasattr(self._canvas, "_stats_overlay"):
                            _overlay = self._canvas._stats_overlay
                            if hasattr(_overlay, "update_live_metrics"):
                                _overlay.update_live_metrics({
                                    "training": {
                                        "reward_mean": d.get("reward_mean", 0.0),
                                        "ep_len_mean": d.get("ep_len_mean", 0.0),
                                        "fps": d.get("fps", 0.0),
                                    },
                                    "losses": {
                                        "policy_loss": d.get("policy_loss", 0.0),
                                        "value_loss": d.get("value_loss", 0.0),
                                    },
                                    "algorithm": {
                                        "kl": d.get("kl", 0.0),
                                        "entropy": d.get("entropy", 0.0),
                                        "lr": d.get("lr", 0.0),
                                    },
                                })
                    elif msg_type == "finished":
                        d = dict(msg["data"])
                        log_info(f"Training finished: {d}")
                        self._on_isaac_lab_finished(d)
                        if panel:
                            panel.on_finished(d.get("bundle_path", ""))
                    elif msg_type == "cancelled":
                        log_info("Training cancelled")
                        self._set_training_state("idle")
                        if panel:
                            panel.on_cancelled()
                    elif msg_type == "error":
                        err = str(msg["data"])
                        log_error(f"Isaac Lab training error: {err}")
                        self._set_training_state("idle")
                        if panel:
                            panel.on_error(err)
                except Exception as _ex:
                    log_error(f"Isaac Lab message dispatch error ({msg_type}): {_ex}")

        self._il_poll_timer = QTimer()
        self._il_poll_timer.setInterval(200)  # poll every 200ms
        self._il_poll_timer.timeout.connect(_poll_messages)
        self._il_poll_timer.start()

        backend.on_message = on_message
        self._set_training_state("running")
        backend.start()
        log_info(f"backend.start() called, is_running={backend.is_running}")

        # Force the chart overlay to pick up the new run (shared helper used
        # by both SB3 and Isaac Lab start paths).
        self._force_refresh_stats_overlay()

    def _on_il_review_clicked(self) -> None:
        """Launch Isaac Sim viewport to replay the trained policy."""
        from src.system.core.logger import log_info
        backend = getattr(self, "_isaac_lab_backend", None)
        if backend is None or backend.config is None:
            return

        # Find the latest checkpoint directory
        from pathlib import Path
        log_dir = backend.config.log_dir
        il_root = Path(backend.config.isaac_lab_path) / "logs" / "rsl_rl"
        checkpoint_dir = ""

        # Search in RSL-RL log dirs for the latest model_*.pt
        for search_root in [Path(log_dir), il_root]:
            if not search_root.is_dir():
                continue
            models = sorted(search_root.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if models:
                checkpoint_dir = str(models[0].parent)
                break

        if not checkpoint_dir:
            log_info("No trained checkpoint found for review.")
            return

        log_info(f"Launching Isaac Sim review: {checkpoint_dir}")
        backend.launch_review(checkpoint_dir)

    def _on_isaac_lab_finished(self, data: dict) -> None:
        """Handle Isaac Lab training completion — auto-import bundle."""
        # Stop the message poll timer
        if hasattr(self, "_il_poll_timer") and self._il_poll_timer is not None:
            self._il_poll_timer.stop()
            self._il_poll_timer = None

        reward = data.get("reward_mean", float("-inf"))
        step = data.get("step", 0)
        if reward == float("-inf") and step == 0:
            from src.system.core.logger import log_warning
            log_warning("[Isaac Lab] Process exited but no training occurred (config error?)")
            self._set_training_state("idle")
            return
        self._set_training_state("done")
        # Final scan then stop the checkpoint watcher
        panel = self._training_panel if hasattr(self, "_training_panel") else None
        if panel:
            panel.stop_checkpoint_watcher()
        bundle_path = data.get("bundle_path", "")
        if not bundle_path:
            return
        try:
            from pathlib import Path
            from src.system.service.checkpoint_registry import CheckpointRegistry

            # RSL-RL saves inside its own log subdirectory, not our log_dir.
            # Search for env.yaml / model_*.pt in subdirectories.
            bp = Path(bundle_path)
            if not (bp / "env.yaml").exists():
                # Look in nested subdirs (RSL-RL structure: logs/rsl_rl/<exp>/<timestamp>/)
                candidates = list(bp.rglob("env.yaml"))
                if not candidates:
                    # Also check the RSL-RL default log location
                    il_path = getattr(self, "_isaac_lab_backend", None)
                    if il_path and hasattr(il_path, "config"):
                        rsl_logs = Path(il_path.config.isaac_lab_path) / "logs" / "rsl_rl"
                        if rsl_logs.is_dir():
                            candidates = sorted(rsl_logs.rglob("env.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    bp = candidates[0].parent
                    # RSL-RL + our launcher writes env.yaml into <log_dir>/params/
                    # while model_*.pt checkpoints sit in <log_dir>/ itself.
                    # Climb out of params/ so the registry sees both.
                    if bp.name == "params":
                        bp = bp.parent
                else:
                    from src.system.core.logger import log_info
                    log_info("Isaac Lab training complete. Bundle auto-import skipped (env.yaml not found).")
                    return

            # CheckpointRegistry picks up the active project via
            # ProjectSession (set by MainWindow) so the bundle lands in
            # ``projects/<active_id>/checkpoints/<bundle_name>/`` rather
            # than the deprecated ``custom_mods/training/checkpoints/``
            # path. ``policy_id`` honours the user-chosen name set on
            # the IL Policy Exporter node — without it the bundle would
            # be named after the throwaway "il_run_<hash>" run id.
            registry = CheckpointRegistry()
            desired_pid = str(getattr(self, "_desired_il_bundle_name", "") or "").strip() or None
            # Thread the gym task ID through so the bundle's source.json
            # remembers what task it was trained on. The Isaac Sim review
            # fallback in vis_check_runner reads this back to launch play.py
            # against the matching env (otherwise it has to guess from the
            # bundle name and gets it wrong for non-default tasks).
            il_task_name = ""
            try:
                il_backend_obj = getattr(self, "_isaac_lab_backend", None)
                if il_backend_obj is not None and getattr(il_backend_obj, "config", None) is not None:
                    il_task_name = str(il_backend_obj.config.task_name or "")
            except Exception:
                il_task_name = ""
            entry = registry.import_isaac_lab_bundle(
                bp,
                policy_id=desired_pid,
                task_name=il_task_name or None,
            )
            self._last_export_bundle_path = str(entry.bundle_path)
            # Refresh registries + canvas nodes so Export's bundle picker
            # and Review button see the imported bundle immediately.
            self._refresh_checkpoint_registry()
            self._refresh_canvas_node_widgets()
            self.checkpoint_exported.emit(str(entry.bundle_path))
        except Exception as exc:
            try:
                from src.system.core.logger import log_error
                log_error(f"Isaac Lab bundle import failed: {exc}")
            except Exception:
                pass

    # ==================================================================
    # Cloud SSH Training (remote Isaac Lab via SSH)
    # ==================================================================

    def _on_cloud_train_requested(self) -> None:
        """Handle Start click when Cloud backend is selected.

        Reuses the same Isaac Lab graph compilation as the local path,
        then prompts the user to select a remote server and launches
        training via RemoteSSHBackend.
        """
        scene = None
        if hasattr(self, "_canvas"):
            scene = getattr(self._canvas, "scene", None) or getattr(self._canvas, "_scene", None)

        has_granular = False
        if scene and hasattr(scene, "serialize_training_graph"):
            graph = scene.serialize_training_graph(for_compiler=False)
            node_types = [n.get("node_type", "") for n in graph.get("nodes", [])]
            for nt in node_types:
                if nt.startswith("il_") or nt in ("rewards", "terminations", "domain_rand"):
                    has_granular = True
                    break

        if not has_granular:
            try:
                from src.system.core.logger import log_error
                log_error("Cloud training requires Isaac Lab (granular IL) nodes on the canvas.")
            except Exception:
                pass
            return

        # Resolve selected cloud server from the float bar target combo.
        float_bar = getattr(self._canvas, "_float_bar", None) if hasattr(self, "_canvas") else None
        server_name = float_bar.selected_cloud_server_name() if float_bar else ""

        if not server_name:
            # Fallback: pop the dialog if somehow no server is pre-selected
            from bin.pages.training.remote_server_dialog import RemoteServerSelectDialog
            dlg = RemoteServerSelectDialog(self)
            selected_server = [None]

            def _on_selected(server):
                selected_server[0] = server

            dlg.server_selected.connect(_on_selected)
            if dlg.exec() != dlg.DialogCode.Accepted or selected_server[0] is None:
                return
            self._on_cloud_granular_train(graph, selected_server[0])
            return

        from src.system.training.remote_server_store import get_remote_server_store
        server = get_remote_server_store().get_server(server_name)
        if server is None:
            try:
                from src.system.core.logger import log_error
                log_error(f"Cloud server '{server_name}' not found in registry.")
            except Exception:
                pass
            return

        # Password-auth servers need runtime password (never stored on disk).
        if server.auth_method == "password" and not server.password:
            from PySide6.QtWidgets import QInputDialog, QLineEdit
            pwd, ok = QInputDialog.getText(
                self, "SSH Password",
                f"Password for {server.username}@{server.host}:",
                QLineEdit.EchoMode.Password,
            )
            if not ok or not pwd:
                return
            server.password = pwd

        # Compile the graph (same path as _on_isaac_lab_granular_train).
        self._on_cloud_granular_train(graph, server)

    def _on_cloud_granular_train(self, graph: dict, server) -> None:
        """Compile the granular IL graph and launch cloud training."""
        from src.system.core.logger import log_info
        from typing import List

        algorithm = "PPO"
        il_bundle_name = ""
        for n in graph.get("nodes", []):
            nt = n.get("node_type", "")
            if nt in ("il_ppo_trainer", "amp_ppo_trainer"):
                p = n.get("parameters", {}) or {}
                mode = str(p.get("training_mode", "PPO") or "PPO").strip()
                if nt == "amp_ppo_trainer" or mode == "AMP_PPO":
                    algorithm = "AMP_PPO"
            elif nt == "il_policy_exporter":
                raw = str(
                    n.get("parameters", {}).get("bundle_name", "") or ""
                ).strip()
                if raw and raw != "<NEW>":
                    il_bundle_name = raw

        from src.system.training.isaac_lab_config_compiler import IsaacLabConfigCompiler
        compiler = IsaacLabConfigCompiler(graph)

        body_errors = compiler.validate_body_mapping()
        if body_errors:
            from src.system.core.logger import log_error
            for e in body_errors:
                log_error(e)
            log_error(
                "Training blocked: resolve all body mapping roles before starting."
            )
            return

        config_path = compiler.compile_to_file()
        config_file = str(config_path)

        task_name = self._infer_il_task_name(graph)

        num_envs = 4096
        max_iterations = 1500
        robot_asset_id = ""
        amp_motion_files: List[str] = []
        amp_reward_coef = 2.0
        amp_num_preload_transitions = 2_000_000
        amp_transition_dt = 0.02
        amp_task_reward_lerp = 0.5
        amp_replay_buffer_size = 1_000_000
        amp_disc_grad_penalty = 10.0
        amp_disc_lr = 1e-4
        amp_disc_label_smoothing = 0.9
        amp_lerp_schedule = ""
        stage_schedule_json = ""
        stage_checkpoint_strategy = "both"
        resume_checkpoint = ""
        warm_start_actor = False

        for n in graph.get("nodes", []):
            nt = n.get("node_type", "")
            p = n.get("parameters", {}) or {}
            if nt in ("il_ppo_trainer", "amp_ppo_trainer"):
                num_envs = int(p.get("num_envs", num_envs) or num_envs)
                max_iterations = int(p.get("max_iterations", max_iterations) or max_iterations)
                amp_reward_coef = float(p.get("amp_reward_coef", amp_reward_coef) or amp_reward_coef)
                amp_num_preload_transitions = int(p.get("amp_num_preload_transitions", amp_num_preload_transitions) or amp_num_preload_transitions)
                amp_transition_dt = float(p.get("amp_transition_dt", amp_transition_dt) or amp_transition_dt)
                amp_task_reward_lerp = float(p.get("amp_task_reward_lerp", amp_task_reward_lerp) or amp_task_reward_lerp)
                amp_replay_buffer_size = int(p.get("amp_replay_buffer_size", amp_replay_buffer_size) or amp_replay_buffer_size)
                amp_disc_grad_penalty = float(p.get("amp_disc_grad_penalty", amp_disc_grad_penalty) or amp_disc_grad_penalty)
                amp_disc_lr = float(p.get("amp_disc_lr", amp_disc_lr) or amp_disc_lr)
                amp_disc_label_smoothing = float(p.get("amp_disc_label_smoothing", amp_disc_label_smoothing) or amp_disc_label_smoothing)
                amp_lerp_schedule = str(p.get("amp_lerp_schedule", "") or "")
            elif nt == "il_robot_asset":
                robot_asset_id = str(p.get("asset_id", "") or "")
            elif nt == "il_reference_motion":
                raw_files = p.get("motion_files", "")
                if isinstance(raw_files, str) and raw_files:
                    amp_motion_files = [f.strip() for f in raw_files.split(",") if f.strip()]
                elif isinstance(raw_files, list):
                    amp_motion_files = [str(f) for f in raw_files if f]
            elif nt == "il_base_asset":
                resume_checkpoint = str(p.get("checkpoint_path", "") or "")
                warm_start_actor = bool(p.get("warm_start_actor", False))
            elif nt == "il_stage_schedule":
                stage_schedule_json = str(p.get("stage_schedule_json", "") or "")
                stage_checkpoint_strategy = str(p.get("checkpoint_strategy", "both") or "both")

        self.start_cloud_training(
            server=server,
            task_name=task_name,
            num_envs=num_envs,
            max_iterations=max_iterations,
            config_file=config_file,
            bundle_name=il_bundle_name,
            resume_checkpoint=resume_checkpoint,
            warm_start_actor=warm_start_actor,
            algorithm=algorithm,
            robot_asset_id=robot_asset_id,
            amp_motion_files=amp_motion_files,
            amp_reward_coef=amp_reward_coef,
            amp_num_preload_transitions=amp_num_preload_transitions,
            amp_transition_dt=amp_transition_dt,
            amp_task_reward_lerp=amp_task_reward_lerp,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_disc_grad_penalty=amp_disc_grad_penalty,
            amp_disc_lr=amp_disc_lr,
            amp_disc_label_smoothing=amp_disc_label_smoothing,
            amp_lerp_schedule=amp_lerp_schedule,
            stage_schedule_json=stage_schedule_json,
            stage_checkpoint_strategy=stage_checkpoint_strategy,
        )

    def start_cloud_training(
        self,
        server,
        task_name: str = "Isaac-Velocity-Flat-Go2-v0",
        num_envs: int = 4096,
        max_iterations: int = 1500,
        config_file: str = "",
        bundle_name: str = "",
        resume_checkpoint: str = "",
        warm_start_actor: bool = False,
        *,
        algorithm: str = "PPO",
        robot_asset_id: str = "",
        amp_motion_files=None,
        amp_reward_coef: float = 2.0,
        amp_num_preload_transitions: int = 2_000_000,
        amp_transition_dt: float = 0.02,
        amp_task_reward_lerp: float = 0.5,
        amp_replay_buffer_size: int = 1_000_000,
        amp_disc_grad_penalty: float = 10.0,
        amp_disc_lr: float = 1e-4,
        amp_disc_label_smoothing: float = 0.9,
        amp_lerp_schedule: str = "",
        stage_schedule_json: str = "",
        stage_checkpoint_strategy: str = "both",
    ) -> None:
        """Launch Isaac Lab training on a remote server via SSH.

        Mirrors ``start_isaac_lab_training`` but uses ``RemoteSSHBackend``
        instead of ``IsaacLabBackend``. The same message polling pattern
        ensures the Training Panel works identically.
        """
        import uuid
        from src.system.training.isaac_lab_config import IsaacLabConfig
        from src.system.training.remote_ssh_backend import RemoteSSHBackend

        # Guard: one active training at a time.
        if self._active_thread is not None and self._active_thread.isRunning():
            return

        # Layout v2 run id.
        from datetime import datetime as _dt, timezone as _tz
        _exp = ""
        try:
            _ws_meta = self._ws_store.load_workspace(self._workspace_name) if self._ws_store else None
            if _ws_meta and _ws_meta.active_experiment_id:
                _exp = _ws_meta.active_experiment_id
        except Exception:
            pass
        if not _exp:
            _exp = (bundle_name or self._workspace_name or "cloud_run")
        from src.system.core.project_store import slugify as _slug
        _exp_slug = _slug(_exp, fallback="cloud_run")
        _ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{_exp_slug}__{_ts}"
        self._active_run_id = run_id
        self._desired_il_bundle_name = str(bundle_name or "").strip()

        # Build log directory under the active project.
        log_dir = ""
        if self._ws_store is not None:
            try:
                project_id = self._ws_store._resolve_project_id(self._workspace_name)
                if not project_id:
                    self._ws_store.ensure_workspace(self._workspace_name)
                    project_id = self._ws_store._resolve_project_id(self._workspace_name)
                if project_id:
                    runs_dir = self._ws_store._ps._runs_dir(project_id)
                    ws_root = runs_dir / run_id
                    ws_root.mkdir(parents=True, exist_ok=True)
                    log_dir = str(ws_root)
            except Exception:
                pass

        panel = self._training_panel if hasattr(self, "_training_panel") else None

        config = IsaacLabConfig(
            task_name=task_name,
            num_envs=num_envs,
            max_iterations=max_iterations,
            log_dir=log_dir,
            run_id=run_id,
            headless=True,  # always headless on remote
            config_file=config_file,
            resume_checkpoint=(resume_checkpoint or None),
            warm_start_actor=bool(warm_start_actor),
            algorithm=algorithm,
            robot_asset_id=robot_asset_id,
            amp_motion_files=list(amp_motion_files or []),
            amp_reward_coef=amp_reward_coef,
            amp_num_preload_transitions=amp_num_preload_transitions,
            amp_transition_dt=amp_transition_dt,
            amp_task_reward_lerp=amp_task_reward_lerp,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_disc_grad_penalty=amp_disc_grad_penalty,
            amp_disc_lr=amp_disc_lr,
            amp_disc_label_smoothing=amp_disc_label_smoothing,
            amp_lerp_schedule=amp_lerp_schedule,
            stage_schedule_json=stage_schedule_json,
            stage_checkpoint_strategy=stage_checkpoint_strategy,
        )

        backend = RemoteSSHBackend(config, server)
        self._cloud_backend = backend

        if panel:
            panel.start_run(run_id)
            if log_dir:
                panel.start_checkpoint_watcher(log_dir)

        from src.system.core.logger import log_info, log_error
        log_info("Cloud training started")
        log_info(f"Starting cloud training run {run_id}")
        log_info(f"Server: {server.username}@{server.host}:{server.port}")
        log_info(f"task={task_name}  envs={num_envs}  iters={max_iterations}")
        log_info(f"config_file={config_file or '(none)'}")
        log_info(f"log_dir={log_dir or '(none)'}")

        import queue
        from PySide6.QtCore import QTimer

        _msg_queue: queue.Queue = queue.Queue()

        from src.system.training.training_run_cache import get_training_run_cache
        _run_cache = get_training_run_cache()
        _cloud_best_reward = [float("-inf")]
        _ws_policy = getattr(self, "_workspace_name", "") or ""
        _run_cache.ensure_run(
            run_id,
            policy_id=_ws_policy,
            algorithm=f"{algorithm} (Cloud)",
            total_timesteps=max_iterations,
            status="running",
        )

        def on_message(msg):
            _msg_queue.put(msg)

        def _poll_messages():
            while not _msg_queue.empty():
                try:
                    msg = _msg_queue.get_nowait()
                except queue.Empty:
                    break
                msg_type = msg.get("type", "")
                try:
                    if msg_type == "log":
                        text = str(msg["data"])
                        if panel:
                            panel.on_log(text)
                    elif msg_type == "progress":
                        step, total, reward, best, ep_len, status_str = msg["data"]
                        self._on_progress_strip(step, total, reward, best, ep_len, status_str)
                        if panel:
                            panel.on_progress(step, total, reward, best, ep_len, status_str)
                    elif msg_type == "metrics":
                        d = dict(msg["data"])
                        _cur_reward = d.get("reward_mean", 0.0)
                        if _cur_reward > _cloud_best_reward[0]:
                            _cloud_best_reward[0] = _cur_reward
                        _run_cache.record_progress(
                            run_id,
                            step=d.get("iteration", 0),
                            total=d.get("total", max_iterations),
                            reward_mean=_cur_reward,
                            best_reward=_cloud_best_reward[0],
                            ep_len_mean=d.get("ep_len_mean", 0.0),
                            status="running",
                            metrics={
                                "policy_loss": d.get("policy_loss", 0.0),
                                "value_loss": d.get("value_loss", 0.0),
                                "kl": d.get("kl", 0.0),
                                "entropy": d.get("entropy", 0.0),
                                "lr": d.get("lr", 0.0),
                            },
                        )
                        if panel:
                            panel.on_metrics(d)
                        if hasattr(self, "_canvas") and hasattr(self._canvas, "_stats_overlay"):
                            _overlay = self._canvas._stats_overlay
                            if hasattr(_overlay, "update_live_metrics"):
                                _overlay.update_live_metrics({
                                    "training": {
                                        "reward_mean": d.get("reward_mean", 0.0),
                                        "ep_len_mean": d.get("ep_len_mean", 0.0),
                                        "fps": d.get("fps", 0.0),
                                    },
                                    "losses": {
                                        "policy_loss": d.get("policy_loss", 0.0),
                                        "value_loss": d.get("value_loss", 0.0),
                                    },
                                    "algorithm": {
                                        "kl": d.get("kl", 0.0),
                                        "entropy": d.get("entropy", 0.0),
                                        "lr": d.get("lr", 0.0),
                                    },
                                })
                    elif msg_type == "finished":
                        d = dict(msg["data"])
                        log_info(f"Cloud training finished: {d}")
                        self._on_cloud_training_finished(d)
                        if panel:
                            panel.on_finished(d.get("bundle_path", ""))
                    elif msg_type == "cancelled":
                        log_info("Cloud training cancelled")
                        self._set_training_state("idle")
                        if panel:
                            panel.on_cancelled()
                    elif msg_type == "error":
                        err = str(msg["data"])
                        log_error(f"Cloud training error: {err}")
                        self._set_training_state("idle")
                        if panel:
                            panel.on_error(err)
                except Exception as _ex:
                    log_error(f"Cloud message dispatch error ({msg_type}): {_ex}")

        self._cloud_poll_timer = QTimer()
        self._cloud_poll_timer.setInterval(200)
        self._cloud_poll_timer.timeout.connect(_poll_messages)
        self._cloud_poll_timer.start()

        backend.on_message = on_message
        self._set_training_state("running")
        backend.start()
        log_info(f"cloud backend.start() called, is_running={backend.is_running}")

        self._force_refresh_stats_overlay()

    def _on_cloud_training_finished(self, data: dict) -> None:
        """Handle cloud training completion."""
        if hasattr(self, "_cloud_poll_timer") and self._cloud_poll_timer is not None:
            self._cloud_poll_timer.stop()
            self._cloud_poll_timer = None

        reward = data.get("reward_mean", float("-inf"))
        step = data.get("step", 0)
        if reward == float("-inf") and step == 0:
            from src.system.core.logger import log_warning
            log_warning("[Cloud] Process exited but no training occurred (config error?)")
            self._set_training_state("idle")
            return
        self._set_training_state("done")

        panel = self._training_panel if hasattr(self, "_training_panel") else None
        if panel:
            panel.stop_checkpoint_watcher()

        bundle_path = data.get("bundle_path", "")
        if not bundle_path:
            return

        # Attempt auto-import (same logic as _on_isaac_lab_finished).
        try:
            from pathlib import Path
            from src.system.service.checkpoint_registry import CheckpointRegistry

            bp = Path(bundle_path)
            if not (bp / "env.yaml").exists():
                candidates = list(bp.rglob("env.yaml"))
                if candidates:
                    bp = candidates[0].parent
                    if bp.name == "params":
                        bp = bp.parent
                else:
                    from src.system.core.logger import log_info
                    log_info("Cloud training complete. Bundle auto-import skipped (env.yaml not found).")
                    return

            registry = CheckpointRegistry()
            desired_pid = str(getattr(self, "_desired_il_bundle_name", "") or "").strip() or None
            il_task_name = ""
            try:
                cloud_backend_obj = getattr(self, "_cloud_backend", None)
                if cloud_backend_obj is not None and getattr(cloud_backend_obj, "config", None) is not None:
                    il_task_name = str(cloud_backend_obj.config.task_name or "")
            except Exception:
                il_task_name = ""
            entry = registry.import_isaac_lab_bundle(
                bp,
                policy_id=desired_pid,
                task_name=il_task_name or None,
            )
            self._last_export_bundle_path = str(entry.bundle_path)
            self._refresh_checkpoint_registry()
            self._refresh_canvas_node_widgets()
            self.checkpoint_exported.emit(str(entry.bundle_path))
        except Exception as exc:
            try:
                from src.system.core.logger import log_error
                log_error(f"Cloud training bundle import failed: {exc}")
            except Exception:
                pass







