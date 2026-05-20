"""Live system monitor — CPU% / RAM / GPU% / VRAM with ' | ' separators.

Ported from DEMO ``bin/pages/training/training_workspace_window.py:SysMonitorWidget``.
Theme integration is rewritten to go through ``Config.get_color`` /
``Config.get_font_size`` per RELEASE/CLAUDE.md §5 (no hex literals in widgets).

Optional runtime deps for live readings:
    psutil  — CPU/RAM (mandatory for any meaningful output)
    pynvml  — NVIDIA GPU (preferred, in-process)
    nvidia-smi  — NVIDIA GPU (subprocess fallback)
    rocm-smi    — AMD GPU
When none are present the GPU/VRAM fields display ``N/A`` gracefully.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from unitport_sdk import Config, LaviLabel


class SysMonitorWidget(QWidget):
    """Compact live system monitor with ' | ' text separators."""

    _INTERVAL_MS = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sysMonitorWidget")
        # Required for QSS background-color to render on a plain QWidget.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Cached title-tier color (sub_t2). Set in apply_theme so _refresh
        # can emit rich-text spans for the "CPU"/"RAM"/"GPU"/"VRAM" prefixes.
        self._title_color: str = Config.get_color("sub_t2", "#777777")
        self._build_ui()
        self.apply_theme()
        self._timer = QTimer(self)
        self._timer.setInterval(self._INTERVAL_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(6, 0, 6, 0)
        # Wider gap between adjacent groups so the "CPU | RAM | GPU | VRAM"
        # row reads as distinct columns rather than one dense run.
        hbox.setSpacing(12)

        small = Config.get_font_size("size_small", 12)
        # Initialise with widest possible text so adjustSize() measures correctly.
        # GB values use 4-digit max (1000G ≈ 1 TB) to avoid overflow on high-mem hosts.
        self._cpu_lbl = LaviLabel("", default="CPU 100%", kind="content", size=small)
        self._ram_lbl = LaviLabel("", default="RAM 1000.00/1000G", kind="content", size=small)
        self._gpu_lbl = LaviLabel("", default="GPU 100%", kind="content", size=small)
        self._vram_lbl = LaviLabel("", default="VRAM 1000.00/1000G", kind="content", size=small)
        for lbl in (self._cpu_lbl, self._ram_lbl, self._gpu_lbl, self._vram_lbl):
            lbl.setObjectName("sysMonitorLabel")
            # Titles ("CPU"/"RAM"/...) are rendered via inline <span> spans
            # so they can carry a distinct theme color from the value text.
            lbl.setTextFormat(Qt.TextFormat.RichText)

        self._sep_labels: list[LaviLabel] = []
        for lbl in (self._cpu_lbl, self._ram_lbl, self._gpu_lbl, self._vram_lbl):
            hbox.addWidget(lbl)
            if lbl is not self._vram_lbl:
                sep = LaviLabel("", default=" | ", kind="content", size=small)
                sep.setObjectName("sysMonitorSep")
                self._sep_labels.append(sep)
                hbox.addWidget(sep)

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------
    def _title(self, name: str) -> str:
        """Render a title prefix ("CPU"/"RAM"/...) in the sub_t2 tier."""
        return f"<span style=\"color:{self._title_color};\">{name}</span>"

    def _refresh(self) -> None:
        # ── CPU / RAM (psutil) ────────────────────────────────────────
        try:
            import psutil
            cpu = int(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            ram_used_gb = mem.used / 1024 ** 3
            ram_total_gb = mem.total / 1024 ** 3
            self._cpu_lbl.setText(f"{self._title('CPU')} {cpu}%")
            self._ram_lbl.setText(
                f"{self._title('RAM')} {ram_used_gb:.2f}/{ram_total_gb:.0f}G"
            )
        except Exception:
            self._cpu_lbl.setText(f"{self._title('CPU')} -")
            self._ram_lbl.setText(f"{self._title('RAM')} -")

        # ── GPU / VRAM — cross-vendor (NVIDIA → AMD → fallback) ──────
        gpu_pct: float | None = None
        vram_used_gb: float | None = None
        vram_total_gb: float | None = None

        # 1) pynvml — NVIDIA preferred (no subprocess)
        if gpu_pct is None:
            try:
                import pynvml  # nvidia-ml-py3
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_pct = float(util.gpu)
                vram_used_gb = meminfo.used / 1024 ** 3
                vram_total_gb = meminfo.total / 1024 ** 3
            except Exception:
                pass

        # 2) nvidia-smi subprocess — NVIDIA fallback
        if gpu_pct is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        gpu_pct = float(parts[0])
                        vram_used_gb = float(parts[1]) / 1024  # MiB → GiB
                        vram_total_gb = float(parts[2]) / 1024
            except Exception:
                pass

        # 3) rocm-smi — AMD
        if gpu_pct is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        # header: GPU use (%),VRAM Total (B),VRAM Used (B)
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
                            gpu_pct = float(parts[0])
                            vram_used_gb = float(parts[2]) / 1024 ** 3
                            vram_total_gb = float(parts[1]) / 1024 ** 3
                            break
            except Exception:
                pass

        if gpu_pct is not None and vram_used_gb is not None and vram_total_gb is not None:
            self._gpu_lbl.setText(f"{self._title('GPU')} {int(gpu_pct)}%")
            self._vram_lbl.setText(
                f"{self._title('VRAM')} {vram_used_gb:.2f}/{vram_total_gb:.0f}G"
            )
        else:
            self._gpu_lbl.setText(f"{self._title('GPU')} N/A")
            self._vram_lbl.setText(f"{self._title('VRAM')} N/A")

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        text_color = Config.get_color("sub_t1", "#A0A0A0")
        sep_color = Config.get_color("sub_t2", "#777777")
        font_size = Config.get_font_size("size_small", 12)
        # Refresh cached title color; re-emit text so the title-tier span
        # picks up the new palette without waiting for the next QTimer tick.
        self._title_color = sep_color
        self.setStyleSheet(
            f"#sysMonitorWidget {{ background: transparent; border: none; }}"
            f"QLabel#sysMonitorLabel {{ color: {text_color}; font-size: {font_size}px; "
            f"background: transparent; border: none; }}"
            f"QLabel#sysMonitorSep {{ color: {sep_color}; font-size: {font_size}px; "
            f"background: transparent; border: none; }}"
        )
        # Only re-render after the value labels exist (apply_theme is called
        # from __init__ once before the first _refresh() — guard against that).
        if getattr(self, "_cpu_lbl", None) is not None:
            self._refresh()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()


__all__ = ["SysMonitorWidget"]
