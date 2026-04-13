#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainingPanel - Circle 7 S2 mock training UI.

The monitor area now renders only the training log. Progress/reward/status
widgets are retained as non-visual attributes so existing signal wiring and
tests can continue to drive them.
"""

from __future__ import annotations

from typing import Optional

from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from src.system.core.theme_manager import get_color_for_theme
from src.system.core.logger import log_error, log_info, log_success, log_warning


def get_color(color_key: str, fallback: str = "#FFFFFF") -> str:
    """Training Ground widgets intentionally stay on the dark palette for now."""
    return get_color_for_theme(color_key, "dark", fallback)


class TrainingPanel(QWidget):
    """
    Training progress panel (S2 shell).

    Signals
    -------
    cancel_requested()
        Emitted when the user clicks the Cancel button.
    """

    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trainingPanel")
        self._build_ui()
        self.apply_theme()
        self.reset()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(0)

        # Retained for compatibility with existing update flow.
        self.title_label = QLabel("Training")
        self.title_label.setObjectName("trainingTitle")

        self.run_id_label = QLabel("")
        self.run_id_label.setObjectName("trainingRunId")
        self.run_id_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("trainingProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")

        self.step_label = QLabel("Step: 0 / 0")
        self.step_label.setObjectName("trainingStepLabel")

        self.reward_mean_label = QLabel("Reward Mean: —")
        self.reward_mean_label.setObjectName("trainingRewardLabel")

        self.reward_best_label = QLabel("Best: —")
        self.reward_best_label.setObjectName("trainingRewardLabel")

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("trainingStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.log_area = QPlainTextEdit()
        self.log_area.setObjectName("trainingLogArea")
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Training log will appear here...")
        self.log_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self.log_area, 1)

        # ── Isaac Lab visualization UI (Phase A shell) ────────────────
        from bin.pages.training.checkpoint_timeline import CheckpointTimeline

        # Checkpoint timeline strip
        self.checkpoint_timeline = CheckpointTimeline(self)
        self.checkpoint_timeline.preview_requested.connect(self._on_preview_requested)
        root.addWidget(self.checkpoint_timeline)

        # Button bar: preview button, charts toggle
        # (headless toggle lives on the il_ppo_trainer node parameter now,
        #  so it is persisted with the canvas graph.)
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 6, 0, 6)
        btn_bar.setSpacing(8)

        self.preview_btn = QPushButton("Preview in MuJoCo")
        self.preview_btn.setObjectName("previewMujocoBtn")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip("Select a checkpoint first")
        self.preview_btn.setCursor(Qt.PointingHandCursor)
        self.preview_btn.clicked.connect(self._on_preview_btn_clicked)
        btn_bar.addWidget(self.preview_btn)

        self.review_btn = QPushButton("Review in Isaac Sim")
        self.review_btn.setObjectName("reviewIsaacBtn")
        self.review_btn.setEnabled(False)
        self.review_btn.setToolTip("Replay trained policy in Isaac Sim viewport")
        self.review_btn.setCursor(Qt.PointingHandCursor)
        self.review_btn.setVisible(False)  # shown only for Isaac Lab backend
        btn_bar.addWidget(self.review_btn)

        btn_bar.addStretch(1)

        self.charts_btn = QPushButton("Charts")
        self.charts_btn.setObjectName("chartsToggleBtn")
        self.charts_btn.setCheckable(True)
        self.charts_btn.setChecked(False)
        self.charts_btn.setToolTip("Toggle real-time training metrics charts")
        self.charts_btn.setCursor(Qt.PointingHandCursor)
        self.charts_btn.toggled.connect(self._on_charts_toggled)
        btn_bar.addWidget(self.charts_btn)

        root.addLayout(btn_bar)

        # Charts area — 2x3 pyqtgraph grid
        self.charts_area = QFrame(self)
        self.charts_area.setObjectName("chartsArea")
        self.charts_area.setFrameShape(QFrame.Shape.StyledPanel)
        self.charts_area.setFixedHeight(360)
        self._build_charts(self.charts_area)
        self.charts_area.setVisible(False)
        root.addWidget(self.charts_area)

        # Track selected checkpoint path for preview button
        self._selected_checkpoint_path: str = ""

        self.vis_banner = QFrame()
        self.vis_banner.setObjectName("visCheckBanner")
        self.vis_banner.setFrameShape(QFrame.Shape.StyledPanel)
        vis_banner_layout = QHBoxLayout(self.vis_banner)
        vis_banner_layout.setContentsMargins(12, 8, 12, 8)
        vis_banner_layout.setSpacing(10)
        self.vis_banner_label = QLabel(
            "MuJoCo viewer open - close the window to resume training"
        )
        self.vis_banner_label.setObjectName("visCheckBannerLabel")
        self.vis_banner_label.setWordWrap(True)
        vis_banner_layout.addWidget(self.vis_banner_label, 1)
        self.vis_banner.setVisible(False)

        self.cancel_btn = QPushButton("Cancel Training")
        self.cancel_btn.setObjectName("trainingCancelButton")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)

        self.bundle_label = QLabel("")
        self.bundle_label.setObjectName("trainingBundleLabel")
        self.bundle_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def apply_theme(self) -> None:
        """Apply theme-driven colors from ui.ini."""
        panel_bg = get_color("training_monitor_bg", get_color("panel_bg", "#111827"))
        border = get_color("training_monitor_border", get_color("border", "#374151"))
        log_bg = get_color("training_monitor_log_bg", get_color("cmd_bg", "#0f172a"))
        log_text = get_color("training_monitor_log_text", get_color("text_primary", "#e5e7eb"))

        self.setStyleSheet(
            f"""
            #trainingPanel {{
                background: {panel_bg};
            }}
            #trainingLogArea {{
                background: {log_bg};
                color: {log_text};
                border: 1px solid {border};
                border-radius: 6px;
            }}
            """
        )

    def reset(self) -> None:
        """Clear all display state back to idle."""
        self.progress_bar.setValue(0)
        self.step_label.setText("Step: 0 / 0")
        self.reward_mean_label.setText("Reward Mean: —")
        self.reward_best_label.setText("Best: —")
        self.status_label.setText("Idle")
        self.bundle_label.setText("")
        self.log_area.clear()
        # Reset chart data
        if hasattr(self, "_chart_data"):
            for v in self._chart_data.values():
                v.clear()
            self._metrics_received = False
        self.cancel_btn.setEnabled(False)
        self.vis_banner.setVisible(False)

    def start_run(self, run_label: str = "") -> None:
        """Prepare panel for an incoming training run."""
        self.reset()
        self.run_id_label.setText(run_label)
        self.status_label.setText("Running...")
        self.cancel_btn.setEnabled(True)

    def on_progress(
        self,
        step: int,
        total: int,
        reward_mean: float,
        best_reward: float,
        ep_len_mean: float = 0.0,
        status: str = "",
    ) -> None:
        """Update progress bar, step counter, and reward display."""
        pct = int(step / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.step_label.setText(f"Step: {step:,} / {total:,}")
        self.reward_mean_label.setText(f"Reward Mean: {reward_mean:.3f}")
        self.reward_best_label.setText(f"Best: {best_reward:.3f}")
        if status:
            self.status_label.setText(status)

    def on_log(self, line: str) -> None:
        """Append a log line to the log area + route to the right severity.

        The Isaac Lab subprocess emits a mixed stream — Carb [Error]/
        [Warning] lines, Python tracebacks, our own ``[UnitPort]`` info,
        and stock INFO from Isaac Lab. Without classification every line
        previously went through ``log_info`` and the user couldn't tell
        a fatal error from a routine progress message.
        """
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.appendPlainText(f"[{ts}] {line}")
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

        severity = self._classify_log_severity(line)
        if severity == "error":
            log_error(line)
        elif severity == "warning":
            log_warning(line)
        else:
            log_info(line)

    # Track python-traceback state across calls so continuation lines
    # (``  File "...", line N, in func``) get logged as errors too.
    _in_traceback: bool = False

    def _classify_log_severity(self, line: str) -> str:
        """Return ``"info"``, ``"warning"``, or ``"error"`` for *line*.

        Stateful: once we see ``Traceback (most recent call last):`` we
        flip into ``in_traceback`` mode and treat indented continuation
        lines as errors until a non-indented non-error line resets us.
        """
        stripped = line.strip()
        if not stripped:
            return "info"

        # ── Hard error markers (always escalate) ──
        if "Traceback (most recent call last)" in stripped:
            self._in_traceback = True
            return "error"

        # Common Python exception type prefixes — ``ValueError:``,
        # ``TypeError:``, ``RuntimeError:``, ``KeyError:``, etc. The
        # cheap way to spot them: line starts with an identifier ending
        # in "Error:" or "Exception:".
        first_token = stripped.split(":", 1)[0] if ":" in stripped else ""
        if first_token and (
            first_token.endswith("Error") or first_token.endswith("Exception")
        ):
            self._in_traceback = False
            return "error"

        # Carbonite / Isaac Sim style: ``[Error]`` / ``[error]``
        # ``[UnitPort][ABORT]`` is our launcher's hard-fail prefix.
        low = stripped.lower()
        if "[error]" in low or "[abort]" in low or "fatal exception" in low:
            return "error"

        # ── Traceback continuation lines (indented) ──
        if self._in_traceback and (
            line.startswith("  ") or line.startswith("\t")
            or line.startswith("File \"")
        ):
            return "error"
        else:
            # First non-indented line after a traceback ends the block
            self._in_traceback = False

        # ── Warning markers ──
        if "[warning]" in low:
            return "warning"
        if stripped.startswith("WARNING:") or " WARNING:" in stripped:
            return "warning"
        # Isaac Lab inline warnings like:
        #   "[simulation_context.py] WARNING: ..."
        if "WARNING:" in stripped and stripped.startswith("["):
            return "warning"

        return "info"

    def on_finished(self, bundle_path: str) -> None:
        """Display completion state."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Complete")
        self.bundle_label.setText(f"Bundle: {bundle_path}")
        self.cancel_btn.setEnabled(False)
        # Enable review button if Isaac Lab backend
        if self.review_btn.isVisible():
            self.review_btn.setEnabled(True)
            self.review_btn.setToolTip("Replay trained policy in Isaac Sim viewport")
        log_success(f"Training finished: {bundle_path}")

    def set_backend_mode(self, backend: str) -> None:
        """Toggle review button visibility based on backend."""
        is_il = backend == "isaac_lab"
        self.review_btn.setVisible(is_il)
        self.review_btn.setEnabled(False)

    def on_cancelled(self) -> None:
        """Display cancelled state."""
        self.status_label.setText("Cancelled")
        self.cancel_btn.setEnabled(False)
        log_warning("Training cancelled")

    def on_error(self, message: str) -> None:
        """Display error state."""
        self.status_label.setText(f"Error: {message}")
        self.cancel_btn.setEnabled(False)
        self.log_area.appendPlainText(f"[ERROR] {message}")
        log_error(message)

    def show_vis_check_banner(self, check_num: int) -> None:
        """Show the vis check milestone banner with the given check number."""
        self.vis_banner_label.setText(
            f"Vis Check #{check_num} - MuJoCo viewer is open. "
            "Watch the current policy, then close the viewer to resume training."
        )
        self.vis_banner.setVisible(True)

    def hide_vis_check_banner(self) -> None:
        """Hide the vis check banner after the milestone completes."""
        self.vis_banner.setVisible(False)

    # ------------------------------------------------------------------
    # Pyqtgraph chart grid
    # ------------------------------------------------------------------

    def _build_charts(self, parent: QFrame) -> None:
        """Build 2x3 grid of PlotWidgets inside *parent*."""
        pg.setConfigOptions(antialias=True, background="#0f172a", foreground="#94a3b8")

        layout = QGridLayout(parent)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._chart_data: Dict[str, List[float]] = {
            "x": [],
            "reward_mean": [], "reward_min": [], "reward_max": [],
            "ep_len": [],
            "policy_loss": [], "value_loss": [],
            "kl": [], "entropy": [],
            # AMP-specific extras (only populated when an AMP_PPO run is
            # active; the corresponding plots stay empty for vanilla PPO).
            "amp_loss": [], "grad_pen": [],
            "policy_pred": [], "expert_pred": [],
        }
        # Last-known-good reward across iterations. AMP emits ``nan`` for
        # reward/ep_len during the warmup period before the deque is
        # populated; instead of plotting a literal 0 (which crushes the
        # y-axis) we carry the previous value forward.
        self._last_reward_mean: float = 0.0
        self._last_ep_len: float = 0.0
        self._is_amp_run: bool = False

        def make_plot(title: str, row: int, col: int) -> pg.PlotWidget:
            pw = pg.PlotWidget(title=title)
            pw.setFixedHeight(110)
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.getAxis("bottom").setStyle(showValues=False)
            pw.setLabel("bottom", "")
            layout.addWidget(pw, row, col)
            return pw

        # Row 0
        self._reward_plot = make_plot("Reward", 0, 0)
        self._reward_mean_curve = self._reward_plot.plot(pen=pg.mkPen("#4FC3F7", width=2), name="mean")
        self._reward_min_curve = self._reward_plot.plot(pen=pg.mkPen("#4FC3F7", width=1, style=pg.QtCore.Qt.PenStyle.DotLine))
        self._reward_max_curve = self._reward_plot.plot(pen=pg.mkPen("#4FC3F7", width=1, style=pg.QtCore.Qt.PenStyle.DotLine))

        self._ep_len_plot = make_plot("Episode Length", 0, 1)
        self._ep_len_curve = self._ep_len_plot.plot(pen=pg.mkPen("#66BB6A", width=2))

        # Row 1
        self._policy_loss_plot = make_plot("Policy Loss", 1, 0)
        self._policy_loss_curve = self._policy_loss_plot.plot(pen=pg.mkPen("#FF7043", width=2))

        self._value_loss_plot = make_plot("Value Loss", 1, 1)
        self._value_loss_curve = self._value_loss_plot.plot(pen=pg.mkPen("#FFB74D", width=2))

        # Row 2 — vanilla PPO diagnostics
        self._kl_plot = make_plot("KL Divergence", 2, 0)
        self._kl_curve = self._kl_plot.plot(pen=pg.mkPen("#CE93D8", width=2))

        self._entropy_plot = make_plot("Entropy", 2, 1)
        self._entropy_curve = self._entropy_plot.plot(pen=pg.mkPen("#4DB6AC", width=2))

        # Row 3 — AMP-only series. Created at distinct grid cells so the
        # layout never has two widgets fighting for the same row/col
        # (QGridLayout silently breaks size hints when that happens).
        # The AMP row stays hidden until the first AMP-tagged metric
        # arrives, at which point _switch_to_amp_layout() flips it on
        # and hides the KL/Entropy row.
        self._amp_loss_plot = make_plot("AMP Loss / Grad Pen", 3, 0)
        self._amp_loss_curve = self._amp_loss_plot.plot(
            pen=pg.mkPen("#F06292", width=2), name="amp_loss"
        )
        self._grad_pen_curve = self._amp_loss_plot.plot(
            pen=pg.mkPen("#F06292", width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
            name="grad_pen",
        )

        self._discriminator_plot = make_plot("Discriminator (pol vs exp)", 3, 1)
        self._policy_pred_curve = self._discriminator_plot.plot(
            pen=pg.mkPen("#FFD54F", width=2), name="policy_pred"
        )
        self._expert_pred_curve = self._discriminator_plot.plot(
            pen=pg.mkPen("#81C784", width=2), name="expert_pred"
        )

        # Default to vanilla PPO layout — AMP plots stay hidden until
        # _switch_to_amp_layout() reveals them.
        self._amp_loss_plot.setVisible(False)
        self._discriminator_plot.setVisible(False)
        self._charts_layout = layout

        self._metrics_received = False

    def _switch_to_amp_layout(self) -> None:
        """Hide KL/Entropy row, reveal AMP row. Idempotent.

        Both rows live at distinct grid cells (row 2 for vanilla PPO,
        row 3 for AMP), so the swap is just a visibility toggle — no
        addWidget/removeWidget gymnastics needed.
        """
        if getattr(self, "_amp_layout_active", False):
            return
        self._amp_layout_active = True
        self._kl_plot.setVisible(False)
        self._entropy_plot.setVisible(False)
        self._amp_loss_plot.setVisible(True)
        self._discriminator_plot.setVisible(True)

    def on_metrics(self, data: dict) -> None:
        """Receive a MSG_METRICS data dict and update all chart plots.

        Handles two payload formats:

        - Vanilla PPO (rsl_rl multi-line block): keys are
          ``reward_mean / reward_min / reward_max / ep_len_mean /
          policy_loss / value_loss / kl / entropy``.
        - AMP_PPO (single-line ``[AMP] it=…``): adds
          ``amp_loss / grad_pen / policy_pred / expert_pred`` and tags
          ``algorithm == "AMP_PPO"``. Also carries ``reward_valid`` so
          we can carry-forward the previous reward instead of plotting
          a literal 0 during AMP's warmup period.
        """
        if not self._metrics_received:
            self._metrics_received = True
            if not self.charts_area.isVisible():
                self.charts_area.setVisible(True)
                self.charts_btn.setChecked(True)

        is_amp = data.get("algorithm") == "AMP_PPO"
        if is_amp and not self._is_amp_run:
            self._is_amp_run = True
            self._switch_to_amp_layout()

        cd = self._chart_data
        cd["x"].append(data.get("iteration", 0))

        # Reward / ep_len with carry-forward for AMP warmup. ``reward_valid``
        # is False on AMP iterations where the reward deque was still empty
        # — in that case we plot the previous value instead of zero.
        reward_valid = data.get("reward_valid", True)
        if reward_valid:
            self._last_reward_mean = float(data.get("reward_mean", 0.0))
            self._last_ep_len = float(data.get("ep_len_mean", 0.0))
        cd["reward_mean"].append(self._last_reward_mean)
        cd["reward_min"].append(float(data.get("reward_min", self._last_reward_mean)))
        cd["reward_max"].append(float(data.get("reward_max", self._last_reward_mean)))
        cd["ep_len"].append(self._last_ep_len)

        cd["policy_loss"].append(float(data.get("policy_loss", 0.0)))
        cd["value_loss"].append(float(data.get("value_loss", 0.0)))
        cd["kl"].append(float(data.get("kl", 0.0)))
        cd["entropy"].append(float(data.get("entropy", 0.0)))

        # AMP-only series — append regardless so the lists stay aligned
        # with cd["x"]; for vanilla PPO they just sit at 0.
        cd["amp_loss"].append(float(data.get("amp_loss", 0.0)))
        cd["grad_pen"].append(float(data.get("grad_pen", 0.0)))
        cd["policy_pred"].append(float(data.get("policy_pred", 0.0)))
        cd["expert_pred"].append(float(data.get("expert_pred", 0.0)))

        x = cd["x"]
        self._reward_mean_curve.setData(x, cd["reward_mean"])
        self._reward_min_curve.setData(x, cd["reward_min"])
        self._reward_max_curve.setData(x, cd["reward_max"])
        self._ep_len_curve.setData(x, cd["ep_len"])
        self._policy_loss_curve.setData(x, cd["policy_loss"])
        self._value_loss_curve.setData(x, cd["value_loss"])
        if self._is_amp_run:
            self._amp_loss_curve.setData(x, cd["amp_loss"])
            self._grad_pen_curve.setData(x, cd["grad_pen"])
            self._policy_pred_curve.setData(x, cd["policy_pred"])
            self._expert_pred_curve.setData(x, cd["expert_pred"])
        else:
            self._kl_curve.setData(x, cd["kl"])
            self._entropy_curve.setData(x, cd["entropy"])

        # Key-data text labels (mirrors what on_progress does for the
        # vanilla path). The MSG_PROGRESS emit in isaac_lab_backend
        # already handles the progress bar, but the reward labels live
        # here so AMP carry-forward stays consistent across the chart
        # and the header text.
        try:
            self.reward_mean_label.setText(
                f"Reward Mean: {self._last_reward_mean:.3f}"
            )
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Isaac Lab visualization handlers
    # ------------------------------------------------------------------

    def _on_preview_requested(self, path: str) -> None:
        """Handle checkpoint timeline click."""
        self._selected_checkpoint_path = path
        self.preview_btn.setEnabled(True)
        self.preview_btn.setToolTip(f"Preview: {path}")

    def _on_preview_btn_clicked(self) -> None:
        """Launch MuJoCo preview for the selected checkpoint."""
        if not self._selected_checkpoint_path:
            return
        self._launch_preview(self._selected_checkpoint_path)

    def _launch_preview(self, checkpoint_path: str) -> None:
        """Start PolicyPreviewRunner in a background thread."""
        # Stop any existing preview
        runner = getattr(self, "_preview_runner", None)
        if runner is not None and runner.isRunning():
            runner.request_stop()
            runner.wait(3000)

        from bin.pages.training.policy_preview_runner import PolicyPreviewRunner
        self._preview_runner = PolicyPreviewRunner(checkpoint_path)
        self._preview_runner.preview_started.connect(self._on_preview_started)
        self._preview_runner.preview_stopped.connect(self._on_preview_stopped)
        self._preview_runner.preview_error.connect(self._on_preview_error)
        self._preview_runner.start()

    def _on_preview_started(self) -> None:
        self.preview_btn.setText("Preview Running...")
        self.preview_btn.setEnabled(False)
        log_info("MuJoCo preview started")

    def _on_preview_stopped(self) -> None:
        self.preview_btn.setText("Preview in MuJoCo")
        self.preview_btn.setEnabled(bool(self._selected_checkpoint_path))
        log_info("MuJoCo preview stopped")

    def _on_preview_error(self, message: str) -> None:
        self.preview_btn.setText("Preview in MuJoCo")
        self.preview_btn.setEnabled(bool(self._selected_checkpoint_path))
        log_error(f"Preview error: {message}")

    def _on_charts_toggled(self, checked: bool) -> None:
        """Toggle charts area visibility."""
        self.charts_area.setVisible(checked)

    # ------------------------------------------------------------------
    # Checkpoint watcher
    # ------------------------------------------------------------------

    def start_checkpoint_watcher(self, log_dir: str) -> None:
        """Start watching *log_dir* for new checkpoint .pt files."""
        self.stop_checkpoint_watcher()
        # Clear Phase-A mock data
        self.checkpoint_timeline.clear()

        from src.system.training.checkpoint_watcher import CheckpointWatcher
        self._checkpoint_watcher = CheckpointWatcher(log_dir)
        self._checkpoint_watcher.checkpoint_found.connect(self._on_checkpoint_found)
        self._checkpoint_watcher.start()

    def stop_checkpoint_watcher(self) -> None:
        """Stop the checkpoint watcher if running."""
        watcher = getattr(self, "_checkpoint_watcher", None)
        if watcher is not None and watcher.isRunning():
            watcher.stop()
            watcher.wait(3000)
        self._checkpoint_watcher = None

    def _on_checkpoint_found(self, iteration: int, reward: float, path: str) -> None:
        """Slot for CheckpointWatcher.checkpoint_found signal."""
        self.checkpoint_timeline.add_checkpoint(iteration, reward, path)

    # ------------------------------------------------------------------
    # Thread wiring
    # ------------------------------------------------------------------

    def connect_thread(self, thread) -> None:
        """Wire a TrainRunThread to this panel and start the run display."""
        self.start_run(getattr(thread, "_policy_id_out", ""))
        thread.progress.connect(self.on_progress)
        thread.log_line.connect(self.on_log)
        thread.finished.connect(self.on_finished)
        thread.error.connect(self.on_error)
        thread.cancelled.connect(self.on_cancelled)
        self.cancel_requested.connect(thread.cancel)
