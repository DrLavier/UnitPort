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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from bin.core.theme_manager import get_color_for_theme
from bin.core.logger import log_error, log_info, log_success, log_warning


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
        """Append a log line to the log area."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.appendPlainText(f"[{ts}] {line}")
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())
        log_info(line)

    def on_finished(self, bundle_path: str) -> None:
        """Display completion state."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Complete")
        self.bundle_label.setText(f"Bundle: {bundle_path}")
        self.cancel_btn.setEnabled(False)
        log_success(f"Training finished: {bundle_path}")

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

    def connect_thread(self, thread) -> None:
        """Wire a TrainRunThread to this panel and start the run display."""
        self.start_run(getattr(thread, "_policy_id_out", ""))
        thread.progress.connect(self.on_progress)
        thread.log_line.connect(self.on_log)
        thread.finished.connect(self.on_finished)
        thread.error.connect(self.on_error)
        thread.cancelled.connect(self.on_cancelled)
        self.cancel_requested.connect(thread.cancel)
