# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ConnectionResultDialog — final outcome of one AutoConnect run.

Phase 3.7's terminal modal. AutoRepairLoop ships a
:class:`ConnectionResult` and the connect task fires
:pyqtsig:`AppSignals.connection_result`; the
:class:`RealRobotConnectionCard` opens this dialog exactly once.

Three layouts based on ``result.ok / result.has_partial``:

* **Pass**: green head + bullet list of applied repairs +
  ``[Continue]`` (default action; closes the dialog).
* **Partial**: yellow head + applied list + unresolved bucket list
  (safe / invasive / manual). Buttons:
    - ``[Apply Invasive Fixes]`` — only when ``unresolved_invasive`` is
      non-empty; opens a confirm modal then submits a RepairTask with
      ``safe_only=False`` containing the deduped invasive actions.
    - ``[Retry Auto-Repair]`` — calls ``controller.reconnect()`` to
      re-run the whole brownfield sequence including a fresh P9.
    - ``[View Details]`` — opens the existing
      :class:`ConnectionDiagnosticsDialog` (read-only) showing the raw
      finding list.
    - ``[Continue Anyway]`` — closes the dialog; the connection is up,
      the user accepts the partial state.
* **Fail**: red head + same buttons as partial (minus invasive when
  none applies). Used when the connect task never reached steady-state.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    log_info,
    log_warning,
    setButton,
    tr,
)

from application.service.connection.result import ConnectionResult
from application.service.diagnostics.results import (
    DiagnosticFinding,
    DiagnosticReport,
    Severity,
)


_SEVERITY_SLOT = {
    Severity.OK: "safe_zone",
    Severity.INFO: "checked_1",
    Severity.WARNING: "mission_diag_severity_warning",
    Severity.ERROR: "mission_diag_severity_error",
    Severity.CRITICAL: "mission_diag_severity_critical",
}


class ConnectionResultDialog(QDialog):
    """Modal showing one ConnectionResult outcome + next-step buttons."""

    def __init__(
        self,
        result: ConnectionResult,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionConnResultDialog")
        self.setModal(True)
        self.setWindowTitle(self._window_title(result))
        self.resize(560, 420)
        self._result = result
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _window_title(self, result: ConnectionResult) -> str:
        if result.ok:
            return tr(
                "mission.conn.result.title.ok", default="Connection Ready",
            )
        if result.has_partial:
            return tr(
                "mission.conn.result.title.partial",
                default="Connection Partial — Action Suggested",
            )
        return tr(
            "mission.conn.result.title.fail", default="Connection Failed",
        )

    def _headline_text(self) -> str:
        r = self._result
        if r.ok:
            if r.applied:
                return tr(
                    "mission.conn.result.head.ok_fixed",
                    default="Connection ready. Auto-fixed {n} issue(s).",
                ).format(n=len(r.applied))
            return tr(
                "mission.conn.result.head.ok_clean",
                default="Connection ready. No issues found.",
            )
        if r.has_partial:
            return tr(
                "mission.conn.result.head.partial",
                default=(
                    "Connection is up, but {n} issue(s) need attention."
                ),
            ).format(n=r.total_unresolved)
        return tr(
            "mission.conn.result.head.fail",
            default="Connection could not complete.",
        )

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        self._headline = QLabel(self._headline_text(), self)
        self._headline.setObjectName("missionConnResultHeadline")
        self._headline.setWordWrap(True)
        outer.addWidget(self._headline, 0)

        meta = QLabel(
            tr(
                "mission.conn.result.meta",
                default=(
                    "Auto-repair attempts: {a}   "
                    "Applied: {ap}   Failed: {fl}   "
                    "Duration: {d:.1f}s"
                ),
            ).format(
                a=self._result.attempts,
                ap=len(self._result.applied),
                fl=len(self._result.failed),
                d=self._result.duration_s,
            ),
            self,
        )
        meta.setObjectName("missionConnResultMeta")
        outer.addWidget(meta, 0)

        # Scroll body holding applied + unresolved sections.
        scroll = QScrollArea(self)
        scroll.setObjectName("missionConnResultScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)

        if self._result.applied:
            body_layout.addWidget(
                self._section(
                    tr(
                        "mission.conn.result.section.applied",
                        default="Applied (auto-fixed)",
                    ),
                    [
                        f"✓ {a.name}: {a.describe}"
                        for a in self._result.applied
                    ],
                    severity=Severity.OK,
                ),
                0,
            )
        if self._result.failed:
            body_layout.addWidget(
                self._section(
                    tr(
                        "mission.conn.result.section.failed",
                        default="Failed during auto-repair",
                    ),
                    [
                        f"✗ {a.name}: {err}"
                        for a, err in self._result.failed
                    ],
                    severity=Severity.ERROR,
                ),
                0,
            )
        if self._result.unresolved_safe:
            body_layout.addWidget(
                self._finding_section(
                    tr(
                        "mission.conn.result.section.unresolved_safe",
                        default="Unresolved (safe-fixable; consider Retry)",
                    ),
                    self._result.unresolved_safe,
                    Severity.WARNING,
                ),
                0,
            )
        if self._result.unresolved_invasive:
            body_layout.addWidget(
                self._finding_section(
                    tr(
                        "mission.conn.result.section.unresolved_invasive",
                        default=(
                            "Unresolved (invasive fix available — needs your"
                            " consent)"
                        ),
                    ),
                    self._result.unresolved_invasive,
                    Severity.WARNING,
                ),
                0,
            )
        if self._result.unresolved_manual:
            body_layout.addWidget(
                self._finding_section(
                    tr(
                        "mission.conn.result.section.unresolved_manual",
                        default=(
                            "Unresolved (no auto fix — manual action required)"
                        ),
                    ),
                    self._result.unresolved_manual,
                    Severity.ERROR,
                ),
                0,
            )
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Footer with dynamic button set.
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)

        if not self._result.ok:
            self._btn_view = setButton(
                "mission.conn.result.btn.view_details", 130, 32,
                kind="normal", spec="none", default="View Details", parent=self,
            )
            self._btn_view.clicked.connect(self._on_view_details)
            footer.addWidget(self._btn_view, 0)

            self._btn_retry = setButton(
                "mission.conn.result.btn.retry", 150, 32,
                kind="normal", spec="none",
                default="Retry Auto-Repair", parent=self,
            )
            self._btn_retry.clicked.connect(self._on_retry)
            footer.addWidget(self._btn_retry, 0)

            if self._result.unresolved_invasive:
                n = len(self._result.invasive_actions())
                self._btn_invasive = setButton(
                    "mission.conn.result.btn.invasive", 200, 32,
                    kind="normal", spec="danger",
                    default=f"Apply Invasive Fixes ({n})", parent=self,
                )
                self._btn_invasive.clicked.connect(self._on_apply_invasive)
                footer.addWidget(self._btn_invasive, 0)

            self._btn_continue = setButton(
                "mission.conn.result.btn.continue_anyway", 150, 32,
                kind="normal", spec="save",
                default="Continue Anyway", parent=self,
            )
        else:
            self._btn_continue = setButton(
                "mission.conn.result.btn.continue", 110, 32,
                kind="normal", spec="save", default="Continue", parent=self,
            )
        self._btn_continue.clicked.connect(self.accept)
        footer.addWidget(self._btn_continue, 0)

        outer.addLayout(footer)

    def _section(
        self,
        title: str,
        lines: List[str],
        *,
        severity: Severity,
    ) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("missionConnResultSection")
        v = QVBoxLayout(frame)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        head = QLabel(title, frame)
        head.setObjectName("missionConnResultSectionTitle")
        head.setProperty("severitySlot", _SEVERITY_SLOT[severity])
        v.addWidget(head, 0)
        for line in lines:
            row = QLabel(line, frame)
            row.setObjectName("missionConnResultSectionRow")
            row.setWordWrap(True)
            v.addWidget(row, 0)
        return frame

    def _finding_section(
        self,
        title: str,
        findings: List[DiagnosticFinding],
        severity: Severity,
    ) -> QFrame:
        lines = []
        for f in findings:
            chunks = [f"• [{f.probe_id}] {f.summary}"]
            if f.repair is not None:
                tag = (
                    tr("mission.conn.result.fix_tag.safe", default="safe")
                    if f.repair.safe
                    else tr(
                        "mission.conn.result.fix_tag.invasive",
                        default="invasive",
                    )
                )
                chunks.append(f"  → fix [{tag}]: {f.repair.describe}")
            lines.append("\n".join(chunks))
        return self._section(title, lines, severity=severity)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        border = Config.get_color("border_2")
        section_bg = Config.get_color("bg_2")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        # Headline color follows the result's overall verdict.
        if self._result.ok:
            head_color = Config.get_color("safe_zone")
        elif self._result.has_partial:
            head_color = Config.get_color("mission_diag_severity_warning")
        else:
            head_color = Config.get_color("mission_diag_severity_error")
        self.setStyleSheet(
            f"QDialog#missionConnResultDialog {{ background-color: {bg}; }}"
            f"QLabel#missionConnResultHeadline {{ color: {head_color}; "
            f"font-size: {font_normal}px; font-weight: 700; }}"
            f"QLabel#missionConnResultMeta {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QFrame#missionConnResultSection {{ background-color: "
            f"{section_bg}; border: 1px solid {border}; border-radius: 4px; }}"
            f"QLabel#missionConnResultSectionTitle {{ color: {main}; "
            f"font-size: {font_small}px; font-weight: 600; }}"
            f"QLabel#missionConnResultSectionRow {{ color: {main}; "
            f"font-size: {font_small}px; }}"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_view_details(self) -> None:
        # Build a synthetic DiagnosticReport from the unresolved findings
        # so the read-only details dialog has something to show. (The raw
        # original report is held by DiagnosticsTask; reaching back for it
        # would couple the dialog to the controller. The unresolved subset
        # is what the user actually wants to inspect anyway.)
        from application.ui.dialogs.connection_diagnostics_dialog import (
            ConnectionDiagnosticsDialog,
        )
        all_findings: List[DiagnosticFinding] = []
        all_findings.extend(self._result.unresolved_safe)
        all_findings.extend(self._result.unresolved_invasive)
        all_findings.extend(self._result.unresolved_manual)
        report = DiagnosticReport(findings=all_findings)
        dlg = ConnectionDiagnosticsDialog(report, parent=self)
        dlg.exec()

    def _on_retry(self) -> None:
        from application.service.ros2_connection_config import (
            get_ros2_connection_controller,
        )
        log_info("[result-dialog] user requested retry — reconnecting")
        try:
            get_ros2_connection_controller().reconnect()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[result-dialog] reconnect raised: {exc}")
        self.accept()

    def _on_apply_invasive(self) -> None:
        from application.service.ros2_connection_config import (
            get_ros2_connection_controller,
        )
        invasive = self._result.invasive_actions()
        if not invasive:
            self.accept()
            return
        msg = tr(
            "mission.conn.result.invasive.confirm",
            default=(
                "About to run {n} invasive repair(s) on the robot. These "
                "may install packages or modify persistent configuration "
                "files.\n\nActions:\n{summary}\n\nProceed?"
            ),
        ).format(
            n=len(invasive),
            summary="\n".join(f"  - {a.describe}" for a in invasive),
        )
        # Default to No so a stray Enter doesn't kick off apt-get install.
        choice = QMessageBox.question(
            self,
            tr(
                "mission.conn.result.invasive.confirm.title",
                default="Confirm Invasive Repair",
            ),
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        log_info(
            f"[result-dialog] user authorised {len(invasive)} invasive fix(es)"
        )
        try:
            get_ros2_connection_controller().apply_repair(
                invasive, safe_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[result-dialog] apply_repair raised: {exc}")
        self.accept()


__all__ = ["ConnectionResultDialog"]
