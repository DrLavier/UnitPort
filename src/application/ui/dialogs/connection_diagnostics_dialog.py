"""ConnectionDiagnosticsDialog — read-only findings detail view.

Phase 3.7 stripped this dialog down to a passive "show me what the probes
found" panel. All the action buttons (Apply Safe Fixes / Apply All Fixes /
re-diagnose / reconnect / hint label / progress strip / SSH section
auto-expand) moved to :class:`ConnectionResultDialog`, which is the
modal that actually drives the AutoConnect resolution flow.

Now opened from exactly one place: the ``[View Details]`` button on
``ConnectionResultDialog``. Used to be auto-popped on every
``diagnostics_ready`` with severity >= WARNING — that behaviour is gone.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, setButton, tr

from application.service.diagnostics import (
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


def _severity_label(s: Severity) -> str:
    return s.name.upper()


class _FindingRow(QFrame):
    """One probe verdict line with severity chip + summary + repair preview."""

    def __init__(
        self,
        finding: DiagnosticFinding,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionDiagFindingRow")
        self._finding = finding
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        self._severity_chip = QLabel(_severity_label(finding.severity), self)
        self._severity_chip.setObjectName("missionDiagSeverityChip")
        self._severity_chip.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter
        )
        self._severity_chip.setFixedWidth(72)
        head.addWidget(self._severity_chip, 0)

        self._probe_id = QLabel(finding.probe_id, self)
        self._probe_id.setObjectName("missionDiagProbeId")
        head.addWidget(self._probe_id, 0)
        head.addStretch(1)

        if finding.requires_ssh:
            ssh_chip = QLabel("SSH", self)
            ssh_chip.setObjectName("missionDiagSshChip")
            ssh_chip.setAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter
            )
            ssh_chip.setFixedWidth(40)
            head.addWidget(ssh_chip, 0)

        outer.addLayout(head)

        self._summary = QLabel(finding.summary, self)
        self._summary.setObjectName("missionDiagSummary")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary, 0)

        if finding.detail:
            self._detail = QLabel(finding.detail, self)
            self._detail.setObjectName("missionDiagDetail")
            self._detail.setWordWrap(True)
            outer.addWidget(self._detail, 0)

        if finding.repair is not None:
            tag_text = (
                tr("mission.diag.repair_safe", default="Safe fix")
                if finding.repair.safe
                else tr("mission.diag.repair_invasive", default="Invasive fix")
            )
            repair_line = QLabel(
                f"→ [{tag_text}] {finding.repair.describe}", self,
            )
            repair_line.setObjectName(
                "missionDiagRepairSafe" if finding.repair.safe
                else "missionDiagRepairInvasive"
            )
            repair_line.setWordWrap(True)
            outer.addWidget(repair_line, 0)

        self.apply_theme()

    def apply_theme(self) -> None:
        sev = self._finding.severity
        chip_fg = Config.get_color(_SEVERITY_SLOT[sev])
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        chip_bg = Config.get_color("row_1")
        chip_border = Config.get_color("border_2")
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        self.setStyleSheet(
            f"QFrame#missionDiagFindingRow {{ background-color: "
            f"{Config.get_color('bg_2')}; border: 1px solid {chip_border}; "
            f"border-radius: 4px; }}"
            f"QLabel#missionDiagSeverityChip {{ background-color: {chip_bg}; "
            f"color: {chip_fg}; border: 1px solid {chip_border}; "
            f"border-radius: 3px; padding: 1px 4px; "
            f"font-size: {font_small}px; font-weight: 600; }}"
            f"QLabel#missionDiagSshChip {{ background-color: "
            f"{Config.get_color('mission_diag_ssh_section_header_bg')}; "
            f"color: {main}; border-radius: 3px; padding: 1px 4px; "
            f"font-size: {font_small}px; }}"
            f"QLabel#missionDiagProbeId {{ color: {main}; "
            f"font-size: {font_small}px; font-weight: 600; }}"
            f"QLabel#missionDiagSummary {{ color: {main}; "
            f"font-size: {font_normal}px; }}"
            f"QLabel#missionDiagDetail {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QLabel#missionDiagRepairSafe {{ color: "
            f"{Config.get_color('safe_zone')}; "
            f"font-size: {font_small}px; }}"
            f"QLabel#missionDiagRepairInvasive {{ color: "
            f"{Config.get_color('mission_diag_repair_btn_invasive_bg')}; "
            f"font-size: {font_small}px; }}"
        )


class ConnectionDiagnosticsDialog(QDialog):
    """Read-only modal listing every finding on a DiagnosticReport."""

    def __init__(
        self,
        report: DiagnosticReport,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionDiagDialog")
        self.setWindowTitle(
            tr("mission.diag.title", default="Connection Diagnostics")
        )
        self.setModal(True)
        self.resize(640, 520)
        self._report = report
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self._summary_label = QLabel(self._summary_text(self._report), self)
        self._summary_label.setObjectName("missionDiagDialogSummary")
        outer.addWidget(self._summary_label, 0)

        scroll = QScrollArea(self)
        scroll.setObjectName("missionDiagScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        for f in self._report.findings:
            row = _FindingRow(f, parent=body)
            body_layout.addWidget(row, 0)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        self._btn_close = setButton(
            "mission.diag.btn.close", 90, 32,
            kind="normal", spec="none", default="Close",
            parent=self,
        )
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close, 0)
        outer.addLayout(footer)

    def _summary_text(self, report: DiagnosticReport) -> str:
        if report.passed:
            head = tr("mission.diag.summary.pass", default="All checks passed.")
        elif report.has_warnings:
            head = tr("mission.diag.summary.warn", default="Warnings detected.")
        else:
            head = tr(
                "mission.diag.summary.error", default="Errors detected.",
            )
        return f"{head} ({len(report.findings)} finding(s))"

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        main = Config.get_color("main_t1")
        font_normal = Config.get_font_size("size_normal")
        self.setStyleSheet(
            f"QDialog#missionDiagDialog {{ background-color: {bg}; }}"
            f"QLabel#missionDiagDialogSummary {{ color: {main}; "
            f"font-size: {font_normal}px; font-weight: 600; }}"
        )


__all__ = ["ConnectionDiagnosticsDialog"]
