# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CanvasErrorDialog — unified popup for canvas self-check failures.

When the user hits ▶ Play and the spec compiler / Isaac Lab env_cfg
compiler rejects the canvas, the strict no-fallback policy means we must
``raise`` rather than silently coerce a bad value. But a raw traceback in
the cmd log reads like an application crash. This dialog is the polite
surface: structured per-issue rows showing **where** in the canvas the
problem is (node id + schema + parameter key) and **why** it failed.

Two exception families feed in:

* :class:`application.training.spec_validator.SpecValidationError` —
  carries a list of :class:`ValidationIssue` (code / severity / node_id
  / field / message).
* :class:`application.training.isaac_lab.env_cfg_compiler.CanvasConfigError`
  — single-issue: ``nid`` / ``schema_id`` / ``key`` / ``reason``.

Anything else (real programming bugs) still ends up in the cmd log via
the caller's broad ``except Exception`` so they remain debuggable.

Theme: colours come from ``Config.get_color`` slots only (CLAUDE.md §1.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    log_info,
    setButton,
    tr,
)


# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanvasIssue:
    """One canvas-side problem ready for display.

    ``severity`` is ``"error"`` or ``"warning"`` — anything else falls
    back to error styling. ``location`` is a short human-readable string
    pointing at the canvas node (e.g. ``"node '1009' · terminations · "
    "termination_conditions"``); empty when the issue is canvas-wide.
    ``code`` is the stable IssueCode value when the source is a
    ``ValidationIssue`` — surfaced as a small chip so the user can search
    docs / past sessions for the same identifier.
    """

    message: str
    location: str = ""
    severity: str = "error"
    code: str = ""

    def as_text_line(self) -> str:
        """Single-line plain-text form, used by Copy-to-Clipboard."""
        head = f"[{self.severity.upper()}]"
        if self.code:
            head += f" {self.code}"
        if self.location:
            head += f" {self.location}"
        return f"{head}\n  {self.message}"


# ---------------------------------------------------------------------------
# Adapters: exception → List[CanvasIssue]
# ---------------------------------------------------------------------------


def _issues_from_spec_validation(exc: Any) -> List[CanvasIssue]:
    """Convert ``SpecValidationError.issues`` to display-side issues."""
    out: List[CanvasIssue] = []
    issues = getattr(exc, "issues", None) or []
    for iss in issues:
        sev_enum = getattr(iss, "severity", None)
        sev = getattr(sev_enum, "value", str(sev_enum) if sev_enum else "error")
        code_enum = getattr(iss, "code", None)
        code = getattr(code_enum, "value", str(code_enum) if code_enum else "")
        node_id = str(getattr(iss, "node_id", "") or "")
        field = str(getattr(iss, "field", "") or "")
        loc_bits: List[str] = []
        if node_id:
            loc_bits.append(f"node '{node_id}'")
        if field:
            loc_bits.append(f"field '{field}'")
        out.append(CanvasIssue(
            message=str(getattr(iss, "message", "") or str(iss)),
            location=" · ".join(loc_bits),
            severity=str(sev or "error"),
            code=str(code or ""),
        ))
    return out


def _issues_from_canvas_config_error(exc: Any) -> List[CanvasIssue]:
    """Convert a single ``CanvasConfigError`` to one display issue."""
    nid = str(getattr(exc, "nid", "") or "")
    schema = str(getattr(exc, "schema_id", "") or "")
    key = str(getattr(exc, "key", "") or "")
    reason = str(getattr(exc, "reason", "") or str(exc))
    loc_bits: List[str] = []
    if nid:
        loc_bits.append(f"node '{nid}'")
    if schema:
        loc_bits.append(schema)
    if key:
        loc_bits.append(f"key '{key}'")
    return [CanvasIssue(
        message=reason,
        location=" · ".join(loc_bits),
        severity="error",
        code="canvas_config",
    )]


def issues_from_exception(exc: BaseException) -> Optional[List[CanvasIssue]]:
    """Return display issues if *exc* is a known canvas-side failure.

    Returns ``None`` when *exc* is not a canvas-side check failure — the
    caller should then fall through to its generic error path (cmd-log
    traceback) so real bugs stay debuggable.

    Imports the exception types lazily because this module is loaded
    early during UI construction and we don't want to drag the training
    stack in just to define the dialog.
    """
    try:
        from application.training.spec_validator import SpecValidationError
    except Exception:
        SpecValidationError = ()  # type: ignore[assignment]
    try:
        from application.training.isaac_lab.env_cfg_compiler import (
            CanvasConfigError,
        )
    except Exception:
        CanvasConfigError = ()  # type: ignore[assignment]

    if SpecValidationError and isinstance(exc, SpecValidationError):
        return _issues_from_spec_validation(exc)
    if CanvasConfigError and isinstance(exc, CanvasConfigError):
        return _issues_from_canvas_config_error(exc)
    return None


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


_SEVERITY_SLOT = {
    "error": "mission_diag_severity_error",
    "warning": "mission_diag_severity_warning",
    "info": "checked_1",
}


class CanvasErrorDialog(QDialog):
    """Modal showing one or more canvas self-check failures.

    Constructed via :func:`show_canvas_error_dialog`; direct instantiation
    is fine when the caller has already built a ``List[CanvasIssue]`` and
    wants control over modality.
    """

    def __init__(
        self,
        issues: Sequence[CanvasIssue],
        *,
        title: Optional[str] = None,
        headline: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("canvasErrorDialog")
        self.setModal(True)
        self.resize(640, 420)
        self._issues: List[CanvasIssue] = list(issues)
        self._headline_override = headline
        self.setWindowTitle(title or tr(
            "canvas.error.title", default="Canvas Check Failed",
        ))
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        n_err = sum(1 for i in self._issues if i.severity == "error")
        n_warn = sum(1 for i in self._issues if i.severity == "warning")
        if self._headline_override is not None:
            head_text = self._headline_override
        elif n_err and n_warn:
            head_text = tr(
                "canvas.error.headline.mixed",
                default=(
                    "Canvas has {e} error(s) and {w} warning(s). "
                    "Fix the errors below and re-run."
                ),
            ).format(e=n_err, w=n_warn)
        elif n_err:
            head_text = tr(
                "canvas.error.headline.errors",
                default=(
                    "Canvas has {e} issue(s) that must be fixed before "
                    "training can start."
                ),
            ).format(e=n_err)
        elif n_warn:
            head_text = tr(
                "canvas.error.headline.warnings",
                default="Canvas has {w} warning(s).",
            ).format(w=n_warn)
        else:
            head_text = tr(
                "canvas.error.headline.unknown",
                default="Canvas check reported an issue.",
            )
        self._headline = QLabel(head_text, self)
        self._headline.setObjectName("canvasErrorHeadline")
        self._headline.setWordWrap(True)
        outer.addWidget(self._headline, 0)

        hint = QLabel(
            tr(
                "canvas.error.hint",
                default=(
                    "Each entry below points at the canvas node and the "
                    "parameter that triggered the check. Open the node, "
                    "adjust the value, save the canvas, and try ▶ again."
                ),
            ),
            self,
        )
        hint.setObjectName("canvasErrorHint")
        hint.setWordWrap(True)
        outer.addWidget(hint, 0)

        # Scrollable issue list.
        scroll = QScrollArea(self)
        scroll.setObjectName("canvasErrorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        if not self._issues:
            empty = QLabel(
                tr(
                    "canvas.error.empty",
                    default="(no structured issues reported)",
                ),
                body,
            )
            empty.setObjectName("canvasErrorEmpty")
            body_layout.addWidget(empty, 0)
        else:
            for iss in self._issues:
                body_layout.addWidget(self._issue_card(iss, parent=body), 0)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Footer.
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)

        self._btn_copy = setButton(
            "canvas.error.btn.copy", 150, 32,
            kind="normal", spec="none",
            default="Copy to Clipboard", parent=self,
        )
        self._btn_copy.clicked.connect(self._on_copy)
        footer.addWidget(self._btn_copy, 0)

        self._btn_close = setButton(
            "canvas.error.btn.close", 110, 32,
            kind="normal", spec="save",
            default="Close", parent=self,
        )
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close, 0)

        outer.addLayout(footer)

    def _issue_card(self, iss: CanvasIssue, *, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setObjectName("canvasErrorIssueCard")
        frame.setProperty("severitySlot", _SEVERITY_SLOT.get(
            iss.severity, _SEVERITY_SLOT["error"],
        ))
        v = QVBoxLayout(frame)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        sev_label = QLabel(
            iss.severity.upper() if iss.severity else "ERROR", frame,
        )
        sev_label.setObjectName("canvasErrorSeverityChip")
        sev_label.setProperty("severitySlot", _SEVERITY_SLOT.get(
            iss.severity, _SEVERITY_SLOT["error"],
        ))
        header_row.addWidget(sev_label, 0)

        if iss.code:
            code_label = QLabel(iss.code, frame)
            code_label.setObjectName("canvasErrorCodeChip")
            header_row.addWidget(code_label, 0)

        if iss.location:
            loc_label = QLabel(iss.location, frame)
            loc_label.setObjectName("canvasErrorLocation")
            header_row.addWidget(loc_label, 1)
        else:
            header_row.addStretch(1)

        v.addLayout(header_row)

        msg_label = QLabel(iss.message, frame)
        msg_label.setObjectName("canvasErrorMessage")
        msg_label.setWordWrap(True)
        msg_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        v.addWidget(msg_label, 0)

        return frame

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        section_bg = Config.get_color("bg_2")
        main = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        border = Config.get_color("border_2")
        chip_bg = Config.get_color("row_1")
        chip_fg = Config.get_color("main_t1")
        head_color = Config.get_color("mission_diag_severity_error")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        sev_err = Config.get_color("mission_diag_severity_error")
        sev_warn = Config.get_color("mission_diag_severity_warning")
        sev_info = Config.get_color("checked_1")
        self.setStyleSheet(
            f"QDialog#canvasErrorDialog {{ background-color: {bg}; }}"
            f"QLabel#canvasErrorHeadline {{ color: {head_color}; "
            f"font-size: {font_normal}px; font-weight: 700; }}"
            f"QLabel#canvasErrorHint {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QFrame#canvasErrorIssueCard {{ background-color: "
            f"{section_bg}; border: 1px solid {border}; border-radius: 4px; }}"
            f"QLabel#canvasErrorSeverityChip[severitySlot=\""
            f"{_SEVERITY_SLOT['error']}\"] {{ color: {sev_err}; "
            f"font-size: {font_small}px; font-weight: 700; }}"
            f"QLabel#canvasErrorSeverityChip[severitySlot=\""
            f"{_SEVERITY_SLOT['warning']}\"] {{ color: {sev_warn}; "
            f"font-size: {font_small}px; font-weight: 700; }}"
            f"QLabel#canvasErrorSeverityChip[severitySlot=\""
            f"{_SEVERITY_SLOT['info']}\"] {{ color: {sev_info}; "
            f"font-size: {font_small}px; font-weight: 700; }}"
            f"QLabel#canvasErrorCodeChip {{ color: {chip_fg}; "
            f"background-color: {chip_bg}; border-radius: 3px; "
            f"padding: 1px 6px; font-size: {font_small}px; }}"
            f"QLabel#canvasErrorLocation {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QLabel#canvasErrorMessage {{ color: {main}; "
            f"font-size: {font_normal}px; }}"
            f"QLabel#canvasErrorEmpty {{ color: {sub}; "
            f"font-size: {font_small}px; font-style: italic; }}"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_copy(self) -> None:
        lines = [iss.as_text_line() for iss in self._issues]
        text = "\n\n".join(lines) if lines else ""
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        log_info(tr(
            "canvas.error.copy.done",
            default="Canvas error report copied to clipboard.",
        ))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def _mark_offending_nodes_on_canvas(
    parent: Optional[QWidget],
    raw_issues: List[Any],
) -> None:
    """Paint the danger-zone outline on every canvas node referenced by an
    issue's ``node_id`` field. Node lookup accepts BOTH the canvas
    instance id (e.g. ``"1010"``) AND the manifest schema id (e.g.
    ``"actor_setting"``) — validator authors use whichever is convenient,
    and this resolver tolerates both so highlights always land.

    Best-effort: missing parent / page / NodeItem class → silent no-op
    (the dialog message itself still surfaces; we just lose the visual
    cue). Called from :func:`show_canvas_error_dialog` before the dialog
    blocks on ``exec``.
    """
    if parent is None:
        return
    page = None
    cursor = parent
    for _ in range(6):
        page = getattr(cursor, "_canvas_page", None)
        if page is not None:
            break
        cursor = getattr(cursor, "parent", lambda: None)()
        if cursor is None:
            break
    if page is None or not hasattr(page, "_instances"):
        return
    try:
        from application.ui.canvas.items import NodeItem
    except Exception:  # noqa: BLE001
        return

    targets: set[str] = set()
    for iss in raw_issues:
        nid = str(getattr(iss, "node_id", "") or "").strip()
        if nid:
            targets.add(nid)
    if not targets:
        return

    for ni in page._instances.values():
        if not isinstance(ni, NodeItem):
            continue
        schema_id = str(getattr(ni.manifest, "id", "") or "")
        inst_id = str(getattr(ni, "node_id", "") or "")
        if schema_id in targets or inst_id in targets:
            try:
                ni.mark_diagnostic("danger")
            except Exception:  # noqa: BLE001
                pass


def show_canvas_error_dialog(
    exc: BaseException,
    *,
    parent: Optional[QWidget] = None,
    title: Optional[str] = None,
    headline: Optional[str] = None,
) -> bool:
    """Open the dialog for a canvas-side check failure.

    Returns ``True`` when *exc* was a known canvas-side failure (and the
    dialog was shown), ``False`` otherwise — the caller can then fall
    through to its generic error path (e.g. cmd-log traceback).
    """
    issues = issues_from_exception(exc)
    if issues is None:
        return False
    # Also reach into the raw exception's `issues` so node_id strings
    # propagate to the canvas highlighter even though they're flattened
    # away in the dialog-side CanvasIssue dataclass.
    raw = getattr(exc, "issues", None) or []
    _mark_offending_nodes_on_canvas(parent, raw)
    dlg = CanvasErrorDialog(
        issues, title=title, headline=headline, parent=parent,
    )
    dlg.exec()
    return True


def clear_canvas_diagnostic_marks(parent: Optional[QWidget]) -> None:
    """Clear all danger-zone marks on the active canvas. Called from
    main_window before submitting a fresh training run so stale marks
    from a previous failed attempt don't bleed into the new check.
    """
    if parent is None:
        return
    page = None
    cursor = parent
    for _ in range(6):
        page = getattr(cursor, "_canvas_page", None)
        if page is not None:
            break
        cursor = getattr(cursor, "parent", lambda: None)()
        if cursor is None:
            break
    if page is None or not hasattr(page, "_instances"):
        return
    try:
        from application.ui.canvas.items import NodeItem
    except Exception:  # noqa: BLE001
        return
    for ni in page._instances.values():
        if isinstance(ni, NodeItem):
            try:
                ni.mark_diagnostic(None)
            except Exception:  # noqa: BLE001
                pass


__all__ = [
    "CanvasErrorDialog",
    "CanvasIssue",
    "issues_from_exception",
    "show_canvas_error_dialog",
    "clear_canvas_diagnostic_marks",
]
