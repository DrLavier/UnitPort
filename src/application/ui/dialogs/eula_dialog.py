"""EulaDialog — modal-blocking three-tab licence acknowledgement.

Surfaced by the InstallConfigWizard when the user selects the
"fresh install Isaac Lab" radio button. Blocks the wizard with
``exec()`` (not the wizard's non-modal ``open()``) so the user must
explicitly accept or reject before the install configuration can
proceed.

Three tabs in the order returned by :func:`eula.list_required_eulas`:
  1. NVIDIA Omniverse License Agreement (NVOLA).
  2. Isaac Lab BSD-3-Clause licence.
  3. Silent-prompts disclosure — every interactive prompt + env-var
     the installer will pass-through on the user's behalf.

Per-tab "I have read and accept" checkbox; the bottom "Accept" button
remains disabled until ALL three checkboxes are ticked. Cancel /
Reject is always enabled and the modal returns ``QDialog.Rejected``.

On accept:
  - Each licence is persisted via :func:`eula.record_acceptance`
    (writes ``<USER_CONFIG_DIR>/eula_acceptance.json``).
  - The dialog signals :attr:`accepted_ids` so the wizard can echo
    the acceptance into its ``selections.backend`` block.

Hash-mismatch UX:
  - If a licence's vendored text doesn't match the SHA256 pinned in
    ``installers/_constants``, the corresponding tab shows a banner
    above the text pointing at the upstream URL. The "I accept"
    checkbox stays enabled — the user is still allowed to accept,
    but they have been visibly warned the text differs from what
    UnitPort verified at build time.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, log_info, log_warning, tr

from application.service.installers import eula


class _LicenceTab(QWidget):
    """One tab: optional hash-mismatch banner + scrolling licence text + checkbox."""

    # ``bool`` payload = new checked state so the dialog can distinguish
    # a rising edge (auto-advance to next tab) from an un-tick (just
    # re-evaluate the Accept-button enabled state, do not jump).
    accept_toggled = pyqtSignal(bool)

    def __init__(
        self,
        loaded: eula.EulaText,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._loaded = loaded
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if not self._loaded.hash_ok:
            banner = QLabel(self)
            url_hint = (
                f"  ({self._loaded.spec.upstream_url})"
                if self._loaded.spec.upstream_url
                else ""
            )
            banner.setText(
                tr(
                    "eula.banner.hash_mismatch",
                    "WARNING: The vendored text differs from the SHA256 "
                    "UnitPort verified at build time. Review the upstream "
                    "source before accepting.",
                )
                + url_hint
            )
            banner.setWordWrap(True)
            banner.setStyleSheet(
                f"QLabel {{ "
                f"background-color: {Config.get_color('bg_2')}; "
                f"color: {Config.get_color('log_warning')}; "
                f"border: 1px solid {Config.get_color('log_warning')}; "
                f"border-radius: 4px; padding: 6px 10px; "
                f"font-size: {Config.get_font_size('size_mini')}pt; "
                f"}}"
            )
            layout.addWidget(banner)

        body = QTextEdit(self)
        body.setReadOnly(True)
        body.setPlainText(self._loaded.text or tr(
            "eula.body.missing",
            "(licence text not found — install is blocked until text is shipped)",
        ))
        body.setStyleSheet(
            f"QTextEdit {{ "
            f"background-color: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 8px; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}}"
        )
        # Word-wrap mode + monospaced family help legal text legibility.
        body.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(body, 1)

        self._cb_accept = QCheckBox(self)
        self._cb_accept.setText(
            tr(
                "eula.checkbox.i_accept",
                "I have read and accept the terms above",
            )
        )
        self._cb_accept.setStyleSheet(
            f"QCheckBox {{ "
            f"color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}}"
        )
        self._cb_accept.toggled.connect(self.accept_toggled.emit)
        layout.addWidget(self._cb_accept)

    # ------------------------------------------------------------------
    def is_accepted(self) -> bool:
        return self._cb_accept.isChecked()

    @property
    def spec(self) -> eula.EulaTextSpec:
        return self._loaded.spec


class EulaDialog(QDialog):
    """Three-tab licence dialog. Blocking via ``exec()``.

    Returns :class:`QDialog.DialogCode.Accepted` only after every tab's
    checkbox is ticked AND the user clicks Accept. Reject (X / Reject
    button / Esc) returns :class:`QDialog.DialogCode.Rejected` regardless
    of checkbox state.

    Accepted licences are persisted via :func:`eula.record_acceptance`
    before :meth:`accept` returns; on persistence I/O failure the dialog
    still returns Accepted (the wizard has the ids in memory) but a
    warning is logged — the user may be re-prompted next launch.
    """

    accepted_ids = pyqtSignal(list)   # list[str] of eula_id values on accept

    def __init__(
        self,
        *,
        user_email: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._user_email = str(user_email or "")
        self._tabs: List[_LicenceTab] = []
        self.setModal(True)
        self.resize(720, 560)
        self.setWindowTitle(
            tr(
                "eula.dialog.title",
                "Review and accept licences before installing Isaac Lab",
            )
        )
        self._build_ui()
        self._refresh_accept_enabled()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(10)

        intro = QLabel(self)
        intro.setText(
            tr(
                "eula.dialog.intro",
                "Installing Isaac Lab will download Isaac Sim 5.x and run "
                "the official isaaclab installer on your behalf. Review "
                "the agreements below; you must accept each tab before "
                "continuing. You can cancel and use the 'Locate existing "
                "installation' option if you prefer to install manually.",
            )
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            f"QLabel {{ "
            f"color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}}"
        )
        root.addWidget(intro)

        self._tab_widget = QTabWidget(self)
        for spec in eula.list_required_eulas():
            loaded = eula.read_eula_text(spec)
            tab = _LicenceTab(loaded, self._tab_widget)
            tab.accept_toggled.connect(self._on_tab_accept_toggled)
            self._tabs.append(tab)
            # Tab label: short human form, not the eula_id slug. Falls
            # back to the slug if no i18n entry exists.
            self._tab_widget.addTab(
                tab,
                tr(f"eula.tab.{spec.eula_id}", spec.eula_id),
            )
        # Selected-tab highlight uses ``safe_zone`` (green) instead of
        # ``highlight`` (amber). The amber slot reads as a warning hue;
        # an EULA tab the user has consented to should feel "this is
        # cleared" — green communicates that more honestly.
        self._tab_widget.setStyleSheet(
            f"QTabWidget::pane {{ "
            f"background-color: {Config.get_color('bg_1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; "
            f"}} "
            f"QTabBar::tab {{ "
            f"background-color: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"padding: 6px 14px; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}} "
            f"QTabBar::tab:selected {{ "
            f"background-color: {Config.get_color('safe_zone')}; "
            f"color: {Config.get_color('bg_1')}; "
            f"}}"
        )
        root.addWidget(self._tab_widget, 1)

        # Footer: Reject (left) — stretch — Accept (right, disabled until all ticked)
        footer = QHBoxLayout()
        footer.setSpacing(8)

        reject_btn = QPushButton(self)
        reject_btn.setText(tr("eula.btn.reject", "Reject"))
        reject_btn.setStyleSheet(self._button_qss("muted"))
        reject_btn.clicked.connect(self.reject)
        footer.addWidget(reject_btn)

        footer.addStretch(1)

        self._accept_btn = QPushButton(self)
        self._accept_btn.setText(
            tr("eula.btn.accept", "Accept and continue")
        )
        self._accept_btn.setStyleSheet(self._button_qss("primary"))
        self._accept_btn.clicked.connect(self._on_accept_clicked)
        footer.addWidget(self._accept_btn)

        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

    # ------------------------------------------------------------------
    def _on_tab_accept_toggled(self, checked: bool) -> None:
        """Per-tab checkbox handler.

        Two responsibilities:

        1. Re-evaluate the bottom Accept button enabled-state — needed
           on both check and un-check.
        2. On a rising edge (``checked=True``) and only when there is a
           next tab, advance the active tab. Lets the user power through
           the three licences with a single click each; on the last tab
           there is nowhere to advance to, so we stop. Un-ticking never
           jumps — the user is reconsidering, not progressing.
        """
        self._refresh_accept_enabled()
        if not checked:
            return
        idx = self._tab_widget.currentIndex()
        if 0 <= idx < self._tab_widget.count() - 1:
            self._tab_widget.setCurrentIndex(idx + 1)

    def _refresh_accept_enabled(self) -> None:
        all_ticked = all(t.is_accepted() for t in self._tabs)
        self._accept_btn.setEnabled(bool(self._tabs) and all_ticked)
        # Subtle visual cue when disabled: lighter button.
        self._accept_btn.setStyleSheet(
            self._button_qss("primary" if all_ticked else "primary_disabled")
        )

    def _on_accept_clicked(self) -> None:
        # Defence-in-depth: enabled state can be desynced if a programmatic
        # caller forces accept(). Re-check before persisting.
        if not all(t.is_accepted() for t in self._tabs):
            log_warning(
                "[eula] accept clicked with not all checkboxes ticked; ignoring"
            )
            return
        accepted: List[str] = []
        for tab in self._tabs:
            spec = tab.spec
            eula.record_acceptance(
                spec.eula_id,
                spec.version,
                user_email=self._user_email,
            )
            accepted.append(spec.eula_id)
        log_info(
            f"[eula] user accepted {len(accepted)} licence(s): "
            f"{', '.join(accepted)}"
        )
        self.accepted_ids.emit(accepted)
        self.accept()

    # ------------------------------------------------------------------
    @staticmethod
    def _button_qss(kind: str) -> str:
        if kind == "primary":
            # ``safe_zone`` (green) matches the selected-tab highlight —
            # acceptance is an affirmative ("you have cleared all gates")
            # action, not an alert (which is what ``highlight`` amber
            # would suggest).
            return (
                f"QPushButton {{ "
                f"background-color: {Config.get_color('safe_zone')}; "
                f"color: {Config.get_color('bg_1')}; "
                f"border: none; border-radius: 4px; "
                f"padding: 7px 16px; font-weight: bold; "
                f"font-size: {Config.get_font_size('size_small')}pt; "
                f"}} "
                f"QPushButton:hover {{ "
                f"background-color: {Config.get_color('safe_zone_hover')}; "
                f"}} "
                f"QPushButton:pressed {{ "
                f"background-color: {Config.get_color('safe_zone_checked')}; "
                f"}}"
            )
        if kind == "primary_disabled":
            return (
                f"QPushButton {{ "
                f"background-color: {Config.get_color('btn_1')}; "
                f"color: {Config.get_color('sub_t2')}; "
                f"border: none; border-radius: 4px; "
                f"padding: 7px 16px; font-weight: bold; "
                f"font-size: {Config.get_font_size('size_small')}pt; "
                f"}}"
            )
        return (
            f"QPushButton {{ "
            f"background-color: {Config.get_color('btn_1')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 6px 14px; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}} "
            f"QPushButton:hover {{ "
            f"background-color: {Config.get_color('hover_1')}; "
            f"}}"
        )


__all__ = ["EulaDialog"]
