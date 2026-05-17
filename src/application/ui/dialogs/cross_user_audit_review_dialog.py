"""CrossUserAuditReviewDialog — sign-in-time review of pending audit entries.

Fires from ``MainWindow._on_workspace_changed`` (deferred one event-loop
tick) whenever the just-activated user has any rows in
``<workspace_root>/<active_uid>/.audit/pending.jsonl``. The user steps
through each entry and either:

* **Accept** — keep the change. Entry drops from the queue (mirrored to
  ``history.jsonl`` for an eventual history viewer).
* **Reject** — restore the previous version from the user's own cloud
  namespace (``CloudSyncService.pull_single``). Requires the user to
  have ever pushed this file via cloud sync; otherwise Reject fails
  with a status line on the card and the entry stays pending.

Layout (clones ``canvas_error_dialog`` for visual continuity):

.. code-block:: text

    +------------------------------------------------------+
    | Pending changes by other users                       |
    | While you were signed out, 3 change(s) were made … |
    +------------------------------------------------------+
    | [scroll]                                             |
    | +-- card --------------------------------------+     |
    | | (op badge) {actor} edited {file}             |     |
    | | {time}                                       |     |
    | | (status line — populated after Reject)       |     |
    | |                       [Reject] [Accept]      |     |
    | +----------------------------------------------+     |
    | …                                                    |
    +------------------------------------------------------+
    | [Accept all]  [Reject all]                  [Later]  |
    +------------------------------------------------------+

All colours / font sizes go through ``Config`` per CLAUDE.md §1.5.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Dict, List, Optional

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

from unitport_sdk import Config, log_warning, setButton, tr

from application.service import cross_user_audit
from application.service.cross_user_audit import AuditEntry, RejectResult


# Op → theme slot used to tint the op badge. Overwrite is "warning-ish"
# (the file still exists), delete is full danger. We borrow the existing
# severity slots used by the canvas-error dialog rather than minting a
# new ``warning_zone`` key — keeps the theme catalogue lean.
_OP_SLOT = {
    "overwrite": "mission_diag_severity_warning",
    "delete": "danger_zone",
}
_OP_SLOT_FALLBACK = "mission_diag_severity_warning"


class CrossUserAuditReviewDialog(QDialog):
    """Modal review surface for pending cross-user audit entries."""

    def __init__(
        self,
        target_uid: str,
        entries: List[AuditEntry],
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._target_uid = target_uid
        self._entries: List[AuditEntry] = list(entries)
        # entry_id → its rendered card (so we can hide / annotate one row
        # in-place after a per-card decision without rebuilding the whole
        # scroll body).
        self._cards: Dict[str, QFrame] = {}
        self._status_labels: Dict[str, QLabel] = {}

        self.setObjectName("crossUserAuditReviewDialog")
        self.setModal(True)
        self.setWindowTitle(tr(
            "audit.review.dialog_title",
            default="Pending changes by other users",
        ))
        self.resize(680, 460)

        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(12)

        title = QLabel(
            tr(
                "audit.review.dialog_title",
                default="Pending changes by other users",
            ),
            self,
        )
        title.setObjectName("crossUserAuditTitle")
        outer.addWidget(title, 0)

        intro_text = tr(
            "audit.review.dialog_intro",
            default=(
                "While you were signed out, {n} change(s) were made to "
                "files in your workspace by other users on this machine. "
                "Review each one and choose Accept (keep their change) or "
                "Reject (restore from your cloud copy)."
            ),
        ).format(n=len(self._entries))
        intro = QLabel(intro_text, self)
        intro.setObjectName("crossUserAuditIntro")
        intro.setWordWrap(True)
        outer.addWidget(intro, 0)

        # Scrollable card list.
        scroll = QScrollArea(self)
        scroll.setObjectName("crossUserAuditScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget(scroll)
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        for entry in self._entries:
            card = self._build_card(entry, parent=body)
            self._cards[entry.id] = card
            self._body_layout.addWidget(card, 0)
        self._body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Footer.
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)

        self._btn_accept_all = setButton(
            "audit.review.btn_accept_all",
            120, 32, kind="border", spec="save",
            default="Accept all", parent=self,
        )
        self._btn_accept_all.clicked.connect(self._on_accept_all)
        footer.addWidget(self._btn_accept_all, 0)

        self._btn_reject_all = setButton(
            "audit.review.btn_reject_all",
            120, 32, kind="border", spec="danger",
            default="Reject all", parent=self,
        )
        self._btn_reject_all.clicked.connect(self._on_reject_all)
        footer.addWidget(self._btn_reject_all, 0)

        footer.addStretch(1)

        self._btn_later = setButton(
            "audit.review.btn_later",
            120, 32, kind="light", spec="none",
            default="Decide later", parent=self,
        )
        self._btn_later.clicked.connect(self.accept)
        footer.addWidget(self._btn_later, 0)

        outer.addLayout(footer)

    def _build_card(self, entry: AuditEntry, *, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setObjectName("crossUserAuditCard")
        slot = _OP_SLOT.get(entry.op, _OP_SLOT_FALLBACK)
        frame.setProperty("opSlot", slot)

        v = QVBoxLayout(frame)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        # ---- header: op badge + actor + file ------------------------
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(8)

        badge = QLabel(self._op_label(entry.op), frame)
        badge.setObjectName("crossUserAuditOpBadge")
        badge.setProperty("opSlot", slot)
        head_row.addWidget(badge, 0)

        actor = QLabel(
            tr("audit.review.actor_line", default="{actor}").format(
                actor=entry.actor_label or entry.actor_uid or "(unknown)",
            ),
            frame,
        )
        actor.setObjectName("crossUserAuditActor")
        head_row.addWidget(actor, 0)
        head_row.addStretch(1)

        when = QLabel(
            tr("audit.review.time_line", default="{time}").format(
                time=self._human_time(entry.ts),
            ),
            frame,
        )
        when.setObjectName("crossUserAuditTime")
        head_row.addWidget(when, 0)
        v.addLayout(head_row)

        # ---- file path (full row, eligible for wrap) ----------------
        file_lbl = QLabel(
            tr("audit.review.file_line", default="{path}").format(
                path=entry.rel_path,
            ),
            frame,
        )
        file_lbl.setObjectName("crossUserAuditFile")
        file_lbl.setWordWrap(True)
        file_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        v.addWidget(file_lbl, 0)

        # ---- status row (empty by default; populated after a click) -
        status = QLabel("", frame)
        status.setObjectName("crossUserAuditStatus")
        status.setWordWrap(True)
        status.setVisible(False)
        self._status_labels[entry.id] = status
        v.addWidget(status, 0)

        # ---- per-card buttons (Reject / Accept) ---------------------
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addStretch(1)

        btn_reject = setButton(
            f"audit.review.btn_reject.{entry.id}",
            100, 28, kind="border", spec="danger",
            default=tr("audit.review.btn_reject", default="Reject"),
            parent=frame,
        )
        btn_reject.clicked.connect(lambda _=False, e=entry: self._on_reject(e))
        btn_row.addWidget(btn_reject, 0)

        btn_accept = setButton(
            f"audit.review.btn_accept.{entry.id}",
            100, 28, kind="normal", spec="save",
            default=tr("audit.review.btn_accept", default="Accept"),
            parent=frame,
        )
        btn_accept.clicked.connect(lambda _=False, e=entry: self._on_accept(e))
        btn_row.addWidget(btn_accept, 0)

        v.addLayout(btn_row)
        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _op_label(op: str) -> str:
        if op == "delete":
            return tr("audit.review.op_delete", default="deleted").upper()
        if op == "overwrite":
            return tr("audit.review.op_overwrite", default="edited").upper()
        return op.upper()

    @staticmethod
    def _human_time(iso_ts: str) -> str:
        """Render the audit ts as a relative-time string when recent."""
        if not iso_ts:
            return ""
        try:
            ts = _dt.datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ")
            ts = ts.replace(tzinfo=_dt.timezone.utc)
            epoch = ts.timestamp()
        except (TypeError, ValueError):
            return iso_ts
        delta = max(0.0, time.time() - epoch)
        if delta < 60:
            return tr("homepage.projects.time_just_now", default="just now")
        if delta < 3600:
            n = int(delta // 60)
            return tr(
                "homepage.projects.time_min_n", default="{n} minutes ago",
            ).format(n=n)
        if delta < 86400:
            n = int(delta // 3600)
            return tr(
                "homepage.projects.time_hr_n", default="{n} hours ago",
            ).format(n=n)
        if delta < 86400 * 2:
            return tr("homepage.projects.time_yesterday", default="yesterday")
        if delta < 86400 * 7:
            n = int(delta // 86400)
            return tr(
                "homepage.projects.time_day_n", default="{n} days ago",
            ).format(n=n)
        return ts.strftime("%Y-%m-%d %H:%M")

    def _hide_card(self, entry_id: str) -> None:
        card = self._cards.pop(entry_id, None)
        self._status_labels.pop(entry_id, None)
        if card is not None:
            card.setVisible(False)
            card.deleteLater()
        # Closing the dialog when nothing's left avoids leaving an empty
        # shell on screen.
        if not self._cards:
            self.accept()

    def _show_status(self, entry_id: str, text: str, *, ok: bool) -> None:
        lbl = self._status_labels.get(entry_id)
        if lbl is None:
            return
        lbl.setText(text)
        lbl.setVisible(bool(text))
        # Recolour the status line so success vs failure is immediately
        # readable (theme slots — no literals).
        slot = "safe_zone" if ok else "danger_zone"
        lbl.setStyleSheet(
            f"QLabel#crossUserAuditStatus {{ color: {Config.get_color(slot)}; "
            f"font-size: {Config.get_font_size('size_small')}px; }}"
        )

    # ------------------------------------------------------------------
    # Slots: per-card actions
    # ------------------------------------------------------------------

    def _on_accept(self, entry: AuditEntry) -> None:
        # No second-confirm for Accept — it's the non-destructive choice
        # (we keep the actor's change). One-click decisions feel right
        # for batch review.
        ok = cross_user_audit.accept(self._target_uid, entry.id)
        if ok:
            self._hide_card(entry.id)
        else:
            self._show_status(
                entry.id,
                tr(
                    "audit.review.reject_failed_network",
                    default="Could not update review queue (see log).",
                ).format(err="accept failed"),
                ok=False,
            )

    def _on_reject(self, entry: AuditEntry) -> None:
        # Confirm: cloud pull will overwrite local — irreversible from
        # the user's perspective if they don't have another copy.
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(tr(
            "audit.review.reject_confirm_title",
            default="Restore from cloud?",
        ))
        confirm.setText(tr(
            "audit.review.reject_confirm_body",
            default=(
                "This will pull the previous version of '{name}' from "
                "your cloud namespace and overwrite the local file.\n\n"
                "Continue?"
            ),
        ).format(name=entry.rel_path))
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Ok:
            return

        result: RejectResult = cross_user_audit.reject(self._target_uid, entry.id)
        if result.success:
            self._hide_card(entry.id)
            return

        # Failure branch — keep the card visible and explain in-place.
        if "no cloud copy" in result.error.lower():
            msg = tr(
                "audit.review.reject_failed_no_cloud",
                default=(
                    "No cloud copy available — Reject not possible. "
                    "(Was this file ever synced?)"
                ),
            )
        else:
            msg = tr(
                "audit.review.reject_failed_network",
                default="Cloud fetch failed (network or auth error): {err}",
            ).format(err=result.error or "unknown")
        self._show_status(entry.id, msg, ok=False)
        log_warning(f"[audit-review] reject failed for {entry.id}: {result.error}")

    # ------------------------------------------------------------------
    # Slots: footer bulk actions
    # ------------------------------------------------------------------

    def _on_accept_all(self) -> None:
        # Snapshot ids first — _hide_card mutates self._cards.
        for entry in list(self._entries):
            if entry.id not in self._cards:
                continue
            if cross_user_audit.accept(self._target_uid, entry.id):
                self._hide_card(entry.id)

    def _on_reject_all(self) -> None:
        # One big confirm at the top, not per-row — bulk Reject is an
        # explicit batch action.
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(tr(
            "audit.review.reject_confirm_title",
            default="Restore from cloud?",
        ))
        confirm.setText(tr(
            "audit.review.reject_confirm_body",
            default=(
                "This will pull the previous version of '{name}' from "
                "your cloud namespace and overwrite the local file.\n\n"
                "Continue?"
            ),
        ).format(name=tr(
            "audit.review.btn_reject_all", default="Reject all",
        )))
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Ok:
            return

        for entry in list(self._entries):
            if entry.id not in self._cards:
                continue
            result = cross_user_audit.reject(self._target_uid, entry.id)
            if result.success:
                self._hide_card(entry.id)
            else:
                if "no cloud copy" in result.error.lower():
                    msg = tr(
                        "audit.review.reject_failed_no_cloud",
                        default="No cloud copy available.",
                    )
                else:
                    msg = tr(
                        "audit.review.reject_failed_network",
                        default="Cloud fetch failed: {err}",
                    ).format(err=result.error or "unknown")
                self._show_status(entry.id, msg, ok=False)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        card_bg = Config.get_color("bg_2")
        border = Config.get_color("border_2")
        main = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        op_overwrite = Config.get_color(_OP_SLOT["overwrite"])
        op_delete = Config.get_color(_OP_SLOT["delete"])
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QDialog#crossUserAuditReviewDialog {{ background-color: {bg}; }}"
            f"QLabel#crossUserAuditTitle {{ color: {main}; "
            f"font-size: {font_normal}px; font-weight: 700; }}"
            f"QLabel#crossUserAuditIntro {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QFrame#crossUserAuditCard {{ background-color: {card_bg}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
            f"QLabel#crossUserAuditOpBadge[opSlot=\""
            f"{_OP_SLOT['overwrite']}\"] {{ color: {op_overwrite}; "
            f"font-size: {font_small}px; font-weight: 700; "
            f"letter-spacing: 1px; }}"
            f"QLabel#crossUserAuditOpBadge[opSlot=\""
            f"{_OP_SLOT['delete']}\"] {{ color: {op_delete}; "
            f"font-size: {font_small}px; font-weight: 700; "
            f"letter-spacing: 1px; }}"
            f"QLabel#crossUserAuditActor {{ color: {main}; "
            f"font-size: {font_small}px; font-weight: 600; }}"
            f"QLabel#crossUserAuditTime {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QLabel#crossUserAuditFile {{ color: {main}; "
            f"font-size: {font_small}px; }}"
        )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def show_audit_review_if_pending(
    target_uid: str,
    *,
    parent: Optional[QWidget] = None,
) -> bool:
    """Convenience wrapper used by MainWindow.

    Reads the pending queue for ``target_uid``; if non-empty, opens the
    review dialog modally. Returns True iff the dialog was shown.
    """
    if not target_uid:
        return False
    entries = cross_user_audit.list_pending_for(target_uid)
    if not entries:
        return False
    dlg = CrossUserAuditReviewDialog(target_uid, entries, parent=parent)
    dlg.exec()
    return True
