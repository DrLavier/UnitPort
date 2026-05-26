# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacInstallsDialog — manage the multi-version Isaac Lab install registry.

Opened from the User panel's Engines → Isaac Lab gear button. Shows every
registered Isaac Lab installation as a row:

    [Root]0.54.2                                  [🗑]
    D: 0.40.1                                     [🗑]

* The short label is drive + version (``[Root]<ver>`` for the project-owned
  base under ``Engines/isaac_lab/``, ``<drive> <ver>`` for external pins). The
  full absolute path is intentionally hidden — it surfaces only on hover
  (tooltip), keeping the list scannable.
* The trash button means **unbind** for an external pin (registry-only remove,
  files untouched) and **uninstall** for the base (delete the directory after a
  confirm, via :class:`IsaacUninstallTask` so the UI never blocks on rmtree).
* The top-right **Add** button opens a directory picker, validates Isaac
  markers, rejects an already-registered root, then registers it.

Versions are refreshed by :class:`IsaacInstallProbeTask` (one subprocess per
root) kicked on open and after each Add — rows render immediately from cached
values and update when the probe lands.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    get_task_signal,
    get_tasks_manager,
    log_warning,
    tr,
)

from registers import backends

from application.service.engines import format_isaac_install_label, get_engine_service
from application.tools.isaac_install_admin_task import (
    IsaacInstallProbeTask,
    IsaacUninstallTask,
)


class _InstallRow(QFrame):
    """One install: short drive/version label + (path on hover) + trash button."""

    delete_clicked = pyqtSignal(str)  # emits the install root

    def __init__(self, entry: Dict[str, object], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._root = str(entry.get("root") or "")
        self._is_base = bool(entry.get("is_base"))
        self.setObjectName("isaacInstallRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(34)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 0, 8, 0)
        row.setSpacing(8)

        sz = int(Config.get_font_size("size_small"))
        self._name = QLabel(format_isaac_install_label(entry), self)
        self._name.setObjectName("isaacInstallName")
        # Full path lives in the tooltip only — list stays clean by default.
        self._name.setToolTip(self._root)
        self._name.setStyleSheet(f"font-size: {sz}px; background: transparent;")
        row.addWidget(self._name, 1)

        if not bool(entry.get("exists", True)):
            warn = QLabel(tr("engines.install_missing", "(missing)"), self)
            warn.setObjectName("isaacInstallMissing")
            warn.setStyleSheet(
                f"color: {Config.get_color('danger_zone')}; "
                f"font-size: {sz}px; background: transparent;"
            )
            row.addWidget(warn, 0)

        self._del = QPushButton(self)
        self._del.setObjectName("isaacInstallRowBtn")
        self._del.setFixedSize(24, 24)
        self._del.setFlat(True)
        self._del.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon = Assets.find_icon("icon_delete")
        if icon is not None:
            self._del.setIcon(QIcon(str(icon)))
            self._del.setIconSize(QSize(14, 14))
        else:
            log_warning("[isaac_installs_dialog] icon missing: icon_delete")
        # Base → uninstall (deletes files); external → unbind (registry only).
        self._del.setToolTip(
            tr("engines.uninstall_tooltip", "Uninstall (delete files)")
            if self._is_base
            else tr("engines.unbind_tooltip", "Unbind (keep files)")
        )
        self._del.clicked.connect(lambda: self.delete_clicked.emit(self._root))
        row.addWidget(self._del, 0)


class IsaacInstallsDialog(QDialog):
    """Modal manager for the registered Isaac Lab installations."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._svc = get_engine_service()
        # task_id → callback(ok, result)
        self._inflight: Dict[str, Callable[[bool, object], None]] = {}

        self.setWindowTitle(tr("engines.isaac_installs_title", "Isaac Lab versions"))
        self.setModal(True)
        self.setMinimumSize(460, 320)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        # Header row: title hint + right-aligned Add button.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(
            tr("engines.isaac_installs_hint", "Registered Isaac Lab installations"),
            self,
        )
        hint.setObjectName("isaacInstallHint")
        header.addWidget(hint, 1)
        self._btn_add = QPushButton(tr("engines.add_install", "Add"), self)
        self._btn_add.setObjectName("isaacInstallAdd")
        self._btn_add.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_add.clicked.connect(self._on_add)
        header.addWidget(self._btn_add, 0, Qt.AlignmentFlag.AlignRight)
        outer.addLayout(header)

        # Scrollable list of rows.
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("isaacInstallScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_host = QWidget(self._scroll)
        self._list_col = QVBoxLayout(self._list_host)
        self._list_col.setContentsMargins(0, 2, 0, 2)
        self._list_col.setSpacing(4)
        self._list_col.addStretch(1)
        self._scroll.setWidget(self._list_host)
        outer.addWidget(self._scroll, 1)

        self._empty = QLabel(
            tr("engines.isaac_installs_empty",
               "No Isaac Lab installed — Add an existing install root."),
            self._list_host,
        )
        self._empty.setObjectName("isaacInstallEmpty")
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._apply_theme()
        get_task_signal().task_finished.connect(self._on_task_finished)

        self._render()
        self._kick_probe()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_rows(self) -> None:
        for i in reversed(range(self._list_col.count())):
            item = self._list_col.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is None or w is self._empty:
                continue
            self._list_col.removeWidget(w)
            w.setParent(None)
            w.deleteLater()

    def _render(self) -> None:
        self._clear_rows()
        installs = self._svc.list_isaac_installations()
        if not installs:
            self._empty.setVisible(True)
            # Park the empty label as the first item.
            self._list_col.insertWidget(0, self._empty)
            return
        self._empty.setVisible(False)
        for idx, entry in enumerate(installs):
            row = _InstallRow(entry, self._list_host)
            row.delete_clicked.connect(self._on_delete)
            self._list_col.insertWidget(idx, row)

    # ------------------------------------------------------------------
    # Background version probe
    # ------------------------------------------------------------------

    def _kick_probe(self) -> None:
        try:
            tid = get_tasks_manager().submit(IsaacInstallProbeTask())
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[isaac_installs_dialog] probe submit failed: {exc!r}")
            return
        self._inflight[tid] = self._on_probe_done

    def _on_probe_done(self, ok: bool, _result: object) -> None:
        if not ok:
            log_warning("[isaac_installs_dialog] version probe failed")
        # Re-render from the freshly persisted registry either way.
        self._render()

    # ------------------------------------------------------------------
    # Add / delete
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            tr("engines.pick_isaac", "Select Isaac Lab installation root"),
            "",
        )
        if not chosen:
            return
        if any(
            backends.install_roots_equal(e.get("root", ""), chosen)
            for e in self._svc.list_isaac_installations()
        ):
            QMessageBox.information(
                self,
                tr("engines.already_registered_title", "Already registered"),
                tr("engines.already_registered_text",
                   "This Isaac Lab root is already registered."),
            )
            return
        if not self._svc.register_isaac_local(chosen):
            QMessageBox.warning(
                self,
                tr("engines.invalid_root_title", "Not an Isaac Lab root"),
                tr("engines.invalid_root_text",
                   "The selected folder is not a valid Isaac Lab installation "
                   "(missing isaaclab.sh/.bat and source/)."),
            )
            return
        self._render()
        self._kick_probe()

    def _on_delete(self, root: str) -> None:
        if backends.is_base_isaac_root(root):
            confirm = QMessageBox.question(
                self,
                tr("engines.confirm_uninstall_title", "Uninstall Isaac Lab"),
                tr("engines.confirm_uninstall_text",
                   "This permanently deletes the base Isaac Lab installation "
                   "from disk:\n\n{root}\n\nThis cannot be undone. Continue?")
                .format(root=root),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            try:
                tid = get_tasks_manager().submit(IsaacUninstallTask(root=root))
            except Exception as exc:                              # noqa: BLE001
                log_warning(f"[isaac_installs_dialog] uninstall submit failed: {exc!r}")
                return
            self._btn_add.setEnabled(False)
            self._inflight[tid] = self._on_uninstall_done
            return
        # External pin → unbind only.
        if not self._svc.unbind_isaac_installation(root):
            log_warning(f"[isaac_installs_dialog] unbind failed for {root!r}")
        self._render()

    def _on_uninstall_done(self, ok: bool, result: object) -> None:
        self._btn_add.setEnabled(True)
        if not ok:
            data = result if isinstance(result, dict) else {}
            QMessageBox.warning(
                self,
                tr("engines.uninstall_failed_title", "Uninstall failed"),
                str(data.get("error") or result
                    or tr("engines.uninstall_failed_text", "Could not delete the "
                          "Isaac Lab installation. See logs.")),
            )
        self._render()

    # ------------------------------------------------------------------
    # Task plumbing / theme
    # ------------------------------------------------------------------

    def _on_task_finished(self, task_id: str, ok: bool, result: object) -> None:
        cb = self._inflight.pop(task_id, None)
        if cb is not None:
            cb(ok, result)

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        row_hover = Config.get_color("hover_1")
        border = Config.get_color("border_1")
        text = Config.get_color("main_t1")
        muted = Config.get_color("main_c2")
        btn_bg = Config.get_color("bg_3")
        sz = int(Config.get_font_size("size_small"))
        self.setStyleSheet(
            f"QDialog {{ background: {bg}; }}"
            f"QLabel#isaacInstallHint {{ color: {muted}; font-size: {sz}px; }}"
            f"QLabel#isaacInstallEmpty {{ color: {muted}; font-size: {sz}px; }}"
            f"QLabel#isaacInstallName {{ color: {text}; }}"
            f"QFrame#isaacInstallRow {{ background: transparent; border: 1px "
            f"solid transparent; border-radius: 4px; }}"
            f"QFrame#isaacInstallRow:hover {{ background: {row_hover}; }}"
            f"QPushButton#isaacInstallRowBtn {{ background: transparent; border: "
            f"none; border-radius: 4px; padding: 2px; }}"
            f"QPushButton#isaacInstallRowBtn:hover {{ background: {btn_bg}; }}"
            f"QPushButton#isaacInstallAdd {{ background: {btn_bg}; color: {text}; "
            f"border: 1px solid {border}; border-radius: 6px; padding: 4px 14px; "
            f"font-size: {sz}px; }}"
            f"QPushButton#isaacInstallAdd:hover {{ background: {row_hover}; }}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._inflight.clear()
        try:
            get_task_signal().task_finished.disconnect(self._on_task_finished)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)


def prompt_no_local_isaac(parent: Optional[QWidget]) -> bool:
    """Ask the user what to do when no local Isaac Lab is registered.

    Returns True if the user wants to register/locate a local install (caller
    should then open :class:`IsaacInstallsDialog`); False if they chose
    cloud-only, in which case we persist that opt-out so we stop prompting.
    """
    svc = get_engine_service()
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("engines.no_local_title", "No local Isaac Lab"))
    box.setText(tr(
        "engines.no_local_text",
        "No Isaac Lab installation is registered on this machine. Register a "
        "local install, or use cloud training only?",
    ))
    btn_register = box.addButton(
        tr("engines.no_local_register", "Register local…"),
        QMessageBox.ButtonRole.AcceptRole,
    )
    box.addButton(
        tr("engines.no_local_cloud", "Cloud only"),
        QMessageBox.ButtonRole.RejectRole,
    )
    box.exec()
    if box.clickedButton() is btn_register:
        return True
    # Cloud-only: remember the choice and switch the target.
    svc.set_no_isaac_prompt_dismissed(True)
    svc.set_link_target("cloud")
    return False


__all__ = ["IsaacInstallsDialog", "prompt_no_local_isaac"]
