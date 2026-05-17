"""EngineSettingsDialog — manage cloud servers for a single engine.

Opened from the User panel's "cloud settings" gear button. Lets the user add /
edit / remove SSH-like training servers and pick the default. Passwords are
NEVER persisted (use ``SecureCredentialStore.ssh_password`` from a separate
flow when the time comes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, I18n, tr

from application.service.engines import get_engine_service


class _ServerEditor(QDialog):
    """Modal sub-dialog for adding / editing one server entry."""

    def __init__(
        self,
        existing: Optional[Dict[str, Any]] = None,
        forbidden_names: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._forbidden = set(forbidden_names or [])
        self.setWindowTitle(
            tr("engines.server_editor.edit", "Edit server")
            if existing
            else tr("engines.server_editor.add", "Add server")
        )
        self.setModal(True)
        self.setMinimumWidth(360)

        self._name = QLineEdit()
        self._host = QLineEdit()
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(22)
        self._user = QLineEdit()

        if existing:
            self._name.setText(str(existing.get("name", "")))
            self._host.setText(str(existing.get("host", "")))
            self._port.setValue(int(existing.get("port", 22) or 22))
            self._user.setText(str(existing.get("user", "")))

        form = QFormLayout()
        form.addRow(tr("engines.server_editor.name", "Display name"), self._name)
        form.addRow(tr("engines.server_editor.host", "Host"), self._host)
        form.addRow(tr("engines.server_editor.port", "Port"), self._port)
        form.addRow(tr("engines.server_editor.user", "User"), self._user)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                tr("engines.server_editor.invalid_title", "Invalid"),
                tr("engines.server_editor.name_required", "Display name is required."),
            )
            return
        if name in self._forbidden:
            QMessageBox.warning(
                self,
                tr("engines.server_editor.invalid_title", "Invalid"),
                tr("engines.server_editor.name_taken",
                   "A server named {name} already exists.").format(name=name),
            )
            return
        host = self._host.text().strip()
        if not host:
            QMessageBox.warning(
                self,
                tr("engines.server_editor.invalid_title", "Invalid"),
                tr("engines.server_editor.host_required", "Host is required."),
            )
            return
        self.accept()

    def value(self) -> Dict[str, Any]:
        return {
            "name": self._name.text().strip(),
            "host": self._host.text().strip(),
            "port": int(self._port.value()),
            "user": self._user.text().strip(),
        }


class EngineSettingsDialog(QDialog):
    """List + CRUD over an engine's cloud servers."""

    def __init__(self, engine_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine_id = engine_id
        self._svc = get_engine_service()

        # Title 含 {engine} 占位，不能直接 i18n_bind setWindowTitle。
        # 自带闭包 + alive flag（pattern 同 i18n_bind）实现 dead-widget
        # 自清，并在 language_changed 时重新 format。
        sig = I18n.instance().language_changed
        state = {"alive": True}

        def _disconnect():
            if not state["alive"]:
                return
            state["alive"] = False
            try:
                sig.disconnect(_set_title)
            except (TypeError, RuntimeError):
                pass

        def _set_title(*_):
            if not state["alive"]:
                return
            try:
                self.setWindowTitle(
                    tr("engines.settings_title", "Cloud servers — {engine}")
                    .format(engine=engine_id)
                )
            except RuntimeError:
                _disconnect()

        _set_title()
        sig.connect(_set_title)
        self.destroyed.connect(lambda *_: _disconnect())
        self.setModal(True)
        self.setMinimumSize(480, 360)

        self._build_ui()
        self._apply_styles()
        self._refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        header = QLabel(tr(
            "engines.settings_hint",
            "SSH-style training servers for {engine}. Passwords live in the OS keyring; nothing here is persisted in plaintext.",
        ).format(engine=self._engine_id))
        header.setWordWrap(True)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(10)

        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        body.addWidget(self._list, 1)

        col = QVBoxLayout()
        self._btn_add = QPushButton(tr("engines.add_server", "Add..."))
        self._btn_edit = QPushButton(tr("engines.edit_server", "Edit..."))
        self._btn_remove = QPushButton(tr("engines.remove_server", "Remove"))
        self._btn_default = QPushButton(tr("engines.set_default", "Set as default"))
        for b in (self._btn_add, self._btn_edit, self._btn_remove, self._btn_default):
            b.setMinimumHeight(28)
            col.addWidget(b)
        col.addStretch(1)
        body.addLayout(col, 0)

        root.addLayout(body, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self._btn_close = QPushButton(tr("auth.close_button", "Close"))
        close_row.addWidget(self._btn_close)
        root.addLayout(close_row)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_remove.clicked.connect(self._on_remove)
        self._btn_default.clicked.connect(self._on_set_default)
        self._btn_close.clicked.connect(self.accept)

    def _apply_styles(self) -> None:
        bg = Config.get_color("bg_2", "#1A1A1A")
        border = Config.get_color("border_1", "#444444")
        fg = Config.get_color("main_t1", "#D6D3C7")
        accent = Config.get_color("safe_zone", "#36E38E")
        self.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
            f"QListWidget {{ background: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 4px; }}"
            f"QLineEdit, QSpinBox {{ background: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 4px; padding: 4px 6px; }}"
            f"QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 4px; padding: 4px 10px; }}"
            f"QPushButton:hover {{ border-color: {accent}; }}"
        )

    # ----- list / state ---------------------------------------------------

    def _refresh(self) -> None:
        self._list.clear()
        servers = self._svc.list_servers(self._engine_id)
        default_name = self._svc.get_default_server_name(self._engine_id)
        for srv in servers:
            label = f"{srv.get('name', '?')}  —  {srv.get('user','')}@{srv.get('host','')}:{srv.get('port', 22)}"
            if srv.get("name") == default_name:
                label = "* " + label
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, srv)
            self._list.addItem(item)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        sel = bool(self._list.selectedItems())
        self._btn_edit.setEnabled(sel)
        self._btn_remove.setEnabled(sel)
        self._btn_default.setEnabled(sel)

    def _selected_server(self) -> Optional[Dict[str, Any]]:
        items = self._list.selectedItems()
        if not items:
            return None
        return dict(items[0].data(Qt.ItemDataRole.UserRole) or {})

    # ----- actions --------------------------------------------------------

    def _on_add(self) -> None:
        existing_names = [s.get("name") for s in self._svc.list_servers(self._engine_id)]
        editor = _ServerEditor(forbidden_names=existing_names, parent=self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            try:
                self._svc.add_server(self._engine_id, editor.value())
            except (ValueError, KeyError) as exc:
                QMessageBox.warning(self, tr("engines.error_title", "Error"), str(exc))
            self._refresh()

    def _on_edit(self) -> None:
        srv = self._selected_server()
        if not srv:
            return
        old_name = srv.get("name", "")
        existing_names = [
            s.get("name") for s in self._svc.list_servers(self._engine_id)
            if s.get("name") != old_name
        ]
        editor = _ServerEditor(existing=srv, forbidden_names=existing_names, parent=self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            try:
                self._svc.update_server(self._engine_id, old_name, editor.value())
            except (ValueError, KeyError) as exc:
                QMessageBox.warning(self, tr("engines.error_title", "Error"), str(exc))
            self._refresh()

    def _on_remove(self) -> None:
        srv = self._selected_server()
        if not srv:
            return
        name = srv.get("name", "")
        confirm = QMessageBox.question(
            self,
            tr("engines.confirm_remove_title", "Remove server"),
            tr("engines.confirm_remove_text", "Remove server {name}?").format(name=name),
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._svc.remove_server(self._engine_id, name)
            self._refresh()

    def _on_set_default(self) -> None:
        srv = self._selected_server()
        if not srv:
            return
        self._svc.set_default_server(self._engine_id, srv.get("name", ""))
        self._refresh()


__all__ = ["EngineSettingsDialog"]
