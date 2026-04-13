"""CloudServerManagerDialog — manage cloud servers for an engine from the sidebar.

Full CRUD: list servers, add/edit/delete, set default, view config summary.
Opened from the User panel cloud settings button.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.system.core.theme_manager import get_color
from src.system.training.remote_ssh_config import RemoteServerConfig


class CloudServerManagerDialog(QDialog):
    """Modal dialog for managing cloud servers of a specific engine."""

    servers_changed = Signal()  # emitted on any add/edit/delete/default change

    def __init__(self, engine_id: str = "isaac_lab", parent=None) -> None:
        super().__init__(parent)
        self._engine_id = engine_id
        self.setWindowTitle(f"Cloud Servers — {engine_id.replace('_', ' ').title()}")
        self.setMinimumSize(560, 400)
        self.setModal(True)
        self._build_ui()
        self._refresh_list()
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        root.addWidget(QLabel("Registered cloud servers:"))

        body = QHBoxLayout()
        body.setSpacing(10)

        # Server list
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selection_changed)
        body.addWidget(self._list, 1)

        # Action buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        self._btn_add = QPushButton("Add...")
        self._btn_add.clicked.connect(self._on_add)
        btn_col.addWidget(self._btn_add)

        self._btn_edit = QPushButton("Edit...")
        self._btn_edit.clicked.connect(self._on_edit)
        btn_col.addWidget(self._btn_edit)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._on_delete)
        btn_col.addWidget(self._btn_delete)

        btn_col.addSpacing(10)

        self._btn_default = QPushButton("Set Default")
        self._btn_default.setToolTip("Set the selected server as the default for new training runs")
        self._btn_default.clicked.connect(self._on_set_default)
        btn_col.addWidget(self._btn_default)

        btn_col.addStretch()
        body.addLayout(btn_col)
        root.addLayout(body, 1)

        # Info summary
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._info_label)

        # Close
        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _store(self):
        from src.system.training.remote_server_store import get_remote_server_store
        return get_remote_server_store()

    def _refresh_list(self) -> None:
        self._list.clear()
        self._servers: List[RemoteServerConfig] = self._store().list_servers()

        from src.system.engines.registry import get_engine_registry
        default_name = get_engine_registry().get_default_server_name(self._engine_id)

        for srv in self._servers:
            badge = ""
            if srv.docker_enabled:
                badge = "[Docker] "
            if srv.name == default_name:
                badge += "(default) "
            label = f"{badge}{srv.name}  —  {srv.username}@{srv.host}:{srv.port}"
            self._list.addItem(label)

        self._on_selection_changed(self._list.currentRow())

    def _selected_server(self) -> Optional[RemoteServerConfig]:
        idx = self._list.currentRow()
        if 0 <= idx < len(self._servers):
            return self._servers[idx]
        return None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self, row: int) -> None:
        has_sel = 0 <= row < len(self._servers)
        self._btn_edit.setEnabled(has_sel)
        self._btn_delete.setEnabled(has_sel)
        self._btn_default.setEnabled(has_sel)

        if has_sel:
            srv = self._servers[row]
            lines = [
                f"Name:  {srv.name}",
                f"Host:  {srv.username}@{srv.host}:{srv.port}",
                f"Auth:  {srv.auth_method}",
            ]
            if srv.docker_enabled:
                lines.append(f"Mode:  Docker")
                lines.append(f"Image: {srv.docker_image}")
                if srv.docker_container_name:
                    lines.append(f"Container: {srv.docker_container_name} (docker exec)")
                else:
                    lines.append(f"Container: ephemeral (docker run)")
                lines.append(f"Host dir:  {srv.docker_host_data_dir}  ->  {srv.docker_container_data_dir}")
            else:
                lines.append(f"Mode:  Bare-metal")
                lines.append(f"Isaac Lab: {srv.remote_isaac_lab_path}")
                if srv.conda_env:
                    lines.append(f"Conda: {srv.conda_env}")
                lines.append(f"Work dir:  {srv.remote_work_dir}")
            self._info_label.setText("\n".join(lines))
        else:
            self._info_label.setText("No server selected")

    def _on_add(self) -> None:
        from bin.pages.training.remote_server_dialog import RemoteServerDialog
        dlg = RemoteServerDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_list()
            self.servers_changed.emit()

    def _on_edit(self) -> None:
        srv = self._selected_server()
        if not srv:
            return
        from bin.pages.training.remote_server_dialog import RemoteServerDialog
        dlg = RemoteServerDialog(parent=self, server=srv, edit_mode=True)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_list()
            self.servers_changed.emit()

    def _on_delete(self) -> None:
        srv = self._selected_server()
        if not srv:
            return
        reply = QMessageBox.question(
            self, "Delete Server",
            f"Remove server '{srv.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._store().remove_server(srv.name)
            self._refresh_list()
            self.servers_changed.emit()

    def _on_set_default(self) -> None:
        srv = self._selected_server()
        if not srv:
            return
        from src.system.engines.registry import get_engine_registry
        get_engine_registry().set_default_server(self._engine_id, srv.name)
        self._refresh_list()
        self.servers_changed.emit()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = get_color("dialog_bg", get_color("button_bg", "#2a2a2a"))
        text = get_color("text_primary", "#e2e8f0")
        border = get_color("border", "#475569")
        input_bg = get_color("input_bg", "#1a1a2e")
        hover = get_color("hover_bg", "#3d3d3d")
        accent = get_color("accent", "#60a5fa")

        self.setStyleSheet(f"""
            QDialog {{ background-color: {bg}; color: {text}; }}
            QLabel {{ color: {text}; background: transparent; }}
            QListWidget {{
                background-color: {input_bg}; color: {text};
                border: 1px solid {border}; border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 6px 8px; border-radius: 3px;
            }}
            QListWidget::item:selected {{
                background-color: {accent}; color: #fff;
            }}
            QListWidget::item:hover {{
                background-color: {hover};
            }}
            QPushButton {{
                background-color: {bg}; color: {text};
                border: 1px solid {border}; border-radius: 6px;
                padding: 6px 14px; min-height: 26px; min-width: 90px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ color: #666; }}
        """)
