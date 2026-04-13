"""RemoteServerDialog — PySide6 dialog for configuring remote SSH servers.

Allows the user to add, edit, test, and save remote server configurations
for cloud-based Isaac Lab training.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.system.core.theme_manager import get_color
from src.system.training.remote_ssh_config import RemoteServerConfig


class RemoteServerDialog(QDialog):
    """Modal dialog for configuring a remote SSH server."""

    server_saved = Signal(object)  # emits RemoteServerConfig

    def __init__(
        self,
        parent=None,
        *,
        server: RemoteServerConfig | None = None,
        edit_mode: bool = False,
    ):
        super().__init__(parent)
        self._edit_mode = edit_mode
        self._original_name = server.name if server and edit_mode else ""
        self.setWindowTitle("Edit Server" if edit_mode else "Add Remote Server")
        self.setMinimumWidth(520)
        self.setModal(True)

        self._build_ui()
        if server:
            self._populate(server)
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # ── Connection group ────────────────────────────────────────────
        conn_group = QGroupBox("Connection")
        conn_form = QFormLayout(conn_group)
        conn_form.setSpacing(8)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Lab GPU Server")
        conn_form.addRow("Name:", self._name_edit)

        self._host_edit = QLineEdit()
        self._host_edit.setPlaceholderText("192.168.1.100 or server.example.com")
        conn_form.addRow("Host:", self._host_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(22)
        conn_form.addRow("Port:", self._port_spin)

        self._username_edit = QLineEdit()
        self._username_edit.setPlaceholderText("username")
        conn_form.addRow("Username:", self._username_edit)

        self._auth_combo = QComboBox()
        self._auth_combo.addItems(["SSH Key", "Password"])
        self._auth_combo.currentIndexChanged.connect(self._on_auth_changed)
        conn_form.addRow("Auth:", self._auth_combo)

        # Key path row
        key_row = QHBoxLayout()
        self._key_edit = QLineEdit()
        self._key_edit.setPlaceholderText("~/.ssh/id_rsa")
        self._key_browse_btn = QPushButton("Browse...")
        self._key_browse_btn.setFixedWidth(80)
        self._key_browse_btn.clicked.connect(self._on_browse_key)
        key_row.addWidget(self._key_edit, 1)
        key_row.addWidget(self._key_browse_btn)
        self._key_label = QLabel("Key File:")
        conn_form.addRow(self._key_label, key_row)

        # Password row
        self._password_edit = QLineEdit()
        self._password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_label = QLabel("Password:")
        conn_form.addRow(self._password_label, self._password_edit)

        layout.addWidget(conn_group)

        # ── Remote Environment group ────────────────────────────────────
        env_group = QGroupBox("Remote Environment")
        env_form = QFormLayout(env_group)
        env_form.setSpacing(8)

        # Docker toggle
        self._docker_check = QCheckBox("Use Docker container (official Isaac Lab image)")
        self._docker_check.toggled.connect(self._on_docker_toggled)
        env_form.addRow(self._docker_check)

        # Docker image
        self._docker_image_edit = QLineEdit()
        self._docker_image_edit.setPlaceholderText("nvcr.io/nvidia/isaac-lab:2.3.2")
        self._docker_image_label = QLabel("Image:")
        env_form.addRow(self._docker_image_label, self._docker_image_edit)

        # Container name (if set → docker exec on existing; if empty → docker run new)
        self._docker_container_edit = QLineEdit()
        self._docker_container_edit.setPlaceholderText("isaac-lab-5.1 (leave empty for ephemeral docker run)")
        self._docker_container_label = QLabel("Container:")
        env_form.addRow(self._docker_container_label, self._docker_container_edit)

        # Host data dir (the local_data_dir from the cloud panel)
        self._docker_host_dir_edit = QLineEdit()
        self._docker_host_dir_edit.setPlaceholderText("/home/user/isaaclab_data")
        self._docker_host_dir_label = QLabel("Host Data Dir:")
        env_form.addRow(self._docker_host_dir_label, self._docker_host_dir_edit)

        # Container data dir (docker_data_dir from the cloud panel)
        self._docker_container_dir_edit = QLineEdit()
        self._docker_container_dir_edit.setText("/data")
        self._docker_container_dir_label = QLabel("Container Dir:")
        env_form.addRow(self._docker_container_dir_label, self._docker_container_dir_edit)

        # Docker extra args
        self._docker_extra_edit = QLineEdit()
        self._docker_extra_edit.setPlaceholderText("--shm-size=8g (optional)")
        self._docker_extra_label = QLabel("Extra Args:")
        env_form.addRow(self._docker_extra_label, self._docker_extra_edit)

        # Bare-metal fields (hidden in Docker mode)
        self._isaac_path_edit = QLineEdit()
        self._isaac_path_edit.setPlaceholderText("/home/user/IsaacLab")
        self._isaac_path_label = QLabel("Isaac Lab Path:")
        env_form.addRow(self._isaac_path_label, self._isaac_path_edit)

        self._launcher_edit = QLineEdit()
        self._launcher_edit.setPlaceholderText("/home/user/IsaacLab/isaaclab.sh (auto-detected if empty)")
        self._launcher_label = QLabel("Launcher:")
        env_form.addRow(self._launcher_label, self._launcher_edit)

        self._conda_edit = QLineEdit()
        self._conda_edit.setPlaceholderText("isaaclab (optional)")
        self._conda_label = QLabel("Conda Env:")
        env_form.addRow(self._conda_label, self._conda_edit)

        self._work_dir_edit = QLineEdit()
        self._work_dir_edit.setText("/tmp/unitport_training")
        env_form.addRow("Work Dir:", self._work_dir_edit)

        layout.addWidget(env_group)

        # Initial state: Docker off
        self._on_docker_toggled(False)

        # ── Bottom buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test)
        btn_row.addWidget(self._test_btn)

        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)

        layout.addLayout(btn_row)

        # Initial auth toggle.
        self._on_auth_changed(0)

    # ------------------------------------------------------------------
    # Auth toggle
    # ------------------------------------------------------------------

    def _on_auth_changed(self, index: int) -> None:
        is_key = index == 0
        self._key_edit.setVisible(is_key)
        self._key_browse_btn.setVisible(is_key)
        self._key_label.setVisible(is_key)
        self._password_edit.setVisible(not is_key)
        self._password_label.setVisible(not is_key)

    def _on_docker_toggled(self, checked: bool) -> None:
        """Show Docker fields or bare-metal fields based on the toggle."""
        # Docker fields
        for w in (self._docker_image_edit, self._docker_image_label,
                  self._docker_container_edit, self._docker_container_label,
                  self._docker_host_dir_edit, self._docker_host_dir_label,
                  self._docker_container_dir_edit, self._docker_container_dir_label,
                  self._docker_extra_edit, self._docker_extra_label):
            w.setVisible(checked)
        # Bare-metal fields
        for w in (self._isaac_path_edit, self._isaac_path_label,
                  self._launcher_edit, self._launcher_label,
                  self._conda_edit, self._conda_label):
            w.setVisible(not checked)

    def _on_browse_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Private Key", "",
            "All Files (*)",
        )
        if path:
            self._key_edit.setText(path)

    # ------------------------------------------------------------------
    # Populate / collect
    # ------------------------------------------------------------------

    def _populate(self, server: RemoteServerConfig) -> None:
        self._name_edit.setText(server.name)
        self._host_edit.setText(server.host)
        self._port_spin.setValue(server.port)
        self._username_edit.setText(server.username)
        if server.auth_method == "password":
            self._auth_combo.setCurrentIndex(1)
            self._password_edit.setText(server.password)
        else:
            self._auth_combo.setCurrentIndex(0)
            self._key_edit.setText(server.private_key_path)
        self._docker_check.setChecked(server.docker_enabled)
        self._docker_image_edit.setText(server.docker_image)
        self._docker_container_edit.setText(server.docker_container_name)
        self._docker_host_dir_edit.setText(server.docker_host_data_dir)
        self._docker_container_dir_edit.setText(server.docker_container_data_dir or "/data")
        self._docker_extra_edit.setText(server.docker_extra_args)
        self._isaac_path_edit.setText(server.remote_isaac_lab_path)
        self._launcher_edit.setText(server.remote_isaac_lab_launcher)
        self._conda_edit.setText(server.conda_env)
        self._work_dir_edit.setText(server.remote_work_dir or "/tmp/unitport_training")

    def _collect(self) -> RemoteServerConfig:
        auth = "key" if self._auth_combo.currentIndex() == 0 else "password"
        use_docker = self._docker_check.isChecked()
        return RemoteServerConfig(
            name=self._name_edit.text().strip(),
            host=self._host_edit.text().strip(),
            port=self._port_spin.value(),
            username=self._username_edit.text().strip(),
            auth_method=auth,
            private_key_path=self._key_edit.text().strip() if auth == "key" else "",
            password=self._password_edit.text() if auth == "password" else "",
            docker_enabled=use_docker,
            docker_image=self._docker_image_edit.text().strip() if use_docker else "",
            docker_container_name=self._docker_container_edit.text().strip() if use_docker else "",
            docker_host_data_dir=self._docker_host_dir_edit.text().strip() if use_docker else "",
            docker_container_data_dir=self._docker_container_dir_edit.text().strip() or "/data" if use_docker else "",
            docker_extra_args=self._docker_extra_edit.text().strip() if use_docker else "",
            remote_isaac_lab_path=self._isaac_path_edit.text().strip() if not use_docker else "",
            remote_isaac_lab_launcher=self._launcher_edit.text().strip() if not use_docker else "",
            remote_work_dir=self._work_dir_edit.text().strip() or "/tmp/unitport_training",
            conda_env=self._conda_edit.text().strip() if not use_docker else "",
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_test(self) -> None:
        server = self._collect()
        err = server.validate()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self._test_btn.setEnabled(False)
        self._test_btn.setText("Testing...")
        try:
            from src.system.training.remote_server_store import get_remote_server_store
            ok, msg = get_remote_server_store().test_connection(server)
            if ok:
                QMessageBox.information(self, "Connection Test", msg)
            else:
                QMessageBox.warning(self, "Connection Test Failed", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            self._test_btn.setEnabled(True)
            self._test_btn.setText("Test Connection")

    def _on_save(self) -> None:
        server = self._collect()
        err = server.validate()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        try:
            from src.system.training.remote_server_store import get_remote_server_store
            store = get_remote_server_store()
            if self._edit_mode:
                store.update_server(self._original_name, server)
            else:
                store.add_server(server)
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Save Error", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.server_saved.emit(server)
        self.accept()

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
            QDialog {{
                background-color: {bg};
                color: {text};
            }}
            QGroupBox {{
                color: {text};
                font-weight: 600;
                border: 1px solid {border};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                padding: 0 6px;
            }}
            QLabel {{
                color: {text};
                background: transparent;
            }}
            QLineEdit, QSpinBox, QComboBox {{
                background-color: {input_bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }}
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
                border-color: {accent};
            }}
            QPushButton {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 16px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:pressed {{
                background-color: {input_bg};
            }}
        """)


class RemoteServerSelectDialog(QDialog):
    """Quick dialog to pick a saved server before starting cloud training."""

    server_selected = Signal(object)  # emits RemoteServerConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Remote Server")
        self.setMinimumWidth(400)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(QLabel("Choose a remote server for training:"))

        self._combo = QComboBox()
        self._refresh_list()
        layout.addWidget(self._combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        add_btn = QPushButton("Add New...")
        add_btn.clicked.connect(self._on_add_new)
        btn_row.addWidget(add_btn)

        btn_row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton("Start Training")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)

        layout.addLayout(btn_row)

        bg = get_color("dialog_bg", get_color("button_bg", "#2a2a2a"))
        text = get_color("text_primary", "#e2e8f0")
        border = get_color("border", "#475569")
        input_bg = get_color("input_bg", "#1a1a2e")
        accent = get_color("accent", "#60a5fa")
        hover = get_color("hover_bg", "#3d3d3d")

        self.setStyleSheet(f"""
            QDialog {{ background-color: {bg}; color: {text}; }}
            QLabel {{ color: {text}; background: transparent; }}
            QComboBox {{
                background-color: {input_bg}; color: {text};
                border: 1px solid {border}; border-radius: 4px;
                padding: 4px 8px; min-height: 24px;
            }}
            QPushButton {{
                background-color: {bg}; color: {text};
                border: 1px solid {border}; border-radius: 6px;
                padding: 6px 16px; min-height: 28px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
        """)

    def _refresh_list(self) -> None:
        self._combo.clear()
        from src.system.training.remote_server_store import get_remote_server_store
        servers = get_remote_server_store().list_servers()
        self._servers = servers
        for s in servers:
            self._combo.addItem(f"{s.name}  ({s.username}@{s.host}:{s.port})")

    def _on_add_new(self) -> None:
        dlg = RemoteServerDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_list()

    def _on_ok(self) -> None:
        idx = self._combo.currentIndex()
        if idx < 0 or idx >= len(self._servers):
            QMessageBox.warning(self, "No Server", "Please select or add a server first.")
            return
        server = self._servers[idx]

        # If password auth, prompt for password (not stored on disk).
        if server.auth_method == "password" and not server.password:
            from PySide6.QtWidgets import QInputDialog
            pw, ok = QInputDialog.getText(
                self, "SSH Password",
                f"Password for {server.username}@{server.host}:",
                QLineEdit.EchoMode.Password,
            )
            if not ok or not pw:
                return
            server.password = pw

        self.server_selected.emit(server)
        self.accept()
