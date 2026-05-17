"""ConnectionSettingsDialog — modal for ROS2 Domain ID + SSH credentials.

Phase 3.8 collapses RealRobotConnectionCard sec1 by extracting the
"set once and forget" knobs into this dialog. The card body now only
shows the per-connect choices (Robot / Adapter strategy / Middleware /
Transport+Address+Port) plus a [Settings] button that opens this dialog.

All edits inside the dialog persist immediately:
- Domain ID writes to ``user.ini[RobotConn].domain_id`` on combo change.
- SSH user writes to ``ssh_user`` on text change.
- SSH password / sudo password write to :class:`SecureCredentialStore`
  on the [Save] button click (matching the previous in-card behaviour).

There is no Save/Cancel split — every change is durable, so the footer
only carries [Close]. The dialog is modal so the card cannot be edited
while it is open (avoids racy reads of ``Ros2ConnectionConfig.load()``).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18nLabel,
    LaviComboBox,
    LaviLineEdit,
    log_info,
    log_warning,
    setButton,
    setComboBox,
    setLineEdit,
    tr,
)

from application.service.auth.secure import get_secure_store
from application.service.ros2_connection_config import (
    DEFAULT_DOMAIN_ID,
    Ros2ConnectionConfig,
    build_profile_from_info,
    ssh_server_name_for_profile,
)


# Domain choices — matches the card's previous range. Brand-specific
# defaults (e.g. Mangdang's 42) are added on demand by _ensure_domain_choice.
_DOMAIN_CHOICES: tuple = tuple(str(i) for i in range(11))


class ConnectionSettingsDialog(QDialog):
    """Modal: Domain ID combo + SSH credentials block."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionConnSettingsDialog")
        self.setModal(True)
        self.setWindowTitle(
            tr("mission.conn.dialog.title", default="Connection Settings")
        )
        self.resize(480, 420)
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        info = Ros2ConnectionConfig.load()

        # --- Domain ID section -----------------------------------------
        domain_title = I18nLabel(
            "mission.conn.dialog.section.domain", default="ROS2 Domain ID",
            parent=self,
        )
        domain_title.setObjectName("missionConnSettingsSectionTitle")
        outer.addWidget(domain_title, 0)

        self._cb_domain: LaviComboBox = setComboBox(
            list(_DOMAIN_CHOICES), height=28, i18n=False, parent=self,
        )
        self._ensure_domain_choice(int(info.domain_id))
        self._set_combo_text(
            self._cb_domain, str(info.domain_id), str(DEFAULT_DOMAIN_ID),
        )
        self._cb_domain.currentTextChanged.connect(self._on_domain_changed)
        outer.addWidget(self._cb_domain, 0)

        # --- SSH credentials section -----------------------------------
        ssh_title = I18nLabel(
            "mission.conn.dialog.section.ssh", default="SSH Credentials",
            parent=self,
        )
        ssh_title.setObjectName("missionConnSettingsSectionTitle")
        outer.addWidget(ssh_title, 0)

        self._ssh_block = QFrame(self)
        self._ssh_block.setObjectName("missionConnSettingsSshBlock")
        ssh_layout = QVBoxLayout(self._ssh_block)
        ssh_layout.setContentsMargins(10, 8, 10, 10)
        ssh_layout.setSpacing(8)

        # Server label (read-only — driven by the active profile).
        profile = build_profile_from_info(info)
        self._server_key = ssh_server_name_for_profile(profile)
        self._server_label = QLabel(self._ssh_block)
        self._server_label.setObjectName("missionConnSettingsSshServer")
        self._server_label.setWordWrap(True)
        self._server_label.setText(
            tr("mission.conn.dialog.ssh.server", default="Server: {server}")
            .format(server=self._server_key)
        )
        ssh_layout.addWidget(self._server_label, 0)

        # SSH user.
        self._le_user: LaviLineEdit = setLineEdit(
            text=str(profile.ssh_user or "ubuntu"),
            placeholder="ubuntu",
            parent=self._ssh_block,
        )
        self._le_user.setFixedHeight(28)
        self._le_user.textChanged.connect(self._on_user_changed)
        ssh_layout.addWidget(self._kv_row(
            tr("mission.conn.ssh.user", default="SSH user"), self._le_user,
        ), 0)

        # Password row.
        store = get_secure_store()
        self._has_password = bool(store.ssh_password(self._server_key).get())
        self._has_sudo = bool(store.ssh_sudo_password(self._server_key).get())

        self._le_pw: LaviLineEdit = setLineEdit(
            placeholder=tr(
                "mission.conn.ssh.password.placeholder",
                default="(enter password and press Save)",
            ),
            parent=self._ssh_block,
        )
        self._le_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self._le_pw.setFixedHeight(28)
        self._btn_save_pw = setButton(
            "mission.conn.ssh.btn.save_password", 88, 28,
            kind="normal", spec="save", default="Save", parent=self._ssh_block,
        )
        self._btn_save_pw.clicked.connect(self._on_save_password)
        ssh_layout.addWidget(self._kv_row(
            tr("mission.conn.ssh.password", default="Password"),
            self._inline_widget(self._le_pw, self._btn_save_pw),
        ), 0)

        # Sudo password row.
        self._le_sudo: LaviLineEdit = setLineEdit(
            placeholder=tr(
                "mission.conn.ssh.sudo.placeholder",
                default="(needed for invasive repairs)",
            ),
            parent=self._ssh_block,
        )
        self._le_sudo.setEchoMode(QLineEdit.EchoMode.Password)
        self._le_sudo.setFixedHeight(28)
        self._btn_save_sudo = setButton(
            "mission.conn.ssh.btn.save_sudo", 88, 28,
            kind="normal", spec="save", default="Save", parent=self._ssh_block,
        )
        self._btn_save_sudo.clicked.connect(self._on_save_sudo)
        ssh_layout.addWidget(self._kv_row(
            tr("mission.conn.ssh.sudo", default="Sudo password"),
            self._inline_widget(self._le_sudo, self._btn_save_sudo),
        ), 0)

        # Status line.
        self._status_label = QLabel("", self._ssh_block)
        self._status_label.setObjectName("missionConnSettingsSshStatus")
        ssh_layout.addWidget(self._status_label, 0)
        self._refresh_status_label()

        outer.addWidget(self._ssh_block, 0)
        outer.addStretch(1)

        # --- Footer ----------------------------------------------------
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        self._btn_close = setButton(
            "mission.conn.dialog.btn.close", 100, 32,
            kind="normal", spec="none", default="Close", parent=self,
        )
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close, 0)
        outer.addLayout(footer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kv_row(self, label_text: str, control: QWidget) -> QWidget:
        row = QWidget(self._ssh_block)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        lbl = QLabel(label_text, row)
        lbl.setObjectName("missionConnSettingsKv")
        lbl.setFixedWidth(120)
        rl.addWidget(lbl, 0)
        rl.addWidget(control, 1)
        return row

    def _inline_widget(self, edit: QWidget, btn: QWidget) -> QWidget:
        host = QWidget(self._ssh_block)
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(edit, 1)
        layout.addWidget(btn, 0)
        return host

    @staticmethod
    def _set_combo_text(cb: LaviComboBox, text: str, fallback: str) -> None:
        for i in range(cb.count()):
            if cb.itemText(i) == text:
                cb.setCurrentIndex(i)
                return
        for i in range(cb.count()):
            if cb.itemText(i) == fallback:
                cb.setCurrentIndex(i)
                return

    def _ensure_domain_choice(self, domain_id: int) -> None:
        """Append ``domain_id`` to the combo if not already present.

        Brand-default domains (Mangdang = 42) live outside the static
        0..10 range; this helper makes them addressable without erasing
        the static choices. Idempotent.
        """
        target = str(int(domain_id))
        existing: List[str] = []
        try:
            for i in range(self._cb_domain.count()):
                item = self._cb_domain._src_model.item(i)
                if item is not None:
                    existing.append(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        except Exception:
            existing = [self._cb_domain.itemText(i) for i in range(self._cb_domain.count())]
        if target in existing:
            return
        new_items = sorted({*existing, target}, key=lambda s: int(s))
        self._cb_domain.blockSignals(True)
        try:
            self._cb_domain.setItems(new_items)
        finally:
            self._cb_domain.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots — every edit persists immediately
    # ------------------------------------------------------------------

    def _on_domain_changed(self, text: str) -> None:
        try:
            v = int(str(text).strip())
        except (TypeError, ValueError):
            v = DEFAULT_DOMAIN_ID
        Ros2ConnectionConfig.set_field("domain_id", v)

    def _on_user_changed(self, text: str) -> None:
        Ros2ConnectionConfig.set_field("ssh_user", str(text or "ubuntu"))

    def _on_save_password(self) -> None:
        text = str(self._le_pw.text() or "")
        if not text:
            log_warning("[mission.conn.dialog] SSH password save: empty")
            return
        get_secure_store().ssh_password(self._server_key).set(text)
        self._has_password = True
        self._le_pw.setText("")
        log_info(
            f"[mission.conn.dialog] SSH password saved for {self._server_key!r}"
        )
        self._refresh_status_label()

    def _on_save_sudo(self) -> None:
        text = str(self._le_sudo.text() or "")
        if not text:
            log_warning("[mission.conn.dialog] SSH sudo password save: empty")
            return
        get_secure_store().ssh_sudo_password(self._server_key).set(text)
        self._has_sudo = True
        self._le_sudo.setText("")
        log_info(
            f"[mission.conn.dialog] SSH sudo password saved for {self._server_key!r}"
        )
        self._refresh_status_label()

    def _refresh_status_label(self) -> None:
        bits: List[str] = []
        if self._has_password:
            bits.append(tr("mission.conn.ssh.has_password", default="pw✓"))
        if self._has_sudo:
            bits.append(tr("mission.conn.ssh.has_sudo", default="sudo✓"))
        if not bits:
            bits.append(tr("mission.conn.ssh.no_creds", default="(no creds)"))
        self._status_label.setText(" · ".join(bits))

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        border = Config.get_color("border_2")
        block_bg = Config.get_color("bg_3")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QDialog#missionConnSettingsDialog {{ background-color: {bg}; }}"
            f"QLabel#missionConnSettingsSectionTitle {{ color: {main}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
            f"QFrame#missionConnSettingsSshBlock {{ background-color: {block_bg}; "
            f"border: 1px solid {border}; border-radius: 4px; }}"
            f"QLabel#missionConnSettingsSshServer {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#missionConnSettingsSshStatus {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#missionConnSettingsKv {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


__all__ = ["ConnectionSettingsDialog"]
