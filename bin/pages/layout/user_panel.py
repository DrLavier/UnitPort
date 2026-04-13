"""User sidebar panel — profile card + engine configuration overview."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.system.core.theme_manager import get_color, get_icon
from src.system.engines.registry import get_engine_registry


# ── Engine descriptors ────────────────────────────────────────────────

_ENGINES = [
    {
        "id": "sb3",
        "name": "SB3",
        "description": "Stable-Baselines3",
        "has_local_root": False,   # pip package, no root dir to configure
        "has_cloud": False,        # no cloud support
    },
    {
        "id": "isaac_lab",
        "name": "Isaac Lab",
        "description": "NVIDIA Isaac Lab",
        "has_local_root": True,    # needs an installation root directory
        "has_cloud": True,         # supports cloud training via SSH
    },
]


def _query_local(engine_id: str) -> Tuple[bool, str]:
    """Read local status from the engine registry (single source of truth).

    Returns (enabled/registered, path_or_message).
    """
    reg = get_engine_registry()
    local = reg.get_local(engine_id)
    if not local:
        return False, "Not registered"

    # SB3 has no ``root`` — enabled flag alone is sufficient.
    if engine_id == "sb3":
        enabled = bool(local.get("enabled", False))
        return enabled, "Built-in (pip)" if enabled else "Not enabled"

    # Engines with a local root path (Isaac Lab, Newton, …).
    registered = bool(local.get("registered", False))
    root = str(local.get("root", ""))
    if registered and root:
        return True, root
    return False, "Not registered"


def _query_cloud(engine_id: str) -> Tuple[bool, str]:
    """Read cloud status from the engine registry."""
    reg = get_engine_registry()
    servers = reg.list_servers(engine_id)
    if servers:
        default = reg.get_default_server_name(engine_id) or servers[0].get("name", "")
        return True, f"{len(servers)} server(s) — {default}"
    return False, "No servers"


# ── Widgets ───────────────────────────────────────────────────────────

class _EngineItemWidget(QWidget):
    """Single engine row in the User panel.

    ::

        [Engine Name]
        Local: [status] [path]              [settings]
        [check] Cloud: [status]             [settings]
    """

    settings_requested = Signal(str, str)  # engine_id, context ("local" / "cloud")

    _ICON_SIZE = 18

    def __init__(self, engine: dict, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._engine_id: str = engine["id"]
        self._has_local_root: bool = engine.get("has_local_root", False)
        self._has_cloud: bool = engine.get("has_cloud", False)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        # ── Engine name ───────────────────────────────────────────
        self._name_label = QLabel(engine["name"], self)
        self._name_label.setObjectName("engineItemName")
        root.addWidget(self._name_label)

        _CHECK_W = 16

        # ── Local row ─────────────────────────────────────────────
        local_row = QHBoxLayout()
        local_row.setContentsMargins(0, 0, 0, 0)
        local_row.setSpacing(6)

        self._local_check = QLabel("", self)
        self._local_check.setObjectName("engineItemCheck")
        self._local_check.setFixedWidth(_CHECK_W)
        local_row.addWidget(self._local_check)

        self._local_prefix = QLabel("Local:", self)
        self._local_prefix.setObjectName("engineItemLabel")
        local_row.addWidget(self._local_prefix)

        self._local_status = QLabel("", self)
        self._local_status.setObjectName("engineItemStatus")
        local_row.addWidget(self._local_status, 1)

        self._local_settings_btn = QPushButton(self)
        self._local_settings_btn.setObjectName("engineSettingsBtn")
        self._local_settings_btn.setFixedSize(24, 24)
        self._local_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._local_settings_btn.setToolTip("Configure local engine path")
        self._local_settings_btn.clicked.connect(self._on_local_settings)
        # Engines without a configurable root (SB3) disable the button
        self._local_settings_btn.setEnabled(self._has_local_root)
        local_row.addWidget(self._local_settings_btn)
        root.addLayout(local_row)

        # ── Cloud row ─────────────────────────────────────────────
        cloud_row = QHBoxLayout()
        cloud_row.setContentsMargins(0, 0, 0, 0)
        cloud_row.setSpacing(6)

        self._cloud_check = QLabel("", self)
        self._cloud_check.setObjectName("engineItemCheck")
        self._cloud_check.setFixedWidth(_CHECK_W)
        cloud_row.addWidget(self._cloud_check)

        self._cloud_prefix = QLabel("Cloud:", self)
        self._cloud_prefix.setObjectName("engineItemLabel")
        cloud_row.addWidget(self._cloud_prefix)

        self._cloud_status = QLabel("", self)
        self._cloud_status.setObjectName("engineItemStatus")
        cloud_row.addWidget(self._cloud_status, 1)

        self._cloud_settings_btn = QPushButton(self)
        self._cloud_settings_btn.setObjectName("engineSettingsBtn")
        self._cloud_settings_btn.setFixedSize(24, 24)
        self._cloud_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if self._has_cloud:
            self._cloud_settings_btn.setToolTip("Manage cloud servers")
            self._cloud_settings_btn.clicked.connect(self._on_cloud_settings)
        else:
            self._cloud_settings_btn.setToolTip("Cloud not available for this engine")
            self._cloud_settings_btn.setEnabled(False)
        cloud_row.addWidget(self._cloud_settings_btn)
        root.addLayout(cloud_row)

    # ── Public API ────────────────────────────────────────────────

    def refresh_status(self) -> None:
        """Read local + cloud status from the engine registry."""
        local_ok, local_detail = _query_local(self._engine_id)
        self._local_check.setText("\u2714" if local_ok else "")
        self._local_status.setText("Registered" if local_ok else "Missing")
        self._local_status.setToolTip(local_detail)

        if self._has_cloud:
            cloud_ok, cloud_label = _query_cloud(self._engine_id)
            self._cloud_check.setText("\u2714" if cloud_ok else "")
            self._cloud_status.setText(cloud_label)
        else:
            self._cloud_check.setText("")
            self._cloud_status.setText("Unavailable")
            self._cloud_status.setToolTip("Cloud training not supported for this engine")

    # ── Settings handlers ─────────────────────────────────────────

    def _on_local_settings(self) -> None:
        """Open a directory picker to register / re-register the local engine root."""
        reg = get_engine_registry()
        current = reg.get_local(self._engine_id)
        start_dir = current.get("root", "") or str(Path.home())

        chosen = QFileDialog.getExistingDirectory(
            self, f"Select {self._engine['name']} Root Directory", start_dir,
        )
        if not chosen:
            return

        # Validate: Isaac Lab needs isaaclab.sh/bat or source/ dir
        p = Path(chosen)
        if self._engine_id == "isaac_lab":
            markers = ("isaaclab.sh", "isaaclab.bat", "source", "isaaclab")
            if not any((p / m).exists() for m in markers):
                QMessageBox.warning(
                    self,
                    "Invalid Path",
                    f"Could not verify Isaac Lab at:\n{chosen}\n\n"
                    "Expected isaaclab.sh, isaaclab.bat, or source/ directory.",
                )
                return
            reg.register_isaac_local(chosen)
        else:
            reg.set_local(self._engine_id, {"root": chosen, "registered": True})

        self.refresh_status()

    def _on_cloud_settings(self) -> None:
        """Open the remote server management dialog."""
        self.settings_requested.emit(self._engine_id, "cloud")

    def refresh_icons(self) -> None:
        icon = get_icon("setting")
        icon_size = QSize(self._ICON_SIZE, self._ICON_SIZE)
        for btn in (self._local_settings_btn, self._cloud_settings_btn):
            if not icon.isNull():
                btn.setIcon(icon)
                btn.setIconSize(icon_size)
                btn.setText("")
            else:
                btn.setText("\u2699")

    def apply_theme(self) -> None:
        name_color = get_color("text_primary", "#e2e8f0")
        label_color = get_color("text_secondary", "#9ca3af")
        check_color = get_color("accent", "#3b82f6")
        hover = get_color("hover_bg", "#3d3d3d")

        self._name_label.setStyleSheet(
            f"QLabel#engineItemName {{ color: {name_color}; font-size: 13px; font-weight: bold; background: transparent; border: none; }}"
        )
        for lbl in (self._local_prefix, self._cloud_prefix):
            lbl.setStyleSheet(
                f"QLabel#engineItemLabel {{ color: {label_color}; font-size: 12px; background: transparent; border: none; }}"
            )
        for lbl in (self._local_status, self._cloud_status):
            lbl.setStyleSheet(
                f"QLabel#engineItemStatus {{ color: {label_color}; font-size: 12px; background: transparent; border: none; }}"
            )
        for lbl in (self._local_check, self._cloud_check):
            lbl.setStyleSheet(
                f"QLabel#engineItemCheck {{ color: {check_color}; font-size: 13px; background: transparent; border: none; }}"
            )
        for btn in (self._local_settings_btn, self._cloud_settings_btn):
            btn.setStyleSheet(f"""
                QPushButton#engineSettingsBtn {{
                    background: transparent;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton#engineSettingsBtn:hover {{
                    background-color: {hover};
                }}
            """)

        self.refresh_icons()
        self.refresh_status()


class UserPanel(QWidget):
    """Sidebar panel showing user info card and engine configuration."""

    engine_settings_requested = Signal(str, str)  # engine_id, context

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("userPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── User info card ────────────────────────────────────────
        card = QWidget(self)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 10)
        card_layout.setSpacing(10)

        self._avatar = QLabel(card)
        self._avatar.setObjectName("userPanelAvatar")
        self._avatar.setFixedSize(40, 40)
        self._avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self._avatar, 0, Qt.AlignmentFlag.AlignVCenter)

        self._username = QLabel("User Name", card)
        self._username.setObjectName("userPanelUsername")
        card_layout.addWidget(self._username, 1, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(card)

        # ── "Engines" title ───────────────────────────────────────
        title_host = QWidget(self)
        title_layout = QHBoxLayout(title_host)
        title_layout.setContentsMargins(12, 6, 12, 4)
        title_layout.setSpacing(0)
        self._engines_title = QLabel("Engines", title_host)
        self._engines_title.setObjectName("userPanelSectionTitle")
        title_layout.addWidget(self._engines_title)
        root.addWidget(title_host)

        # 1px divider
        self._divider = QFrame(self)
        self._divider.setFrameShape(QFrame.Shape.HLine)
        self._divider.setFixedHeight(1)
        root.addWidget(self._divider)

        # ── Scrollable engine list ────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setObjectName("userPanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        list_host = QWidget()
        self._list_layout = QVBoxLayout(list_host)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)

        self._engine_items: list[_EngineItemWidget] = []
        for eng in _ENGINES:
            item = _EngineItemWidget(eng, list_host)
            item.settings_requested.connect(self.engine_settings_requested.emit)
            self._list_layout.addWidget(item)
            self._engine_items.append(item)

        self._list_layout.addStretch(1)
        scroll.setWidget(list_host)
        root.addWidget(scroll, 1)

        self.apply_theme()

    # ── Public API ────────────────────────────────────────────────

    def set_username(self, name: str) -> None:
        self._username.setText(name or "User Name")

    def refresh_engines(self) -> None:
        for item in self._engine_items:
            item.refresh_status()

    def apply_theme(self) -> None:
        text = get_color("text_primary", "#e2e8f0")
        secondary = get_color("text_secondary", "#9ca3af")
        border = get_color("border", "#475569")
        card_bg = get_color("card_bg", "#2E2E2E")

        # Avatar: rounded square with border, icon placeholder
        self._avatar.setStyleSheet(
            f"QLabel#userPanelAvatar {{"
            f" background: transparent;"
            f" border: 1px solid {border};"
            f" border-radius: 5px;"
            f"}}"
        )
        icon = get_icon("acc")
        if not icon.isNull():
            self._avatar.setPixmap(icon.pixmap(QSize(28, 28)))
        else:
            self._avatar.setText("U")

        self._username.setStyleSheet(
            f"QLabel#userPanelUsername {{ color: {text}; font-size: 14px; font-weight: bold; background: transparent; border: none; }}"
        )
        self._engines_title.setStyleSheet(
            f"QLabel#userPanelSectionTitle {{ color: {secondary}; font-size: 12px; font-weight: 600; background: transparent; border: none; }}"
        )
        self._divider.setStyleSheet(f"background-color: {border}; border: none;")

        for item in self._engine_items:
            item.apply_theme()
