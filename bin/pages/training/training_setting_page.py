"""TrainingSettingPage: buffer landing page shown when the Training area
is entered with no active experiment.

Offers two backend entry buttons — SB3 and Isaac — each of which signals
``new_training_requested(backend)`` so the host can prompt the user for a
filename and then enter the corresponding training canvas.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.system.core.theme_manager import get_color


class TrainingSettingPage(QWidget):
    """Landing buffer for the Training area.

    Layout: centered column with a title + subtitle + three large backend
    buttons. Emits ``new_training_requested(backend)`` where backend is
    ``"sb3"``, ``"isaac"``, or ``"cloud"``.
    """

    new_training_requested = Signal(str)

    _OBJECT_NAME = "trainingSettingPage"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(self._OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.setSpacing(0)
        outer.addStretch(1)

        # Centred card
        card = QWidget(self)
        card.setObjectName("trainingSettingCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(48, 36, 48, 36)
        card_layout.setSpacing(18)

        self._title = QLabel("Training Ground", card)
        self._title.setObjectName("trainingSettingTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._subtitle = QLabel(
            "Select a backend to start a new training session.",
            card,
        )
        self._subtitle.setObjectName("trainingSettingSubtitle")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setWordWrap(True)

        card_layout.addWidget(self._title)
        card_layout.addWidget(self._subtitle)
        card_layout.addSpacing(12)

        # Buttons row
        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(0, 0, 0, 0)
        buttons_row.setSpacing(16)

        self._sb3_btn = self._make_backend_button("New SB3 Training")
        self._sb3_btn.clicked.connect(
            lambda: self.new_training_requested.emit("sb3")
        )
        buttons_row.addWidget(self._sb3_btn, 1)

        self._isaac_btn = self._make_backend_button("New Isaac Training")
        self._isaac_btn.clicked.connect(
            lambda: self.new_training_requested.emit("isaac")
        )
        buttons_row.addWidget(self._isaac_btn, 1)

        self._cloud_btn = self._make_backend_button("New Cloud Training")
        self._cloud_btn.clicked.connect(
            lambda: self.new_training_requested.emit("cloud")
        )
        buttons_row.addWidget(self._cloud_btn, 1)

        card_layout.addLayout(buttons_row)

        # "Configure Servers..." link
        self._configure_link = QPushButton("Configure Servers...", card)
        self._configure_link.setObjectName("trainingSettingConfigureLink")
        self._configure_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self._configure_link.setFlat(True)
        self._configure_link.clicked.connect(self._on_configure_servers)
        card_layout.addWidget(self._configure_link, alignment=Qt.AlignmentFlag.AlignCenter)

        # Horizontal centering wrapper for the card
        centerer = QHBoxLayout()
        centerer.setContentsMargins(0, 0, 0, 0)
        centerer.addStretch(1)
        centerer.addWidget(card)
        centerer.addStretch(1)
        outer.addLayout(centerer)
        outer.addStretch(2)

        self._card = card
        self.apply_theme()

    def _on_configure_servers(self) -> None:
        """Open the remote server configuration dialog."""
        from bin.pages.training.remote_server_dialog import RemoteServerDialog
        dlg = RemoteServerDialog(self)
        dlg.exec()

    def _make_backend_button(self, label: str) -> QPushButton:
        btn = QPushButton(label, self)
        btn.setObjectName("trainingSettingBackendBtn")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(72)
        btn.setMinimumWidth(220)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return btn

    def apply_theme(self) -> None:
        bg = get_color("main_row_training_bg", get_color("main_row_bg", "#1f2937"))
        card_bg = get_color("button_bg", "#2a2a2a")
        border = get_color("border", "#475569")
        text = get_color("text_primary", "#e2e8f0")
        sub = get_color("text_secondary", "#9ca3af")
        hover = get_color("hover_bg", "#3d3d3d")

        self.setStyleSheet(f"""
            QWidget#{self._OBJECT_NAME} {{
                background-color: {bg};
            }}
            QWidget#trainingSettingCard {{
                background-color: {card_bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel#trainingSettingTitle {{
                color: {text};
                font-size: 22px;
                font-weight: 600;
                background: transparent;
                border: none;
            }}
            QLabel#trainingSettingSubtitle {{
                color: {sub};
                font-size: 13px;
                background: transparent;
                border: none;
            }}
            QPushButton#trainingSettingBackendBtn {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
            }}
            QPushButton#trainingSettingBackendBtn:hover {{
                background-color: {hover};
            }}
            QPushButton#trainingSettingBackendBtn:pressed {{
                background-color: {card_bg};
            }}
            QPushButton#trainingSettingConfigureLink {{
                color: {sub};
                background: transparent;
                border: none;
                font-size: 12px;
                text-decoration: underline;
                padding: 4px 0px;
            }}
            QPushButton#trainingSettingConfigureLink:hover {{
                color: {text};
            }}
        """)
