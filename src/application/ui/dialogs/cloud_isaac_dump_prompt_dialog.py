# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CloudIsaacDumpPromptDialog — surfaced when only cloud IsaacLab is present.

Fired by ``main.py._finalize`` when
:class:`application.tools.isaac_lab_body_dump_task.IsaacLabBodyAutoDumpTask`
returned ``requires_cloud_prompt=True`` — i.e. the user has no local
IsaacLab installation, but has at least one cloud Isaac server configured
under ``EngineService.list_servers("isaac_lab")``.

The dialog tells the user that the standard USD body tables (which feed R5
⊆-coverage / sensor wiring / canvas pre-realization) have not yet been
dumped on this install, and that the local-IsaacLab dump path requires a
local install they don't have. It offers a single action — **"Start cloud
server and dump"** — whose handler is currently a §8 fail-loud no-op
(``log_error`` + return; tells the user the cloud dump RPC is not yet
implemented). The dialog *framework* is in place so wiring the cloud action
later is a handler-only edit.

§5 compliance: every colour goes through ``Config.get_color``; every string
goes through ``tr()``; font sizes via ``Config.get_font_size``.

§8 compliance: the cloud-action handler is **not** a silent skip — it calls
``log_error`` with a clear "not yet wired" directive so the user
understands the action did nothing.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, log_error, log_info, tr


class CloudIsaacDumpPromptDialog(QDialog):
    """Modal dialog asking the user to launch the cloud IsaacLab dump."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            tr(
                "cloud_isaac_dump.window_title",
                "Cloud IsaacLab — USD Body Dump Required",
            )
        )
        self.setModal(True)
        self.setMinimumWidth(520)
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        self._headline = QLabel(self)
        self._headline.setText(
            tr(
                "cloud_isaac_dump.headline",
                "Standard robot USD body tables have not been dumped on "
                "this install.",
            )
        )
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._body = QLabel(self)
        self._body.setText(
            tr(
                "cloud_isaac_dump.body",
                "UnitPort needs USD body topology for every standard robot "
                "(Unitree Go2, Boston Dynamics Spot, Unitree H1, etc.) so "
                "contact sensors, canvas pre-realization, and training "
                "validation can wire reliably. The local-dump pathway "
                "requires a local IsaacLab installation; you have only "
                "cloud IsaacLab servers configured. Start your cloud "
                "server, then click the button below to run the dump "
                "remotely.",
            )
        )
        self._body.setWordWrap(True)
        layout.addWidget(self._body)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)

        self._dismiss_btn = QPushButton(self)
        self._dismiss_btn.setText(
            tr("cloud_isaac_dump.dismiss", "Dismiss")
        )
        self._dismiss_btn.clicked.connect(self.reject)
        button_row.addWidget(self._dismiss_btn)

        self._action_btn = QPushButton(self)
        self._action_btn.setText(
            tr(
                "cloud_isaac_dump.action",
                "Start cloud server and dump",
            )
        )
        self._action_btn.clicked.connect(self._on_action_clicked)
        button_row.addWidget(self._action_btn)

        layout.addLayout(button_row)

    # ------------------------------------------------------------------
    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        fg = Config.get_color("main_t1")
        sub = Config.get_color("sub_t1")
        btn_bg = Config.get_color("btn_1")
        btn_hover = Config.get_color("hover_1")
        accent_bg = Config.get_color("highlight")
        accent_fg = Config.get_color("bg_3")
        border = Config.get_color("border_1")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QDialog {{ background-color: {bg}; color: {fg}; }}"
        )
        self._headline.setStyleSheet(
            f"QLabel {{ color: {fg}; font-size: {font_normal}pt; "
            f"font-weight: 600; }}"
        )
        self._body.setStyleSheet(
            f"QLabel {{ color: {sub}; font-size: {font_small}pt; }}"
        )
        common_btn = (
            f"font-size: {font_small}pt; "
            f"padding: 6px 14px; border-radius: 4px; "
            f"border: 1px solid {border}; "
        )
        self._dismiss_btn.setStyleSheet(
            "QPushButton { " + common_btn
            + f"background-color: {btn_bg}; color: {fg}; "
            f"}} QPushButton:hover {{ background-color: {btn_hover}; }}"
        )
        self._action_btn.setStyleSheet(
            "QPushButton { " + common_btn
            + f"background-color: {accent_bg}; color: {accent_fg}; "
            f"font-weight: 600; "
            f"}} QPushButton:hover {{ background-color: "
            f"{Config.get_color('highlight_checked')}; }}"
        )

    # ------------------------------------------------------------------
    def _on_action_clicked(self) -> None:
        """§8 fail-loud no-op: the cloud-dump RPC is not yet wired.

        We close the dialog and emit a clear log line so the user
        understands the action did nothing — silently flashing the
        dialog and closing it would look identical to "successful
        dispatch". When the cloud-dump pathway is implemented, replace
        the log_error block with the actual RPC call; the dialog UI
        does not need to change.
        """
        log_error(
            "[cloud-isaac-dump] cloud-dump pathway is not yet wired in "
            "this build — your standard-SKU USD body tables remain "
            "absent. Install a local IsaacLab (Settings → Engines) to "
            "trigger the dump on next launch, or wait for the cloud-dump "
            "RPC to be implemented."
        )
        log_info(
            "[cloud-isaac-dump] dialog dismissed without dump (cloud "
            "pathway pending)"
        )
        self.reject()
