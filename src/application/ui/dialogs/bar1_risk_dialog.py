# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Bar1RiskDialog — pre-launch GPU BAR1 aperture risk confirmation.

Headed Isaac Lab training renders the viewport through the GPU's PCIe BAR1
aperture; exhausting it aborts deep inside Kit with an opaque device-lost /
access-violation crash that leaves the user with no idea why. Before a headed
run is submitted, ``MainWindow._on_start_training`` probes BAR1
(:func:`application.training.isaac_lab.bar1_preflight.assess_bar1_risk`) and,
when the verdict is ``warn`` / ``block``, shows this modal so the user makes an
informed Continue / Abort choice instead of hitting a cryptic crash.

The verdict message (GPU name, num_envs, BAR1 free, fix hints) is built in
``bar1_preflight`` and shown verbatim as the technical body; the title, intro
line and buttons are localized.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, setButton, tr


class Bar1RiskDialog(QDialog):
    """Modal Continue / Abort confirmation for a BAR1-risky headed run."""

    def __init__(self, verdict, *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._verdict = verdict
        self.setObjectName("bar1RiskDialog")
        self.setModal(True)
        self.setMinimumWidth(520)

        is_block = getattr(verdict, "level", "warn") == "block"
        self.setWindowTitle(
            tr("bar1.title_block", "Training likely to crash (GPU BAR1)")
            if is_block
            else tr("bar1.title_warn", "GPU memory warning (BAR1)")
        )

        self._build_ui(verdict, is_block)
        self._apply_theme(is_block)

    # ------------------------------------------------------------------
    def _build_ui(self, verdict, is_block: bool) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(10)

        headline = QLabel(
            tr(
                "bar1.headline",
                "Headed training renders through the GPU BAR1 aperture, which "
                "looks too small / saturated for this run. It may crash with a "
                "GPU device-lost error.",
            ),
            self,
        )
        headline.setObjectName("bar1Headline")
        headline.setWordWrap(True)
        outer.addWidget(headline, 0)

        body = QLabel(str(getattr(verdict, "message", "") or ""), self)
        body.setObjectName("bar1Body")
        body.setWordWrap(True)
        outer.addWidget(body, 1)

        outer.addSpacing(2)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)

        # Left button: Continue (proceed despite the risk).
        self._btn_continue = setButton(
            "bar1.btn_continue", 120, 32,
            kind="normal", spec="save",
            default="Continue anyway", parent=self,
        )
        self._btn_continue.clicked.connect(self.accept)
        footer.addWidget(self._btn_continue, 0)

        # Right button: Abort (do not launch). Default / safe choice.
        self._btn_abort = setButton(
            "bar1.btn_abort", 120, 32,
            kind="normal", spec="danger",
            default="Abort", parent=self,
        )
        self._btn_abort.clicked.connect(self.reject)
        footer.addWidget(self._btn_abort, 0)

        outer.addLayout(footer)

    # ------------------------------------------------------------------
    def _apply_theme(self, is_block: bool) -> None:
        bg = Config.get_color("bg_2")
        body_color = Config.get_color("main_c2")
        # Block = near-certain crash → danger tone; warn → softer notice tone.
        headline_color = (
            Config.get_color("danger_zone")
            if is_block
            else Config.get_color("notice_1", fallback=Config.get_color("main_t1"))
        )
        sz_head = int(Config.get_font_size("size_normal"))
        sz_body = int(Config.get_font_size("size_small"))
        self.setStyleSheet(
            f"QDialog#bar1RiskDialog {{ background-color: {bg}; }}"
            f"QLabel#bar1Headline {{ color: {headline_color}; "
            f"font-size: {sz_head}px; font-weight: 700; background: transparent; }}"
            f"QLabel#bar1Body {{ color: {body_color}; "
            f"font-size: {sz_body}px; background: transparent; }}"
        )
