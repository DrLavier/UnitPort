#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MissionOverviewContent — Mission page content for the OverviewPanel (stub v1)."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MissionOverviewContent(QWidget):
    """Placeholder content shown when the Mission page is active."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionOverviewContent")

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Mission overview \u2014 coming soon")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "color: #555555; font-size: 13px; font-style: italic;"
        )
        vl.addWidget(label)

        self.setStyleSheet("#missionOverviewContent { background: #1a1a1a; }")
