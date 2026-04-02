#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainingOverviewContent — Training page content for the OverviewPanel.

Layout::

    ┌───────────────────────────────────────────────────────────────────┐
    │  Setup Zone (scrollable card row)              │  Result Zone     │
    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │  ┌────────────┐  │
    │  │Robot │ │Action│ │ Env  │ │Ctrl  │          │  │  Idle /    │  │
    │  │      │ │      │ │      │ │      │          │  │  Progress/ │  │
    │  │      │ │      │ │      │ │      │          │  │  SkillCard │  │
    │  └──────┘ └──────┘ └──────┘ └──────┘          │  └────────────┘  │
    └───────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Card base
# ---------------------------------------------------------------------------

_CARD_W = 180


class _OverviewCard(QFrame):
    """Base card widget for the Setup Zone."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("overviewCard")
        self.setFixedWidth(_CARD_W)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(10, 8, 10, 8)
        vl.setSpacing(6)

        # Card title
        title_lbl = QLabel(title)
        title_lbl.setObjectName("cardTitle")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: #e0e0e0;")
        vl.addWidget(title_lbl, 0)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3d3d3d;")
        vl.addWidget(sep, 0)

        # Body area — subclasses add content here
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(4)
        vl.addWidget(self._body, 1)

    def _add_row(self, label: str, value: str) -> QLabel:
        """Add a label: value row and return the value QLabel for later updates."""
        row = QWidget(self._body)
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #888888; font-size: 10px;")
        hl.addWidget(lbl, 0)

        hl.addStretch(1)

        val = QLabel(value)
        val.setStyleSheet("color: #cccccc; font-size: 10px; font-weight: bold;")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hl.addWidget(val, 0)

        self._body_layout.addWidget(row, 0)
        return val

    def _add_badge(self, text: str, color: str = "#3b82f6") -> QLabel:
        """Add a colored badge label."""
        badge = QLabel(text)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(20)
        badge.setStyleSheet(
            f"background: {color}; color: #ffffff; font-size: 10px; "
            f"font-weight: bold; border-radius: 3px; padding: 1px 8px;"
        )
        self._body_layout.addWidget(badge, 0)
        return badge


# ---------------------------------------------------------------------------
# Concrete cards (placeholder data for Phase 1)
# ---------------------------------------------------------------------------

class _RobotCard(_OverviewCard):
    def __init__(self, parent=None):
        super().__init__("Robot", parent)
        self._robot_type = self._add_row("Type", "go2")
        self._joints = self._add_row("Joints", "12")
        self._family = self._add_badge("Quadruped", "#2563eb")
        self._body_layout.addStretch(1)


class _ActionCard(_OverviewCard):
    def __init__(self, parent=None):
        super().__init__("Action", parent)
        self._action_type = self._add_badge("JP  joint_position", "#3b82f6")
        self._action_dim = self._add_row("Dimension", "12")
        self._frequency = self._add_row("Frequency", "50 Hz")
        self._motion = self._add_row("Ref Motion", "None")
        self._body_layout.addStretch(1)


class _EnvironmentCard(_OverviewCard):
    def __init__(self, parent=None):
        super().__init__("Environment", parent)
        self._scene = self._add_row("Scene", "flat")
        self._gravity = self._add_row("Gravity", "9.81 m/s\u00b2")
        self._sim_dt = self._add_row("Sim dt", "0.002s")
        self._domain_rand = self._add_badge("Domain Rand  ON", "#16a34a")
        self._body_layout.addStretch(1)


class _ControlCard(_OverviewCard):
    def __init__(self, parent=None):
        super().__init__("Control", parent)
        self._algorithm = self._add_badge("PPO", "#8b5cf6")
        self._total_steps = self._add_row("Total Steps", "1M")
        self._obs_dim = self._add_row("Obs Dim", "45")
        self._sensors = self._add_row("Sensors", "IMU, JointEnc")
        self._body_layout.addStretch(1)


# ---------------------------------------------------------------------------
# Result Zone
# ---------------------------------------------------------------------------

class _ResultZone(QFrame):
    """Training result display — idle / progress / skill card."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultZone")
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(10, 8, 10, 8)
        vl.setSpacing(6)

        title = QLabel("Result")
        title.setObjectName("cardTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #e0e0e0;")
        vl.addWidget(title, 0)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3d3d3d;")
        vl.addWidget(sep, 0)

        # Idle state placeholder
        self._idle_label = QLabel("Run training to see results")
        self._idle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._idle_label.setStyleSheet("color: #555555; font-size: 11px; font-style: italic;")
        vl.addWidget(self._idle_label, 1)


# ---------------------------------------------------------------------------
# TrainingOverviewContent (main container)
# ---------------------------------------------------------------------------

class TrainingOverviewContent(QWidget):
    """Training page content for OverviewPanel.

    Horizontally: Setup Zone (scrollable cards) | Result Zone.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trainingOverviewContent")

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        # ── Setup Zone (left, scrollable) ─────────────────────────────
        self._setup_scroll = QScrollArea()
        self._setup_scroll.setObjectName("setupScrollArea")
        self._setup_scroll.setWidgetResizable(True)
        self._setup_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._setup_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._setup_scroll.setFrameShape(QFrame.Shape.NoFrame)

        cards_container = QWidget()
        cards_layout = QHBoxLayout(cards_container)
        cards_layout.setContentsMargins(8, 4, 8, 4)
        cards_layout.setSpacing(8)

        self.robot_card = _RobotCard()
        self.action_card = _ActionCard()
        self.env_card = _EnvironmentCard()
        self.control_card = _ControlCard()

        cards_layout.addWidget(self.robot_card)
        cards_layout.addWidget(self.action_card)
        cards_layout.addWidget(self.env_card)
        cards_layout.addWidget(self.control_card)
        cards_layout.addStretch(1)

        self._setup_scroll.setWidget(cards_container)
        hl.addWidget(self._setup_scroll, 1)

        # ── Vertical separator ────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #3d3d3d;")
        hl.addWidget(sep, 0)

        # ── Result Zone (right) ───────────────────────────────────────
        self.result_zone = _ResultZone()
        hl.addWidget(self.result_zone, 0)

        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            #trainingOverviewContent {
                background: #1a1a1a;
            }
            #overviewCard {
                background: #222222;
                border: 1px solid #333333;
                border-radius: 6px;
            }
            #resultZone {
                background: #222222;
                border: 1px solid #333333;
                border-radius: 6px;
                margin: 4px 8px 4px 0px;
            }
            #setupScrollArea {
                background: transparent;
            }
        """)
