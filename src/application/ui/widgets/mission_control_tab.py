"""Mission Control segmented switch (Training / Script / Control).

Thin wrapper over the SDK ``SliderSwitch`` that fixes the option set used
in MainRow. Theme + i18n + animation come for free from ``SliderSwitch``.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QWidget

from unitport_sdk import SliderSwitch


class MissionControlTab(SliderSwitch):
    """Three-way segmented switch: Training / Script / Control."""

    OPTIONS = [
        ("controlbar.tab.training", "Training"),
        ("controlbar.tab.script", "Script"),
        ("controlbar.tab.control", "Control"),
    ]

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        height: int = 30,
        min_segment_width: int = 72,
    ) -> None:
        super().__init__(
            self.OPTIONS,
            parent=parent,
            height=height,
            min_segment_width=min_segment_width,
        )
        self.setObjectName("missionControlTab")


__all__ = ["MissionControlTab"]
