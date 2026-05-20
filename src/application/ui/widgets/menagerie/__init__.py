"""Shared Menagerie card-grid widgets.

Co-host for two callers:
- :class:`MenagerieSelectPage` inside :class:`InstallConfigWizard` (first-launch).
- :class:`MenagerieBrowserDialog` inside the sidebar Robot Asset panel.

Both UIs reuse :class:`MenagerieCard` and :class:`MenagerieCardGrid` from this
package; styling lives in ``system.ini[Theme]`` slots prefixed ``menagerie_*``.
"""

from .card_grid import (
    CARD_W,
    CARD_H,
    ICON_W,
    ICON_H,
    GRID_COLUMNS,
    GRID_HSPACE,
    GRID_VSPACE,
    GRID_MARGIN,
    MenagerieCard,
    MenagerieCardGrid,
)

__all__ = [
    "CARD_W",
    "CARD_H",
    "ICON_W",
    "ICON_H",
    "GRID_COLUMNS",
    "GRID_HSPACE",
    "GRID_VSPACE",
    "GRID_MARGIN",
    "MenagerieCard",
    "MenagerieCardGrid",
]
