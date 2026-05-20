"""Sidebar panel factory.

Each top-level rail entry maps to a builder. Lazy imports keep app startup
cheap — service singletons are constructed on first panel open, not at
module load time.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QWidget


def build_panel(key: str, parent: Optional[QWidget] = None) -> Optional[QWidget]:
    """Return the populated content widget for a sidebar key (or None if unknown)."""
    if key == "user":
        from .user_panel import UserPanel
        return UserPanel(parent=parent)
    if key == "projects":
        from .projects_panel import ProjectsPanel
        return ProjectsPanel(parent=parent)
    if key == "robot_assets":
        from .robot_assets_panel import RobotAssetsPanel
        return RobotAssetsPanel(parent=parent)
    if key == "controller":
        from .controller_panel import ControllerPanel
        return ControllerPanel(parent=parent)
    if key == "resources":
        from .resources_panel import ResourcesPanel
        return ResourcesPanel(parent=parent)
    if key == "node_library":
        from .node_library_panel_wrapper import SidebarNodeLibraryPanel
        return SidebarNodeLibraryPanel(parent=parent)
    if key == "training":
        from .scripts_panels import TrainingScriptsPanel
        return TrainingScriptsPanel(parent=parent)
    if key in ("rewards", "terminations", "observations"):
        from .scripts_panels import RegistryPresetPanel
        kind = {
            "rewards": "reward",
            "terminations": "termination",
            "observations": "observation",
        }[key]
        return RegistryPresetPanel(kind=kind, parent=parent)
    return None


__all__ = ["build_panel"]
