"""Project store -- directory-level scanner + per-project file listings."""

from __future__ import annotations

from .file_index import (
    list_canvas_files,
    list_canvas_groups,
    list_script_groups,
    resolve_file,
)
from .paths import (
    current_project_canvas_path,
    ensure_project_canvas_path,
    find_default_canvas,
    is_project_canvas_path,
    project_canvas_dir,
)
from .store import (
    ProjectInfo,
    ProjectStore,
    current_project_info,
    get_project_store,
    set_current_project_info,
)

__all__ = [
    "ProjectInfo",
    "ProjectStore",
    "get_project_store",
    "current_project_info",
    "set_current_project_info",
    "list_canvas_files",
    "list_canvas_groups",
    "list_script_groups",
    "resolve_file",
    # paths
    "current_project_canvas_path",
    "ensure_project_canvas_path",
    "find_default_canvas",
    "is_project_canvas_path",
    "project_canvas_dir",
]
