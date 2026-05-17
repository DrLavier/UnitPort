"""application.service.projects.paths — canvas / training path helpers.

Per RELEASE/CLAUDE.md §1.4: project-internal artifacts (canvas DAGs, training
runs) live under the project root (``ProjectInfo.path``); user-globally-shared
artifacts (exported policy bundles, login tokens) live under
``Paths.USER_CONFIG_DIR``. Canvas files are project-internal — they are part
of the project's source.

Helpers here keep the path layout consistent with what
``application.service.projects.file_index`` already scans, so a saved canvas
is immediately discovered by the Mission Control panel without manual
re-registration.

Layout under ``<ProjectInfo.path>/``:

    canvas/
        <backend_subdir>/<name>.canvas.json   # IR-shape JSON (page.to_workflow_dict)

Each canvas is stored under a sub-directory named after its bound training
backend (e.g. ``IsaacLab/``, ``SB3/`` — matching
``registers.backends.canvas_subdir(engine_id)``). The subdir is the storage
contract enforced by :class:`CanvasPage.save_to_project`; this module's path
helpers stay format-agnostic so they remain reusable.

A path is *project-canvas-valid* if it lies under ``<project>/canvas/`` and
ends in ``.canvas.json``. ``ensure_project_canvas_path`` raises on anything
else; ``is_project_canvas_path`` is the boolean variant for soft assertions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .store import ProjectInfo

_CANVAS_DIR_NAME = "canvas"
_CANVAS_FILE_SUFFIX = ".canvas.json"
_DEFAULT_CANVAS_NAME = "main"


def project_canvas_dir(project: ProjectInfo) -> Path:
    """Return ``<project>/canvas/`` (does not create the directory)."""
    return project.path / _CANVAS_DIR_NAME


def current_project_canvas_path(
    project: ProjectInfo,
    name: str = _DEFAULT_CANVAS_NAME,
) -> Path:
    """Return ``<project>/canvas/<name>.canvas.json``.

    *name* may include subdirectories (e.g. ``"IsaacLab/amp_walk"``) but must
    not start with ``/`` and must not contain ``..``. By the storage contract,
    the first segment must equal the bound backend's
    ``registers.backends.canvas_subdir(engine_id)`` — that policy is enforced
    in :class:`CanvasPage.save_to_project`, not here, so this helper stays
    composable for tests and tools that legitimately need raw paths.
    The ``.canvas.json`` suffix is appended unconditionally — passing a name
    that already has the suffix collapses to a single one.
    """
    if not name:
        name = _DEFAULT_CANVAS_NAME
    name = str(name).strip().lstrip("/").lstrip("\\")
    if ".." in Path(name).parts:
        raise ValueError(f"canvas name {name!r} 不允许包含 '..'")
    if name.lower().endswith(_CANVAS_FILE_SUFFIX):
        name = name[: -len(_CANVAS_FILE_SUFFIX)]
    return project.path / _CANVAS_DIR_NAME / f"{name}{_CANVAS_FILE_SUFFIX}"


def is_project_canvas_path(project: ProjectInfo, path: Path) -> bool:
    """Return True if *path* is under ``<project>/canvas/`` and is a canvas file.

    Soft check (returns False rather than raising). Use this for dev-time
    assertions where you want to log a warning instead of aborting.
    """
    try:
        resolved = Path(path).resolve()
        canvas_root = project_canvas_dir(project).resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(canvas_root)
    except ValueError:
        return False
    return resolved.name.lower().endswith(_CANVAS_FILE_SUFFIX)


def ensure_project_canvas_path(project: ProjectInfo, path: Path) -> Path:
    """Raise ``ValueError`` if *path* is not inside the project's canvas dir.

    Hard check — used at write boundaries where saving outside the project
    tree would break ``file_index.list_canvas_files`` discovery.

    Returns the resolved absolute path.
    """
    resolved = Path(path).resolve()
    canvas_root = project_canvas_dir(project).resolve()
    try:
        resolved.relative_to(canvas_root)
    except ValueError:
        raise ValueError(
            f"canvas 存盘路径 {path} 必须在 {canvas_root} 下；"
            f"当前项目 = {project.project_id}"
        )
    if not resolved.name.lower().endswith(_CANVAS_FILE_SUFFIX):
        raise ValueError(
            f"canvas 存盘文件名 {resolved.name} 必须以 {_CANVAS_FILE_SUFFIX} 结尾"
        )
    return resolved


def find_default_canvas(project: ProjectInfo) -> Optional[Path]:
    """Return ``<project>/canvas/main.canvas.json`` if it exists, else None.

    Used by the app shell to decide whether a project has a "primary" canvas
    to auto-open on project switch.
    """
    p = current_project_canvas_path(project, _DEFAULT_CANVAS_NAME)
    return p if p.exists() else None


__all__ = [
    "project_canvas_dir",
    "current_project_canvas_path",
    "is_project_canvas_path",
    "ensure_project_canvas_path",
    "find_default_canvas",
]
