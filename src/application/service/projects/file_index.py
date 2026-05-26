# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Per-project file listing for MissionControlPanel.

Pure functions over :class:`ProjectInfo` -- no Qt, no global state.

* Canvas ids stay project-root-relative POSIX paths so they round-trip
  through :func:`resolve_file` back to absolute paths under the project.
* Script ids are bucket-tagged: ``project:<rel>`` (under the current
  project's ``scripts/`` dir) or ``system:<rel>`` (under the SDK's
  ``src/scripts/`` dir). The prefix tells :func:`resolve_file` which
  root to anchor against.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_warning

from .store import ProjectInfo


_EXCLUDED_DIRS = {"cache"}
_ROOT_GROUP_ID_SUFFIX = "__root__"
_ROOT_GROUP_LABEL = "(root)"


def list_canvas_files(info: ProjectInfo) -> List[Tuple[str, str]]:
    """Return ``[(rel_id, label), ...]`` for the project's canvas files.

    Manifest-declared ``assets.workflows`` come first (stable ids); loose
    ``*.canvas.json`` files under ``canvas/`` are appended for visibility
    so editor-saved workflows that haven't yet been registered in the
    manifest still appear in the panel.
    """
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()

    workflows = (info.manifest.get("assets") or {}).get("workflows") or []
    for entry in workflows:
        if not isinstance(entry, dict):
            continue
        rel = str(entry.get("file") or "").replace("\\", "/").strip()
        wid = str(entry.get("id") or "").strip()
        if not rel or not wid:
            continue
        out.append((rel, wid))
        seen.add(rel.lower())

    canvas_root = info.path / "canvas"
    if canvas_root.is_dir():
        for p in sorted(canvas_root.rglob("*.canvas.json")):
            rel = p.relative_to(info.path).as_posix()
            if rel.lower() in seen:
                continue
            label = p.relative_to(canvas_root).as_posix()
            if label.endswith(".canvas.json"):
                label = label[: -len(".canvas.json")]
            out.append((rel, label))
            seen.add(rel.lower())

    return out


# ---------------------------------------------------------------------------
# Canvas tab — two-level groups spec (backend → sub-dir/(root))
# ---------------------------------------------------------------------------
def _canvas_items_from_dir(
    info: ProjectInfo,
    canvas_root: Path,
    sub_dir: Path,
    backend_display: str,
    backend_theme_slot: str,
) -> List[Dict]:
    """``[{"id":<rel>, "default":<basename>, "prefix":"[<bd>]", "prefix_color":<slot>}, ...]``
    for ``*.canvas.json`` under ``sub_dir`` (recursive)."""
    out: List[Dict] = []
    for p in sorted(sub_dir.rglob("*.canvas.json")):
        try:
            parts = p.relative_to(canvas_root).parts
        except ValueError:
            continue
        if any(seg in _EXCLUDED_DIRS for seg in parts[:-1]):
            continue
        rel = p.relative_to(info.path).as_posix()
        basename = p.name
        if basename.endswith(".canvas.json"):
            basename = basename[: -len(".canvas.json")]
        out.append({
            "id": rel,
            "default": basename,
            "prefix": f"[{backend_display}]",
            "prefix_color": backend_theme_slot,
        })
    return out


def _canvas_items_from_root(
    info: ProjectInfo,
    backend_dir: Path,
    backend_display: str,
    backend_theme_slot: str,
) -> List[Dict]:
    """Loose ``*.canvas.json`` directly under ``backend_dir`` (non-recursive)."""
    out: List[Dict] = []
    for p in sorted(backend_dir.glob("*.canvas.json")):
        if not p.is_file():
            continue
        rel = p.relative_to(info.path).as_posix()
        basename = p.name
        if basename.endswith(".canvas.json"):
            basename = basename[: -len(".canvas.json")]
        out.append({
            "id": rel,
            "default": basename,
            "prefix": f"[{backend_display}]",
            "prefix_color": backend_theme_slot,
        })
    return out


def _build_backend_canvas_group(
    info: ProjectInfo,
    canvas_root: Path,
    backend_dir: Path,
    engine_id: str,
    backend_display: str,
    backend_theme_slot: str,
) -> Dict:
    """One top-level canvas group ``{id, default, expanded, groups: [...]}``.

    Sub-groups: one per direct subdirectory of ``backend_dir`` (excluding
    ``_EXCLUDED_DIRS``), each carrying its descendant ``*.canvas.json``
    files; plus a trailing ``(root)`` sub-group for loose files at
    ``backend_dir`` root. Empty sub-groups are dropped. The top group is
    always returned (header renders even when ``groups`` is ``[]``).
    """
    sub_groups: List[Dict] = []

    if backend_dir.is_dir():
        for child in sorted(backend_dir.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            if child.name in _EXCLUDED_DIRS:
                continue
            items = _canvas_items_from_dir(
                info, canvas_root, child, backend_display, backend_theme_slot
            )
            if not items:
                continue
            sub_groups.append({
                "id": f"mc.canvas.cat.{engine_id}.{child.name}",
                "default": child.name,
                "expanded": True,
                "items": items,
            })

        root_items = _canvas_items_from_root(
            info, backend_dir, backend_display, backend_theme_slot
        )
        if root_items:
            sub_groups.append({
                "id": f"mc.canvas.cat.{engine_id}.{_ROOT_GROUP_ID_SUFFIX}",
                "default": _ROOT_GROUP_LABEL,
                "expanded": True,
                "items": root_items,
            })

    return {
        "id": f"mc.canvas.backend.{engine_id}",
        "default": backend_display,
        "expanded": True,
        "groups": sub_groups,
    }


def list_canvas_groups(info: ProjectInfo) -> List[Dict]:
    """Build the Canvas tab's two-level groups spec — backend → sub-dir.

    Layout:
      * Top-level group per known backend whose ``canvas/<backend_subdir>/``
        exists or whose row appears in ``backends_installed.json``.
      * Within each backend group, one sub-group per direct subdir of
        ``canvas/<backend_subdir>/``, plus a trailing ``(root)`` for files
        directly under it.
      * Each item carries ``prefix`` (``[<display_name>]``) and
        ``prefix_color`` (theme slot) so the SDK ``LaviTabTable`` paints
        the prefix in the backend's signature theme color while the
        basename follows hover/checked styling.

    Manifest-declared ``assets.workflows`` are NOT integrated into this
    grouped view (current spec: a canvas's identity = its on-disk
    ``canvas/<backend>/<file>`` location, full stop). If a manifest
    workflow points outside ``canvas/`` it is silently dropped.

    Subdirectories under ``canvas/`` whose name does not resolve to a
    known backend (via ``registers.backends.resolve_engine_id_from_subdir``)
    are skipped with a warning — this is the storage rule:
    ``canvas/{backend_subdir}/...``.
    """
    # Local import — registers depends on unitport_sdk only, so this stays
    # cycle-free; deferred to keep service.projects import-time light.
    from registers import backends as backends_registry

    out: List[Dict] = []
    canvas_root = info.path / "canvas"

    # Only backends declared in ``_CANVAS_SUBDIRS`` own canvas storage —
    # component-probe rows in ``backends_installed.json`` (sb3 / mujoco /
    # gymnasium individually) are not canvas owners and must not render
    # their own bucket. Probe the source of truth via canvas_subdir() over
    # a fixed list of known canvas-owning ids; the dance lets us still
    # surface a top-level header for backends with empty folders.
    canvas_owner_ids: List[str] = []
    for eid in ("isaac_lab", "sb3_mujoco"):
        if backends_registry.canvas_subdir(eid):
            canvas_owner_ids.append(eid)

    # On-disk subdirs not matching a canvas-owner subdir get a warning
    # (storage rule violation) — already-known subdirs are silently used.
    expected_subdirs = {backends_registry.canvas_subdir(eid) for eid in canvas_owner_ids}
    if canvas_root.is_dir():
        for child in sorted(canvas_root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            if child.name in _EXCLUDED_DIRS:
                continue
            if child.name in expected_subdirs:
                continue
            eid = backends_registry.resolve_engine_id_from_subdir(child.name)
            if eid is None:
                log_warning(
                    f"[file_index] canvas subdir {child.name!r} does not match any "
                    f"registered backend; skipping. Storage rule: "
                    f"canvas/<backend_subdir>/<file>.canvas.json"
                )
                continue
            if eid not in canvas_owner_ids:
                canvas_owner_ids.append(eid)

    for eid in canvas_owner_ids:
        display = backends_registry.canvas_subdir(eid)
        slot = backends_registry.get_theme_slot(eid)
        backend_dir = canvas_root / display
        out.append(_build_backend_canvas_group(
            info=info,
            canvas_root=canvas_root,
            backend_dir=backend_dir,
            engine_id=eid,
            backend_display=display,
            backend_theme_slot=slot,
        ))

    return out


# ---------------------------------------------------------------------------
# Script tab — two-level groups spec
# ---------------------------------------------------------------------------
def _scripts_root_for_project(info: ProjectInfo) -> Path:
    return info.path / "scripts"


def _scripts_root_for_system() -> Path:
    return Paths.SRC_ROOT / "scripts"


def _build_bucket_groups(
    bucket: str,
    bucket_id: str,
    bucket_label: str,
    scripts_root: Path,
) -> Dict:
    """Build one top-level group spec ``{id, default, expanded, groups: [...]}``.

    Sub-groups: one per direct subdirectory of ``scripts_root`` (excluding
    any folder named in :data:`_EXCLUDED_DIRS`), plus a trailing ``(root)``
    sub-group for loose ``*.py`` files at ``scripts_root`` itself. Sub-groups
    that contain zero ``*.py`` items are dropped. If ``scripts_root`` does
    not exist or has no usable scripts, the top-group is returned with an
    empty ``groups`` list (its header still renders).
    """
    sub_groups: List[Dict] = []

    if scripts_root.is_dir():
        for child in sorted(scripts_root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            if child.name in _EXCLUDED_DIRS:
                continue
            items = _items_from_dir(bucket, scripts_root, child)
            if not items:
                continue
            sub_groups.append(
                {
                    "id": f"mc.script.cat.{bucket}.{child.name}",
                    "default": child.name,
                    "expanded": True,
                    "items": items,
                }
            )

        # Trailing (root) sub-group for loose .py files at the bucket root.
        root_items = _items_from_root(bucket, scripts_root)
        if root_items:
            sub_groups.append(
                {
                    "id": f"mc.script.cat.{bucket}.{_ROOT_GROUP_ID_SUFFIX}",
                    "default": _ROOT_GROUP_LABEL,
                    "expanded": True,
                    "items": root_items,
                }
            )

    return {
        "id": bucket_id,
        "default": bucket_label,
        "expanded": True,
        "groups": sub_groups,
    }


def _items_from_dir(
    bucket: str, scripts_root: Path, sub_dir: Path
) -> List[Dict]:
    """``[{"id": "<bucket>:<rel>", "default": "<label>"}, ...]`` for *.py under ``sub_dir``."""
    out: List[Dict] = []
    for p in sorted(sub_dir.rglob("*.py")):
        # Skip any *.py living under an excluded folder (e.g. nested cache/).
        try:
            parts = p.relative_to(scripts_root).parts
        except ValueError:
            continue
        if any(seg in _EXCLUDED_DIRS for seg in parts[:-1]):
            continue
        rel = p.relative_to(scripts_root).as_posix()
        label = p.relative_to(sub_dir).as_posix()
        if label.endswith(".py"):
            label = label[:-3]
        out.append({"id": f"{bucket}:{rel}", "default": label})
    return out


def _items_from_root(bucket: str, scripts_root: Path) -> List[Dict]:
    """``[{"id": "<bucket>:<file>", "default": "<stem>"}, ...]`` for loose top-level *.py."""
    out: List[Dict] = []
    for p in sorted(scripts_root.glob("*.py")):
        if not p.is_file():
            continue
        rel = p.relative_to(scripts_root).as_posix()
        label = p.stem
        out.append({"id": f"{bucket}:{rel}", "default": label})
    return out


def list_script_groups(info: ProjectInfo) -> List[Dict]:
    """Build the Script tab's two-level groups spec.

    Returns a list slot-in-able as the Script tab's ``groups`` field. Two
    top-level groups are always emitted so the bucket structure stays
    visible even when one side is empty:

      * ``Project Script`` — ``<Paths.PROJECTS_DIR>/<cur>/scripts/``
      * ``Global Script``  — ``Paths.SRC_ROOT / 'scripts'``

    Each top-level group contains one sub-group per direct subdirectory
    (excluding ``cache``), plus a trailing ``(root)`` sub-group for loose
    ``*.py`` files at the bucket's scripts root. Item ids are encoded as
    ``project:<rel>`` / ``system:<rel>`` so :func:`resolve_file` can
    dispatch back to the correct on-disk root.
    """
    return [
        _build_bucket_groups(
            bucket="project",
            bucket_id="mc.script.bucket.project",
            bucket_label="Project Script",
            scripts_root=_scripts_root_for_project(info),
        ),
        _build_bucket_groups(
            bucket="system",
            bucket_id="mc.script.bucket.system",
            bucket_label="Global Script",
            scripts_root=_scripts_root_for_system(),
        ),
    ]


# ---------------------------------------------------------------------------
# resolve_file
# ---------------------------------------------------------------------------
def resolve_file(info: ProjectInfo, file_id: str) -> Path:
    """Resolve a file_id to an absolute on-disk path.

    Dispatch by prefix:
      * ``project:<rel>`` — anchored under ``info.path / 'scripts'``.
      * ``system:<rel>``  — anchored under ``Paths.SRC_ROOT / 'scripts'``.
      * (no prefix)       — legacy canvas id; project-root-relative POSIX path.

    Raises :class:`ValueError` on empty id or path traversal that escapes
    the corresponding root (defence against ``../``).
    """
    if not file_id:
        raise ValueError("empty file_id")

    if file_id.startswith("project:"):
        rel = file_id[len("project:"):]
        root = (info.path / "scripts").resolve()
        candidate = (info.path / "scripts" / rel).resolve()
    elif file_id.startswith("system:"):
        rel = file_id[len("system:"):]
        root = (Paths.SRC_ROOT / "scripts").resolve()
        candidate = (Paths.SRC_ROOT / "scripts" / rel).resolve()
    else:
        rel = file_id
        root = info.path.resolve()
        candidate = (info.path / rel).resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"file_id {file_id!r} escapes its root {root!s}")
    return candidate


__all__ = [
    "list_canvas_files",
    "list_canvas_groups",
    "list_script_groups",
    "resolve_file",
]
