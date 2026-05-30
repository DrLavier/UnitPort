# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""robot_assets.discovery — local asset auto-discovery.

Scans ``custom_mods/models/`` (incl. ``menagerie/`` sparse-checkout) for
USD / URDF / XACRO files belonging to robots that exist in
``registers.robots``, and returns the discovered relative paths so
:class:`RobotAssetService` can persist them into
``<USER_CONFIG_DIR>/registers/robots_custom.json`` (the user-overlay layer that
``registers.robots.merge_user_extensions`` already merges into the
hub at load time).

Why discovery is needed: factory ``robots_canonical.json`` ships with
``USD/URDF/XACRO`` set to ``null`` for every robot — those are
per-machine facts (depend on whether the user installed mujoco_menagerie,
which USD packs they have, etc.). Without this scanner the side panel
shows "(none)" for every USD slot and the canvas Robot Node never sees
USD/URDF, even when the files are on disk.

Single source of truth rule: discovery results land **only** in
``robots_custom.json`` (the registry-overlay layer). They are NEVER
written into ``state.json`` or any panel/canvas-local cache — those
modules read back from ``registers.robots`` like every other consumer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from unitport_sdk import Paths, log_warning

from registers import robots as _robots_registry


# Same root list as RobotAssetService._canonical_asset_roots — kept in
# sync deliberately so a relative path discovered here resolves through
# the same lookup chain that resolve() uses.
def scan_roots() -> List[Path]:
    return [
        Paths.CUSTOM_MODS_DIR / "models",
        Paths.PROJECT_ROOT.parent / "DEMO" / "custom_mods" / "models",
        Paths.ASSETS_DIR,
    ]


# Deprecated underscore alias kept until 2026-07 in case external callers
# imported the private name. WHY KEPT (§8 (c)): consistent rename without
# breaking forward imports during the same release cycle.
_scan_roots = scan_roots


def _pick_usd(dir_: Path) -> Optional[Path]:
    if not dir_.is_dir():
        return None
    usds = sorted(dir_.glob("*.usd")) + sorted(dir_.glob("*.usda"))
    return usds[0] if usds else None


def _pick_urdf(dir_: Path) -> Optional[Path]:
    if not dir_.is_dir():
        return None
    urdfs = sorted(dir_.glob("*.urdf"))
    return urdfs[0] if urdfs else None


def _pick_xacro(dir_: Path) -> Optional[Path]:
    if not dir_.is_dir():
        return None
    xacros = sorted(dir_.glob("*.xacro")) + sorted(dir_.glob("*.urdf.xacro"))
    return xacros[0] if xacros else None


def _to_rel_posix(abs_path: Path, root: Path) -> str:
    """Convert ``abs_path`` to a POSIX-style relpath under ``root``.

    Falls back to absolute POSIX if the path is not under ``root`` (e.g.
    when discovered via the DEMO sibling root) — resolve() handles
    absolute candidates too.
    """
    try:
        rel = abs_path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return abs_path.resolve().as_posix()


def _resolve_dir_to_sku(dir_name: str) -> Optional[str]:
    """Map a menagerie folder name to a canonical SKU via aliases.

    Examples:
        ``unitree_go2`` → SKU for ``unitree.go2``
        ``boston_dynamics_spot`` → SKU for ``boston_dynamics.spot``
    Returns ``None`` if the folder doesn't match any registered robot.
    """
    return _robots_registry.resolve_id(dir_name)


def enumerate_candidate_dirs(root: Path) -> List[Path]:
    """Top-level subdirs that look like robot packages.

    Includes ``<root>/menagerie/<robot>/`` (the common case after
    sparse-checkout) AND ``<root>/<brand>/<model>/`` (custom drops).
    """
    candidates: List[Path] = []
    if not root.is_dir():
        return candidates
    # Direct subdirs (e.g. custom drops at custom_mods/models/<brand>/<model>/)
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name.startswith(".") or sub.name.startswith("_"):
            continue
        candidates.append(sub)
        # One nested level for brand/model layout.
        for sub2 in sub.iterdir() if sub.is_dir() else []:
            if not sub2.is_dir() or sub2.name.startswith(".") or sub2.name.startswith("_"):
                continue
            candidates.append(sub2)
    return candidates


# Deprecated underscore alias — see ``scan_roots`` for the rename rationale.
_enumerate_candidate_dirs = enumerate_candidate_dirs


def has_robot_description_file(dir_: Path) -> bool:
    """True iff ``dir_`` contains at least one MJCF / USD / URDF file.

    Used by :meth:`RobotAssetService.scan_unregistered_packages` to filter
    out non-robot directories (pure asset libraries, image-only previews,
    nested geometry packs) when scanning all ``custom_mods/models/*`` subdirs
    for unregistered packages. A robot package without any of these three
    is not actionable for the registration editor.
    """
    if not dir_.is_dir():
        return False
    # Single-level glob; nested files don't count — the registration
    # editor needs a top-level scene/usd/urdf file to pre-fill from.
    for ext in ("*.xml", "*.usd", "*.usda", "*.urdf"):
        for _ in dir_.glob(ext):
            return True
    return False


def discover_local_assets() -> Dict[str, Dict[str, str]]:
    """Scan all canonical roots for USD/URDF/XACRO files belonging to known SKUs.

    Returns ``{sku: {kind: rel_path_posix, ...}, ...}``. Only kinds with
    a discovered file are present. ``rel_path_posix`` is relative to the
    root that contained the file (matches the canonical schema like
    ``menagerie/unitree_go2/go2.usd``); falls back to absolute POSIX
    when discovered outside any of the canonical roots.

    This function does **not** write anywhere — it's pure scan. The
    caller (``RobotAssetService.scan_and_merge_assets``) decides what to
    persist and applies the "fill-only-empty-slots" rule.
    """
    out: Dict[str, Dict[str, str]] = {}
    seen_pairs: Set[str] = set()

    for root in scan_roots():
        if not root.exists():
            continue
        for cand_dir in enumerate_candidate_dirs(root):
            try:
                sku = _resolve_dir_to_sku(cand_dir.name)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[robot_assets.discovery] resolve_id({cand_dir.name!r}) failed: {exc}")
                continue
            if not sku:
                continue

            for kind, picker in (
                ("USD", _pick_usd),
                ("URDF", _pick_urdf),
                ("XACRO", _pick_xacro),
            ):
                pair_key = f"{sku}:{kind}"
                if pair_key in seen_pairs:
                    continue
                hit = picker(cand_dir)
                if hit is None:
                    continue
                seen_pairs.add(pair_key)
                rel = _to_rel_posix(hit, root)
                out.setdefault(sku, {})[kind] = rel

    return out


__all__ = [
    "discover_local_assets",
    "scan_roots",
    "enumerate_candidate_dirs",
    "has_robot_description_file",
    "_to_rel_posix",
]


# Re-export the path-relativiser too so the service can reuse it when
# building hint dicts for unregistered packages without duplicating the
# resolve/relpath fallback logic.
