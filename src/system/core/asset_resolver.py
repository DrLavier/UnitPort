#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AssetResolver — merges project-local + shared checkpoint/bundle views.

Resolution rules:
  1. Scan project-local checkpoints/   → scope = "local"
  2. Scan shared/checkpoints/          → scope = "shared"
  3. Merge:
     - Same policy_id in both → shared display_name gets "(GLOB)" suffix
     - Only in local  → display as-is
     - Only in shared → display as-is
  4. resolve_policy: local-first, shared-fallback

No Qt.  No training logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from src.system.core.project_store import ProjectStore


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ResolvedAsset:
    """A checkpoint entry visible to the user, with scope metadata."""
    policy_id: str = ""
    display_name: str = ""
    scope: str = ""           # "local" | "shared"
    path: Path = field(default_factory=Path)
    published: bool = False   # for local: whether already published to shared


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class AssetResolver:
    """Merge project-local + shared asset views.

    Parameters
    ----------
    project_store : ProjectStore
        The store instance that owns path layout.
    project_id : str
        Active project to resolve against.
    """

    def __init__(self, project_store: ProjectStore, project_id: str) -> None:
        self._store = project_store
        self._project_id = project_id

    # -- public API ----------------------------------------------------------

    def list_available_checkpoints(self) -> List[ResolvedAsset]:
        """Return merged list of local + shared checkpoints.

        When the same policy_id exists in both scopes, the shared entry's
        display_name gets a ``(GLOB)`` suffix.
        """
        local_map = self._scan_local_checkpoints()
        shared_map = self._scan_shared_checkpoints()

        local_ids = set(local_map)
        shared_ids = set(shared_map)
        collision_ids = local_ids & shared_ids

        results: List[ResolvedAsset] = []

        # Local entries
        for pid, (path, published) in local_map.items():
            results.append(ResolvedAsset(
                policy_id=pid,
                display_name=pid,
                scope="local",
                path=path,
                published=published,
            ))

        # Shared entries
        for pid, path in shared_map.items():
            display = f"{pid} (GLOB)" if pid in collision_ids else pid
            results.append(ResolvedAsset(
                policy_id=pid,
                display_name=display,
                scope="shared",
                path=path,
                published=False,
            ))

        return results

    def resolve_policy(self, policy_id: str) -> Path:
        """Resolve a policy_id to its bundle directory path.

        Local-first, shared-fallback.  Raises FileNotFoundError if neither.
        """
        # Try local
        local_path = (
            self._store._checkpoints_dir(self._project_id) / policy_id
        )
        if local_path.is_dir():
            return local_path

        # Try shared
        shared_path = self._store._shared_checkpoints_dir() / policy_id
        if shared_path.is_dir():
            return shared_path

        raise FileNotFoundError(
            f"Policy '{policy_id}' not found in project "
            f"'{self._project_id}' or shared checkpoints"
        )

    # -- internals -----------------------------------------------------------

    def _scan_local_checkpoints(self) -> dict:
        """Return {policy_id: (Path, published_bool)} for project-local."""
        ckpt_dir = self._store._checkpoints_dir(self._project_id)
        result: dict = {}
        if not ckpt_dir.exists():
            return result

        # Build published lookup from project meta
        try:
            meta = self._store.open_project(self._project_id)
            published_set = {
                c.policy_id for c in meta.assets.checkpoints if c.published
            }
        except FileNotFoundError:
            published_set = set()

        for entry in ckpt_dir.iterdir():
            if entry.is_dir():
                pid = entry.name
                result[pid] = (entry, pid in published_set)
        return result

    def _scan_shared_checkpoints(self) -> dict:
        """Return {policy_id: Path} for shared checkpoints."""
        shared_dir = self._store._shared_checkpoints_dir()
        result: dict = {}
        if not shared_dir.exists():
            return result
        for entry in shared_dir.iterdir():
            if entry.is_dir():
                result[entry.name] = entry
        return result
