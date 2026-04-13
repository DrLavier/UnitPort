#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — Workspace persistence layer.

.. deprecated::
    This module is a thin compatibility wrapper around
    :class:`~src.system.core.project_store.ProjectStore`.
    New code should use ``ProjectStore`` directly.

Layout v2 (``knowledge_base/project_design.yaml``) defines a project as a
*task workspace* that contains many experiments, each producing many
*policies*. A trained-policy id is therefore an artifact identifier that
lives **inside** an experiment run — it is not a workspace key.

To make that distinction explicit, every method on this wrapper takes a
``workspace_name`` argument that maps to ``ProjectMeta.name`` via lookup.
Earlier revisions of this file called the same parameter ``policy_id``,
which was misleading; it has been renamed to ``workspace_name``. Callers
that still hold a true trained-policy id should reach for ``ProjectStore``
+ ``ExperimentMeta`` directly instead of routing through this wrapper.

API::

    store = TrainingWorkspaceStore(root)
    store.bind_project_id(workspace_name, project_id)   # optional, recommended
    store.ensure_workspace(workspace_name)
    meta  = store.load_workspace(workspace_name)
    exps  = store.list_experiments(workspace_name)
    store.save_experiment(workspace_name, experiment_id, canvas_data, metadata)
    canvas= store.load_experiment(workspace_name, experiment_id)
    info  = store.create_experiment(workspace_name, name)
    runs  = store.list_runs(workspace_name)

Internally each workspace_name resolves to a ProjectStore project via
``ProjectMeta.name`` lookup; the v2 ``project_id`` (filesystem slug) is
recovered through ``_resolve_project_id`` and cached.

No Qt.  No real training.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.system.core.project_store import (
    ProjectStore,
    ProjectMeta,
    ExperimentMeta,
    RunMeta,
    _atomic_write,
    _gen_id,
    _now_iso,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_WORKSPACE_SCHEMA = "training_workspace_v1"
_CANVAS_SCHEMA    = "training_canvas_v1"


# ---------------------------------------------------------------------------
# Data classes — kept for backward compatibility
# ---------------------------------------------------------------------------

# Re-export from project_store for any callers that import from here
ExperimentMeta = ExperimentMeta
RunMeta = RunMeta


@dataclass
class WorkspaceMeta:
    """Contents of workspace.json — compatibility shim.

    Maps onto ProjectMeta internally, exposing the workspace as a flat view
    of (workspace_name, active_experiment_id, experiments[]).

    The on-disk legacy ``workspace_meta.json`` (only ever read during
    migration) stored the workspace handle under the misleading key
    ``policy_id``; ``from_dict`` accepts that legacy key as a fallback.
    """
    schema: str = _WORKSPACE_SCHEMA
    workspace_name: str = ""   # == ProjectMeta.name; the human-readable workspace handle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active_experiment_id: str = ""
    experiments: List[ExperimentMeta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "workspace_name": self.workspace_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_experiment_id": self.active_experiment_id,
            "experiments": [e.to_dict() for e in self.experiments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceMeta":
        # Accept legacy "policy_id" key for unmigrated workspace_meta.json files.
        ws_name = str(d.get("workspace_name") or d.get("policy_id") or "")
        return cls(
            schema=str(d.get("schema", _WORKSPACE_SCHEMA)),
            workspace_name=ws_name,
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
            active_experiment_id=str(d.get("active_experiment_id", "")),
            experiments=[
                ExperimentMeta.from_dict(e)
                for e in d.get("experiments", [])
            ],
        )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMeta]:
        for e in self.experiments:
            if e.experiment_id == experiment_id:
                return e
        return None


# ---------------------------------------------------------------------------
# Store — thin wrapper around ProjectStore
# ---------------------------------------------------------------------------

class TrainingWorkspaceStore:
    """
    Persist and retrieve training workspace data.

    .. deprecated::
        Delegates all operations to :class:`ProjectStore`.
        New code should use ``ProjectStore`` directly.

    Every method below takes a ``workspace_name`` parameter — the human
    readable workspace handle that maps onto ``ProjectMeta.name``. This is
    *not* a trained-policy id. Callers that already know the active
    ``project_id`` (filesystem slug) should call :meth:`bind_project_id`
    first; this guarantees ``ensure_workspace`` reuses the existing project
    rather than name-matching against ``ProjectMeta.name``.

    Parameters
    ----------
    root:
        Base directory under which ``projects/`` and ``shared/`` live.
        Defaults to the current working directory.
    """

    def __init__(self, root: Optional[str] = None):
        if root is None:
            root = os.getcwd()
        self._root = Path(root)
        self._ps = ProjectStore(root=self._root)
        # Legacy path for migration detection
        self._legacy_root = self._root / "training_workspaces"
        # Cache: workspace_name → project_id mapping
        self._pid_map: Dict[str, str] = {}
        # Cache: workspace_name → workspace.json path (for active_experiment_id)
        self._ws_meta_cache: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Internal: workspace_name → project_id resolution
    # ------------------------------------------------------------------

    def bind_project_id(self, workspace_name: str, project_id: str) -> None:
        """Pre-bind a workspace_name to a known project_id.

        Allows callers that already know the active project to skip the
        name-based scan and, more importantly, prevents ``ensure_workspace``
        from accidentally creating a duplicate project.
        """
        self._pid_map[workspace_name] = project_id

    def _resolve_project_id(self, workspace_name: str) -> str:
        """Return the project_id for a given workspace_name.

        Projects are looked up by ``ProjectMeta.name == workspace_name``.
        If not found, returns "" (caller should call ensure_workspace first).
        """
        if workspace_name in self._pid_map:
            return self._pid_map[workspace_name]
        for p in self._ps.list_projects():
            if p.name == workspace_name:
                self._pid_map[workspace_name] = p.project_id
                return p.project_id
        return ""

    def _ws_meta_path(self, project_id: str) -> Path:
        """Legacy path to workspace_meta.json — kept only for migration reads."""
        return self._ps._project_dir(project_id) / "workspace_meta.json"

    def _load_ws_meta(self, project_id: str, workspace_name: str) -> WorkspaceMeta:
        """Load workspace state from project.yaml (v2) with legacy fallback.

        Layout v2 absorbs the old workspace_meta.json fields
        (active_experiment_id, experiments[]) into project.yaml itself, so
        the canonical read path is ProjectMeta. The old sidecar is still
        consulted on first access so an unmigrated project keeps working.
        """
        # v2: read from project.yaml
        try:
            pmeta = self._ps.open_project(project_id)
            return WorkspaceMeta(
                workspace_name=workspace_name or pmeta.name or pmeta.project_id,
                created_at=time.time(),
                updated_at=time.time(),
                active_experiment_id=pmeta.active_experiment,
                experiments=list(pmeta.experiments),
            )
        except FileNotFoundError:
            pass
        # Legacy: workspace_meta.json sidecar
        meta_path = self._ws_meta_path(project_id)
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return WorkspaceMeta.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        return WorkspaceMeta(
            workspace_name=workspace_name,
            created_at=time.time(),
            updated_at=time.time(),
            experiments=self._ps.list_experiments(project_id),
        )

    def _save_ws_meta(self, project_id: str, meta: WorkspaceMeta) -> None:
        """Persist workspace state into project.yaml.

        In v2 there is no sidecar — active_experiment + experiments live
        directly in ProjectMeta. We mirror the wrapper's WorkspaceMeta back
        into ProjectMeta and call save_meta(). Any pre-existing
        workspace_meta.json is removed to avoid stale duplicates.
        """
        meta.updated_at = time.time()
        try:
            pmeta = self._ps.open_project(project_id)
        except FileNotFoundError:
            return
        pmeta.active_experiment = meta.active_experiment_id
        pmeta.experiments = list(meta.experiments)
        self._ps.save_meta(pmeta)
        legacy = self._ws_meta_path(project_id)
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Workspace lifecycle
    # ------------------------------------------------------------------

    def list_workspaces(self) -> List[str]:
        """Return all workspace names (== project names)."""
        return [p.name for p in self._ps.list_projects()]

    def rename_workspace(self, old_workspace_name: str, new_workspace_name: str) -> None:
        """Rename a workspace (project)."""
        pid = self._resolve_project_id(old_workspace_name)
        if not pid:
            raise FileNotFoundError(f"Workspace '{old_workspace_name}' not found.")
        existing = self._resolve_project_id(new_workspace_name)
        if existing:
            raise ValueError(f"Workspace '{new_workspace_name}' already exists.")
        meta = self._ps.open_project(pid)
        meta.name = new_workspace_name
        self._ps.save_meta(meta)
        # Mirror the rename into the WorkspaceMeta shim
        ws_meta = self._load_ws_meta(pid, new_workspace_name)
        ws_meta.workspace_name = new_workspace_name
        self._save_ws_meta(pid, ws_meta)
        # Invalidate cache
        self._pid_map.pop(old_workspace_name, None)
        self._pid_map[new_workspace_name] = pid

    def delete_workspace(self, workspace_name: str) -> None:
        """Delete a workspace (project) entirely."""
        pid = self._resolve_project_id(workspace_name)
        if not pid:
            raise FileNotFoundError(f"Workspace '{workspace_name}' not found.")
        self._ps.delete_project(pid)
        self._pid_map.pop(workspace_name, None)

    def workspace_exists(self, workspace_name: str) -> bool:
        """Return True if a workspace named *workspace_name* already exists."""
        return bool(self._resolve_project_id(workspace_name))

    def ensure_workspace(self, workspace_name: str) -> WorkspaceMeta:
        """Open the workspace named *workspace_name*; create it on demand if absent.

        Auto-creation exists for the "New Workspace" UI flow that prompts
        the user for a workspace name. Callers that already hold a known
        ``project_id`` should call :meth:`bind_project_id` first to attach
        this wrapper to that project rather than creating a new one.
        """
        pid = self._resolve_project_id(workspace_name)
        if pid:
            return self._load_ws_meta(pid, workspace_name)
        # No existing workspace under this name — create one.
        proj = self._ps.create_project(name=workspace_name)
        self._pid_map[workspace_name] = proj.project_id
        meta = WorkspaceMeta(
            workspace_name=workspace_name,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._save_ws_meta(proj.project_id, meta)
        return meta

    def load_workspace(self, workspace_name: str) -> WorkspaceMeta:
        """Load workspace metadata.  Raises FileNotFoundError."""
        pid = self._resolve_project_id(workspace_name)
        if not pid:
            raise FileNotFoundError(
                f"Workspace not found for workspace_name={workspace_name!r}. "
                "Call ensure_workspace() first."
            )
        return self._load_ws_meta(pid, workspace_name)

    def _save_workspace_meta(self, meta: WorkspaceMeta) -> None:
        pid = self._resolve_project_id(meta.workspace_name)
        if pid:
            self._save_ws_meta(pid, meta)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def list_experiments(self, workspace_name: str) -> List[ExperimentMeta]:
        """Return experiment metadata list."""
        try:
            meta = self.load_workspace(workspace_name)
        except FileNotFoundError:
            return []
        return list(meta.experiments)

    def create_experiment(
        self,
        workspace_name: str,
        name: str = "",
        initial_canvas: Optional[dict] = None,
    ) -> ExperimentMeta:
        """Create a new experiment entry."""
        meta = self.ensure_workspace(workspace_name)
        pid = self._resolve_project_id(workspace_name)
        if not name:
            name = f"Experiment {len(meta.experiments) + 1}"

        canvas_data = initial_canvas or {
            "schema": _CANVAS_SCHEMA,
            "nodes": [],
            "edges": [],
        }
        exp = self._ps.create_experiment(pid, name, canvas_data)

        # Update workspace meta
        meta.experiments.append(exp)
        if not meta.active_experiment_id:
            meta.active_experiment_id = exp.experiment_id
        self._save_ws_meta(pid, meta)
        return exp

    def save_experiment(
        self,
        workspace_name: str,
        experiment_id: str,
        canvas_data: dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist canvas snapshot for an existing experiment."""
        meta = self.ensure_workspace(workspace_name)
        pid = self._resolve_project_id(workspace_name)

        payload = dict(canvas_data)
        if metadata:
            payload["_meta"] = metadata

        self._ps.save_experiment(pid, experiment_id, payload)

        # Update timestamps in workspace meta
        exp_meta = meta.get_experiment(experiment_id)
        if exp_meta is None:
            exp_meta = ExperimentMeta(
                experiment_id=experiment_id,
                name=experiment_id,
                created_at=time.time(),
            )
            meta.experiments.append(exp_meta)
        exp_meta.updated_at = time.time()
        self._save_ws_meta(pid, meta)

    def load_experiment(self, workspace_name: str, experiment_id: str) -> dict:
        """Load and return the canvas snapshot for an experiment."""
        pid = self._resolve_project_id(workspace_name)
        if not pid:
            raise FileNotFoundError(
                f"Experiment {experiment_id!r} not found in workspace {workspace_name!r}"
            )
        return self._ps.load_experiment(pid, experiment_id)

    def set_active_experiment(self, workspace_name: str, experiment_id: str) -> None:
        """Set the active (last-opened) experiment for a workspace."""
        meta = self.load_workspace(workspace_name)
        meta.active_experiment_id = experiment_id
        self._save_workspace_meta(meta)

    def rename_experiment(
        self, workspace_name: str, experiment_id: str, new_name: str
    ) -> None:
        """Rename an experiment (updates workspace meta only)."""
        meta = self.load_workspace(workspace_name)
        exp_meta = meta.get_experiment(experiment_id)
        if exp_meta is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        exp_meta.name = new_name
        exp_meta.updated_at = time.time()
        self._save_workspace_meta(meta)

    def delete_experiment(self, workspace_name: str, experiment_id: str) -> None:
        """Delete an experiment entry and its persisted canvas snapshot."""
        meta = self.load_workspace(workspace_name)
        exp_meta = meta.get_experiment(experiment_id)
        if exp_meta is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")

        meta.experiments = [
            e for e in meta.experiments if e.experiment_id != experiment_id
        ]
        if meta.active_experiment_id == experiment_id:
            meta.active_experiment_id = (
                meta.experiments[0].experiment_id if meta.experiments else ""
            )

        pid = self._resolve_project_id(workspace_name)
        if pid:
            self._ps.delete_experiment(pid, experiment_id)

        self._save_workspace_meta(meta)

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def create_run(self, workspace_name: str, run_meta: "RunMeta") -> "RunMeta":
        """Persist a new run record under the workspace's project."""
        self.ensure_workspace(workspace_name)
        pid = self._resolve_project_id(workspace_name)
        return self._ps.create_run(pid, run_meta)

    def update_run(self, workspace_name: str, run_id: str, fields: Dict[str, Any]) -> None:
        """Update specific fields of an existing run record."""
        pid = self._resolve_project_id(workspace_name)
        if not pid:
            raise FileNotFoundError(
                f"Run {run_id!r} not found in workspace {workspace_name!r}"
            )
        self._ps.update_run(pid, run_id, **fields)

    def list_runs(self, workspace_name: str) -> List[Dict[str, Any]]:
        """Return run metadata dicts, sorted newest-first."""
        pid = self._resolve_project_id(workspace_name)
        if not pid:
            return []
        return self._ps.list_runs(pid)
