#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — Workspace persistence layer.

.. deprecated::
    This module is a thin compatibility wrapper around
    :class:`~src.system.core.project_store.ProjectStore`.
    New code should use ``ProjectStore`` directly.

API (unchanged)::

    store = TrainingWorkspaceStore(root)
    store.ensure_workspace(policy_id)
    meta  = store.load_workspace(policy_id)
    exps  = store.list_experiments(policy_id)
    store.save_experiment(policy_id, experiment_id, canvas_data, metadata)
    canvas= store.load_experiment(policy_id, experiment_id)
    info  = store.create_experiment(policy_id, name)
    runs  = store.list_runs(policy_id)

Internally, each ``policy_id`` workspace maps to a ProjectStore project
where ``project_id == policy_id``.

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

    Maps onto ProjectMeta internally, but preserves the legacy API surface
    (policy_id, active_experiment_id, experiments list).
    """
    schema: str = _WORKSPACE_SCHEMA
    policy_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active_experiment_id: str = ""
    experiments: List[ExperimentMeta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_experiment_id": self.active_experiment_id,
            "experiments": [e.to_dict() for e in self.experiments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceMeta":
        return cls(
            schema=str(d.get("schema", _WORKSPACE_SCHEMA)),
            policy_id=str(d.get("policy_id", "")),
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
        # Cache: policy_id → project_id mapping
        self._pid_map: Dict[str, str] = {}
        # Cache: policy_id → workspace.json path (for active_experiment_id)
        self._ws_meta_cache: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Internal: policy_id → project_id resolution
    # ------------------------------------------------------------------

    def _resolve_project_id(self, policy_id: str) -> str:
        """Return the project_id for a given policy_id.

        Projects are looked up by name == policy_id.  If not found,
        returns "" (caller should call ensure_workspace first).
        """
        if policy_id in self._pid_map:
            return self._pid_map[policy_id]
        for p in self._ps.list_projects():
            if p.name == policy_id:
                self._pid_map[policy_id] = p.project_id
                return p.project_id
        return ""

    def _ws_meta_path(self, project_id: str) -> Path:
        """Path to the workspace_meta.json inside a project dir.

        Stores active_experiment_id and experiment list — legacy compat.
        """
        return self._ps._project_dir(project_id) / "workspace_meta.json"

    def _load_ws_meta(self, project_id: str, policy_id: str) -> WorkspaceMeta:
        """Load or synthesize a WorkspaceMeta for a project."""
        meta_path = self._ws_meta_path(project_id)
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return WorkspaceMeta.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        # Synthesize from project experiments
        experiments = self._ps.list_experiments(project_id)
        return WorkspaceMeta(
            policy_id=policy_id,
            created_at=time.time(),
            updated_at=time.time(),
            experiments=experiments,
        )

    def _save_ws_meta(self, project_id: str, meta: WorkspaceMeta) -> None:
        """Persist WorkspaceMeta alongside the project."""
        meta.updated_at = time.time()
        _atomic_write(self._ws_meta_path(project_id), meta.to_dict())

    # ------------------------------------------------------------------
    # Workspace lifecycle
    # ------------------------------------------------------------------

    def list_workspaces(self) -> List[str]:
        """Return all workspace policy_ids (= project names)."""
        return [p.name for p in self._ps.list_projects()]

    def rename_workspace(self, old_policy_id: str, new_policy_id: str) -> None:
        """Rename a workspace (project)."""
        pid = self._resolve_project_id(old_policy_id)
        if not pid:
            raise FileNotFoundError(f"Workspace '{old_policy_id}' not found.")
        existing = self._resolve_project_id(new_policy_id)
        if existing:
            raise ValueError(f"Workspace '{new_policy_id}' already exists.")
        meta = self._ps.open_project(pid)
        meta.name = new_policy_id
        self._ps.save_meta(meta)
        # Update workspace_meta.json
        ws_meta = self._load_ws_meta(pid, new_policy_id)
        ws_meta.policy_id = new_policy_id
        self._save_ws_meta(pid, ws_meta)
        # Invalidate cache
        self._pid_map.pop(old_policy_id, None)
        self._pid_map[new_policy_id] = pid

    def delete_workspace(self, policy_id: str) -> None:
        """Delete a workspace (project) entirely."""
        pid = self._resolve_project_id(policy_id)
        if not pid:
            raise FileNotFoundError(f"Workspace '{policy_id}' not found.")
        self._ps.delete_project(pid)
        self._pid_map.pop(policy_id, None)

    def workspace_exists(self, policy_id: str) -> bool:
        """Return True if a workspace for *policy_id* already exists."""
        return bool(self._resolve_project_id(policy_id))

    def ensure_workspace(self, policy_id: str) -> WorkspaceMeta:
        """Create or open a workspace for the given policy_id."""
        pid = self._resolve_project_id(policy_id)
        if pid:
            return self._load_ws_meta(pid, policy_id)
        # Create new project
        proj = self._ps.create_project(name=policy_id)
        self._pid_map[policy_id] = proj.project_id
        meta = WorkspaceMeta(
            policy_id=policy_id,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._save_ws_meta(proj.project_id, meta)
        return meta

    def load_workspace(self, policy_id: str) -> WorkspaceMeta:
        """Load workspace metadata.  Raises FileNotFoundError."""
        pid = self._resolve_project_id(policy_id)
        if not pid:
            raise FileNotFoundError(
                f"Workspace not found for policy_id={policy_id!r}. "
                "Call ensure_workspace() first."
            )
        return self._load_ws_meta(pid, policy_id)

    def _save_workspace_meta(self, meta: WorkspaceMeta) -> None:
        pid = self._resolve_project_id(meta.policy_id)
        if pid:
            self._save_ws_meta(pid, meta)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def list_experiments(self, policy_id: str) -> List[ExperimentMeta]:
        """Return experiment metadata list."""
        try:
            meta = self.load_workspace(policy_id)
        except FileNotFoundError:
            return []
        return list(meta.experiments)

    def create_experiment(
        self,
        policy_id: str,
        name: str = "",
        initial_canvas: Optional[dict] = None,
    ) -> ExperimentMeta:
        """Create a new experiment entry."""
        meta = self.ensure_workspace(policy_id)
        pid = self._resolve_project_id(policy_id)
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
        policy_id: str,
        experiment_id: str,
        canvas_data: dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist canvas snapshot for an existing experiment."""
        meta = self.ensure_workspace(policy_id)
        pid = self._resolve_project_id(policy_id)

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

    def load_experiment(self, policy_id: str, experiment_id: str) -> dict:
        """Load and return the canvas snapshot for an experiment."""
        pid = self._resolve_project_id(policy_id)
        if not pid:
            raise FileNotFoundError(
                f"Experiment {experiment_id!r} not found in workspace {policy_id!r}"
            )
        return self._ps.load_experiment(pid, experiment_id)

    def set_active_experiment(self, policy_id: str, experiment_id: str) -> None:
        """Set the active (last-opened) experiment for a workspace."""
        meta = self.load_workspace(policy_id)
        meta.active_experiment_id = experiment_id
        self._save_workspace_meta(meta)

    def rename_experiment(
        self, policy_id: str, experiment_id: str, new_name: str
    ) -> None:
        """Rename an experiment (updates workspace meta only)."""
        meta = self.load_workspace(policy_id)
        exp_meta = meta.get_experiment(experiment_id)
        if exp_meta is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        exp_meta.name = new_name
        exp_meta.updated_at = time.time()
        self._save_workspace_meta(meta)

    def delete_experiment(self, policy_id: str, experiment_id: str) -> None:
        """Delete an experiment entry and its persisted canvas snapshot."""
        meta = self.load_workspace(policy_id)
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

        pid = self._resolve_project_id(policy_id)
        if pid:
            self._ps.delete_experiment(pid, experiment_id)

        self._save_workspace_meta(meta)

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def create_run(self, policy_id: str, run_meta: "RunMeta") -> "RunMeta":
        """Persist a new run record."""
        self.ensure_workspace(policy_id)
        pid = self._resolve_project_id(policy_id)
        run_meta.policy_id = policy_id
        return self._ps.create_run(pid, run_meta)

    def update_run(self, policy_id: str, run_id: str, fields: Dict[str, Any]) -> None:
        """Update specific fields of an existing run record."""
        pid = self._resolve_project_id(policy_id)
        if not pid:
            raise FileNotFoundError(
                f"Run {run_id!r} not found in workspace {policy_id!r}"
            )
        self._ps.update_run(pid, run_id, **fields)

    def list_runs(self, policy_id: str) -> List[Dict[str, Any]]:
        """Return run metadata dicts, sorted newest-first."""
        pid = self._resolve_project_id(policy_id)
        if not pid:
            return []
        return self._ps.list_runs(pid)
