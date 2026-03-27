#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — Workspace persistence layer.

Directory layout::

    training_workspaces/
        <policy_id>/
            workspace.json          # workspace metadata + active experiment id
            experiments/
                <experiment_id>.canvas.json   # canvas snapshot
            runs/
                <run_id>.json                 # run metadata (Phase C writer)

API::

    store = TrainingWorkspaceStore(root)
    store.ensure_workspace(policy_id)
    meta  = store.load_workspace(policy_id)
    exps  = store.list_experiments(policy_id)
    store.save_experiment(policy_id, experiment_id, canvas_data, metadata)
    canvas= store.load_experiment(policy_id, experiment_id)
    info  = store.create_experiment(policy_id, name)
    runs  = store.list_runs(policy_id)

Atomic writes: JSON is written to a temp file in the same directory, then
renamed over the target.  This avoids partial writes corrupting saved data.

No Qt.  No real training.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_WORKSPACE_SCHEMA = "training_workspace_v1"
_CANVAS_SCHEMA    = "training_canvas_v1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    """Lightweight metadata stored in workspace.json's experiment index."""
    experiment_id: str
    name: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentMeta":
        return cls(
            experiment_id=str(d.get("experiment_id", "")),
            name=str(d.get("name", "")),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
        )


@dataclass
class RunMeta:
    """Metadata for a single training run, persisted as runs/<run_id>.json."""
    run_id: str
    policy_id: str
    experiment_id: str = ""
    experiment_name: str = ""
    status: str = "queued"   # queued | running | finished | cancelled | error
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancelled_at: Optional[float] = None
    algorithm: str = ""
    total_timesteps: int = 0
    policy_id_out: str = ""
    bundle_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "policy_id": self.policy_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancelled_at": self.cancelled_at,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "policy_id_out": self.policy_id_out,
            "bundle_path": self.bundle_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunMeta":
        return cls(
            run_id=str(d.get("run_id", "")),
            policy_id=str(d.get("policy_id", "")),
            experiment_id=str(d.get("experiment_id", "")),
            experiment_name=str(d.get("experiment_name", "")),
            status=str(d.get("status", "queued")),
            created_at=float(d.get("created_at", 0.0)),
            started_at=float(d["started_at"]) if d.get("started_at") is not None else None,
            finished_at=float(d["finished_at"]) if d.get("finished_at") is not None else None,
            cancelled_at=float(d["cancelled_at"]) if d.get("cancelled_at") is not None else None,
            algorithm=str(d.get("algorithm", "")),
            total_timesteps=int(d.get("total_timesteps", 0)),
            policy_id_out=str(d.get("policy_id_out", "")),
            bundle_path=str(d.get("bundle_path", "")),
            error=str(d.get("error", "")),
        )


@dataclass
class WorkspaceMeta:
    """Contents of workspace.json."""
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
# Store
# ---------------------------------------------------------------------------

class TrainingWorkspaceStore:
    """
    Persist and retrieve training workspace data.

    Parameters
    ----------
    root:
        Base directory under which ``training_workspaces/`` lives.
        Defaults to the current working directory.
    """

    def __init__(self, root: Optional[str] = None):
        if root is None:
            root = os.getcwd()
        self._root = Path(root) / "training_workspaces"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _workspace_dir(self, policy_id: str) -> Path:
        return self._root / policy_id

    def _workspace_json(self, policy_id: str) -> Path:
        return self._workspace_dir(policy_id) / "workspace.json"

    def _experiments_dir(self, policy_id: str) -> Path:
        return self._workspace_dir(policy_id) / "experiments"

    def _canvas_json(self, policy_id: str, experiment_id: str) -> Path:
        return self._experiments_dir(policy_id) / f"{experiment_id}.canvas.json"

    def _runs_dir(self, policy_id: str) -> Path:
        return self._workspace_dir(policy_id) / "runs"

    # ------------------------------------------------------------------
    # Atomic write helper
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, data: dict) -> None:
        """Write *data* as JSON to *path* atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            # Atomic rename
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Workspace lifecycle
    # ------------------------------------------------------------------

    def list_workspaces(self) -> List[str]:
        """Return all workspace policy_ids found under training_workspaces/."""
        if not self._root.exists():
            return []
        result = []
        for p in sorted(self._root.iterdir()):
            if p.is_dir() and (p / "workspace.json").exists():
                result.append(p.name)
        return result

    def rename_workspace(self, old_policy_id: str, new_policy_id: str) -> None:
        """
        Rename a workspace directory and update workspace.json policy_id.

        Raises ValueError if new_policy_id already exists.
        Raises FileNotFoundError if old_policy_id does not exist.
        """
        old_dir = self._workspace_dir(old_policy_id)
        new_dir = self._workspace_dir(new_policy_id)
        if not old_dir.exists():
            raise FileNotFoundError(f"Workspace '{old_policy_id}' not found.")
        if new_dir.exists():
            raise ValueError(f"Workspace '{new_policy_id}' already exists.")
        old_dir.rename(new_dir)
        ws_json = new_dir / "workspace.json"
        if ws_json.exists():
            with open(ws_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["policy_id"] = new_policy_id
            self._atomic_write(ws_json, data)

    def delete_workspace(self, policy_id: str) -> None:
        """Delete a workspace directory and all persisted experiments/runs within it."""
        ws_dir = self._workspace_dir(policy_id)
        if not ws_dir.exists():
            raise FileNotFoundError(f"Workspace '{policy_id}' not found.")
        shutil.rmtree(ws_dir)

    def ensure_workspace(self, policy_id: str) -> WorkspaceMeta:
        """
        Create the workspace directory structure if it does not exist.
        Returns the (possibly newly created) WorkspaceMeta.
        """
        ws_json = self._workspace_json(policy_id)
        if ws_json.exists():
            return self.load_workspace(policy_id)

        meta = WorkspaceMeta(policy_id=policy_id, created_at=time.time(),
                             updated_at=time.time())
        self._workspace_dir(policy_id).mkdir(parents=True, exist_ok=True)
        self._experiments_dir(policy_id).mkdir(parents=True, exist_ok=True)
        self._runs_dir(policy_id).mkdir(parents=True, exist_ok=True)
        self._atomic_write(ws_json, meta.to_dict())
        return meta

    def load_workspace(self, policy_id: str) -> WorkspaceMeta:
        """
        Load workspace metadata from disk.

        Raises
        ------
        FileNotFoundError
            If the workspace does not exist (call ensure_workspace first).
        """
        ws_json = self._workspace_json(policy_id)
        if not ws_json.exists():
            raise FileNotFoundError(
                f"Workspace not found for policy_id={policy_id!r}. "
                "Call ensure_workspace() first."
            )
        with open(ws_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return WorkspaceMeta.from_dict(data)

    def _save_workspace_meta(self, meta: WorkspaceMeta) -> None:
        meta.updated_at = time.time()
        self._atomic_write(self._workspace_json(meta.policy_id), meta.to_dict())

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def list_experiments(self, policy_id: str) -> List[ExperimentMeta]:
        """
        Return experiment metadata list from workspace.json.
        Returns [] if the workspace does not exist.
        """
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
        """
        Create a new experiment entry.

        Parameters
        ----------
        policy_id:
            The workspace to add the experiment to.
        name:
            Human-readable experiment name.  Defaults to
            ``"Experiment <n>"`` where ``n`` is the current count + 1.
        initial_canvas:
            Optional canvas snapshot to persist immediately.  When None
            an empty training_canvas_v1 snapshot is stored.

        Returns
        -------
        ExperimentMeta of the newly created experiment.
        """
        meta = self.ensure_workspace(policy_id)
        if not name:
            name = f"Experiment {len(meta.experiments) + 1}"
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        exp_meta = ExperimentMeta(
            experiment_id=experiment_id,
            name=name,
            created_at=time.time(),
            updated_at=time.time(),
        )
        meta.experiments.append(exp_meta)
        if not meta.active_experiment_id:
            meta.active_experiment_id = experiment_id

        # Persist canvas
        canvas_data = initial_canvas or {
            "schema": _CANVAS_SCHEMA,
            "nodes": [],
            "edges": [],
        }
        self._experiments_dir(policy_id).mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._canvas_json(policy_id, experiment_id), canvas_data)

        self._save_workspace_meta(meta)
        return exp_meta

    def save_experiment(
        self,
        policy_id: str,
        experiment_id: str,
        canvas_data: dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist canvas snapshot for an existing experiment.

        Updates the experiment's ``updated_at`` timestamp in workspace.json.
        If the experiment_id does not exist yet it is created.

        Parameters
        ----------
        policy_id:
            Target workspace.
        experiment_id:
            Experiment to update.
        canvas_data:
            Output of ``TrainingGraphScene.serialize_training_graph()``.
        metadata:
            Optional extra metadata merged into the canvas JSON under
            ``"_meta"`` key.  Not required; ignored when None.
        """
        meta = self.ensure_workspace(policy_id)

        # Embed optional metadata
        payload = dict(canvas_data)
        if metadata:
            payload["_meta"] = metadata

        self._experiments_dir(policy_id).mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._canvas_json(policy_id, experiment_id), payload)

        # Update timestamps
        exp_meta = meta.get_experiment(experiment_id)
        if exp_meta is None:
            # Auto-register experiment if not yet tracked
            exp_meta = ExperimentMeta(
                experiment_id=experiment_id,
                name=experiment_id,
                created_at=time.time(),
            )
            meta.experiments.append(exp_meta)
        exp_meta.updated_at = time.time()
        self._save_workspace_meta(meta)

    def load_experiment(self, policy_id: str, experiment_id: str) -> dict:
        """
        Load and return the canvas snapshot for *experiment_id*.

        Raises
        ------
        FileNotFoundError
            If the canvas file does not exist.
        """
        path = self._canvas_json(policy_id, experiment_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Experiment {experiment_id!r} not found in workspace {policy_id!r}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def set_active_experiment(self, policy_id: str, experiment_id: str) -> None:
        """Set the active (last-opened) experiment for a workspace."""
        meta = self.load_workspace(policy_id)
        meta.active_experiment_id = experiment_id
        self._save_workspace_meta(meta)

    def rename_experiment(
        self, policy_id: str, experiment_id: str, new_name: str
    ) -> None:
        """Rename an experiment (updates workspace.json only; file name unchanged)."""
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

        meta.experiments = [e for e in meta.experiments if e.experiment_id != experiment_id]
        if meta.active_experiment_id == experiment_id:
            meta.active_experiment_id = meta.experiments[0].experiment_id if meta.experiments else ""

        path = self._canvas_json(policy_id, experiment_id)
        if path.exists():
            path.unlink()

        self._save_workspace_meta(meta)

    # ------------------------------------------------------------------
    # Run history (Phase C — full read/write API)
    # ------------------------------------------------------------------

    def create_run(self, policy_id: str, run_meta: "RunMeta") -> "RunMeta":
        """
        Persist a new run record atomically under runs/<run_id>.json.

        Parameters
        ----------
        policy_id:
            Target workspace.
        run_meta:
            Populated RunMeta instance (run_id must be set).

        Returns
        -------
        The same RunMeta that was written.
        """
        self.ensure_workspace(policy_id)
        run_meta.policy_id = policy_id
        path = self._runs_dir(policy_id) / f"{run_meta.run_id}.json"
        self._atomic_write(path, run_meta.to_dict())
        return run_meta

    def update_run(self, policy_id: str, run_id: str, fields: Dict[str, Any]) -> None:
        """
        Update specific fields of an existing run record atomically.

        Parameters
        ----------
        policy_id:
            Workspace containing the run.
        run_id:
            ID of the run to update.
        fields:
            Dict of field names → new values to merge into the stored record.

        Raises
        ------
        FileNotFoundError
            If the run JSON does not exist.
        """
        path = self._runs_dir(policy_id) / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Run {run_id!r} not found in workspace {policy_id!r}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.update(fields)
        self._atomic_write(path, data)

    def list_runs(self, policy_id: str) -> List[Dict[str, Any]]:
        """
        Return run metadata dicts for *policy_id*, sorted newest-first
        by ``created_at``.

        Returns [] if the workspace or runs/ directory does not exist.
        """
        runs_dir = self._runs_dir(policy_id)
        if not runs_dir.exists():
            return []
        results = []
        for p in runs_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    results.append(json.load(fh))
            except Exception:
                pass
        results.sort(key=lambda r: r.get("created_at", 0.0), reverse=True)
        return results
