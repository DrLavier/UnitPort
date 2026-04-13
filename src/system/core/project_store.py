#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProjectStore — unified read/write entry point for project workspaces.

Layout v2 (see ``knowledge_base/project_design.yaml``)::

    projects/
        <slug>/                              # human-readable, == project_id
            README.md
            project.yaml                     # single source of truth
            .unitport/                       # tool-private cache
            canvas/
                <wf_id>.canvas.json          # mission canvases
                experiments/
                    <experiment>.canvas.json # training canvases
            behaviors/                       # IR -> Python codegen output
            scenarios/
                default.yaml
            datasets/
            resources/
                robots/<robot>/{urdf,meshes,mjcf}/
            training/
                configs/<experiment>.yaml    # serialized TrainingSpec
                runs/<experiment>__<ts>/     # one dir per training run
                    run.yaml                 # run metadata (replaces flat <run_id>.json)
                    config.snapshot.yaml
                    events.out.tfevents.*
                    checkpoints/model_*.pt
                    log.txt
                exported/<policy_id>/        # bundle: ONNX + TorchScript + manifest
                    manifest.yaml
                    policy.onnx
                    source.json
            logs/
                <date>/run_<timestamp>.log

    shared/
        checkpoints/<policy_id>/             # promoted bundles, cross-project
        bundles/<bundle_id>/

The legacy v1 layout (``project.json`` + ``workspace_meta.json`` +
``workflows/`` + flat ``checkpoints/`` + flat ``bundles/``) is still readable
for migration; converted on disk by ``scripts/migrate_project_layout.py``.

API stays compatible: every old path helper (``_workflows_dir``,
``_checkpoints_dir``, ``_bundles_dir``, ``_runs_dir``, ``_experiments_dir``)
now returns the new location, so callers — CheckpointRegistry,
bundle_exporter, AssetResolver, ProjectSession — keep working unchanged.

No Qt.  No training logic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_PROJECT_SCHEMA = "project_v2"
_LEGACY_PROJECT_SCHEMA = "project_v1"

_PROJECT_YAML = "project.yaml"
_LEGACY_PROJECT_JSON = "project.json"
_LEGACY_WORKSPACE_META = "workspace_meta.json"

_CANVAS_EXT = ".canvas.json"
_LEGACY_WORKFLOW_EXT = ".workflow.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _gen_id(prefix: str = "proj") -> str:
    """Short random id with prefix, e.g. proj_a1b2c3d4. Used for fallbacks only."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def slugify(name: str, *, fallback: str = "") -> str:
    """Convert a human-readable name to a filesystem-safe slug.

    Lowercases, replaces runs of non-alphanumerics with single underscores,
    strips leading/trailing underscores. Returns *fallback* (or a short uuid
    when *fallback* is empty) if the result would be empty.
    """
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "")).strip("_").lower()
    if not s:
        s = fallback or _gen_id("proj")
    return s


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".pw_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".pw_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f, sort_keys=False, allow_unicode=True, indent=2
            )
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# Back-compat alias for one-line JSON dumps used by other modules.
_atomic_write = _atomic_write_json


# ---------------------------------------------------------------------------
# Dataclasses — Project metadata
# ---------------------------------------------------------------------------

@dataclass
class RobotTarget:
    """Robot brand + model binding for a project."""
    brand: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return {"brand": self.brand, "model": self.model}

    @classmethod
    def from_dict(cls, d: dict) -> "RobotTarget":
        return cls(
            brand=str(d.get("brand", "")),
            model=str(d.get("model", "")),
        )


@dataclass
class WorkflowEntry:
    """Pointer to a workflow file inside the project."""
    id: str = ""
    file: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "file": self.file}

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowEntry":
        return cls(
            id=str(d.get("id", "")),
            file=str(d.get("file", "")),
        )


@dataclass
class CheckpointRef:
    """Reference to a project-local exported policy bundle."""
    policy_id: str = ""
    path: str = ""
    published: bool = False
    published_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "path": self.path,
            "published": self.published,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointRef":
        return cls(
            policy_id=str(d.get("policy_id", "")),
            path=str(d.get("path", "")),
            published=bool(d.get("published", False)),
            published_at=d.get("published_at"),
        )


@dataclass
class BundleRef:
    """Reference to an exported bundle inside the project."""
    bundle_id: str = ""
    path: str = ""

    def to_dict(self) -> dict:
        return {"bundle_id": self.bundle_id, "path": self.path}

    @classmethod
    def from_dict(cls, d: dict) -> "BundleRef":
        return cls(
            bundle_id=str(d.get("bundle_id", "")),
            path=str(d.get("path", "")),
        )


@dataclass
class AssetIndex:
    """Index of all assets tracked by a project."""
    workflows: List[WorkflowEntry] = field(default_factory=list)
    checkpoints: List[CheckpointRef] = field(default_factory=list)
    bundles: List[BundleRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "workflows": [w.to_dict() for w in self.workflows],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "bundles": [b.to_dict() for b in self.bundles],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssetIndex":
        return cls(
            workflows=[
                WorkflowEntry.from_dict(w) for w in d.get("workflows", [])
            ],
            checkpoints=[
                CheckpointRef.from_dict(c) for c in d.get("checkpoints", [])
            ],
            bundles=[
                BundleRef.from_dict(b) for b in d.get("bundles", [])
            ],
        )

    def find_workflow(self, wf_id: str) -> Optional[WorkflowEntry]:
        for w in self.workflows:
            if w.id == wf_id:
                return w
        return None

    def find_checkpoint(self, policy_id: str) -> Optional[CheckpointRef]:
        for c in self.checkpoints:
            if c.policy_id == policy_id:
                return c
        return None

    def find_bundle(self, bundle_id: str) -> Optional[BundleRef]:
        for b in self.bundles:
            if b.bundle_id == bundle_id:
                return b
        return None


@dataclass
class ExperimentMeta:
    """Metadata for a training experiment within a project."""
    experiment_id: str = ""           # == name in v2; kept for v1 compat
    name: str = ""
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
class ProjectMeta:
    """Root metadata for a project workspace, persisted as project.yaml."""
    schema_version: str = _PROJECT_SCHEMA
    project_id: str = ""               # == filesystem slug
    uuid: str = ""                     # stable cross-system id
    name: str = ""                     # human-readable display name
    created_at: str = ""
    updated_at: str = ""
    robot: RobotTarget = field(default_factory=RobotTarget)
    assets: AssetIndex = field(default_factory=AssetIndex)
    active_experiment: str = ""
    experiments: List[ExperimentMeta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "uuid": self.uuid,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "robot": self.robot.to_dict(),
            "assets": self.assets.to_dict(),
            "active_experiment": self.active_experiment,
            "experiments": [e.to_dict() for e in self.experiments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectMeta":
        return cls(
            schema_version=str(d.get("schema_version", _PROJECT_SCHEMA)),
            project_id=str(d.get("project_id", "")),
            uuid=str(d.get("uuid", "")),
            name=str(d.get("name", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            robot=RobotTarget.from_dict(d.get("robot", {})),
            assets=AssetIndex.from_dict(d.get("assets", {})),
            active_experiment=str(
                d.get("active_experiment")
                or d.get("active_experiment_id", "")
            ),
            experiments=[
                ExperimentMeta.from_dict(e)
                for e in d.get("experiments", [])
            ],
        )

    def get_experiment(self, exp_id: str) -> Optional[ExperimentMeta]:
        for e in self.experiments:
            if e.experiment_id == exp_id or e.name == exp_id:
                return e
        return None

    def save(self, path: Path) -> None:
        """Persist to project.yaml atomically."""
        _atomic_write_yaml(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "ProjectMeta":
        """Load from project.yaml or legacy project.json."""
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Dataclasses — Training run records
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    """Metadata for a single training run."""
    run_id: str = ""
    policy_id: str = ""
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
class PublishResult:
    """Result of publishing a checkpoint to shared space."""
    action: str = ""      # "created" | "overwritten" | "versioned"
    shared_path: Path = field(default_factory=Path)
    policy_id: str = ""


# ---------------------------------------------------------------------------
# ProjectStore
# ---------------------------------------------------------------------------

class ProjectStore:
    """
    Unified read/write entry point for project workspaces.

    The ``project_id`` value used by every method is the project's
    *filesystem slug* — e.g. ``"go2_walk_v1"``. The slug doubles as the
    directory name under ``projects/``. A separate ``uuid`` field inside
    ``project.yaml`` provides a stable cross-system identifier.

    Parameters
    ----------
    root : Path or str, optional
        Base directory containing ``projects/`` and ``shared/`` subdirs.
        Defaults to ``<src_parent>/`` (i.e. the repo root).
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            # Default: repo root (parent of src/)
            root = Path(__file__).resolve().parent.parent.parent.parent
        self._root = Path(root)
        self._projects_dir = self._root / "projects"
        self._shared_dir = self._root / "shared"

    # -- properties ----------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def projects_dir(self) -> Path:
        return self._projects_dir

    @property
    def shared_dir(self) -> Path:
        return self._shared_dir

    # -- path helpers (v2 layout) -------------------------------------------

    def _project_dir(self, project_id: str) -> Path:
        return self._projects_dir / project_id

    def _project_yaml(self, project_id: str) -> Path:
        return self._project_dir(project_id) / _PROJECT_YAML

    def _legacy_project_json(self, project_id: str) -> Path:
        return self._project_dir(project_id) / _LEGACY_PROJECT_JSON

    def _legacy_workspace_meta(self, project_id: str) -> Path:
        return self._project_dir(project_id) / _LEGACY_WORKSPACE_META

    # canvas
    def _canvas_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "canvas"

    def _canvas_experiments_dir(self, project_id: str) -> Path:
        return self._canvas_dir(project_id) / "experiments"

    # behaviors / scenarios / datasets / resources
    def _behaviors_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "behaviors"

    def _scenarios_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "scenarios"

    def _datasets_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "datasets"

    def _resources_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "resources"

    # training
    def _training_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "training"

    def _training_configs_dir(self, project_id: str) -> Path:
        return self._training_dir(project_id) / "configs"

    def _training_runs_dir(self, project_id: str) -> Path:
        return self._training_dir(project_id) / "runs"

    def _training_exported_dir(self, project_id: str) -> Path:
        return self._training_dir(project_id) / "exported"

    # runtime logs
    def _runtime_logs_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "logs"

    # -- legacy path aliases (kept so CheckpointRegistry / bundle_exporter /
    #    AssetResolver / project_session do not need to change) -------------

    def _workflows_dir(self, project_id: str) -> Path:
        """Alias: legacy ``workflows/`` -> new ``canvas/``."""
        return self._canvas_dir(project_id)

    def _experiments_dir(self, project_id: str) -> Path:
        """Alias: legacy ``training/experiments/`` -> new ``canvas/experiments/``."""
        return self._canvas_experiments_dir(project_id)

    def _runs_dir(self, project_id: str) -> Path:
        """Alias: legacy ``training/runs/`` (path unchanged in v2)."""
        return self._training_runs_dir(project_id)

    def _checkpoints_dir(self, project_id: str) -> Path:
        """Alias: legacy top-level ``checkpoints/`` -> new ``training/exported/``.

        In v2 a "checkpoint" (= an exported policy bundle) lives inside the
        ``training/exported/<policy_id>/`` directory. CheckpointRegistry and
        bundle_exporter both treat this dir as a flat list of bundles, so the
        alias is a drop-in replacement.
        """
        return self._training_exported_dir(project_id)

    def _bundles_dir(self, project_id: str) -> Path:
        """Alias: legacy top-level ``bundles/`` -> new ``training/exported/``."""
        return self._training_exported_dir(project_id)

    def _shared_checkpoints_dir(self) -> Path:
        return self._shared_dir / "checkpoints"

    def _shared_bundles_dir(self) -> Path:
        return self._shared_dir / "bundles"

    # -- project lifecycle ---------------------------------------------------

    def _scaffold_dirs(self, project_id: str) -> None:
        """Create the v2 directory tree under projects/<project_id>/."""
        for sub in [
            self._canvas_dir(project_id),
            self._canvas_experiments_dir(project_id),
            self._behaviors_dir(project_id),
            self._scenarios_dir(project_id),
            self._datasets_dir(project_id),
            self._resources_dir(project_id) / "robots",
            self._training_configs_dir(project_id),
            self._training_runs_dir(project_id),
            self._training_exported_dir(project_id),
            self._runtime_logs_dir(project_id),
            self._project_dir(project_id) / ".unitport",
        ]:
            sub.mkdir(parents=True, exist_ok=True)

    def _write_readme(self, project_id: str, name: str) -> None:
        readme = self._project_dir(project_id) / "README.md"
        if readme.exists():
            return
        readme.write_text(
            f"# {name}\n\n"
            f"UnitPort project — created {_now_iso()}\n\n"
            f"See `project.yaml` for metadata. Layout follows the v2 spec\n"
            f"in `knowledge_base/project_design.yaml`.\n",
            encoding="utf-8",
        )

    def create_project(
        self,
        name: str,
        robot_brand: str = "",
        robot_model: str = "",
    ) -> ProjectMeta:
        """Create a new project with v2 directory layout.

        The slugified *name* becomes the directory name AND the project_id.
        If a project with the same slug already exists, a short uuid suffix
        is appended.
        """
        slug = slugify(name)
        if (self._projects_dir / slug).exists():
            slug = f"{slug}_{uuid.uuid4().hex[:6]}"
        project_id = slug
        now = _now_iso()

        meta = ProjectMeta(
            schema_version=_PROJECT_SCHEMA,
            project_id=project_id,
            uuid=uuid.uuid4().hex,
            name=name,
            created_at=now,
            updated_at=now,
            robot=RobotTarget(brand=robot_brand, model=robot_model),
            assets=AssetIndex(),
        )

        self._scaffold_dirs(project_id)
        self._write_readme(project_id, name)
        meta.save(self._project_yaml(project_id))
        return meta

    def open_project(self, project_id: str) -> ProjectMeta:
        """Load an existing project. Tries v2 (project.yaml), then v1 (project.json)."""
        py = self._project_yaml(project_id)
        if py.exists():
            return ProjectMeta.load(py)
        # Legacy v1 fallback
        pj = self._legacy_project_json(project_id)
        if pj.exists():
            meta = ProjectMeta.load(pj)
            meta.schema_version = _LEGACY_PROJECT_SCHEMA
            return meta
        raise FileNotFoundError(
            f"Project not found: {project_id} (no project.yaml or project.json under {self._project_dir(project_id)})"
        )

    def list_projects(self) -> List[ProjectMeta]:
        """Discover all projects under projects/. Reads v2 first, falls back to v1."""
        results: List[ProjectMeta] = []
        if not self._projects_dir.exists():
            return results
        for entry in sorted(self._projects_dir.iterdir()):
            if not entry.is_dir():
                continue
            py = entry / _PROJECT_YAML
            pj = entry / _LEGACY_PROJECT_JSON
            try:
                if py.exists():
                    meta = ProjectMeta.load(py)
                elif pj.exists():
                    meta = ProjectMeta.load(pj)
                    meta.schema_version = _LEGACY_PROJECT_SCHEMA
                else:
                    continue
                # Backfill project_id from directory name when missing
                if not meta.project_id:
                    meta.project_id = entry.name
                results.append(meta)
            except (json.JSONDecodeError, yaml.YAMLError, KeyError, OSError):
                continue
        return results

    def delete_project(self, project_id: str) -> None:
        """Remove a project directory entirely.

        Does NOT remove any published shared assets.
        """
        proj_dir = self._project_dir(project_id)
        if proj_dir.exists():
            shutil.rmtree(proj_dir)

    def save_meta(self, meta: ProjectMeta) -> None:
        """Persist updated ProjectMeta back to project.yaml."""
        meta.updated_at = _now_iso()
        meta.schema_version = _PROJECT_SCHEMA
        meta.save(self._project_yaml(meta.project_id))

    # -- workflow persistence ------------------------------------------------

    def save_workflow(
        self,
        project_id: str,
        workflow_id: str,
        canvas_data: dict,
    ) -> Path:
        """Save a workflow canvas to ``canvas/<workflow_id>.canvas.json``.

        Updates the asset index in project.yaml.
        Returns the absolute path of the saved file.
        """
        canvas_dir = self._canvas_dir(project_id)
        canvas_dir.mkdir(parents=True, exist_ok=True)
        rel_file = f"canvas/{workflow_id}{_CANVAS_EXT}"
        abs_path = self._project_dir(project_id) / rel_file

        _atomic_write_json(abs_path, canvas_data)

        # If a legacy .workflow.json sits alongside, drop it now that the
        # canonical file has been written.
        legacy = canvas_dir / f"{workflow_id}{_LEGACY_WORKFLOW_EXT}"
        if legacy.exists() and legacy != abs_path:
            try:
                legacy.unlink()
            except OSError:
                pass

        # Update asset index
        meta = self.open_project(project_id)
        existing = meta.assets.find_workflow(workflow_id)
        if existing is None:
            meta.assets.workflows.append(
                WorkflowEntry(id=workflow_id, file=rel_file)
            )
        else:
            existing.file = rel_file
        self.save_meta(meta)
        return abs_path

    def load_workflow(self, project_id: str, workflow_id: str) -> dict:
        """Load a workflow canvas dict.

        Reads ``canvas/<id>.canvas.json`` first, then falls back to the legacy
        ``canvas/<id>.workflow.json`` (and the even older ``workflows/`` dir).
        Raises FileNotFoundError if neither exists.
        """
        canvas_dir = self._canvas_dir(project_id)
        candidates = [
            canvas_dir / f"{workflow_id}{_CANVAS_EXT}",
            canvas_dir / f"{workflow_id}{_LEGACY_WORKFLOW_EXT}",
            self._project_dir(project_id) / "workflows" / f"{workflow_id}{_LEGACY_WORKFLOW_EXT}",
        ]
        for cand in candidates:
            if cand.exists():
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"Workflow not found: {workflow_id} in project {project_id}"
        )

    def list_workflows(self, project_id: str) -> List[WorkflowEntry]:
        """Return all workflow entries registered in the project."""
        meta = self.open_project(project_id)
        return list(meta.assets.workflows)

    def workflow_path(self, project_id: str, workflow_id: str) -> Path:
        """Return the canonical filesystem path for a workflow id.

        Resolution order (returns the first that exists; if none exists,
        returns the v2 canonical path so the caller can use it as a save
        target):

          1. ``canvas/<id>.canvas.json``       (v2 canonical)
          2. ``canvas/<id>.workflow.json``     (legacy extension, in v2 dir)
          3. ``workflows/<id>.workflow.json``  (legacy v1 directory)
        """
        canvas_dir = self._canvas_dir(project_id)
        candidates = [
            canvas_dir / f"{workflow_id}{_CANVAS_EXT}",
            canvas_dir / f"{workflow_id}{_LEGACY_WORKFLOW_EXT}",
            self._project_dir(project_id) / "workflows" / f"{workflow_id}{_LEGACY_WORKFLOW_EXT}",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    @staticmethod
    def workflow_id_from_path(path: str) -> str:
        """Extract the workflow id from a *.canvas.json or *.workflow.json filename."""
        base = os.path.basename(str(path))
        if base.endswith(_CANVAS_EXT):
            return base[: -len(_CANVAS_EXT)]
        if base.endswith(_LEGACY_WORKFLOW_EXT):
            return base[: -len(_LEGACY_WORKFLOW_EXT)]
        return ""

    # -- training experiments ------------------------------------------------

    def create_experiment(
        self,
        project_id: str,
        name: str = "",
        canvas: Optional[dict] = None,
    ) -> ExperimentMeta:
        """Create a new training experiment in the project.

        In v2 the experiment_id == name (slugified). The canvas is stored at
        ``canvas/experiments/<experiment_id>.canvas.json`` and an entry is
        added to ``project.yaml``.
        """
        meta = self.open_project(project_id)
        exp_id = slugify(name) if name else _gen_id("exp")
        # Disambiguate against existing experiments
        existing_ids = {e.experiment_id for e in meta.experiments}
        base = exp_id
        n = 2
        while exp_id in existing_ids:
            exp_id = f"{base}_{n}"
            n += 1

        now = time.time()
        exp = ExperimentMeta(
            experiment_id=exp_id,
            name=name or exp_id,
            created_at=now,
            updated_at=now,
        )

        # Save canvas if provided
        exp_dir = self._canvas_experiments_dir(project_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        if canvas is not None:
            canvas_path = exp_dir / f"{exp_id}{_CANVAS_EXT}"
            _atomic_write_json(canvas_path, canvas)

        meta.experiments.append(exp)
        if not meta.active_experiment:
            meta.active_experiment = exp_id
        self.save_meta(meta)
        return exp

    def save_experiment(
        self,
        project_id: str,
        exp_id: str,
        canvas: dict,
    ) -> None:
        """Save/overwrite a training experiment canvas."""
        exp_dir = self._canvas_experiments_dir(project_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        canvas_path = exp_dir / f"{exp_id}{_CANVAS_EXT}"
        _atomic_write_json(canvas_path, canvas)
        # Drop any legacy filename for the same id
        legacy = exp_dir / f"{exp_id}{_LEGACY_WORKFLOW_EXT}"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass
        # Touch experiment metadata
        meta = self.open_project(project_id)
        exp = meta.get_experiment(exp_id)
        if exp is None:
            exp = ExperimentMeta(
                experiment_id=exp_id, name=exp_id, created_at=time.time()
            )
            meta.experiments.append(exp)
        exp.updated_at = time.time()
        self.save_meta(meta)

    def load_experiment(self, project_id: str, exp_id: str) -> dict:
        """Load a training experiment canvas. Raises FileNotFoundError."""
        exp_dir = self._canvas_experiments_dir(project_id)
        for cand in [
            exp_dir / f"{exp_id}{_CANVAS_EXT}",
            exp_dir / f"{exp_id}{_LEGACY_WORKFLOW_EXT}",
        ]:
            if cand.exists():
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"Experiment not found: {exp_id} in project {project_id}"
        )

    def list_experiments(self, project_id: str) -> List[ExperimentMeta]:
        """Return experiment metadata list from project.yaml."""
        try:
            meta = self.open_project(project_id)
        except FileNotFoundError:
            return []
        if meta.experiments:
            return list(meta.experiments)
        # Fallback: scan canvas/experiments/ for orphaned canvases
        exp_dir = self._canvas_experiments_dir(project_id)
        if not exp_dir.exists():
            return []
        results: List[ExperimentMeta] = []
        for fp in sorted(exp_dir.iterdir()):
            if fp.is_file() and (
                fp.name.endswith(_CANVAS_EXT)
                or fp.name.endswith(_LEGACY_WORKFLOW_EXT)
            ):
                exp_id = fp.name.replace(_CANVAS_EXT, "").replace(
                    _LEGACY_WORKFLOW_EXT, ""
                )
                results.append(ExperimentMeta(
                    experiment_id=exp_id,
                    name=exp_id,
                    created_at=fp.stat().st_ctime,
                    updated_at=fp.stat().st_mtime,
                ))
        return results

    def delete_experiment(self, project_id: str, exp_id: str) -> None:
        """Delete an experiment canvas file and remove it from project.yaml."""
        exp_dir = self._canvas_experiments_dir(project_id)
        for cand in [
            exp_dir / f"{exp_id}{_CANVAS_EXT}",
            exp_dir / f"{exp_id}{_LEGACY_WORKFLOW_EXT}",
        ]:
            if cand.exists():
                try:
                    cand.unlink()
                except OSError:
                    pass
        try:
            meta = self.open_project(project_id)
        except FileNotFoundError:
            return
        meta.experiments = [
            e for e in meta.experiments if e.experiment_id != exp_id
        ]
        if meta.active_experiment == exp_id:
            meta.active_experiment = (
                meta.experiments[0].experiment_id if meta.experiments else ""
            )
        self.save_meta(meta)

    def rename_experiment(
        self,
        project_id: str,
        exp_id: str,
        new_name: str,
    ) -> None:
        """Rename an experiment's display name (file id stays stable)."""
        meta = self.open_project(project_id)
        exp = meta.get_experiment(exp_id)
        if exp is None:
            return
        exp.name = new_name
        exp.updated_at = time.time()
        self.save_meta(meta)

    # -- training runs -------------------------------------------------------

    def _run_dir(self, project_id: str, run_id: str) -> Path:
        return self._training_runs_dir(project_id) / run_id

    def _run_yaml(self, project_id: str, run_id: str) -> Path:
        return self._run_dir(project_id, run_id) / "run.yaml"

    def create_run(self, project_id: str, run_meta: RunMeta) -> RunMeta:
        """Persist a new training run record at runs/<run_id>/run.yaml."""
        run_dir = self._run_dir(project_id, run_meta.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        _atomic_write_yaml(self._run_yaml(project_id, run_meta.run_id), run_meta.to_dict())
        return run_meta

    def update_run(
        self,
        project_id: str,
        run_id: str,
        **fields: Any,
    ) -> None:
        """Partial update of a training run record."""
        run_yaml = self._run_yaml(project_id, run_id)
        # Legacy fallback: flat <run_id>.json under runs/
        legacy_json = self._training_runs_dir(project_id) / f"{run_id}.json"
        if run_yaml.exists():
            with open(run_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data.update(fields)
            _atomic_write_yaml(run_yaml, data)
            return
        if legacy_json.exists():
            with open(legacy_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.update(fields)
            _atomic_write_json(legacy_json, data)
            return
        raise FileNotFoundError(
            f"Run not found: {run_id} in project {project_id}"
        )

    def list_runs(self, project_id: str) -> List[dict]:
        """List all training runs, sorted newest-first.

        Reads run.yaml from each subdir, plus legacy flat <run_id>.json
        records (kept readable so historical UI views still work).
        """
        runs_dir = self._training_runs_dir(project_id)
        if not runs_dir.exists():
            return []
        results: List[dict] = []
        for entry in runs_dir.iterdir():
            try:
                if entry.is_dir():
                    ry = entry / "run.yaml"
                    if ry.exists():
                        with open(ry, "r", encoding="utf-8") as f:
                            results.append(yaml.safe_load(f) or {})
                elif entry.suffix == ".json":
                    with open(entry, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
            except (json.JSONDecodeError, yaml.YAMLError, OSError):
                continue
        results.sort(key=lambda d: d.get("created_at", 0), reverse=True)
        return results

    # -- checkpoint storage (project-local exported bundles) ----------------

    def save_checkpoint(
        self,
        project_id: str,
        policy_id: str,
        artifacts: Dict[str, Any],
    ) -> Path:
        """Register an exported policy bundle in the project.

        Writes to ``training/exported/<policy_id>/`` (the v2 location;
        ``_checkpoints_dir`` is now an alias for that path).

        ``artifacts`` is a dict mapping filename → content:
          - For binary files (bytes): written as-is
          - For dict/list: written as JSON
          - For str: written as text

        Returns the bundle directory path.
        """
        ckpt_dir = self._training_exported_dir(project_id) / policy_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in artifacts.items():
            out_path = ckpt_dir / filename
            if isinstance(content, bytes):
                out_path.write_bytes(content)
            elif isinstance(content, (dict, list)):
                _atomic_write_json(out_path, content)
            else:
                out_path.write_text(str(content), encoding="utf-8")

        # Update asset index
        meta = self.open_project(project_id)
        existing = meta.assets.find_checkpoint(policy_id)
        if existing is None:
            rel_path = f"training/exported/{policy_id}/"
            meta.assets.checkpoints.append(
                CheckpointRef(policy_id=policy_id, path=rel_path)
            )
            self.save_meta(meta)

        return ckpt_dir

    def get_checkpoint(
        self,
        project_id: str,
        policy_id: str,
    ) -> Optional[Path]:
        """Return the bundle directory path, or None if not found."""
        ckpt_dir = self._training_exported_dir(project_id) / policy_id
        if ckpt_dir.is_dir():
            return ckpt_dir
        return None

    # -- publish to shared ---------------------------------------------------

    def publish_to_shared(
        self,
        project_id: str,
        policy_id: str,
        action: str = "created",
    ) -> PublishResult:
        """Copy a project-local exported bundle to ``shared/checkpoints/``.

        Parameters
        ----------
        project_id : str
            Source project.
        policy_id : str
            Policy to publish.
        action : str
            One of ``"created"``, ``"overwritten"``, ``"versioned"``.
            - ``"created"`` or ``"overwritten"``: target is ``shared/checkpoints/<policy_id>/``
            - ``"versioned"``: find next ``_v{N}`` suffix and create new entry.

        Returns
        -------
        PublishResult
            With the action taken, shared path, and final policy_id.
        """
        src_dir = self._training_exported_dir(project_id) / policy_id
        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint '{policy_id}' not found in project '{project_id}'"
            )

        shared_ckpts = self._shared_checkpoints_dir()
        shared_ckpts.mkdir(parents=True, exist_ok=True)

        if action == "versioned":
            final_pid = self._next_version_id(shared_ckpts, policy_id)
        else:
            final_pid = policy_id

        dest_dir = shared_ckpts / final_pid

        # Copy (overwrite if exists)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

        # Inject published_from into source.json in the shared copy
        source_json = dest_dir / "source.json"
        source_data: dict = {}
        if source_json.exists():
            try:
                with open(source_json, "r", encoding="utf-8") as f:
                    source_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        source_data["published_from"] = {
            "project_id": project_id,
            "original_policy_id": policy_id,
            "timestamp": _now_iso(),
        }
        _atomic_write_json(source_json, source_data)

        # Update project.yaml published flag
        meta = self.open_project(project_id)
        ckpt_ref = meta.assets.find_checkpoint(policy_id)
        if ckpt_ref is not None:
            ckpt_ref.published = True
            ckpt_ref.published_at = _now_iso()
            self.save_meta(meta)

        return PublishResult(
            action=action,
            shared_path=dest_dir,
            policy_id=final_pid,
        )

    @staticmethod
    def _next_version_id(shared_dir: Path, base_id: str) -> str:
        """Find the next available _v{N} suffix for a policy_id."""
        n = 2
        while True:
            candidate = f"{base_id}_v{n}"
            if not (shared_dir / candidate).exists():
                return candidate
            n += 1
