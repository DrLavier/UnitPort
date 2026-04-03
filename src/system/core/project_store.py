#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProjectStore — unified read/write entry point for project workspaces.

Replaces the disconnected Mission (.unitport) + Training (workspace/experiments)
dual structure with a single project-scoped workspace.

Directory layout::

    projects/
        <project_id>/
            project.json                    # metadata + asset index
            workflows/
                <wf_id>.workflow.json       # mission canvases
            training/
                experiments/
                    <exp_id>.canvas.json    # training canvases
                runs/
                    <run_id>.json           # training run records
            checkpoints/
                <policy_id>/
                    manifest.yaml
                    policy.onnx
                    source.json
            bundles/
                <bundle_id>/

    shared/
        checkpoints/
            <policy_id>/
        bundles/
            <bundle_id>/

API::

    store = ProjectStore(root)
    meta  = store.create_project("Go2 Walk", "unitree", "go2")
    meta  = store.open_project(project_id)
    store.list_projects()
    store.delete_project(project_id)

Atomic writes: JSON is written to a temp file in the same directory,
then renamed over the target.  Prevents partial-write corruption.

No Qt.  No training logic.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_PROJECT_SCHEMA = "project_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _gen_id(prefix: str = "proj") -> str:
    """Short random id with prefix, e.g. proj_a1b2c3d4."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".pw_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # On Windows os.replace is atomic within same volume
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    """Reference to a project-local checkpoint."""
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
class ProjectMeta:
    """Root metadata for a project workspace, persisted as project.json."""
    schema_version: str = _PROJECT_SCHEMA
    project_id: str = ""
    name: str = ""
    created_at: str = ""
    updated_at: str = ""
    robot: RobotTarget = field(default_factory=RobotTarget)
    assets: AssetIndex = field(default_factory=AssetIndex)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "robot": self.robot.to_dict(),
            "assets": self.assets.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectMeta":
        return cls(
            schema_version=str(d.get("schema_version", _PROJECT_SCHEMA)),
            project_id=str(d.get("project_id", "")),
            name=str(d.get("name", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            robot=RobotTarget.from_dict(d.get("robot", {})),
            assets=AssetIndex.from_dict(d.get("assets", {})),
        )

    def save(self, path: Path) -> None:
        """Persist to JSON file atomically."""
        _atomic_write(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "ProjectMeta":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Dataclasses — Training (absorbed from TrainingWorkspaceStore)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    """Metadata for a training experiment within a project."""
    experiment_id: str = ""
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

    # -- path helpers --------------------------------------------------------

    def _project_dir(self, project_id: str) -> Path:
        return self._projects_dir / project_id

    def _project_json(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _workflows_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "workflows"

    def _training_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "training"

    def _experiments_dir(self, project_id: str) -> Path:
        return self._training_dir(project_id) / "experiments"

    def _runs_dir(self, project_id: str) -> Path:
        return self._training_dir(project_id) / "runs"

    def _checkpoints_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "checkpoints"

    def _bundles_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "bundles"

    def _shared_checkpoints_dir(self) -> Path:
        return self._shared_dir / "checkpoints"

    def _shared_bundles_dir(self) -> Path:
        return self._shared_dir / "bundles"

    # -- project lifecycle ---------------------------------------------------

    def create_project(
        self,
        name: str,
        robot_brand: str = "",
        robot_model: str = "",
    ) -> ProjectMeta:
        """Create a new project with scaffolded directory structure."""
        project_id = _gen_id("proj")
        now = _now_iso()

        meta = ProjectMeta(
            schema_version=_PROJECT_SCHEMA,
            project_id=project_id,
            name=name,
            created_at=now,
            updated_at=now,
            robot=RobotTarget(brand=robot_brand, model=robot_model),
            assets=AssetIndex(),
        )

        # Scaffold directories
        proj_dir = self._project_dir(project_id)
        for subdir in [
            self._workflows_dir(project_id),
            self._experiments_dir(project_id),
            self._runs_dir(project_id),
            self._checkpoints_dir(project_id),
            self._bundles_dir(project_id),
        ]:
            subdir.mkdir(parents=True, exist_ok=True)

        # Write project.json
        meta.save(self._project_json(project_id))
        return meta

    def open_project(self, project_id: str) -> ProjectMeta:
        """Load an existing project by ID.  Raises FileNotFoundError."""
        pj = self._project_json(project_id)
        if not pj.exists():
            raise FileNotFoundError(
                f"Project not found: {project_id} (expected {pj})"
            )
        return ProjectMeta.load(pj)

    def list_projects(self) -> List[ProjectMeta]:
        """Discover all projects under projects/."""
        results: List[ProjectMeta] = []
        if not self._projects_dir.exists():
            return results
        for entry in sorted(self._projects_dir.iterdir()):
            pj = entry / "project.json"
            if pj.is_file():
                try:
                    results.append(ProjectMeta.load(pj))
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupted
        return results

    def delete_project(self, project_id: str) -> None:
        """Remove a project directory entirely.

        Does NOT remove any published shared assets.
        """
        proj_dir = self._project_dir(project_id)
        if proj_dir.exists():
            shutil.rmtree(proj_dir)

    def save_meta(self, meta: ProjectMeta) -> None:
        """Persist updated ProjectMeta back to project.json."""
        meta.updated_at = _now_iso()
        meta.save(self._project_json(meta.project_id))

    # -- workflow persistence ------------------------------------------------

    def save_workflow(
        self,
        project_id: str,
        workflow_id: str,
        canvas_data: dict,
    ) -> Path:
        """Save a workflow canvas to the project.

        Creates or overwrites ``workflows/<workflow_id>.workflow.json``
        and updates the asset index in project.json.

        Returns the absolute path of the saved file.
        """
        wf_dir = self._workflows_dir(project_id)
        wf_dir.mkdir(parents=True, exist_ok=True)
        rel_file = f"workflows/{workflow_id}.workflow.json"
        abs_path = self._project_dir(project_id) / rel_file

        _atomic_write(abs_path, canvas_data)

        # Update asset index
        meta = self.open_project(project_id)
        existing = meta.assets.find_workflow(workflow_id)
        if existing is None:
            meta.assets.workflows.append(
                WorkflowEntry(id=workflow_id, file=rel_file)
            )
        self.save_meta(meta)
        return abs_path

    def load_workflow(self, project_id: str, workflow_id: str) -> dict:
        """Load a workflow canvas dict.  Raises FileNotFoundError."""
        abs_path = (
            self._workflows_dir(project_id)
            / f"{workflow_id}.workflow.json"
        )
        if not abs_path.exists():
            raise FileNotFoundError(
                f"Workflow not found: {workflow_id} in project {project_id}"
            )
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_workflows(self, project_id: str) -> List[WorkflowEntry]:
        """Return all workflow entries registered in the project."""
        meta = self.open_project(project_id)
        return list(meta.assets.workflows)

    # -- training experiments ------------------------------------------------

    def create_experiment(
        self,
        project_id: str,
        name: str = "",
        canvas: Optional[dict] = None,
    ) -> ExperimentMeta:
        """Create a new training experiment in the project."""
        exp_id = _gen_id("exp")
        now = time.time()
        exp = ExperimentMeta(
            experiment_id=exp_id,
            name=name or exp_id,
            created_at=now,
            updated_at=now,
        )

        # Save canvas if provided
        exp_dir = self._experiments_dir(project_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        if canvas is not None:
            canvas_path = exp_dir / f"{exp_id}.canvas.json"
            _atomic_write(canvas_path, canvas)

        return exp

    def save_experiment(
        self,
        project_id: str,
        exp_id: str,
        canvas: dict,
    ) -> None:
        """Save/overwrite a training experiment canvas."""
        exp_dir = self._experiments_dir(project_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        canvas_path = exp_dir / f"{exp_id}.canvas.json"
        _atomic_write(canvas_path, canvas)

    def load_experiment(self, project_id: str, exp_id: str) -> dict:
        """Load a training experiment canvas.  Raises FileNotFoundError."""
        canvas_path = (
            self._experiments_dir(project_id) / f"{exp_id}.canvas.json"
        )
        if not canvas_path.exists():
            raise FileNotFoundError(
                f"Experiment not found: {exp_id} in project {project_id}"
            )
        with open(canvas_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_experiments(self, project_id: str) -> List[ExperimentMeta]:
        """Return experiment metadata list from project.json."""
        try:
            meta = self.open_project(project_id)
        except FileNotFoundError:
            return []
        # Scan experiment files and return ExperimentMeta for each
        exp_dir = self._experiments_dir(project_id)
        if not exp_dir.exists():
            return []
        results: List[ExperimentMeta] = []
        for fp in sorted(exp_dir.iterdir()):
            if fp.suffix == ".json" and fp.stem.startswith("exp_"):
                exp_id = fp.stem.replace(".canvas", "")
                results.append(ExperimentMeta(
                    experiment_id=exp_id,
                    name=exp_id,
                    created_at=fp.stat().st_ctime,
                    updated_at=fp.stat().st_mtime,
                ))
        return results

    def delete_experiment(self, project_id: str, exp_id: str) -> None:
        """Delete an experiment canvas file."""
        canvas_path = (
            self._experiments_dir(project_id) / f"{exp_id}.canvas.json"
        )
        if canvas_path.exists():
            canvas_path.unlink()

    def rename_experiment(
        self,
        project_id: str,
        exp_id: str,
        new_name: str,
    ) -> None:
        """Rename an experiment (metadata only, file stays the same)."""
        # In the ProjectStore model, experiment names are tracked in the
        # canvas file itself or by the caller.  This is a no-op placeholder
        # for API compatibility with TrainingWorkspaceStore.
        pass

    # -- training runs -------------------------------------------------------

    def create_run(self, project_id: str, run_meta: RunMeta) -> RunMeta:
        """Persist a new training run record."""
        runs_dir = self._runs_dir(project_id)
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_path = runs_dir / f"{run_meta.run_id}.json"
        _atomic_write(run_path, run_meta.to_dict())
        return run_meta

    def update_run(
        self,
        project_id: str,
        run_id: str,
        **fields: Any,
    ) -> None:
        """Partial update of a training run record."""
        run_path = self._runs_dir(project_id) / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(
                f"Run not found: {run_id} in project {project_id}"
            )
        with open(run_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.update(fields)
        _atomic_write(run_path, data)

    def list_runs(self, project_id: str) -> List[dict]:
        """List all training runs, sorted newest-first."""
        runs_dir = self._runs_dir(project_id)
        if not runs_dir.exists():
            return []
        results: List[dict] = []
        for fp in runs_dir.iterdir():
            if fp.suffix == ".json":
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    continue
        results.sort(key=lambda d: d.get("created_at", 0), reverse=True)
        return results

    # -- checkpoint storage (project-local) ----------------------------------

    def save_checkpoint(
        self,
        project_id: str,
        policy_id: str,
        artifacts: Dict[str, Any],
    ) -> Path:
        """Register a checkpoint in the project.

        ``artifacts`` is a dict mapping filename → content:
          - For binary files (bytes): written as-is
          - For dict/list: written as JSON
          - For str: written as text

        Returns the checkpoint directory path.
        """
        ckpt_dir = self._checkpoints_dir(project_id) / policy_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in artifacts.items():
            out_path = ckpt_dir / filename
            if isinstance(content, bytes):
                out_path.write_bytes(content)
            elif isinstance(content, (dict, list)):
                _atomic_write(out_path, content)
            else:
                out_path.write_text(str(content), encoding="utf-8")

        # Update asset index
        meta = self.open_project(project_id)
        existing = meta.assets.find_checkpoint(policy_id)
        if existing is None:
            rel_path = f"checkpoints/{policy_id}/"
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
        """Return the checkpoint directory path, or None if not found."""
        ckpt_dir = self._checkpoints_dir(project_id) / policy_id
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
        """Copy a project-local checkpoint to shared space.

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
        src_dir = self._checkpoints_dir(project_id) / policy_id
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
        _atomic_write(source_json, source_data)

        # Update project.json published flag
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
