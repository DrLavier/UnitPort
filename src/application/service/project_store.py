"""TrainingStore — per-project runs.json + bundles.json index.

REWRITE-WITH-DEMO-REF from DEMO ``training_workspace_store.py`` +
``run_meta.py``, narrowed to what Stage 11 actually needs:

  * A typed record for each training run (run_id, status, algorithm, …)
    appended to ``<project>/training/runs.json``.
  * A typed record for each exported policy bundle (policy_id, path,
    manifest pointer) appended to ``<project>/training/bundles.json``.
  * Atomic-rewrite semantics via ``DataManager.save`` (handles tempfile
    + ``os.replace`` on the same volume).

Project tree convention (Stage 0 user decision: canvas + training inside
project root):

    <project>/
      project.yaml
      canvas/main.canvas.json
      training/
        runs.json                  # this file (append-only index)
        runs/<run_id>/
          progress.jsonl           # tail-able (Stage 10's launcher writes)
          checkpoints/<step>.pt    # see CheckpointRegistry
          stdout.log
        exported/<policy_id>/      # Stage 7 BundleExporter writes here
          policy.onnx
          manifest.yaml
          source.json
        bundles.json               # this file (policy_id index)

Distinct from :class:`application.service.projects.store.ProjectStore`,
which handles project *listing* (scanning ``Paths.PROJECTS_DIR`` for
``project.yaml`` manifests). :class:`TrainingStore` is per-project state.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from unitport_sdk import log_info, log_warning, read_data, save_data

from application.service.projects.store import ProjectInfo


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One row in ``training/runs.json``."""

    run_id: str
    status: str = "submitted"           # submitted | running | finished | cancelled | failed
    algorithm: str = ""
    backend: str = ""
    robot_sku: str = ""
    bundle_name: str = ""
    total_timesteps: int = 0
    seed: int = 0
    submitted_at: str = ""              # ISO-8601 UTC
    finished_at: str = ""
    last_step: int = 0
    best_reward: Optional[float] = None
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=str(d.get("run_id", "")),
            status=str(d.get("status", "submitted")),
            algorithm=str(d.get("algorithm", "")),
            backend=str(d.get("backend", "")),
            robot_sku=str(d.get("robot_sku", "")),
            bundle_name=str(d.get("bundle_name", "")),
            total_timesteps=int(d.get("total_timesteps", 0) or 0),
            seed=int(d.get("seed", 0) or 0),
            submitted_at=str(d.get("submitted_at", "")),
            finished_at=str(d.get("finished_at", "")),
            last_step=int(d.get("last_step", 0) or 0),
            best_reward=(
                float(d["best_reward"])
                if d.get("best_reward") is not None
                else None
            ),
            error=str(d.get("error", "")),
            extra=dict(d.get("extra", {}) or {}),
        )


@dataclass
class BundleRecord:
    """One row in ``training/bundles.json``."""

    policy_id: str
    path: str                           # relative to project root
    version: str = "v1"
    algorithm: str = ""
    obs_dim: int = 0
    action_dim: int = 0
    run_id: str = ""
    created_at: str = ""                # ISO-8601 UTC
    published: bool = False
    published_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BundleRecord":
        return cls(
            policy_id=str(d.get("policy_id", "")),
            path=str(d.get("path", "")),
            version=str(d.get("version", "v1")),
            algorithm=str(d.get("algorithm", "")),
            obs_dim=int(d.get("obs_dim", 0) or 0),
            action_dim=int(d.get("action_dim", 0) or 0),
            run_id=str(d.get("run_id", "")),
            created_at=str(d.get("created_at", "")),
            published=bool(d.get("published", False)),
            published_at=d.get("published_at"),
            extra=dict(d.get("extra", {}) or {}),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_index(path: Path, key: str) -> List[Dict[str, Any]]:
    """Read a JSON index file and return its ``key`` array.

    Returns an empty list when the file is missing or malformed (the
    store treats absence as "no records yet"; callers can distinguish
    via :meth:`TrainingStore.runs_index_path.exists()` if needed).
    """
    if not path.exists():
        return []
    payload = read_data(path)
    if not isinstance(payload, dict):
        log_warning(f"[training_store] {path} is not a JSON object; treating as empty")
        return []
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        log_warning(f"[training_store] {path}#{key} is not a list; treating as empty")
        return []
    return [r for r in rows if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# TrainingStore
# ---------------------------------------------------------------------------


class TrainingStore:
    """Per-project training-side store.

    Construct with a :class:`ProjectInfo` (from
    :class:`application.service.projects.store.ProjectStore`). All paths
    resolve under ``project.path / "training" / ...``.
    """

    SCHEMA_VERSION = "training_index_v1"

    def __init__(self, project: ProjectInfo) -> None:
        if not isinstance(project, ProjectInfo):
            raise TypeError(
                f"TrainingStore: expected ProjectInfo, got {type(project).__name__}"
            )
        self.project = project

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def training_dir(self) -> Path:
        return self.project.path / "training"

    @property
    def runs_dir(self) -> Path:
        return self.training_dir / "runs"

    @property
    def exported_dir(self) -> Path:
        return self.training_dir / "exported"

    @property
    def runs_index_path(self) -> Path:
        return self.training_dir / "runs.json"

    @property
    def bundles_index_path(self) -> Path:
        return self.training_dir / "bundles.json"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / str(run_id)

    def bundle_dir(self, policy_id: str) -> Path:
        return self.exported_dir / str(policy_id)

    def ensure_dirs(self) -> None:
        for d in (self.training_dir, self.runs_dir, self.exported_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def list_runs(self) -> List[RunRecord]:
        rows = _read_index(self.runs_index_path, "runs")
        return [RunRecord.from_dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        for r in self.list_runs():
            if r.run_id == run_id:
                return r
        return None

    def register_run(
        self,
        run_id: str,
        *,
        algorithm: str = "",
        backend: str = "",
        robot_sku: str = "",
        bundle_name: str = "",
        total_timesteps: int = 0,
        seed: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        """Append a new ``submitted`` row to runs.json."""
        self.ensure_dirs()
        rec = RunRecord(
            run_id=str(run_id),
            status="submitted",
            algorithm=str(algorithm),
            backend=str(backend),
            robot_sku=str(robot_sku),
            bundle_name=str(bundle_name),
            total_timesteps=int(total_timesteps),
            seed=int(seed),
            submitted_at=_utc_now_iso(),
            extra=dict(extra or {}),
        )
        self._upsert_run(rec)
        log_info(
            f"[training_store] registered run {rec.run_id} "
            f"({rec.algorithm}/{rec.backend}/{rec.robot_sku})"
        )
        return rec

    def update_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        last_step: Optional[int] = None,
        best_reward: Optional[float] = None,
        error: Optional[str] = None,
        finished: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[RunRecord]:
        """Patch an existing run row. Returns the updated record (or None)."""
        rec = self.get_run(run_id)
        if rec is None:
            log_warning(
                f"[training_store] update_run: unknown run_id {run_id!r}"
            )
            return None
        if status is not None:
            rec.status = str(status)
        if last_step is not None:
            rec.last_step = int(last_step)
        if best_reward is not None:
            rec.best_reward = float(best_reward)
        if error is not None:
            rec.error = str(error)
        if extra:
            rec.extra.update(extra)
        if finished and not rec.finished_at:
            rec.finished_at = _utc_now_iso()
        self._upsert_run(rec)
        return rec

    def _upsert_run(self, rec: RunRecord) -> None:
        rows = _read_index(self.runs_index_path, "runs")
        out: List[Dict[str, Any]] = []
        replaced = False
        for r in rows:
            if r.get("run_id") == rec.run_id:
                out.append(rec.to_dict())
                replaced = True
            else:
                out.append(r)
        if not replaced:
            out.append(rec.to_dict())
        save_data(self.runs_index_path, {
            "schema": self.SCHEMA_VERSION,
            "runs": out,
        })

    # ------------------------------------------------------------------
    # Bundles
    # ------------------------------------------------------------------

    def list_bundles(self) -> List[BundleRecord]:
        rows = _read_index(self.bundles_index_path, "bundles")
        return [BundleRecord.from_dict(r) for r in rows]

    def get_bundle(self, policy_id: str) -> Optional[BundleRecord]:
        for b in self.list_bundles():
            if b.policy_id == policy_id:
                return b
        return None

    def register_bundle(
        self,
        policy_id: str,
        bundle_path: Path,
        *,
        version: str = "v1",
        algorithm: str = "",
        obs_dim: int = 0,
        action_dim: int = 0,
        run_id: str = "",
        published: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> BundleRecord:
        """Add (or replace) a bundles.json row.

        ``bundle_path`` is normalised to a project-relative path when it
        sits under ``<project>/``; otherwise an absolute path is stored
        (legacy out-of-project bundles use this branch — the current
        BundleExporter contract always writes inside the project).
        """
        self.ensure_dirs()
        try:
            rel = str(Path(bundle_path).resolve().relative_to(self.project.path.resolve()))
        except (OSError, ValueError):
            rel = str(Path(bundle_path).resolve())

        rec = BundleRecord(
            policy_id=str(policy_id),
            path=rel,
            version=str(version),
            algorithm=str(algorithm),
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            run_id=str(run_id),
            created_at=_utc_now_iso(),
            published=bool(published),
            published_at=_utc_now_iso() if published else None,
            extra=dict(extra or {}),
        )
        self._upsert_bundle(rec)
        log_info(
            f"[training_store] registered bundle {rec.policy_id} "
            f"({rec.algorithm}, dim={rec.obs_dim}/{rec.action_dim}) → {rec.path}"
        )
        return rec

    def mark_bundle_published(self, policy_id: str, *, published: bool = True) -> Optional[BundleRecord]:
        rec = self.get_bundle(policy_id)
        if rec is None:
            log_warning(
                f"[training_store] mark_bundle_published: unknown {policy_id!r}"
            )
            return None
        rec.published = bool(published)
        rec.published_at = _utc_now_iso() if published else None
        self._upsert_bundle(rec)
        return rec

    def _upsert_bundle(self, rec: BundleRecord) -> None:
        rows = _read_index(self.bundles_index_path, "bundles")
        out: List[Dict[str, Any]] = []
        replaced = False
        for r in rows:
            if r.get("policy_id") == rec.policy_id:
                out.append(rec.to_dict())
                replaced = True
            else:
                out.append(r)
        if not replaced:
            out.append(rec.to_dict())
        save_data(self.bundles_index_path, {
            "schema": self.SCHEMA_VERSION,
            "bundles": out,
        })


__all__ = [
    "TrainingStore",
    "RunRecord",
    "BundleRecord",
]
