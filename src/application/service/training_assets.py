"""TrainingAssetsCache — in-memory cache of <project>/training/{runs,exported}/.

Single source of truth for "what runs / policies does the current project
have". Built per the architecture rule: scan once on project bind, keep in
RAM, expose to consumers; refresh only when (a) the active project changes
or (b) a consumer reports a path miss. No on-disk index file, ever — the
filesystem layout IS authoritative.

Lifecycle::

    [project loaded]                                ← AppSignals.project_changed
        └─ TrainingAssetsCache.bind(info) → re-scan runs/ + exported/
              └─ in-memory lists become the source for all consumers

    [consumer call]
        └─ get_training_assets().runs() / .policies() / .find_*(id)
              └─ if find_*() returns None or asset.path no longer exists:
                    refresh() then retry once

    [project switched / unloaded]
        └─ AppSignals.project_changed → bind(new) or bind(None) → cache cleared

Consumers
---------

* History dropdown (``MainWindow._populate_history_menu``) reads ``runs()``.
* Policies dropdown (``MainWindow._populate_policies_menu``) reads ``policies()``.
* Click handlers (``_on_history_pick`` / ``_on_policy_pick``) resolve via
  ``find_run`` / ``find_policy``, refresh on miss.

Anything else that needs to enumerate per-project training assets MUST go
through this cache — do not re-scan disk in widget code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import List, Optional

from unitport_sdk import log_info, log_warning, read_data

from application.service.projects.store import ProjectInfo


# ---------------------------------------------------------------------------
# Asset records — frozen views into one run/bundle directory.
# ---------------------------------------------------------------------------

_LEGACY_BACKEND = "(legacy)"
"""Backend slot for runs/bundles that live directly under runs/ or exported/
(pre-backend-layer fixtures). Read-only — new writes always go to a real
backend subdir."""


@dataclass(frozen=True)
class RunAsset:
    """One run directory under ``<project>/training/runs/<backend_id>/<run_id>/``.

    ``backend_id`` is the parent directory name (the partition key).
    Pre-backend-layer entries (flat ``runs/<run_id>/``) appear with
    ``backend_id == "(legacy)"`` so the user can still see them.
    """
    run_id: str           # = leaf directory name
    backend_id: str       # = parent directory name (or "(legacy)")
    path: Path            # absolute path to the run directory
    mtime: float          # st_mtime (newest-first ordering)


@dataclass(frozen=True)
class PolicyAsset:
    """One bundle dir under ``<project>/training/exported/<backend_id>/<policy_id>/``.

    Optional manifest-derived fields (``algorithm`` / ``obs_dim`` / ``action_dim``)
    are best-effort: missing or unparseable ``manifest.yaml`` collapses them
    to empty / 0 — the asset is still listed so the user sees what's on disk.
    Pre-backend-layer entries (flat ``exported/<policy_id>/``) appear with
    ``backend_id == "(legacy)"``.
    """
    policy_id: str
    backend_id: str
    path: Path
    mtime: float
    algorithm: str
    obs_dim: int
    action_dim: int


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TrainingAssetsCache:
    """Project-scoped cache of runs/ and exported/ directories.

    Thread-safe (RLock) because ``bind`` may fire from a project-switch
    signal on the main thread while a background scan or consumer holds
    the lock briefly.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._project: Optional[ProjectInfo] = None
        self._runs: List[RunAsset] = []
        self._policies: List[PolicyAsset] = []

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def bind(self, project: Optional[ProjectInfo]) -> None:
        """Switch the cache to ``project`` (or clear when ``None``).

        Wired to ``AppSignals.project_changed``. Idempotent — re-binding
        the same project just re-scans (cheap; refresh semantics).
        """
        with self._lock:
            self._project = project
            self._runs = []
            self._policies = []
            if project is not None:
                self._scan_locked()

    def refresh(self) -> None:
        """Re-scan disk for the currently bound project. No-op if unbound."""
        with self._lock:
            if self._project is None:
                self._runs = []
                self._policies = []
                return
            self._scan_locked()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def project(self) -> Optional[ProjectInfo]:
        with self._lock:
            return self._project

    def runs(self) -> List[RunAsset]:
        """Snapshot of the current runs list (newest mtime first)."""
        with self._lock:
            return list(self._runs)

    def policies(self) -> List[PolicyAsset]:
        """Snapshot of the current policies list (newest mtime first)."""
        with self._lock:
            return list(self._policies)

    def find_run(
        self, run_id: str, backend_id: Optional[str] = None,
    ) -> Optional[RunAsset]:
        """Lookup by ``run_id``. When ``backend_id`` is provided it must
        also match (exact partition lookup) — needed because run_ids are
        unique only within their backend partition. Caller should check
        ``asset.path.exists()`` and call :meth:`refresh` then re-lookup
        on miss — see module docstring "consumer call" lifecycle."""
        with self._lock:
            for r in self._runs:
                if r.run_id == run_id and (
                    backend_id is None or r.backend_id == backend_id
                ):
                    return r
        return None

    def find_policy(
        self, policy_id: str, backend_id: Optional[str] = None,
    ) -> Optional[PolicyAsset]:
        with self._lock:
            for p in self._policies:
                if p.policy_id == policy_id and (
                    backend_id is None or p.backend_id == backend_id
                ):
                    return p
        return None

    # ------------------------------------------------------------------
    # Scan internals — caller MUST hold self._lock
    # ------------------------------------------------------------------

    def _scan_locked(self) -> None:
        proj = self._project
        if proj is None:
            return
        self._runs = self._scan_runs(proj)
        self._policies = self._scan_policies(proj)
        log_info(
            f"[training_assets] refreshed for {proj.name}: "
            f"{len(self._runs)} runs, {len(self._policies)} policies"
        )

    # ------------------------------------------------------------------
    # Scan helpers — runs/ and exported/ now use a per-backend subdir.
    # Layout::
    #
    #     <project>/training/runs/<backend_id>/<run_id>/
    #     <project>/training/exported/<backend_id>/<policy_id>/
    #
    # The scanners walk that 2-level structure. As a backwards-compat
    # fallback they ALSO accept the old flat layout (a direct child that
    # itself looks like a run/bundle): those entries surface with
    # ``backend_id == _LEGACY_BACKEND`` so the user can see them and
    # decide whether to move them.
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_run(d: Path) -> bool:
        """Heuristic: a flat run dir contains at least one of the standard
        per-run artifacts (params/, checkpoints, tfevents). A backend
        partition dir contains only further subdirectories (no such files
        at its own level)."""
        try:
            for child in d.iterdir():
                name = child.name
                if name == "params" or name == "checkpoints":
                    return True
                if name.startswith("events.out.tfevents") or name.endswith(".pt"):
                    return True
        except OSError:
            return False
        return False

    @staticmethod
    def _looks_like_bundle(d: Path) -> bool:
        """A flat bundle dir has manifest.yaml directly inside."""
        return (d / "manifest.yaml").exists()

    @classmethod
    def _scan_runs(cls, proj: ProjectInfo) -> List[RunAsset]:
        runs_dir = proj.path / "training" / "runs"
        out: List[RunAsset] = []
        if not runs_dir.is_dir():
            return out
        try:
            for level1 in runs_dir.iterdir():
                if not level1.is_dir():
                    continue
                # Legacy flat layout: this dir itself is a run.
                if cls._looks_like_run(level1):
                    try:
                        mtime = level1.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    out.append(RunAsset(
                        run_id=level1.name,
                        backend_id=_LEGACY_BACKEND,
                        path=level1,
                        mtime=mtime,
                    ))
                    continue
                # New layout: this is a backend partition; recurse one level.
                try:
                    for run_dir in level1.iterdir():
                        if not run_dir.is_dir():
                            continue
                        try:
                            mtime = run_dir.stat().st_mtime
                        except OSError:
                            mtime = 0.0
                        out.append(RunAsset(
                            run_id=run_dir.name,
                            backend_id=level1.name,
                            path=run_dir,
                            mtime=mtime,
                        ))
                except OSError as exc:                     # pragma: no cover
                    log_warning(f"[training_assets] scan {level1} failed: {exc}")
        except OSError as exc:                             # pragma: no cover
            log_warning(f"[training_assets] scan {runs_dir} failed: {exc}")
        out.sort(key=lambda r: r.mtime, reverse=True)
        return out

    @classmethod
    def _scan_policies(cls, proj: ProjectInfo) -> List[PolicyAsset]:
        exported_dir = proj.path / "training" / "exported"
        out: List[PolicyAsset] = []
        if not exported_dir.is_dir():
            return out

        def _read_manifest(bundle_dir: Path) -> tuple:
            """Return ``(algorithm, obs_dim, action_dim)`` from manifest.yaml.

            Two on-disk shapes coexist:
              • new BundleExporter v1: action_space.dim / observation_space.dim
              • legacy/Isaac Lab bundles: skill.action_dim / skill.observation_dim
            Tries both; missing/unparseable manifest collapses to ('', 0, 0).
            """
            manifest_path = bundle_dir / "manifest.yaml"
            if not manifest_path.exists():
                return ("", 0, 0)
            try:
                m = read_data(manifest_path)
            except Exception:
                return ("", 0, 0)
            if not isinstance(m, dict):
                return ("", 0, 0)
            algo = str(m.get("algorithm") or "")
            skill = m.get("skill") or {}
            obs = int(
                (m.get("observation_space") or {}).get("dim", 0)
                or skill.get("observation_dim", 0) or 0
            )
            act = int(
                (m.get("action_space") or {}).get("dim", 0)
                or skill.get("action_dim", 0) or 0
            )
            return (algo, obs, act)

        try:
            for level1 in exported_dir.iterdir():
                if not level1.is_dir():
                    continue
                # Legacy flat: a bundle dir directly under exported/.
                if cls._looks_like_bundle(level1):
                    try:
                        mtime = level1.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    algo, obs, act = _read_manifest(level1)
                    out.append(PolicyAsset(
                        policy_id=level1.name,
                        backend_id=_LEGACY_BACKEND,
                        path=level1,
                        mtime=mtime,
                        algorithm=algo, obs_dim=obs, action_dim=act,
                    ))
                    continue
                # New layout: backend partition; recurse one level.
                try:
                    for bundle_dir in level1.iterdir():
                        if not bundle_dir.is_dir():
                            continue
                        try:
                            mtime = bundle_dir.stat().st_mtime
                        except OSError:
                            mtime = 0.0
                        algo, obs, act = _read_manifest(bundle_dir)
                        out.append(PolicyAsset(
                            policy_id=bundle_dir.name,
                            backend_id=level1.name,
                            path=bundle_dir,
                            mtime=mtime,
                            algorithm=algo, obs_dim=obs, action_dim=act,
                        ))
                except OSError as exc:                     # pragma: no cover
                    log_warning(f"[training_assets] scan {level1} failed: {exc}")
        except OSError as exc:                             # pragma: no cover
            log_warning(f"[training_assets] scan {exported_dir} failed: {exc}")
        out.sort(key=lambda p: p.mtime, reverse=True)
        return out


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_singleton: Optional[TrainingAssetsCache] = None


def get_training_assets() -> TrainingAssetsCache:
    """Return the process-wide cache; lazy-construct + auto-subscribe.

    First access wires the cache to ``AppSignals.project_changed`` so all
    subsequent project switches refresh it transparently. It also subscribes
    to ``training_run_finished`` so the newly produced run dir + exported
    bundle become visible to consumers immediately, without requiring a
    project reload. First access seeds the cache from ``current_project_info()``
    — covers the case where a project was bound *before* anyone touched the
    cache (the signal would have fired into the void).
    """
    global _singleton
    if _singleton is None:
        cache = TrainingAssetsCache()

        from application.service.signals import get_app_signals
        sigs = get_app_signals()
        sigs.project_changed.connect(cache.bind)
        # Re-scan after every training run completes (success or failure —
        # failed runs still produced a run dir worth listing in History).
        sigs.training_run_finished.connect(
            lambda _run_id, _ok: cache.refresh()
        )

        from application.service.projects.store import current_project_info
        cache.bind(current_project_info())

        _singleton = cache
    return _singleton


__all__ = [
    "RunAsset",
    "PolicyAsset",
    "TrainingAssetsCache",
    "get_training_assets",
]
