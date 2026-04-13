#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckpointRegistry — single source of truth for deployed checkpoint bundles.

Bundles live inside the active project at (Layout v2)::

    projects/<slug>/training/exported/<policy_id>/manifest.yaml

The path is resolved through ``ProjectStore._checkpoints_dir(project_id)``
which is now an alias for ``_training_exported_dir`` — the public method name
is preserved so this module needs no churn.

When no project is active *and* no explicit root is provided, the registry
falls back to scanning the legacy ``custom_mods/training/checkpoints/`` path
in **read-only** mode so historical bundles remain visible during the
transition.  All *write* operations (training export, bundle import) target
the active project's ``training/exported/`` directory and refuse to fall
back to the legacy path.

The active project is resolved via
:mod:`src.system.core.project_session` so call sites do not have to pass a
``ProjectStore + project_id`` pair every time they construct a registry.
The UI shell sets the session whenever the active project changes; the
SB3 training subprocess inherits it through the job spec.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.system.core.project_store import ProjectStore


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CheckpointImportError(Exception):
    """Raised when a checkpoint cannot be imported."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckpointEntry:
    """Metadata for a single discovered checkpoint bundle."""

    policy_id: str
    """Directory name under custom_mods/training/checkpoints/ — used as the canonical ID."""

    bundle_path: Path
    """Absolute path to the bundle directory."""

    display_name: str = ""
    """Human-readable name (from manifest 'name' field, falls back to policy_id)."""

    version: str = ""
    """Bundle version string (from manifest, may be empty)."""

    robot_brand: str = ""
    """Robot brand string (from manifest robot.brand, may be empty)."""

    robot_model: str = ""
    """Robot model string (from manifest robot.model, may be empty)."""

    is_valid: bool = True
    """False when the bundle directory has no manifest.yaml or the YAML is unreadable."""

    error: str = ""
    """Non-empty when is_valid=False; contains the parse/validation error message."""

    source_type: str = ""
    """Bundle origin: 'local' | 'huggingface' | 'training' | '' (unknown/legacy)."""

    # ------------------------------------------------------------------
    # Training lineage fields (Phase B)
    # ------------------------------------------------------------------

    parent_policy_id: str = ""
    """
    policy_id of the base checkpoint this one was trained from.
    Populated for training-derived bundles (source_type='training').
    Empty for locally imported or HuggingFace bundles.
    """

    experiment_id: str = ""
    """
    Training experiment id that produced this checkpoint.
    Matches TrainingWorkspaceStore experiment_id.
    """

    run_id: str = ""
    """
    Training run id that produced this checkpoint.
    Populated by Phase C TrainRunThread when writing the bundle.
    """

    scope: str = "legacy"
    """Asset scope: 'local' | 'shared' | 'legacy' (pre-project-workspace)."""

    skill_manifest: object = None
    """Populated with a :class:`SkillManifest` on discovery (lazy v1→v2 migration)."""

    def label(self) -> str:
        """Short label for list display."""
        name = self.display_name or self.policy_id
        parts = [name]
        if self.version:
            parts.append(f"v{self.version}")
        if self.robot_brand:
            parts.append(f"[{self.robot_brand}]")
        return "  ".join(parts)

    def source_badge(self) -> str:
        """Short badge string representing the bundle origin."""
        return {
            "local": "📁",
            "huggingface": "🌐",
            "training": "🏋",
        }.get(self.source_type, "")


class CheckpointRegistry:
    """
    Scans and indexes checkpoint bundles for the active project.

    Canonical checkpoint storage path (Layout v2):
        projects/<slug>/training/exported/<policy_id>/manifest.yaml

    The active project is resolved via :mod:`project_session` unless an
    explicit ``project_store + project_id`` pair (or a raw ``root`` path
    for tests) is supplied to the constructor.

    Usage::

        registry = CheckpointRegistry()                       # picks active project
        entries  = registry.discover()

        registry.refresh()                                    # re-scan
        all_cp   = registry.list_checkpoints()                # cached list
        unitree  = registry.list_checkpoints(robot_brand="unitree")
        entry    = registry.get(policy_id="go2_v1")           # KeyError if missing
        path     = registry.get_bundle_path("go2_v1")         # Path | KeyError

        # Import a local bundle directory or zip:
        entry = registry.import_local(Path("/downloads/go2_walk/"))
    """

    #: Legacy sub-directory used by older builds.  Kept for **read-only**
    #: discovery so historical bundles remain visible after upgrading.
    LEGACY_STORAGE_SUBDIR = "custom_mods/training/checkpoints"

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        project_store: Optional["ProjectStore"] = None,
        project_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        root:
            Explicit root override (useful for tests).  When provided this
            takes precedence over both the project context and the session.
        project_store, project_id:
            Explicit project context.  When *None* (the default), the
            registry consults :mod:`project_session` for the currently
            active project.
        """
        self._explicit_root: Optional[Path] = root
        self._explicit_store: Optional["ProjectStore"] = project_store
        self._explicit_pid: str = str(project_id or "")
        self._cache: List[CheckpointEntry] = []
        self._discovered: bool = False

    # ------------------------------------------------------------------
    # Root resolution
    # ------------------------------------------------------------------

    def _resolve_active_project(self) -> tuple:
        """Return (store, project_id) for the active project, or (None, '').

        Explicit constructor args win over the session.
        """
        if self._explicit_store is not None and self._explicit_pid:
            return self._explicit_store, self._explicit_pid
        try:
            from src.system.core import project_session
            store = project_session.get_store()
            pid = project_session.get_project_id()
            if store is not None and pid:
                return store, pid
        except Exception:
            pass
        return None, ""

    @property
    def root(self) -> Path:
        """Resolved primary storage directory (always absolute).

        Resolution order:
          1. Explicit ``root`` constructor arg.
          2. Active project's ``checkpoints/`` directory.
          3. Legacy ``custom_mods/training/checkpoints/`` (read-only fallback).

        The legacy path is intentionally NOT auto-created.  Writes always
        go through :meth:`_resolve_write_root` which raises if no project
        is active.
        """
        if self._explicit_root is not None:
            return self._explicit_root
        store, pid = self._resolve_active_project()
        if store is not None and pid:
            return store._checkpoints_dir(pid)
        # Legacy fallback (read-only)
        try:
            from src.system.core.config_manager import ConfigManager  # lazy import
            return ConfigManager().get_path("project_root").parent / self.LEGACY_STORAGE_SUBDIR
        except Exception:
            return Path.cwd() / self.LEGACY_STORAGE_SUBDIR

    def _resolve_write_root(self) -> Path:
        """Return the directory where new bundles should be written.

        Always project-scoped.  Raises ``CheckpointImportError`` when no
        project is active — the legacy ``custom_mods/...`` path is no
        longer a valid write target.
        """
        if self._explicit_root is not None:
            return self._explicit_root
        store, pid = self._resolve_active_project()
        if store is None or not pid:
            raise CheckpointImportError(
                "No active project — cannot write a checkpoint bundle. "
                "Open a project before training/importing, or pass an "
                "explicit project_store + project_id to CheckpointRegistry()."
            )
        return store._checkpoints_dir(pid)

    def _legacy_root(self) -> Optional[Path]:
        """Return the legacy ``custom_mods/training/checkpoints`` path if it exists."""
        try:
            from src.system.core.config_manager import ConfigManager
            legacy = ConfigManager().get_path("project_root").parent / self.LEGACY_STORAGE_SUBDIR
        except Exception:
            legacy = Path.cwd() / self.LEGACY_STORAGE_SUBDIR
        return legacy if legacy.exists() else None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> List[CheckpointEntry]:
        """Return a ``CheckpointEntry`` for every candidate bundle visible.

        Search order:
          1. Active project's ``checkpoints/`` (scope = ``"local"``)
          2. Shared ``shared/checkpoints/``           (scope = ``"shared"``)
          3. Legacy ``custom_mods/training/checkpoints/`` (scope = ``"legacy"``)

        When an explicit ``root`` was passed to the constructor, only that
        directory is scanned (used by tests).  Bundles whose YAML cannot be
        parsed are returned with ``is_valid=False``.

        IDs collide across scopes are resolved local-first (the local entry
        wins; the shared/legacy entry is suppressed) so the registry never
        returns two ``CheckpointEntry`` objects with the same ``policy_id``.
        """
        results: List[CheckpointEntry] = []
        seen_ids: set = set()

        def _scan(directory: Path, scope: str) -> None:
            if not directory.exists():
                return
            for candidate in sorted(directory.iterdir()):
                if not candidate.is_dir():
                    continue
                manifest_path = candidate / "manifest.yaml"
                if not manifest_path.exists():
                    continue
                policy_id = candidate.name
                if policy_id in seen_ids:
                    continue  # local/shared takes precedence over legacy
                entry = self._parse_entry(policy_id, candidate, manifest_path)
                entry.scope = scope
                results.append(entry)
                seen_ids.add(policy_id)

        if self._explicit_root is not None:
            _scan(self._explicit_root, "local")
        else:
            store, pid = self._resolve_active_project()
            if store is not None and pid:
                _scan(store._checkpoints_dir(pid), "local")
                _scan(store._shared_checkpoints_dir(), "shared")
            legacy = self._legacy_root()
            if legacy is not None:
                _scan(legacy, "legacy")

        self._cache = results
        self._discovered = True
        return results

    def _parse_entry(
        self,
        policy_id: str,
        bundle_path: Path,
        manifest_path: Path,
    ) -> CheckpointEntry:
        """Parse manifest.yaml (and optional source.json) and return a CheckpointEntry.

        Also loads (or auto-migrates) the SkillManifest v2 contract from the
        bundle.  If loading fails the entry is still valid — skill_manifest
        will just be ``None``.
        """
        source_info = self._read_source_info(bundle_path)
        source_type = source_info.get("type", "")
        parent_policy_id = source_info.get("parent_policy_id", "")
        experiment_id = source_info.get("experiment_id", "")
        run_id = source_info.get("run_id", "")
        try:
            import yaml
            with manifest_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            robot_section = data.get("robot", {}) or {}
            robot_brand = (
                str(robot_section.get("brand", "") or "")
                or str(data.get("robot_brand", "") or "")
            )
            robot_model = (
                str(robot_section.get("model", "") or "")
                or str(data.get("robot_model", "") or "")
            )

            # SkillManifest v2 — load or auto-migrate from v1
            sm = self._load_skill_manifest(bundle_path)

            return CheckpointEntry(
                policy_id=policy_id,
                bundle_path=bundle_path,
                display_name=str(data.get("name", "") or policy_id),
                version=str(data.get("version", "") or ""),
                robot_brand=robot_brand,
                robot_model=robot_model,
                is_valid=True,
                source_type=source_type,
                parent_policy_id=parent_policy_id,
                experiment_id=experiment_id,
                run_id=run_id,
                skill_manifest=sm,
            )
        except ImportError:
            return CheckpointEntry(
                policy_id=policy_id,
                bundle_path=bundle_path,
                display_name=policy_id,
                is_valid=True,
                robot_model="",
                source_type=source_type,
                parent_policy_id=parent_policy_id,
                experiment_id=experiment_id,
                run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001
            return CheckpointEntry(
                policy_id=policy_id,
                bundle_path=bundle_path,
                is_valid=False,
                error=str(exc),
                robot_model="",
                source_type=source_type,
                parent_policy_id=parent_policy_id,
                experiment_id=experiment_id,
                run_id=run_id,
            )

    @staticmethod
    def _load_skill_manifest(bundle_path: Path):
        """Load SkillManifest from bundle, auto-migrating v1 if needed.

        Returns a SkillManifest instance or None on failure.
        """
        try:
            from src.system.skill.manifest_loader import load_skill_manifest
            return load_skill_manifest(bundle_path)
        except Exception:
            return None

    @staticmethod
    def _read_source_info(bundle_path: Path) -> dict:
        """Read source.json and return its contents as a dict."""
        source_file = bundle_path / "source.json"
        if not source_file.exists():
            return {}
        try:
            return json.loads(source_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _read_source_type(bundle_path: Path) -> str:
        """Read the 'type' field from source.json if present (kept for backward compat)."""
        source_file = bundle_path / "source.json"
        if not source_file.exists():
            return ""
        try:
            data = json.loads(source_file.read_text(encoding="utf-8"))
            return str(data.get("type", ""))
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-run discovery and update the internal cache."""
        self._discovered = False
        self.discover()

    def list_checkpoints(
        self, robot_brand: Optional[str] = None
    ) -> List[CheckpointEntry]:
        """Return all discovered entries (calls ``discover()`` on first use).

        Parameters
        ----------
        robot_brand:
            When provided, return only entries whose ``robot_brand`` matches
            (case-insensitive).  Only ``is_valid=True`` entries are returned
            when filtering by brand.
        """
        if not self._discovered:
            self.discover()

        if robot_brand is None:
            return list(self._cache)

        brand_lower = robot_brand.lower()
        return [
            e for e in self._cache
            if e.is_valid and e.robot_brand.lower() == brand_lower
        ]

    def get(self, policy_id: str) -> CheckpointEntry:
        """Return the entry for *policy_id*.  Raises ``KeyError`` if not found."""
        if not self._discovered:
            self.discover()
        for entry in self._cache:
            if entry.policy_id == policy_id:
                return entry
        raise KeyError(
            f"Checkpoint '{policy_id}' not found in registry (root={self.root})"
        )

    def get_bundle_path(self, policy_id: str) -> Path:
        """Return the ``bundle_path`` for the given *policy_id*.  Raises ``KeyError``."""
        return self.get(policy_id).bundle_path

    def delete(self, policy_id: str) -> None:
        """Delete the bundle directory for *policy_id* and refresh the registry."""
        entry = self.get(policy_id)
        bundle_path = Path(entry.bundle_path)
        if not bundle_path.exists():
            raise KeyError(
                f"Checkpoint '{policy_id}' not found on disk at {bundle_path}"
            )
        shutil.rmtree(bundle_path)
        self.refresh()

    def policy_ids(self) -> List[str]:
        """Return the list of all known policy IDs."""
        if not self._discovered:
            self.discover()
        return [e.policy_id for e in self._cache]

    # ------------------------------------------------------------------
    # Import: local
    # ------------------------------------------------------------------

    def import_local(self, src_path: Path) -> CheckpointEntry:
        """Import a checkpoint bundle from a local directory or zip archive.

        Parameters
        ----------
        src_path:
            Path to a bundle directory (contains ``manifest.yaml``) or a ``.zip``
            archive whose root contains ``manifest.yaml``.

        Returns
        -------
        CheckpointEntry
            The newly registered entry.

        Raises
        ------
        CheckpointImportError
            If the source is not found, manifest is missing/invalid, or the
            policy_id already exists in the registry.
        """
        src_path = Path(src_path)

        # 1. Resolve source: extract zip to temp dir if needed
        tmp_dir: Optional[Path] = None
        try:
            bundle_src = self._resolve_source(src_path)
            tmp_dir = bundle_src if bundle_src != src_path else None

            # 2. Validate manifest present
            manifest_path = bundle_src / "manifest.yaml"
            if not manifest_path.exists():
                raise CheckpointImportError(
                    f"No manifest.yaml found in: {bundle_src}"
                )

            # 3. Parse manifest to get policy_id (= bundle name)
            policy_id = self._read_policy_id(manifest_path, bundle_src)

            # 4. Check no collision (writes always go into the active project)
            write_root = self._resolve_write_root()
            dest = write_root / policy_id
            if dest.exists():
                raise CheckpointImportError(
                    f"A checkpoint with policy_id '{policy_id}' already exists "
                    f"at {dest}.  Rename the source bundle directory to import it "
                    f"under a different name."
                )

            # 5. Write source.json — preserve existing non-local provenance
            #    (e.g. HuggingFace bundles already have source.json written
            #    by the downloader; don't overwrite with "local")
            existing_source = bundle_src / "source.json"
            _preserve = False
            if existing_source.exists():
                try:
                    _existing = json.loads(existing_source.read_text("utf-8"))
                    _preserve = _existing.get("type", "") not in ("", "local")
                except Exception:  # noqa: BLE001
                    pass
            if not _preserve:
                source_meta = {"type": "local", "src": str(src_path.resolve())}
                existing_source.write_text(
                    json.dumps(source_meta, indent=2), encoding="utf-8"
                )

            # 6. Atomic copy: temp staging → final destination
            write_root.mkdir(parents=True, exist_ok=True)
            staging = write_root / f"{policy_id}__importing"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(bundle_src, staging)
            staging.rename(dest)

            # 7. Register in project asset index, refresh, return
            self._register_in_project_meta(policy_id, dest)
            self.refresh()
            return self.get(policy_id)

        except CheckpointImportError:
            raise
        except Exception as exc:
            raise CheckpointImportError(
                f"Import failed: {exc}"
            ) from exc
        finally:
            # Clean up temp extraction dir (if we created one)
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Import: HuggingFace normalized bundle
    # ------------------------------------------------------------------

    def import_hf_bundle(self, bundle_path: Path) -> "CheckpointEntry":
        """Install a pre-normalized HuggingFace bundle into checkpoint storage.

        The *bundle_path* must already contain a valid ``manifest.yaml`` and a
        ``source.json`` with ``type = "huggingface"`` (as written by
        ``system.training.hf_downloader.normalize_hf_snapshot``).

        This is a thin wrapper around :meth:`import_local` that signals intent
        clearly and validates the HuggingFace provenance before delegating.

        Parameters
        ----------
        bundle_path:
            Path to the normalized bundle directory produced by the downloader.

        Returns
        -------
        CheckpointEntry
            The newly registered entry.

        Raises
        ------
        CheckpointImportError
            On validation failure, collision, or copy error.
        """
        bundle_path = Path(bundle_path)
        # Quick sanity: source.json must declare huggingface type
        source_file = bundle_path / "source.json"
        if source_file.exists():
            try:
                data = json.loads(source_file.read_text(encoding="utf-8"))
                if data.get("type", "") not in ("huggingface",):
                    raise CheckpointImportError(
                        f"Expected source.json type='huggingface', "
                        f"got '{data.get('type', '')}' in {source_file}"
                    )
            except CheckpointImportError:
                raise
            except Exception as exc:
                raise CheckpointImportError(
                    f"Cannot read source.json: {exc}"
                ) from exc
        # Delegate to import_local which handles atomic copy + registry refresh
        return self.import_local(bundle_path)

    # ------------------------------------------------------------------
    # Import helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_source(src_path: Path) -> Path:
        """Return the directory to import from.

        For zip files: extract to a temp directory and return its path.
        For directories: return unchanged.
        Raises CheckpointImportError for anything else.
        """
        if not src_path.exists():
            raise CheckpointImportError(f"Source path does not exist: {src_path}")

        if src_path.is_dir():
            return src_path

        if src_path.suffix.lower() == ".zip":
            import zipfile
            if not zipfile.is_zipfile(src_path):
                raise CheckpointImportError(f"File is not a valid zip archive: {src_path}")
            tmp = Path(tempfile.mkdtemp(prefix="unitport_cp_import_"))
            with zipfile.ZipFile(src_path, "r") as zf:
                zf.extractall(tmp)
            # If zip contains a single root directory, descend into it
            children = [c for c in tmp.iterdir()]
            if len(children) == 1 and children[0].is_dir():
                return children[0]
            return tmp

        raise CheckpointImportError(
            f"Unsupported source type '{src_path.suffix}'. "
            "Please select a directory or a .zip archive."
        )

    @staticmethod
    def _read_policy_id(manifest_path: Path, bundle_path: Path) -> str:
        """Parse manifest.yaml and return the bundle directory name as policy_id.

        The policy_id is the *bundle directory name* (= bundle_path.name), not
        the manifest 'name' field — so the user can have custom display names
        without affecting the registry key.
        """
        try:
            import yaml
            with manifest_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise CheckpointImportError("manifest.yaml must contain a YAML mapping.")
            # Minimal required field check
            if "name" not in data:
                raise CheckpointImportError(
                    "manifest.yaml is missing required field 'name'."
                )
        except CheckpointImportError:
            raise
        except Exception as exc:
            raise CheckpointImportError(
                f"Failed to parse manifest.yaml: {exc}"
            ) from exc
        return bundle_path.name

    @staticmethod
    def peek_manifest(src_path: Path) -> dict:
        """Parse and return manifest fields from a source path for preview UI.

        Accepts a directory or .zip.  Returns a dict with keys:
        ``name``, ``version``, ``robot_brand``, ``obs_dim``, ``action_dim``,
        ``policy_id`` (= directory name).

        Raises ``CheckpointImportError`` on any parse failure.
        """
        src_path = Path(src_path)
        tmp_dir: Optional[Path] = None
        try:
            bundle_src = CheckpointRegistry._resolve_source(src_path)
            tmp_dir = bundle_src if bundle_src != src_path else None

            manifest_path = bundle_src / "manifest.yaml"
            if not manifest_path.exists():
                raise CheckpointImportError(
                    f"No manifest.yaml found in: {bundle_src}"
                )
            try:
                import yaml
                with manifest_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception as exc:
                raise CheckpointImportError(
                    f"Failed to parse manifest.yaml: {exc}"
                ) from exc

            robot_section = data.get("robot", {}) or {}
            obs_section = data.get("observation_space", {}) or {}
            act_section = data.get("action_space", {}) or {}
            skill_section = data.get("skill", {}) or {}

            result = {
                "policy_id": bundle_src.name,
                "name": str(data.get("name", "") or bundle_src.name),
                "version": str(data.get("version", "") or ""),
                "robot_brand": str(
                    robot_section.get("brand", "")
                    or data.get("robot_brand", "")
                    or ""
                ),
                "obs_dim": obs_section.get("dim", "—"),
                "action_dim": act_section.get("dim", "—"),
            }

            # SkillManifest v2 fields (from "skill" section or inferred)
            if skill_section:
                result.update({
                    "action_space_type": skill_section.get("action_space_type", ""),
                    "control_frequency_hz": skill_section.get("control_frequency_hz", ""),
                    "target_robot_family": skill_section.get("target_robot_family", ""),
                    "required_sensors": skill_section.get("required_sensors", []),
                    "precondition_posture": (skill_section.get("precondition") or {}).get("posture", ""),
                    "postcondition_posture": (skill_section.get("postcondition") or {}).get("posture", ""),
                    "source_type": skill_section.get("source_type", ""),
                    "inference_backend": skill_section.get("inference_backend", ""),
                    "has_skill_manifest": True,
                })
                # Override dims from skill section if present (more reliable)
                if skill_section.get("observation_dim"):
                    result["obs_dim"] = skill_section["observation_dim"]
                if skill_section.get("action_dim"):
                    result["action_dim"] = skill_section["action_dim"]
            else:
                result["has_skill_manifest"] = False

            return result
        finally:
            if tmp_dir is not None and tmp_dir != src_path and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Project-aware discovery (delegates to AssetResolver)
    # ------------------------------------------------------------------

    def discover_for_project(
        self,
        project_store: "ProjectStore",
        project_id: str,
    ) -> List[CheckpointEntry]:
        """Discover checkpoints merged from project-local + shared.

        Uses :class:`AssetResolver` to combine both scopes.  Each returned
        ``CheckpointEntry`` has its ``scope`` field set to ``"local"`` or
        ``"shared"``, and shared entries that collide with local ones have
        ``display_name`` suffixed with ``(GLOB)``.

        This does NOT update the internal ``_cache`` — the legacy cache
        continues to serve callers that use ``discover()`` / ``list_checkpoints()``
        without a project context.

        Parameters
        ----------
        project_store : ProjectStore
            The store that owns path layout.
        project_id : str
            Active project to resolve against.
        """
        from src.system.core.asset_resolver import AssetResolver

        resolver = AssetResolver(project_store, project_id)
        resolved = resolver.list_available_checkpoints()

        entries: List[CheckpointEntry] = []
        for asset in resolved:
            manifest_path = asset.path / "manifest.yaml"
            if manifest_path.exists():
                entry = self._parse_entry(asset.policy_id, asset.path, manifest_path)
            else:
                entry = CheckpointEntry(
                    policy_id=asset.policy_id,
                    bundle_path=asset.path,
                    display_name=asset.display_name,
                    is_valid=False,
                    error="manifest.yaml not found",
                )
            entry.scope = asset.scope
            entry.display_name = asset.display_name
            entries.append(entry)

        return entries

    def list_checkpoints_for_project(
        self,
        project_store: "ProjectStore",
        project_id: str,
        robot_brand: Optional[str] = None,
    ) -> List[CheckpointEntry]:
        """Project-aware version of :meth:`list_checkpoints`.

        Merges project-local + shared checkpoints.  Optionally filters by
        robot brand.
        """
        entries = self.discover_for_project(project_store, project_id)
        if robot_brand is None:
            return entries
        brand_lower = robot_brand.lower()
        return [
            e for e in entries
            if e.is_valid and e.robot_brand.lower() == brand_lower
        ]

    # ------------------------------------------------------------------
    # Import: Isaac Lab exported bundle
    # ------------------------------------------------------------------

    def import_isaac_lab_bundle(
        self,
        exported_dir: Path,
        *,
        dest_root: Optional[Path] = None,
        policy_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> CheckpointEntry:
        """Import an Isaac Lab ``exported/`` directory into the registry.

        Steps:
          1. Locate ``policy.onnx`` (or ``policy.pt``) and ``env.yaml``
          2. Parse ``env.yaml`` → generate ``manifest.yaml`` v2 via
             :func:`isaac_lab_manifest_parser.parse_isaac_lab_env_yaml`
          3. Copy artifacts into a bundle directory
          4. Write ``source.json`` with ``type = "isaac_lab"``
          5. Register in the checkpoint cache

        Parameters
        ----------
        exported_dir:
            Path to the Isaac Lab export output (contains policy.onnx + env.yaml).
        dest_root:
            Override destination root. When a project is active, pass the
            project-local checkpoints directory. Falls back to ``self.root``.
        policy_id:
            Override bundle name. Defaults to the export directory name.

        Returns
        -------
        CheckpointEntry

        Raises
        ------
        CheckpointImportError
        """
        exported_dir = Path(exported_dir)
        if not exported_dir.is_dir():
            raise CheckpointImportError(
                f"Isaac Lab export directory not found: {exported_dir}"
            )

        # 1. Locate env.yaml — may sit directly in the export dir (play.py's
        # exported/ output) or inside a params/ subfolder (RSL-RL log_dir layout).
        env_yaml: Optional[Path] = None
        for candidate in (exported_dir / "env.yaml", exported_dir / "params" / "env.yaml"):
            if candidate.exists():
                env_yaml = candidate
                break
        if env_yaml is None:
            raise CheckpointImportError(
                f"env.yaml not found in {exported_dir} or {exported_dir / 'params'}"
            )

        # 2. Locate policy file. Preference order:
        #    a) policy.onnx / policy.pt at root (play.py export layout)
        #    b) latest model_<iter>.pt from RSL-RL (raw log_dir layout)
        #    c) exported/policy.onnx nested one level down
        policy_file: Optional[Path] = None
        for name in ("policy.onnx", "policy.pt"):
            candidate = exported_dir / name
            if candidate.exists():
                policy_file = candidate
                break
        if policy_file is None:
            nested_exported = exported_dir / "exported"
            if nested_exported.is_dir():
                for name in ("policy.onnx", "policy.pt"):
                    candidate = nested_exported / name
                    if candidate.exists():
                        policy_file = candidate
                        break
        if policy_file is None:
            # RSL-RL OnPolicyRunner writes model_<iter>.pt directly in log_dir.
            # Pick the one with the highest iteration number.
            def _iter_num(p: Path) -> int:
                try:
                    return int(p.stem.split("_", 1)[1])
                except (IndexError, ValueError):
                    return -1
            model_pts = sorted(
                exported_dir.glob("model_*.pt"), key=_iter_num, reverse=True,
            )
            if model_pts and _iter_num(model_pts[0]) >= 0:
                policy_file = model_pts[0]
        if policy_file is None:
            raise CheckpointImportError(
                f"No policy.onnx, policy.pt, or model_*.pt in {exported_dir}"
            )

        # ── 2.5 Detect raw rsl_rl checkpoint and convert in-process ──
        # When the IL training pipeline's ONNX export step (Isaac Lab
        # play.py) fails or produces output in an unexpected location,
        # we end up here with a raw rsl_rl ``model_*.pt`` checkpoint
        # (PyTorch zip-format state dict). The runtime stack only knows
        # how to load ONNX or TorchScript — a raw state dict makes
        # ``onnxruntime`` blow up with INVALID_PROTOBUF.
        #
        # Solution: convert in-process using torch (which UnitPort venv
        # has) by reconstructing the rsl_rl actor MLP from agent.yaml
        # and exporting via torch.onnx.export. The bundle ends up with
        # a self-contained ``policy.onnx`` and the manifest is rewritten
        # to point at it. No Isaac Lab subprocess required, no torch
        # dependency at replay time.
        policy_write_name = policy_file.name  # what we'll actually copy
        policy_write_format = "onnx"
        rsl_rl_export_meta: Optional[Dict[str, Any]] = None

        suffix = policy_file.suffix.lower()
        if suffix == ".pt" and policy_file.stem.startswith("model_"):
            # rsl_rl raw checkpoint — locate the matching agent.yaml
            agent_yaml_candidates = [
                exported_dir / "params" / "agent.yaml",
                exported_dir / "agent.yaml",
                policy_file.parent / "params" / "agent.yaml",
                policy_file.parent / "agent.yaml",
            ]
            agent_yaml: Optional[Path] = None
            for cand in agent_yaml_candidates:
                if cand.is_file():
                    agent_yaml = cand
                    break
            if agent_yaml is None:
                raise CheckpointImportError(
                    f"Found raw rsl_rl checkpoint {policy_file.name} but no "
                    f"agent.yaml in {exported_dir} or params/. Cannot reconstruct "
                    f"the actor MLP for ONNX export. Looked in: "
                    f"{[str(c) for c in agent_yaml_candidates]}"
                )

            # Stage destination first so we can write straight there.
            policy_write_name = "policy.onnx"

        # 2. Parse env.yaml → SkillManifest
        from src.system.skill.isaac_lab_manifest_parser import (
            parse_isaac_lab_env_yaml,
        )

        pid = policy_id or exported_dir.name
        manifest = parse_isaac_lab_env_yaml(
            env_yaml,
            skill_id=pid,
            model_path=policy_write_name,
        )

        # 3. Prepare destination — always project-scoped unless caller forces.
        storage = Path(dest_root) if dest_root is not None else self._resolve_write_root()
        storage.mkdir(parents=True, exist_ok=True)

        # Auto-version on collision so the user can re-pick the same
        # checkpoint name across multiple training runs without having to
        # manually clear the previous bundle. Append ``_v2``, ``_v3``…
        # until we find a free directory; strip any pre-existing
        # ``_v<N>`` suffix first so repeated imports do not nest into
        # ``foo_v2_v2_v2``. The bumped name becomes the canonical
        # ``policy_id`` everywhere downstream — re-issue the manifest's
        # ``skill_id``/``skill_name`` so the on-disk dir name and the
        # embedded ids stay in sync.
        if (storage / pid).exists():
            import re as _re_iv
            base = pid
            n = 2
            m = _re_iv.match(r"^(?P<stem>.+)_v(?P<n>\d+)$", base)
            if m:
                base = m.group("stem")
                n = int(m.group("n")) + 1
            while (storage / f"{base}_v{n}").exists() and n < 10000:
                n += 1
            pid = f"{base}_v{n}"
            try:
                manifest.skill_id = pid
                manifest.skill_name = pid
            except Exception:
                pass
        dest = storage / pid

        staging = storage / f"{pid}__importing"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        # Write policy file. Two paths:
        #   - native ONNX/JIT: straight copy
        #   - rsl_rl raw .pt:  reconstruct actor MLP + export to staging/policy.onnx
        if policy_write_name == "policy.onnx" and suffix == ".pt":
            from src.system.skill.il_policy_onnx_export import (
                export_rsl_rl_actor_to_onnx,
            )
            try:
                rsl_rl_export_meta = export_rsl_rl_actor_to_onnx(
                    checkpoint_path=policy_file,
                    agent_yaml_path=agent_yaml,  # noqa: F821 — set in the .pt branch above
                    onnx_out_path=staging / "policy.onnx",
                )
            except Exception as exc:
                shutil.rmtree(staging, ignore_errors=True)
                raise CheckpointImportError(
                    f"Failed to convert rsl_rl checkpoint {policy_file} to ONNX: "
                    f"{exc}. The bundle import was rolled back."
                ) from exc
        else:
            shutil.copy2(policy_file, staging / policy_write_name)

        # Copy env.yaml as provenance
        shutil.copy2(env_yaml, staging / "env.yaml")

        # 4. Generate manifest.yaml v2
        manifest_dict = self._skill_manifest_to_dict(manifest)

        # ── B6 (AMP fix plan): AMP-aware bundle enrichment ──────────────
        # When the source run was AMP_PPO we augment the bundle with:
        #   * ``discriminator.pt``  — the trained discriminator state
        #     dict, extracted straight from the source checkpoint.
        #     Needed for offline style-reward debugging and for resuming
        #     AMP training from the bundle (spec §6.discriminator.pt).
        #   * ``amp_metadata`` block in manifest.yaml — the authoritative
        #     record of dataset composition, obs term layout, and
        #     hyperparameters the discriminator was trained against.
        #     Consumed by PolicyRunner (H3) and by the offline debug
        #     tool (M6).
        #   * ``unitport_run_meta.yaml`` — copied verbatim from the source
        #     run directory so the bundle self-describes without requiring
        #     the original run dir to still exist.
        #
        # Dispatch is authoritative: we read ``unitport_run_meta.yaml``
        # first (H2), with a checkpoint-heuristic fallback for runs
        # created before the meta file existed. Non-AMP runs skip this
        # entire block so the PPO / SAC import path is unchanged.
        from src.system.training.run_meta import (
            detect_algorithm_class,
            read_run_meta,
            verify_spec_hash,
            sha256_file,
        )

        source_algorithm = detect_algorithm_class(exported_dir, default="PPO")
        run_meta = read_run_meta(exported_dir)
        manifest_dict["algorithm"] = source_algorithm

        if source_algorithm == "AMP_PPO":
            # H1: verify spec_hash if the run meta carries one. We warn
            # rather than abort on mismatch — the bundle is still usable
            # for inference, only resume-training and offline debugging
            # are affected. Aborting would strand users who legitimately
            # edited their meta file (e.g. to correct a dataset path).
            if run_meta is not None:
                ok, msg = verify_spec_hash(run_meta)
                if not ok:
                    print(
                        f"[UnitPort][AMP] spec_hash verification failed: {msg}. "
                        f"Bundle will still import but ``spec_hash_verified`` "
                        f"will be False in manifest.",
                        flush=True,
                    )

            # Extract the discriminator state dict from the source
            # checkpoint. We intentionally drop the optimizer and AMP
            # replay buffer — those are training-only state and would
            # bloat the bundle for no deployment benefit.
            disc_archived = False
            disc_keys_saved: List[str] = []
            try:
                if suffix == ".pt":
                    import torch as _torch_disc
                    src_ckpt = _torch_disc.load(
                        str(policy_file), map_location="cpu", weights_only=False,
                    )
                    if isinstance(src_ckpt, dict) and "discriminator_state_dict" in src_ckpt:
                        disc_sd = src_ckpt["discriminator_state_dict"]
                        disc_bundle = {
                            "discriminator_state_dict": disc_sd,
                        }
                        # amp_normalizer carries the running mean/std the
                        # discriminator was fed during training — strictly
                        # required for faithful offline replay of the
                        # style reward. Include it when present.
                        if "amp_normalizer" in src_ckpt:
                            disc_bundle["amp_normalizer"] = src_ckpt["amp_normalizer"]
                        _torch_disc.save(disc_bundle, staging / "discriminator.pt")
                        disc_archived = True
                        disc_keys_saved = list(disc_bundle.keys())
                        print(
                            f"[UnitPort][AMP] Archived discriminator.pt "
                            f"({len(disc_sd)} tensors)",
                            flush=True,
                        )
            except Exception as exc:
                print(
                    f"[UnitPort][AMP] WARNING: could not extract "
                    f"discriminator from {policy_file}: {exc}. Bundle "
                    f"will mark discriminator_archived=False.",
                    flush=True,
                )

            # Copy the run meta file verbatim so the bundle is
            # self-contained even after the source run dir is deleted.
            if run_meta is not None:
                try:
                    shutil.copy2(
                        exported_dir / "unitport_run_meta.yaml",
                        staging / "unitport_run_meta.yaml",
                    )
                except OSError:
                    pass

            # Build the amp_metadata block. The discriminator source of
            # truth is the run_meta; we enrich dataset_files with sha256
            # at publish time so consumers can verify the dataset hasn't
            # drifted under them.
            amp_src = (run_meta or {}).get("amp", {}) if isinstance(run_meta, dict) else {}
            raw_dataset_files = amp_src.get("dataset_files") or []
            dataset_files_enriched: List[Dict[str, Any]] = []
            for item in raw_dataset_files:
                if not isinstance(item, dict):
                    continue
                entry: Dict[str, Any] = {
                    "path": str(item.get("path", "")),
                    "basename": str(
                        item.get("basename", "") or Path(str(item.get("path", ""))).name
                    ),
                }
                src_path = Path(entry["path"])
                if src_path.is_file():
                    try:
                        entry["sha256"] = sha256_file(src_path)
                    except OSError as exc:
                        entry["sha256"] = None
                        entry["sha256_error"] = str(exc)
                else:
                    entry["sha256"] = None
                dataset_files_enriched.append(entry)

            manifest_dict["amp_metadata"] = {
                "amp_obs_terms": list(amp_src.get("amp_obs_terms") or []),
                "amp_obs_dim": int(amp_src.get("amp_obs_dim") or 0),
                "num_amp_obs_history": int(amp_src.get("num_amp_obs_history") or 1),
                "amp_reward_coef": float(amp_src.get("amp_reward_coef") or 0.0),
                "task_reward_lerp": float(amp_src.get("task_reward_lerp") or 0.0),
                "transition_dt": float(amp_src.get("transition_dt") or 0.0),
                "dataset_files": dataset_files_enriched,
                "discriminator_archived": bool(disc_archived),
                "discriminator_file": "discriminator.pt" if disc_archived else None,
                "discriminator_keys": disc_keys_saved,
                "spec_hash": (run_meta or {}).get("spec_hash"),
                "spec_hash_verified": (
                    bool(run_meta is not None and verify_spec_hash(run_meta)[0])
                ),
                "source_algorithm": source_algorithm,
                "note": (
                    "AMP metadata for training-time provenance and offline "
                    "discriminator replay. The deployable policy.onnx does "
                    "NOT include the discriminator — inference uses the "
                    "actor alone. See knowledge_base/amp_fix.yaml §B6."
                ),
            }

        import yaml as _yaml
        manifest_yaml_path = staging / "manifest.yaml"
        with manifest_yaml_path.open("w", encoding="utf-8") as fh:
            _yaml.dump(manifest_dict, fh, default_flow_style=False, allow_unicode=True)

        # Write source.json. ``task_name`` is the gym registration ID
        # (e.g. "Isaac-Velocity-Rough-Unitree-Go2-v0") used by play.py at
        # review time. It is NOT in env.yaml — Isaac Lab dumps the env_cfg
        # dataclass, not the gym task ID — so the caller must thread it in
        # explicitly from the training launch site.
        source_meta = {
            "type": "isaac_lab",
            "src": str(exported_dir.resolve()),
            "env_yaml": "env.yaml",
        }
        if task_name:
            source_meta["task_name"] = str(task_name)
        (staging / "source.json").write_text(
            json.dumps(source_meta, indent=2), encoding="utf-8",
        )

        # 5. Atomic rename + refresh + register in project asset index
        staging.rename(dest)
        self._register_in_project_meta(pid, dest)
        self.refresh()
        return self.get(pid)

    def _register_in_project_meta(self, policy_id: str, dest: Path) -> None:
        """Append a CheckpointRef to project.json if a project is active.

        No-op when there is no active project (e.g. tests using an
        ``explicit_root``).  Idempotent — duplicate inserts are skipped.
        """
        try:
            store, pid = self._resolve_active_project()
            if store is None or not pid:
                return
            # Only register when the destination actually lives inside this
            # project — guards against tests that pass an explicit dest_root.
            try:
                ckpt_dir = store._checkpoints_dir(pid)
                dest.relative_to(ckpt_dir)
            except (ValueError, AttributeError):
                return
            from src.system.core.project_store import CheckpointRef
            meta = store.open_project(pid)
            if meta.assets.find_checkpoint(policy_id) is None:
                meta.assets.checkpoints.append(
                    CheckpointRef(
                        policy_id=policy_id,
                        path=f"training/exported/{policy_id}/",
                    )
                )
                store.save_meta(meta)
        except Exception:
            # Registration is best-effort — never fail an import for it.
            pass

    @staticmethod
    def _skill_manifest_to_dict(sm) -> dict:
        """Convert a SkillManifest to a v2 manifest.yaml dict."""
        skill_dict = {
            "skill_id": sm.skill_id,
            "skill_name": sm.skill_name,
            "version": sm.version,
            "source_type": sm.source_type.value if hasattr(sm.source_type, "value") else str(sm.source_type),
            "execution_mode": sm.execution_mode,
            "action_space_type": sm.action_space_type.value if hasattr(sm.action_space_type, "value") else str(sm.action_space_type),
            "action_dim": sm.action_dim,
            "control_frequency_hz": sm.control_frequency_hz,
            "required_sensors": list(sm.required_sensors),
            "observation_space_keys": list(sm.observation_space_keys),
            "observation_dim": sm.observation_dim,
            "target_robot_family": sm.target_robot_family,
            "target_robot_models": list(sm.target_robot_models),
            "joint_mapping": sm.joint_mapping,
            "precondition": {
                "posture": sm.precondition.posture,
                "posture_tolerance_rad": sm.precondition.posture_tolerance_rad,
                "velocity_max_mps": sm.precondition.velocity_max_mps,
            },
            "postcondition": {
                "posture": sm.postcondition.posture,
                "posture_tolerance_rad": sm.postcondition.posture_tolerance_rad,
                "velocity_max_mps": sm.postcondition.velocity_max_mps,
            },
            "inference_backend": sm.inference_backend.value if hasattr(sm.inference_backend, "value") else str(sm.inference_backend),
            "model_path": sm.model_path,
            "normalize_obs": sm.normalize_obs,
            "normalizer_path": sm.normalizer_path,
            "training_source": sm.training_source,
            "description": sm.description,
            "tags": list(sm.tags),
        }

        # command_interface (v2)
        if sm.command_interface is not None:
            ci = sm.command_interface
            skill_dict["command_interface"] = {
                "type": ci.type,
                "fields": [
                    {
                        "name": f.name,
                        "obs_index": f.obs_index,
                        "range": list(f.range),
                        "default": f.default,
                    }
                    for f in ci.fields
                ],
                "input_sources": list(ci.input_sources),
            }

        return {
            "name": sm.skill_name,
            "version": sm.version,
            "skill": skill_dict,
        }
