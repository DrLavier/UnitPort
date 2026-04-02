#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckpointRegistry — single source of truth for deployed checkpoint bundles.

Checkpoints live under  <project_root>/custom_mods/training/checkpoints/  (one sub-directory
per bundle, each containing a manifest.yaml).  This registry provides a
lightweight discovery layer used by the Checkpoints sidebar panel and by
any training / inference code that needs to enumerate bundles without
importing the heavier BundleLoader stack.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
    Scans and indexes checkpoint bundles under ``custom_mods/training/checkpoints/``.

    Canonical checkpoint storage path:
        <project_root>/custom_mods/training/checkpoints/<policy_id>/manifest.yaml

    Usage::

        registry = CheckpointRegistry()
        entries = registry.discover()

        registry.refresh()                                    # re-scan
        all_cp = registry.list_checkpoints()                  # cached list
        unitree = registry.list_checkpoints(robot_brand="unitree")
        entry   = registry.get(policy_id="go2_v1")            # KeyError if missing
        path    = registry.get_bundle_path("go2_v1")          # Path | KeyError

        # Import a local bundle directory or zip:
        entry = registry.import_local(Path("/downloads/go2_walk/"))
    """

    #: Sub-directory of project_root used as checkpoint storage.
    STORAGE_SUBDIR = "custom_mods/training/checkpoints"

    def __init__(self, root: Optional[Path] = None):
        """
        Parameters
        ----------
        root:
            Explicit root override (useful for tests).  When *None* the root is
            resolved lazily from ConfigManager on first access.
        """
        self._explicit_root: Optional[Path] = root
        self._cache: List[CheckpointEntry] = []
        self._discovered: bool = False

    # ------------------------------------------------------------------
    # Root resolution
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Resolved storage root directory (always absolute).

        ``custom_mods/`` lives at the **repo root** (one level above
        ``project_root`` which is ``src/``), so we go up one level.
        """
        if self._explicit_root is not None:
            return self._explicit_root
        try:
            from src.system.core.config_manager import ConfigManager  # lazy import
            return ConfigManager().get_path("project_root").parent / self.STORAGE_SUBDIR
        except Exception:
            return Path.cwd() / self.STORAGE_SUBDIR

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> List[CheckpointEntry]:
        """Scan ``root`` and return a ``CheckpointEntry`` for every candidate bundle.

        A candidate is any direct sub-directory that contains a ``manifest.yaml``.
        Bundles whose YAML cannot be parsed are returned with ``is_valid=False``.
        Returns an empty list when ``root`` does not exist.
        """
        if not self.root.exists():
            self._cache = []
            self._discovered = True
            return []

        results: List[CheckpointEntry] = []
        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir():
                continue
            manifest_path = candidate / "manifest.yaml"
            if not manifest_path.exists():
                continue
            policy_id = candidate.name
            entry = self._parse_entry(policy_id, candidate, manifest_path)
            results.append(entry)

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

            # 4. Check no collision
            dest = self.root / policy_id
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
            self.root.mkdir(parents=True, exist_ok=True)
            staging = self.root / f"{policy_id}__importing"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(bundle_src, staging)
            staging.rename(dest)

            # 7. Refresh and return
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
