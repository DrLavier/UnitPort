"""application.service.exported_bundle_registry — Export bundle discovery.

DEMO 对应：``DEMO/src/system/service/checkpoint_registry.py:CheckpointRegistry``
(reduced to the discover-only subset Export node UI needs).

**重命名理由**：RELEASE 已存在 ``application.service.checkpoint_registry``，
但其语义是 *training run 中的 .pt 检查点* (``runs/<run_id>/checkpoints/``)，
而本模块管理 *训练完成后导出的 bundle 目录* (``training/exported/<policy_id>/``)。
两者都叫 "checkpoint" 但 scope 不同；Stage C 把后者改名 ``ExportedBundleRegistry``
避免歧义。

Consumed by:
    * Export 节点 ``OutputFileListRow`` — auto-detect existing bundle dirs +
      "Will Overwrite" badge.
    * (Future) Export bundle picker dropdowns — choose existing bundles to
      overwrite vs. create new.

Search root:
    Active project's ``<project>/training/exported/`` directory (resolved via
    ``application.service.projects.current_project_info``). With no project
    bound discovery returns an empty list — bundles never live outside the
    project tree (RELEASE/CLAUDE.md §1.4).

Each discovered ``ExportedBundleEntry`` requires a readable ``manifest.yaml``;
parse failures surface as ``is_valid=False`` rather than being silently
dropped, so the picker can show ⚠ for broken bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from unitport_sdk import log_warning


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ExportedBundleEntry:
    """Metadata for one exported bundle directory."""

    policy_id: str
    """Leaf directory name (the bundle id within its backend partition)."""
    bundle_path: Path
    """Absolute path to the bundle directory."""
    backend_id: str = ""
    """Parent partition under ``training/exported/`` — e.g. ``"sb3_mujoco"``,
    ``"isaac_lab"``. ``"(legacy)"`` for pre-backend-layer flat bundles
    discovered directly under ``exported/``."""
    display_name: str = ""
    """Human-readable name from ``manifest.yaml#name`` (falls back to policy_id)."""
    version: str = ""
    """Bundle version string from ``manifest.yaml#version``."""
    robot_sku: str = ""
    """Canonical SKU from ``manifest.yaml#robot.sku``. Empty for legacy
    bundles written before the SKU-only contract — those will fail
    BundleLoader validation; the picker row stays visible but selecting
    them surfaces a clear "re-train" error."""
    is_valid: bool = True
    """False when ``manifest.yaml`` missing/unparseable."""
    error: str = ""
    """Non-empty when ``is_valid=False`` — parse/validation message."""
    source_type: str = ""
    """Bundle origin: 'local' | 'training' | '' (unknown). Read from
    sibling ``source.json#type`` when present."""
    scope: str = ""
    """Discovery root: 'project' | 'user'."""

    def label(self) -> str:
        """Short label for picker rows. Display brand resolved via
        registry from the canonical SKU — never reads brand strings out
        of the manifest directly (those are informational only)."""
        name = self.display_name or self.policy_id
        parts = [name]
        if self.version:
            parts.append(f"v{self.version}")
        if self.robot_sku:
            display = self.robot_sku  # SKU-as-fallback when registry doesn't know
            try:
                from registers.robots import get_robot_spec, resolve_id
                rs = get_robot_spec(resolve_id(self.robot_sku) or self.robot_sku)
                if rs is not None:
                    display = rs.name or rs.brand or self.robot_sku
            except Exception:
                pass
            parts.append(f"[{display}]")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ExportedBundleRegistry:
    """Scan + index export bundles in a stable order."""

    def __init__(self, root: Optional[Path] = None) -> None:
        """Test override: pass an explicit ``root`` to scan only that dir."""
        self._explicit_root: Optional[Path] = root
        self._cache: List[ExportedBundleEntry] = []

    # ---- root resolution ----

    @staticmethod
    def _project_root() -> Optional[Path]:
        """Active project's ``training/exported/`` directory or None.

        Reads ``current_project_info()`` (the canonical "what project is
        loaded" accessor — same one TrainingAssetsCache uses).
        """
        try:
            from application.service.projects import current_project_info
            info = current_project_info()
            if info is not None:
                return Path(info.path) / "training" / "exported"
        except Exception:
            pass
        return None

    # ---- discovery ----

    def discover(self) -> List[ExportedBundleEntry]:
        """Walk ``exported/`` two levels deep::

            exported/<backend_id>/<policy_id>/manifest.yaml

        Each first-level child that itself has a ``manifest.yaml`` is
        also accepted as a *legacy* flat bundle (``backend_id="(legacy)"``).
        Dedup key is ``(backend_id, policy_id)`` so the same leaf name
        across different backends is preserved.
        """
        results: List[ExportedBundleEntry] = []
        seen: set = set()  # (backend_id, policy_id)

        def _emit_bundle(
            backend_id: str, policy_id: str, cand: Path, scope: str,
        ) -> None:
            key = (backend_id, policy_id)
            if key in seen:
                return
            manifest_path = cand / "manifest.yaml"
            if not manifest_path.exists():
                results.append(ExportedBundleEntry(
                    policy_id=policy_id,
                    bundle_path=cand,
                    backend_id=backend_id,
                    is_valid=False,
                    error="manifest.yaml missing",
                    scope=scope,
                ))
                seen.add(key)
                return
            entry = self._parse_entry(policy_id, cand, manifest_path)
            entry.backend_id = backend_id
            entry.scope = scope
            results.append(entry)
            seen.add(key)

        def _scan(directory: Optional[Path], scope: str) -> None:
            if directory is None or not directory.exists():
                return
            try:
                level1_children = sorted(directory.iterdir())
            except OSError as e:
                log_warning(f"ExportedBundleRegistry: scan {directory!s} failed: {e}")
                return
            for level1 in level1_children:
                if not level1.is_dir():
                    continue
                # Legacy flat bundle (manifest.yaml directly inside).
                if (level1 / "manifest.yaml").exists():
                    _emit_bundle("(legacy)", level1.name, level1, scope)
                    continue
                # Backend partition — recurse one level for bundles.
                try:
                    level2_children = sorted(level1.iterdir())
                except OSError as e:
                    log_warning(
                        f"ExportedBundleRegistry: scan {level1!s} failed: {e}"
                    )
                    continue
                for cand in level2_children:
                    if not cand.is_dir():
                        continue
                    _emit_bundle(level1.name, cand.name, cand, scope)

        if self._explicit_root is not None:
            _scan(self._explicit_root, "test")
        else:
            _scan(self._project_root(), "project")

        self._cache = results
        return results

    def _parse_entry(
        self,
        policy_id: str,
        bundle_path: Path,
        manifest_path: Path,
    ) -> ExportedBundleEntry:
        # Integrity check against manifest.sha256 sidecar (Plan finding P2-1).
        # Mismatch ⇒ refuse to load; missing sidecar ⇒ one-shot WARN
        # (back-compat for bundles exported before the sidecar was added).
        sidecar = manifest_path.parent / "manifest.sha256"
        if sidecar.is_file():
            try:
                expected = sidecar.read_text(encoding="utf-8").strip().split()
                expected_hex = expected[0].lower() if expected else ""
                import hashlib
                got_hex = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if expected_hex and got_hex != expected_hex:
                    return ExportedBundleEntry(
                        policy_id=policy_id,
                        bundle_path=bundle_path,
                        is_valid=False,
                        error=(
                            f"manifest.sha256 mismatch — bundle integrity "
                            f"check failed (expected {expected_hex}, got {got_hex})"
                        ),
                    )
            except OSError as exc:
                log_warning(
                    f"ExportedBundleRegistry: {bundle_path!s} sidecar read failed "
                    f"({exc}); proceeding without integrity check."
                )
        else:
            # WHY KEPT: bundles exported before manifest.sha256 was added do
            # not have a sidecar; surface a directive to re-export rather
            # than block loads. CLAUDE.md §1.8 (c) on-disk legacy compat.
            log_warning(
                f"ExportedBundleRegistry: {bundle_path!s} has no manifest.sha256 "
                f"sidecar (legacy bundle?). Re-export to enable integrity "
                f"verification on next load."
            )

        try:
            import yaml
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as e:
            return ExportedBundleEntry(
                policy_id=policy_id,
                bundle_path=bundle_path,
                is_valid=False,
                error=f"manifest parse failed: {e}",
            )
        if not isinstance(data, dict):
            return ExportedBundleEntry(
                policy_id=policy_id,
                bundle_path=bundle_path,
                is_valid=False,
                error="manifest.yaml is not a mapping",
            )
        robot = data.get("robot") if isinstance(data.get("robot"), dict) else {}
        entry = ExportedBundleEntry(
            policy_id=policy_id,
            bundle_path=bundle_path,
            display_name=str(data.get("name", "") or ""),
            version=str(data.get("version", "") or ""),
            robot_sku=str((robot or {}).get("sku", "") or ""),
            is_valid=True,
        )
        # source.json gives bundle origin (training / local-import / ...)
        source_path = bundle_path / "source.json"
        if source_path.exists():
            try:
                import json
                with open(source_path, "r", encoding="utf-8") as fh:
                    src = json.load(fh) or {}
                if isinstance(src, dict):
                    entry.source_type = str(src.get("type", "") or "")
            except Exception:
                pass
        return entry

    @property
    def cache(self) -> List[ExportedBundleEntry]:
        return list(self._cache)


# ---------------------------------------------------------------------------
# Convenience helpers — Export node UI consumes these directly
# ---------------------------------------------------------------------------


def list_exported_bundle_ids(*, valid_only: bool = True) -> List[str]:
    """Sorted policy_ids visible to the active project.

    Used by the Export node's bundle picker (Stage B+) to populate its
    "existing checkpoints" choice list. Returns ``[]`` when no project is
    bound — bundles live only under ``<project>/training/exported/``.
    """
    reg = ExportedBundleRegistry()
    entries = reg.discover()
    if valid_only:
        entries = [e for e in entries if e.is_valid]
    return sorted(e.policy_id for e in entries)


def discover_exported_bundles() -> List[ExportedBundleEntry]:
    """One-shot discovery — convenience wrapper around ``ExportedBundleRegistry().discover()``."""
    return ExportedBundleRegistry().discover()


__all__ = [
    "ExportedBundleEntry",
    "ExportedBundleRegistry",
    "list_exported_bundle_ids",
    "discover_exported_bundles",
]
