"""Disk scanners for the Resources panel.

Two separate scanners because the two asset kinds have different on-disk
shapes and live alongside legacy roots that **already** know how to walk
themselves:

- **Motion**: do NOT re-implement. ``scripts.training_motion.library``
  already scans ``custom_mods/{motions,archives}`` plus the user overlay
  root and emits ``MotionPack`` entries that drive the Training Motion
  node picker. ``ResourceManager.list_motion_packs()`` delegates there so
  the Resources panel and the canvas dropdown agree byte-for-byte.

- **Policy**: new scanner. Walks ``Paths.CUSTOM_MODS_DIR / "policies" /
  <pack_id>/`` looking for ``manifest.yaml`` files. Validation is the
  same schema used by ``application.training.manifest_schema`` (the SKU-
  only contract). Invalid bundles surface as ``is_valid=False`` rather
  than being dropped, so the UI can render a warning row.

Neither scanner persists state — every call is a fresh walk. This is the
project's standing rule (no ``runs.json``-style index files) and the
``.resources.json`` metadata in ``index.py`` is *only* about provenance
(URL / revision / downloaded_at), not about asset existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from unitport_sdk import Paths, log_warning


# ---------------------------------------------------------------------------
# Policy pack — bundles dropped into custom_mods/policies/<pack_id>/
# ---------------------------------------------------------------------------


@dataclass
class PolicyPack:
    """One policy bundle discovered under ``custom_mods/policies/``.

    Mirrors ``ExportedBundleEntry`` deliberately (same field names) so a
    UI widget can render either kind in the same row component without
    branching. ``scope`` is set to ``"custom_mods"`` here vs ``"project"``
    in the exported_bundle_registry.
    """

    pack_id: str
    """Leaf directory name (the bundle id within ``policies/``)."""
    bundle_path: Path
    """Absolute path to the bundle directory."""
    display_name: str = ""
    """Human-readable name from ``manifest.yaml#name`` (falls back to pack_id)."""
    version: str = ""
    """Bundle version from ``manifest.yaml#version``."""
    robot_sku: str = ""
    """Canonical SKU from ``manifest.yaml#robot.sku``."""
    is_valid: bool = True
    error: str = ""
    scope: str = "custom_mods"

    def label(self) -> str:
        name = self.display_name or self.pack_id
        parts = [name]
        if self.version:
            parts.append(f"v{self.version}")
        if self.robot_sku:
            display = self.robot_sku
            try:
                from registers.robots import get_robot_spec, resolve_id
                rs = get_robot_spec(resolve_id(self.robot_sku) or self.robot_sku)
                if rs is not None:
                    display = rs.name or rs.brand or self.robot_sku
            except Exception:
                pass
            parts.append(f"[{display}]")
        return "  ".join(parts)


class PolicyPackScanner:
    """Walk ``custom_mods/policies/`` and yield bundle entries.

    Each immediate subdirectory is treated as a candidate pack. A
    ``manifest.yaml`` directly inside is required; without it the entry
    appears in the result list with ``is_valid=False`` so the UI can show
    a "missing manifest" warning instead of hiding the broken bundle.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._explicit_root = Path(root) if root is not None else None

    def _root(self) -> Path:
        if self._explicit_root is not None:
            return self._explicit_root
        return Paths.CUSTOM_MODS_DIR / "policies"

    def scan(self) -> List[PolicyPack]:
        root = self._root()
        results: List[PolicyPack] = []
        if not root.exists():
            return results
        try:
            children = sorted(root.iterdir())
        except OSError as e:
            log_warning(f"PolicyPackScanner: scan {root!s} failed: {e}")
            return results
        for child in children:
            if not child.is_dir():
                continue
            manifest_path = child / "manifest.yaml"
            if not manifest_path.exists():
                results.append(PolicyPack(
                    pack_id=child.name,
                    bundle_path=child,
                    is_valid=False,
                    error="manifest.yaml missing",
                ))
                continue
            results.append(self._parse(child, manifest_path))
        return results

    def _parse(self, bundle_path: Path, manifest_path: Path) -> PolicyPack:
        try:
            import yaml
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as e:
            return PolicyPack(
                pack_id=bundle_path.name,
                bundle_path=bundle_path,
                is_valid=False,
                error=f"manifest parse failed: {e}",
            )
        if not isinstance(data, dict):
            return PolicyPack(
                pack_id=bundle_path.name,
                bundle_path=bundle_path,
                is_valid=False,
                error="manifest.yaml is not a mapping",
            )
        robot = data.get("robot") if isinstance(data.get("robot"), dict) else {}
        return PolicyPack(
            pack_id=bundle_path.name,
            bundle_path=bundle_path,
            display_name=str(data.get("name", "") or ""),
            version=str(data.get("version", "") or ""),
            robot_sku=str((robot or {}).get("sku", "") or ""),
            is_valid=True,
        )


def scan_policy_packs() -> List[PolicyPack]:
    """One-shot convenience wrapper around :class:`PolicyPackScanner`."""
    return PolicyPackScanner().scan()


__all__ = [
    "PolicyPack",
    "PolicyPackScanner",
    "scan_policy_packs",
]
