"""BundleLoader — read manifest.yaml + verify policy file (port from DEMO).

Changes from DEMO:

* ``manifest_schema`` now lives at ``application.training.manifest_schema``
  per the migration plan.
* The DEMO ``il_manifest_compat`` v2→v1 upgrader is intentionally NOT
  ported — that file is a Phase 5 deletion target. RELEASE bundles must be
  written in v1 layout (the layout RELEASE's BundleExporter will emit). If
  a downstream user feeds an Isaac-v2-shaped manifest, validation fails
  loudly instead of silently rewriting it.
* The YAML read is still ``yaml.safe_load`` on a Path open — DataManager
  has a YAML handler but its dict semantics drop insertion order on some
  legacy paths, and ``deploy_contract.observations`` MUST preserve
  insertion order. Direct ``yaml.safe_load`` keeps that invariant safe.
* ``import yaml`` is deferred into ``_parse_manifest`` (was top-level).
  Any transitive import of this module that happens before
  ``ProvisioningTask`` installs ``pyyaml`` (e.g. UI widgets that type-hint
  ``BundleLoader`` at module load) used to surface a
  ``ModuleNotFoundError: No module named 'yaml'`` warning during
  Stage 2 of startup. Lazy-loading makes the module import yaml-free; the
  actual load path still raises loudly if yaml is missing when a real
  bundle is parsed.

SKU-only contract (2026-05):
* ``CheckpointBundle.robot_sku`` is the canonical robot identity. The
  loader reads ``manifest.robot.sku`` directly; ``validate_manifest``
  rejects bundles missing this field with a clear "re-train" message.
* ``CheckpointBundle.joint_names`` carries **IR role names** (e.g.
  ``"hip_FL"``, ``"thigh_FL"``). The deploy stack translates these to
  physical names via ``ir_roles_to_physical_names(roles, sku)`` — the
  SKU is plumbed explicitly, never reverse-derived from manifest
  display strings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from application.training.manifest_schema import (
    CheckpointBundle,
    ManifestValidationError,
    _get_nested,
    validate_manifest,
)


class BundleLoader:
    """Loads and validates a policy weight bundle from disk."""

    def load(self, bundle_path: Path) -> CheckpointBundle:
        """Load a bundle from *bundle_path*."""
        bundle_path = Path(bundle_path)

        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Bundle path does not exist: {bundle_path}"
            )

        manifest_path = bundle_path / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.yaml not found in bundle: {bundle_path}"
            )

        raw = self._parse_manifest(bundle_path)
        validate_manifest(raw, bundle_path=bundle_path)
        bundle = self._build_bundle(raw, bundle_path)

        if not bundle.policy_file.exists():
            raise FileNotFoundError(
                f"Policy file '{bundle.policy_file.name}' referenced in manifest "
                f"does not exist in bundle: {bundle_path}"
            )

        return bundle

    @staticmethod
    def _parse_manifest(bundle_path: Path) -> Dict[str, Any]:
        """Read and YAML-parse manifest.yaml from *bundle_path*."""
        # Lazy import: keep module-level import-graph yaml-free so this
        # module can be referenced (type hints, `from ... import BundleLoader`
        # at the top of a widget) before ProvisioningTask has installed
        # pyyaml. Missing-yaml at actual load time still raises loudly.
        import yaml

        manifest_path = bundle_path / "manifest.yaml"
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ManifestValidationError(
                f"Failed to parse YAML in {manifest_path}: {exc}",
                bundle_path=bundle_path,
            ) from exc
        if not isinstance(raw, dict):
            raise ManifestValidationError(
                f"manifest.yaml must contain a YAML mapping, got {type(raw).__name__}",
                bundle_path=bundle_path,
            )
        return raw

    @staticmethod
    def _build_bundle(raw: Dict[str, Any], bundle_path: Path) -> CheckpointBundle:
        """Construct CheckpointBundle from a validated raw manifest dict."""
        policy_file = bundle_path / _get_nested(raw, "policy.file")
        return CheckpointBundle(
            name=str(_get_nested(raw, "name")),
            version=str(_get_nested(raw, "version")),
            bundle_path=bundle_path,
            policy_file=policy_file,
            policy_format=str(_get_nested(raw, "policy.format")),
            obs_dim=int(_get_nested(raw, "observation_space.dim")),
            action_dim=int(_get_nested(raw, "action_space.dim")),
            action_type=str(_get_nested(raw, "action_space.type")),
            control_frequency_hz=float(_get_nested(raw, "runtime.control_frequency_hz")),
            decimation=int(_get_nested(raw, "runtime.decimation")),
            robot_sku=str(_get_nested(raw, "robot.sku")),
            num_joints=int(_get_nested(raw, "robot.num_joints")),
            joint_names=list(_get_nested(raw, "robot.joint_names")),
            raw_manifest=raw,
        )
