from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .manifest_schema import (
    ManifestValidationError,
    CheckpointBundle,
    _get_nested,
    validate_manifest,
)


class BundleLoader:
    """Loads and validates a policy weight bundle from disk."""

    def load(self, bundle_path: Path) -> CheckpointBundle:
        """Load a bundle from *bundle_path*.

        Raises:
            FileNotFoundError       if bundle_path or manifest.yaml is missing,
                                    or if the policy file referenced in the manifest
                                    does not exist.
            ManifestValidationError if the manifest fails validation.
        """
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
            robot_brand=str(_get_nested(raw, "robot.brand")),
            num_joints=int(_get_nested(raw, "robot.num_joints")),
            joint_names=list(_get_nested(raw, "robot.joint_names")),
            raw_manifest=raw,
        )
