# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Bundle manifest schema + validator (verbatim port from DEMO).

Lives under ``application.training`` per the migration plan (DEMO had it
under ``policy/`` for historical reasons). The 12 required top-level
fields, the validator, and the ``CheckpointBundle`` dataclass are all
byte-for-byte the same as DEMO's ``src/system/policy/manifest_schema.py``.

DEMO's re-exports of ``SkillManifest`` / ``load_skill_manifest`` from
``src/system/skill/`` are dropped here — those serve the behavior_node /
reactive_loco_node code paths which are out of Phase 3 scope. They land
back in when ``application/skill/`` is populated.

The lazy ``DeployContract`` import path is updated to RELEASE's
``application.service.runtime.policy.deploy_contract``.

SKU-only robot identity (2026-05):
* ``robot.sku`` is the **only** required robot identifier in a bundle. It
  is the canonical key from ``registers.robots`` (``build_sku(brand, model)``)
  and resolves through ``registers.robots.get_robot_spec(sku)`` to all
  display fields (``brand`` / ``model`` / ``name``) and joint contracts.
* ``robot.brand`` / ``robot.model`` / ``robot.name`` are tolerated in the
  YAML for human readability when grepping artifacts, but the deploy
  stack **never consumes them**. Code that reverse-resolves SKU from
  these strings (alias-table guessing) is a bug — the SKU is plumbed
  explicitly from the caller (UI / training spec) or read directly from
  ``manifest.robot.sku``.
* ``robot.joint_names`` carries **IR role names** (e.g. ``["hip_FL",
  "thigh_FL", "calf_FL", ...]``), NOT physical USD/MJCF joint names. The
  deploy stack resolves each IR role to the bound robot's physical name
  via ``ir_roles_to_physical_names(roles, sku)``. This is what enables
  cross-machine bundle reuse — the same trained policy deploys across
  same-family robots whose physical joint names differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from application.service.runtime.policy.deploy_contract import DeployContract


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class ManifestValidationError(ValueError):
    """Raised when a bundle manifest fails validation."""

    def __init__(
        self,
        message: str,
        missing_fields: Optional[List[str]] = None,
        invalid_fields: Optional[Dict[str, str]] = None,
        bundle_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.missing_fields: List[str] = missing_fields or []
        self.invalid_fields: Dict[str, str] = invalid_fields or {}
        self.bundle_path: Optional[Path] = bundle_path


# ---------------------------------------------------------------------------
# Required field list
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_FIELDS = (
    "name",
    "version",
    "policy.file",
    "policy.format",
    "observation_space.dim",
    "action_space.dim",
    "action_space.type",
    "runtime.control_frequency_hz",
    "runtime.decimation",
    "robot.sku",                       # SKU-only contract (2026-05)
    "robot.num_joints",
    "robot.joint_names",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    """Walk *d* using dot-separated segments; raise KeyError if missing."""
    parts = dotted_key.split(".")
    node: Any = d
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_SUPPORTED_FORMATS = {"onnx", "jit"}   # Phase 1: onnx; Phase 2 adds: jit
_REJECTED_FORMATS = {"pt", "safetensors"}


def validate_manifest(raw: Dict[str, Any], bundle_path: Optional[Path] = None) -> None:
    """Validate *raw* manifest dict.

    Raises ManifestValidationError listing ALL problems found in one call.
    """
    missing: List[str] = []
    for key in MANIFEST_REQUIRED_FIELDS:
        try:
            _get_nested(raw, key)
        except KeyError:
            missing.append(key)

    if missing:
        raise ManifestValidationError(
            f"Manifest missing required fields: {missing}",
            missing_fields=missing,
            bundle_path=bundle_path,
        )

    invalid: Dict[str, str] = {}

    fmt = _get_nested(raw, "policy.format")
    if fmt in _REJECTED_FORMATS:
        invalid["policy.format"] = (
            f"unsupported format '{fmt}'; Phase 1 only supports 'onnx'"
        )
    elif fmt not in _SUPPORTED_FORMATS:
        invalid["policy.format"] = (
            f"unsupported format '{fmt}'; Phase 1 only supports 'onnx'"
        )

    obs_dim = _get_nested(raw, "observation_space.dim")
    if not isinstance(obs_dim, int) or obs_dim <= 0:
        invalid["observation_space.dim"] = (
            f"must be a positive integer, got {obs_dim!r}"
        )

    act_dim = _get_nested(raw, "action_space.dim")
    if not isinstance(act_dim, int) or act_dim <= 0:
        invalid["action_space.dim"] = (
            f"must be a positive integer, got {act_dim!r}"
        )

    freq = _get_nested(raw, "runtime.control_frequency_hz")
    if not isinstance(freq, (int, float)) or freq <= 0:
        invalid["runtime.control_frequency_hz"] = (
            f"must be a positive number, got {freq!r}"
        )

    dec = _get_nested(raw, "runtime.decimation")
    if not isinstance(dec, int) or dec <= 0:
        invalid["runtime.decimation"] = (
            f"must be a positive integer, got {dec!r}"
        )

    num_joints = _get_nested(raw, "robot.num_joints")
    joint_names = _get_nested(raw, "robot.joint_names")
    if isinstance(num_joints, int) and isinstance(joint_names, list):
        if num_joints != len(joint_names):
            invalid["robot.num_joints"] = (
                f"num_joints={num_joints} does not match "
                f"len(joint_names)={len(joint_names)}"
            )

    sku = _get_nested(raw, "robot.sku")
    if not isinstance(sku, str) or not sku.strip():
        invalid["robot.sku"] = (
            f"must be a non-empty string (the registry SKU); got {sku!r}. "
            f"Bundles written before the SKU-only contract (2026-05) lack "
            f"this field — re-train to produce a valid bundle."
        )
    else:
        try:
            from registers.robots import get_robot_spec, resolve_id
            canonical = resolve_id(sku.strip()) or sku.strip()
            rs = get_robot_spec(canonical)
            if rs is None:
                invalid["robot.sku"] = (
                    f"sku {sku!r} is not registered (resolved to "
                    f"{canonical!r}). Use registers.robots.list_skus() to "
                    f"enumerate available robots."
                )
            else:
                # Cross-format-aware joint-count check.
                # ``rs.num_joints`` uses ``preferred_format`` which is
                # ambiguous for robots with both MJCF and USD declared
                # (e.g. G1: MJCF=29-DOF stock variant, USD=43-DOF with
                # hands). The bundle's manifest.robot.num_joints reflects
                # whatever format the bundle was trained against; the
                # registry value to compare against is the matching
                # format, not preferred_format.
                #
                # Resolution: the bundle's num_joints must equal AT LEAST
                # ONE of the robot's per-format counts. Falling back to
                # ``rs.num_joints`` (preferred_format) would re-introduce
                # the bug where a USD-trained bundle gets rejected because
                # MJCF happens to be the preferred fallback.
                if isinstance(num_joints, int):
                    candidates: list[int] = []
                    joints_pf = (
                        getattr(rs, "joints_per_format", {}) or {}
                    )
                    for fmt in ("MJCF", "USD", "URDF"):
                        block = joints_pf.get(fmt)
                        if isinstance(block, dict) and block:
                            candidates.append(len(block))
                    # Legacy path: rs.num_joints from preferred_format.
                    # Only kept as a candidate so single-format robots
                    # still match the historical check.
                    if rs.num_joints:
                        candidates.append(int(rs.num_joints))
                    if candidates and num_joints not in candidates:
                        invalid["robot.num_joints"] = (
                            f"manifest declares num_joints={num_joints} "
                            f"but registered robot {canonical!r} has no "
                            f"per-format table with that joint count "
                            f"(available: {sorted(set(candidates))}). "
                            f"The bundle's trained joint set does not "
                            f"correspond to any declared asset variant "
                            f"of this SKU."
                        )
        except ImportError as exc:
            # ``registers.robots`` is part of the deployable RELEASE; its
            # absence at validate time is a deployment bug, not a test-stub
            # nicety. Surface it instead of silently best-efforting.
            raise ManifestValidationError(
                f"manifest validate: registers.robots could not be imported "
                f"({exc}); manifest cannot be validated against the SKU "
                f"registry.",
                invalid_fields={},
                bundle_path=bundle_path,
            ) from exc

    # joint_array_format (optional): articulation order of the bundle's
    # per-joint arrays. When present it MUST be a known asset-format key —
    # the deploy stack keys joint reorder/validation off it (see
    # joint_space.resolve_joint_array_format). Absent = legacy bundle
    # (inferred at load).
    if "joint_array_format" in raw:
        jaf = raw.get("joint_array_format")
        if not isinstance(jaf, str) or jaf.strip().upper() not in (
            "MJCF", "USD", "URDF",
        ):
            invalid["joint_array_format"] = (
                f"must be one of MJCF/USD/URDF when present, got {jaf!r}"
            )

    if invalid:
        raise ManifestValidationError(
            f"Manifest has invalid fields: {list(invalid.keys())}",
            invalid_fields=invalid,
            bundle_path=bundle_path,
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckpointBundle:
    """Fully-validated, fully-typed bundle descriptor.

    Robot identity: ``robot_sku`` is the canonical key. Display fields
    (``robot_brand`` / ``robot_model`` / ``robot_name``) are exposed as
    ``@property`` that look up the registry on demand — they are NEVER
    read from the raw manifest. This guarantees a manifest with stale or
    corrupt brand strings cannot influence deploy decisions.
    """
    name: str
    version: str
    bundle_path: Path
    policy_file: Path          # absolute: bundle_path / manifest["policy"]["file"]
    policy_format: str         # "onnx"
    obs_dim: int
    action_dim: int
    action_type: str
    control_frequency_hz: float
    decimation: int
    robot_sku: str             # canonical from registers.robots; the only identity
    num_joints: int
    joint_names: List[str]
    raw_manifest: Dict[str, Any]   # preserved for forward-compat keys

    @cached_property
    def _registered(self) -> Optional[Any]:
        """Lazy registry lookup. Returns None when SKU isn't registered
        (this should never happen for validated bundles — validate_manifest
        rejects unregistered SKUs — but the loader is defensive)."""
        try:
            from registers.robots import get_robot_spec, resolve_id
            return get_robot_spec(resolve_id(self.robot_sku) or self.robot_sku)
        except Exception:
            return None

    @property
    def robot_brand(self) -> str:
        """Display only — derived from registry, never from manifest."""
        rs = self._registered
        return rs.brand if rs is not None else ""

    @property
    def robot_model(self) -> str:
        """Display only — derived from registry, never from manifest."""
        rs = self._registered
        return rs.model if rs is not None else ""

    @property
    def robot_name(self) -> str:
        """Display only — derived from registry, never from manifest."""
        rs = self._registered
        return rs.name if rs is not None else ""

    @cached_property
    def deploy_contract(self) -> "DeployContract":
        """Lazy-decoded sim2sim deployment contract.

        Strict-canvas: every loadable bundle MUST carry a ``deploy_contract``
        section; the legacy "return None for older bundles" path was removed
        because it silently degraded the deploy stack to a heuristic. If
        access raises ``ValueError`` here, the bundle is from before strict
        mode — re-export it from the current pipeline.
        """
        from application.service.runtime.policy.deploy_contract import DeployContract

        raw = (self.raw_manifest or {}).get("deploy_contract")
        return DeployContract.from_dict(raw)

    @property
    def algorithm(self) -> str:
        """Training algorithm identifier as recorded in the manifest."""
        raw = (self.raw_manifest or {}).get("algorithm")
        if isinstance(raw, str) and raw:
            return raw.upper()
        train = (self.raw_manifest or {}).get("training") or {}
        if isinstance(train, dict):
            algo = train.get("algorithm")
            if isinstance(algo, str) and algo:
                return algo.upper()
        return "UNKNOWN"

    @property
    def amp_metadata(self) -> Optional[Dict[str, Any]]:
        """Return the ``amp_metadata`` block if present, else None."""
        raw = (self.raw_manifest or {}).get("amp_metadata")
        return raw if isinstance(raw, dict) else None


@dataclass
class PolicyInfo:
    """Lightweight summary for CheckpointRegistry listings.

    Robot identity follows the same SKU-only contract as
    :class:`CheckpointBundle`: ``robot_sku`` is the stored field;
    ``robot_brand`` is derived from the registry on demand for display.
    """
    policy_id: str          # directory name under custom_mods/training/checkpoints/
    name: str               # manifest["name"]
    version: str
    robot_sku: str          # canonical SKU from registers.robots
    bundle_path: Path
    is_valid: bool
    error: Optional[str] = None   # set when is_valid=False

    @property
    def robot_brand(self) -> str:
        """Display only — derived from registry."""
        try:
            from registers.robots import get_robot_spec, resolve_id
            rs = get_robot_spec(resolve_id(self.robot_sku) or self.robot_sku)
            return rs.brand if rs is not None else ""
        except Exception:
            return ""
