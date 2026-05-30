# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reference motion alignment override schema + cross-validation.

Each alignment override is a per-``(source_format, target_sku)`` JSON file
stored under ``USER_CONFIG_DIR/registers/motion_alignment/`` with the
canonical filename ``<source_format>__to__<target_sku>.json`` (double
underscore separator avoids collisions with single underscores inside
SKU strings).

Two frozen dataclasses carry the parsed file:

  * :class:`AxisConvention` — sub-block describing root axes (up /
    forward / lateral), per-joint sign flips, and velocity frames.
  * :class:`AlignmentOverride` — top-level block wrapping the joint
    mapping and the axis convention.

Cross-validation
----------------
At load time, ten cross-validation rules run against
``registers.robots`` / ``registers.families`` plus the source format's
IR role table:

   1. target_sku resolves to a canonical SKU via ``resolve_id``
   2. target_family is in ``registers.families.list_families()``
   3. target_family is in ``RobotSpec(target_sku).families``
   4. joint_mapping values are joint names declared by the SKU in at
      least one available asset format
   5. joint_mapping keys equal ``FORMAT_IR_ROLES[source_format]``
      exactly (no missing, no extras — extras would otherwise be a
      silent typo)
   6. axis_convention (up, forward, lateral) form a right-handed
      orthogonal basis (``forward × lateral == up``)
   7. axis_convention.joint_sign_flips keys are a subset of
      joint_mapping values
   8. axis_convention.{linear,angular}_velocity_frame is in
      ``{world, root, body}``
   9. joint_mapping is non-empty
  10. joint_mapping values are distinct (no two IR roles target one
      physical joint)

Rules 1-5, 9, 10 raise :exc:`AlignmentOverrideError`. Rules 6-8 raise
:exc:`AxisConventionError`. Every error message includes the override
file path, the offending field, and an actionable fix-up directive.

The optional fields ``created_at`` / ``created_by`` / ``user_notes`` are
audit metadata; no cross-validation rules apply and the values are
passed through verbatim.

Loader-time entry point is :func:`load_alignment_override`. A
contract-time consistency check (override versus a loaded
``ReferenceMotionContract``) is :func:`validate_alignment_against_contract`.
The fuzzy-match helper :func:`suggest_alignment` is callable
independently for diagnostic messages and UI tooling.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from application.training.motion.contract import ReferenceMotionContract


# ---------------------------------------------------------------------------
# Schema version pins
# ---------------------------------------------------------------------------


#: Pinned top-level schema version. Loaders refuse anything else.
ALIGNMENT_OVERRIDE_SCHEMA_VERSION = "1.0.0"

#: Pinned axis_convention sub-schema version. Tracked independently so
#: the inner schema can evolve without bumping the outer file format.
AXIS_CONVENTION_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enums + defaults
# ---------------------------------------------------------------------------


#: Acceptable literal values for a root axis declaration.
ALLOWED_AXIS_LITERALS: frozenset = frozenset(
    {"x", "-x", "y", "-y", "z", "-z"}
)

#: Acceptable velocity-frame values.
ALLOWED_VELOCITY_FRAMES: frozenset = frozenset({"world", "root", "body"})

#: Default convention values. Match the ``amp_legged_gym`` source
#: convention (z-up, x forward, y lateral, world-frame velocities) so a
#: motion normalizer can treat the default as an identity transform.
DEFAULT_ROOT_UP_AXIS = "z"
DEFAULT_ROOT_FORWARD_AXIS = "x"
DEFAULT_ROOT_LATERAL_AXIS = "y"
DEFAULT_LINEAR_VELOCITY_FRAME = "world"
DEFAULT_ANGULAR_VELOCITY_FRAME = "world"


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

# Override directory beneath USER_CONFIG_DIR.
_OVERRIDE_REL_DIR = Path("registers") / "motion_alignment"

# Filename template — double underscore separator survives SKUs and
# format ids that contain single underscores.
_OVERRIDE_FILENAME_TEMPLATE = "{source_format}__to__{target_sku}.json"


def _override_filename(source_format: str, target_sku: str) -> str:
    return _OVERRIDE_FILENAME_TEMPLATE.format(
        source_format=source_format, target_sku=target_sku,
    )


def _default_override_dir() -> Optional[Path]:
    """Return the canonical override directory, or ``None`` when the user
    config dir has not been established yet (pre-wizard)."""
    from unitport_sdk import Paths
    if not Paths.is_user_config_dir_set():
        return None
    return Paths.require_user_config_dir() / _OVERRIDE_REL_DIR


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlignmentOverrideError(ValueError):
    """Raised when an alignment override file fails schema parsing or
    one of cross-validation rules 1-5, 9, 10."""


class AxisConventionError(ValueError):
    """Raised when the ``axis_convention`` sub-block fails parsing or
    one of cross-validation rules 6-8."""


# ---------------------------------------------------------------------------
# Frozen data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisConvention:
    """Coordinate-frame convention the source motion clip is authored in.

    ``frozen=True`` — field reassignment raises ``FrozenInstanceError``.
    The nested ``joint_sign_flips`` dict is treated as read-only by
    convention (Python frozen dataclasses cannot deep-freeze mappings;
    callers must not mutate it).

    Attributes
    ----------
    schema_version
        Pinned to :data:`AXIS_CONVENTION_SCHEMA_VERSION`.
    root_up_axis, root_forward_axis, root_lateral_axis
        One of :data:`ALLOWED_AXIS_LITERALS`. The three axes must form
        a right-handed orthogonal basis (``forward × lateral == up``).
    joint_sign_flips
        ``{joint_name: sign}`` where ``sign`` is ``-1`` or ``+1``.
        Joint names referenced here must appear in the parent
        :class:`AlignmentOverride`'s ``joint_mapping`` values.
    linear_velocity_frame, angular_velocity_frame
        One of :data:`ALLOWED_VELOCITY_FRAMES`.
    """

    schema_version: str
    root_up_axis: str
    root_forward_axis: str
    root_lateral_axis: str
    joint_sign_flips: Dict[str, int]
    linear_velocity_frame: str
    angular_velocity_frame: str


@dataclass(frozen=True)
class AlignmentOverride:
    """User-supplied alignment between a motion file format and a SKU.

    ``frozen=True`` — field reassignment raises ``FrozenInstanceError``.

    Attributes
    ----------
    schema_version
        Pinned to :data:`ALIGNMENT_OVERRIDE_SCHEMA_VERSION`.
    source_format
        Motion file format identifier; must have an entry in
        ``application.training.motion_ir_mapping.FORMAT_IR_ROLES``.
    target_sku
        Stored as written in the file (could be any alias accepted by
        ``registers.robots.resolve_id``). Rule 1 resolves it to a
        canonical SKU; downstream comparisons in
        :func:`validate_alignment_against_contract` re-resolve both
        sides so alias drift does not cause false mismatches.
    target_family
        Family name; must appear in both
        ``registers.families.list_families()`` and
        ``RobotSpec(target_sku).families``.
    joint_mapping
        ``{ir_role: physical_joint_name}``. Keys must equal
        ``FORMAT_IR_ROLES[source_format]`` exactly. Values must be
        joint names declared on the SKU in at least one available asset
        format, and distinct (no two IR roles map to one joint).
    axis_convention
        Coordinate-frame convention sub-block.
    created_at, created_by, user_notes
        Optional audit metadata; not cross-validated.
    """

    schema_version: str
    source_format: str
    target_sku: str
    target_family: str
    joint_mapping: Dict[str, str]
    axis_convention: AxisConvention
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    user_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Load + parse
# ---------------------------------------------------------------------------


def load_alignment_override(
    source_format: str,
    target_sku: str,
    *,
    override_dir: Optional[Path] = None,
) -> Optional[AlignmentOverride]:
    """Locate, parse, and cross-validate an alignment override file.

    Parameters
    ----------
    source_format
        Motion file format identifier (e.g. ``"amp_legged_gym"``).
    target_sku
        SKU string — the caller's form is used as-is for the filename
        lookup. The override's ``target_sku`` field is later normalized
        through ``resolve_id`` at cross-validation time.
    override_dir
        Override location root. ``None`` resolves to
        ``USER_CONFIG_DIR/registers/motion_alignment/``. Test code
        passes a ``tmp_path``-rooted directory.

    Returns
    -------
    Optional[AlignmentOverride]
        ``None`` when no override file exists for the
        ``(source_format, target_sku)`` pair, or when the user config
        directory has not been established yet. Otherwise a parsed and
        cross-validated override.

    Raises
    ------
    AlignmentOverrideError
        On JSON parse failure, missing required field, schema_version
        mismatch, or cross-validation rules 1-5 / 9 / 10.
    AxisConventionError
        On axis_convention sub-block validation rules 6-8.
    """
    if override_dir is None:
        override_dir = _default_override_dir()
        if override_dir is None:
            # User config dir not yet established. No override possible.
            return None
    path = Path(override_dir) / _override_filename(source_format, target_sku)
    if not path.is_file():
        return None
    return _parse_and_validate(path, source_format, target_sku)


def _parse_and_validate(
    path: Path, source_format: str, target_sku: str,
) -> AlignmentOverride:
    """Parse JSON at ``path`` and run all structural + cross-validation
    rules. Returns a validated :class:`AlignmentOverride`."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: cannot read file ({exc}). "
            "Check filesystem permissions, or remove the file to fall "
            "back to automatic IR matching."
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: invalid JSON ({exc}). Edit "
            "the file to fix the JSON syntax error and re-run."
        ) from exc
    if not isinstance(raw, dict):
        raise AlignmentOverrideError(
            f"alignment_override at {path}: top-level JSON value must be "
            f"an object, got {type(raw).__name__}. Rewrite the file as "
            "a JSON object."
        )

    # Top-level required fields + schema version pin.
    _require_field(path, raw, "schema_version", str, AlignmentOverrideError)
    if raw["schema_version"] != ALIGNMENT_OVERRIDE_SCHEMA_VERSION:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: schema_version="
            f"{raw['schema_version']!r} does not match the supported "
            f"version {ALIGNMENT_OVERRIDE_SCHEMA_VERSION!r}. Re-export "
            "the override with the current schema, or edit the "
            "schema_version field if the content is already compatible."
        )
    for k in ("source_format", "target_sku", "target_family"):
        _require_field(path, raw, k, str, AlignmentOverrideError)
    _require_field(path, raw, "joint_mapping", dict, AlignmentOverrideError)
    _require_field(path, raw, "axis_convention", dict, AlignmentOverrideError)

    # File location vs file content consistency.
    if raw["source_format"] != source_format:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: file source_format="
            f"{raw['source_format']!r} does not match the requested "
            f"source_format={source_format!r}. Rename the file to "
            f"{_override_filename(raw['source_format'], raw['target_sku'])!r} "
            "or correct the source_format field inside."
        )
    if raw["target_sku"] != target_sku:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: file target_sku="
            f"{raw['target_sku']!r} does not match the requested "
            f"target_sku={target_sku!r}. Rename the file to "
            f"{_override_filename(raw['source_format'], raw['target_sku'])!r} "
            "or correct the target_sku field inside."
        )

    # Parse axis_convention sub-block (structural type checks only;
    # rules 6-8 run inside _cross_validate after the parent override
    # is constructed so the joint_mapping context is available).
    axis = _parse_axis_convention(path, raw["axis_convention"])

    # joint_mapping must be {str: str}.
    joint_mapping: Dict[str, str] = {}
    for k, v in raw["joint_mapping"].items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise AlignmentOverrideError(
                f"alignment_override at {path}: joint_mapping must map "
                f"string to string; got entry {k!r}={v!r} (types "
                f"{type(k).__name__}, {type(v).__name__}). Edit the file "
                "so every key and value is a string."
            )
        joint_mapping[k] = v

    override = AlignmentOverride(
        schema_version=raw["schema_version"],
        source_format=raw["source_format"],
        target_sku=raw["target_sku"],
        target_family=raw["target_family"],
        joint_mapping=joint_mapping,
        axis_convention=axis,
        created_at=raw.get("created_at") if isinstance(raw.get("created_at"), str) else None,
        created_by=raw.get("created_by") if isinstance(raw.get("created_by"), str) else None,
        user_notes=raw.get("user_notes") if isinstance(raw.get("user_notes"), str) else None,
    )

    _cross_validate(override, path)
    return override


def _require_field(
    path: Path,
    raw: Mapping[str, Any],
    key: str,
    expected_type: type,
    err_cls: type,
    *,
    context: str = "",
) -> None:
    # ``context`` lets the caller scope the field name (e.g. pass
    # ``"axis_convention."`` from inside the nested-block parser) so a
    # missing ``schema_version`` at the inner level surfaces as
    # ``axis_convention.schema_version`` in the error message instead of
    # the bare ``schema_version`` that would collide with the top-level
    # field.
    field_label = f"{context}{key}" if context else key
    if key not in raw:
        raise err_cls(
            f"alignment_override at {path}: required field {field_label!r} is "
            "missing. Add it to the JSON object."
        )
    if not isinstance(raw[key], expected_type):
        raise err_cls(
            f"alignment_override at {path}: field {field_label!r} must be "
            f"{expected_type.__name__}, got {type(raw[key]).__name__}. "
            "Edit the file so the field has the correct type."
        )


def _parse_axis_convention(path: Path, raw: Mapping[str, Any]) -> AxisConvention:
    _AX_CTX = "axis_convention."
    _require_field(
        path, raw, "schema_version", str, AxisConventionError,
        context=_AX_CTX,
    )
    if raw["schema_version"] != AXIS_CONVENTION_SCHEMA_VERSION:
        raise AxisConventionError(
            f"alignment_override at {path}: axis_convention.schema_version="
            f"{raw['schema_version']!r} does not match the supported "
            f"version {AXIS_CONVENTION_SCHEMA_VERSION!r}. Re-export the "
            "override or edit the field if the content is compatible."
        )
    for k in (
        "root_up_axis", "root_forward_axis", "root_lateral_axis",
        "linear_velocity_frame", "angular_velocity_frame",
    ):
        _require_field(path, raw, k, str, AxisConventionError, context=_AX_CTX)
    _require_field(
        path, raw, "joint_sign_flips", dict, AxisConventionError,
        context=_AX_CTX,
    )

    sign_flips: Dict[str, int] = {}
    for k, v in raw["joint_sign_flips"].items():
        if not isinstance(k, str):
            raise AxisConventionError(
                f"alignment_override at {path}: axis_convention."
                f"joint_sign_flips key must be a string, got {k!r} "
                f"({type(k).__name__}). Edit the file so every key is "
                "a joint name string."
            )
        if v not in (-1, 1):
            raise AxisConventionError(
                f"alignment_override at {path}: axis_convention."
                f"joint_sign_flips[{k!r}]={v!r} must be -1 or +1. Edit "
                "the file so the sign is one of those two values."
            )
        sign_flips[k] = int(v)

    return AxisConvention(
        schema_version=raw["schema_version"],
        root_up_axis=raw["root_up_axis"],
        root_forward_axis=raw["root_forward_axis"],
        root_lateral_axis=raw["root_lateral_axis"],
        joint_sign_flips=sign_flips,
        linear_velocity_frame=raw["linear_velocity_frame"],
        angular_velocity_frame=raw["angular_velocity_frame"],
    )


# ---------------------------------------------------------------------------
# 10 cross-validation rules
# ---------------------------------------------------------------------------


def _cross_validate(override: AlignmentOverride, path: Path) -> None:
    """Run rules 1-5 + 9 + 10 (alignment) then 6-8 (axis convention)."""
    # Lazy imports — keep alignment.py importable without the registers /
    # motion_ir_mapping subpackages being initialised at import time.
    from registers import robots as _robots_registry
    from registers import families as _families_registry
    from application.training.motion_ir_mapping import FORMAT_IR_ROLES

    # Rule 9 — joint_mapping non-empty (check first; later rules rely on
    # the mapping having content to be meaningful).
    if not override.joint_mapping:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: joint_mapping is empty. An "
            "override with no joint entries cannot be applied. Edit the "
            "file to declare a mapping for every IR role required by "
            f"source_format={override.source_format!r}."
        )

    # Rule 1 — target_sku resolves to a canonical SKU.
    canonical_sku = _robots_registry.resolve_id(override.target_sku)
    if canonical_sku is None:
        candidates = difflib.get_close_matches(
            override.target_sku, _robots_registry.list_skus(), n=5,
        )
        raise AlignmentOverrideError(
            f"alignment_override at {path}: target_sku="
            f"{override.target_sku!r} cannot be resolved to a canonical "
            f"SKU via registers.robots.resolve_id. Closest candidates: "
            f"{candidates!r}. Edit the file to set target_sku to one of "
            "the registered SKUs (or aliases that resolve to one), or "
            "register the new SKU in registers/data/robots_canonical.json "
            "or the user overlay before retrying."
        )

    # Rule 2 — target_family in registers.families catalog.
    known_families = _families_registry.list_families()
    if override.target_family not in known_families:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: target_family="
            f"{override.target_family!r} is not in registers.families. "
            f"Known families: {sorted(known_families)!r}. Edit the file "
            "to set target_family to one of the listed values, or add "
            "the family to registers/data/families_catalog.json (or the "
            "user overlay)."
        )

    # Rule 3 — target_family declared for this SKU.
    robot_spec = _robots_registry.get_robot_spec(canonical_sku)
    if robot_spec is None:
        # Defensive: rule 1 already confirmed the SKU is registered.
        raise AlignmentOverrideError(
            f"alignment_override at {path}: get_robot_spec("
            f"{canonical_sku!r}) returned None despite target_sku="
            f"{override.target_sku!r} resolving to it. Reload the "
            "registers (RegistryHub.reload()) or report this as a "
            "registry bug."
        )
    if override.target_family not in robot_spec.families:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: target_family="
            f"{override.target_family!r} is not declared for target_sku="
            f"{override.target_sku!r} (canonical SKU={canonical_sku!r}; "
            f"declared families: {robot_spec.families!r}). Edit the file "
            "to set target_family to one of the declared families, or "
            "pick a different target_sku."
        )

    # Rule 4 — joint_mapping values are joint names declared on the SKU.
    sku_joint_names: set = set()
    for fmt in robot_spec.available_formats:
        sku_joint_names.update(robot_spec.joint_order_for(fmt))
    bad_values = sorted(
        v for v in set(override.joint_mapping.values())
        if v not in sku_joint_names
    )
    if bad_values:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: joint_mapping has "
            f"{len(bad_values)} value(s) not declared on target_sku="
            f"{override.target_sku!r} (canonical SKU={canonical_sku!r}) "
            f"in any of its available asset formats "
            f"{robot_spec.available_formats!r}: {bad_values!r}. Edit the "
            "file to replace these with names from the robot's declared "
            "joint list (call registers.robots.get_joint_order(sku, fmt) "
            "to inspect them)."
        )

    # Rule 5 (modified) — joint_mapping keys equal FORMAT_IR_ROLES exactly.
    expected_ir_roles = FORMAT_IR_ROLES.get(override.source_format)
    if expected_ir_roles is None:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: source_format="
            f"{override.source_format!r} has no IR-role contract in "
            "application.training.motion_ir_mapping.FORMAT_IR_ROLES. "
            f"Known formats with IR contracts: {sorted(FORMAT_IR_ROLES)!r}. "
            "Edit the file to use a source_format with a declared IR-role "
            "table, or extend FORMAT_IR_ROLES first."
        )
    expected_set = set(expected_ir_roles)
    actual_set = set(override.joint_mapping.keys())
    missing = sorted(expected_set - actual_set)
    extras = sorted(actual_set - expected_set)
    if missing or extras:
        raise AlignmentOverrideError(
            f"alignment_override at {path}: joint_mapping keys must "
            f"equal FORMAT_IR_ROLES[{override.source_format!r}] exactly. "
            f"Missing keys: {missing!r}; extra keys: {extras!r}. Edit "
            "the file to add the missing IR roles and remove the extras "
            "(extra keys are rejected to catch typos that would "
            "otherwise be silently ignored)."
        )

    # Rule 10 — joint_mapping values are distinct.
    values_list = list(override.joint_mapping.values())
    if len(set(values_list)) != len(values_list):
        seen: Dict[str, List[str]] = {}
        for k, v in override.joint_mapping.items():
            seen.setdefault(v, []).append(k)
        dups = {v: ks for v, ks in seen.items() if len(ks) > 1}
        raise AlignmentOverrideError(
            f"alignment_override at {path}: joint_mapping has duplicate "
            f"target joint name(s): {dups!r}. Each physical joint may "
            "be the target of at most one IR role. Edit the file so "
            "every joint name appears as a value at most once."
        )

    # Rules 6-8 on axis convention.
    _cross_validate_axis(override.axis_convention, override.joint_mapping, path)


def _cross_validate_axis(
    axis: AxisConvention,
    joint_mapping: Mapping[str, str],
    path: Path,
) -> None:
    """Run rules 6, 7, 8."""
    # Rule 8 — velocity frame enum (early — fast and cheap).
    for fname, fvalue in (
        ("linear_velocity_frame", axis.linear_velocity_frame),
        ("angular_velocity_frame", axis.angular_velocity_frame),
    ):
        if fvalue not in ALLOWED_VELOCITY_FRAMES:
            raise AxisConventionError(
                f"alignment_override at {path}: axis_convention.{fname}="
                f"{fvalue!r} is not in {sorted(ALLOWED_VELOCITY_FRAMES)!r}. "
                "Edit the file to set the frame to one of the allowed values."
            )

    # Rule 6 — root axes form a right-handed orthogonal basis. Check each
    # axis literal individually first so the right-handed check below can
    # assume each value is in ALLOWED_AXIS_LITERALS.
    for axis_key, axis_value in (
        ("root_up_axis", axis.root_up_axis),
        ("root_forward_axis", axis.root_forward_axis),
        ("root_lateral_axis", axis.root_lateral_axis),
    ):
        if axis_value not in ALLOWED_AXIS_LITERALS:
            raise AxisConventionError(
                f"alignment_override at {path}: axis_convention.{axis_key}="
                f"{axis_value!r} is not in {sorted(ALLOWED_AXIS_LITERALS)!r}. "
                "Edit the file to set the axis to one of the allowed literals."
            )
    if not _is_right_handed(
        axis.root_up_axis, axis.root_forward_axis, axis.root_lateral_axis,
    ):
        raise AxisConventionError(
            f"alignment_override at {path}: axis_convention "
            f"(root_up_axis={axis.root_up_axis!r}, "
            f"root_forward_axis={axis.root_forward_axis!r}, "
            f"root_lateral_axis={axis.root_lateral_axis!r}) does not form "
            "a right-handed orthogonal basis (the constraint is "
            "`forward × lateral == up`). Edit the axes so they form a "
            "right-handed triple (example: z-up / x-forward / y-lateral)."
        )

    # Rule 7 — sign_flips keys are physical joint names known to the mapping.
    mapped_joint_names = set(joint_mapping.values())
    unknown_flip_keys = sorted(
        k for k in axis.joint_sign_flips if k not in mapped_joint_names
    )
    if unknown_flip_keys:
        raise AxisConventionError(
            f"alignment_override at {path}: axis_convention."
            f"joint_sign_flips references {len(unknown_flip_keys)} joint "
            f"name(s) not in joint_mapping values: {unknown_flip_keys!r}. "
            "Each sign-flip target must be a joint declared as a "
            "joint_mapping value. Edit the file to remove the unknown "
            "entries or add the corresponding joint_mapping rows first."
        )


# ---------------------------------------------------------------------------
# Right-handed basis check (integer arithmetic; no numpy dependency)
# ---------------------------------------------------------------------------


_AXIS_TO_VEC: Dict[str, Sequence[int]] = {
    "x":  (1, 0, 0),
    "-x": (-1, 0, 0),
    "y":  (0, 1, 0),
    "-y": (0, -1, 0),
    "z":  (0, 0, 1),
    "-z": (0, 0, -1),
}


def _cross_int(a: Sequence[int], b: Sequence[int]) -> Sequence[int]:
    """Integer cross product of two length-3 vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _is_right_handed(up: str, forward: str, lateral: str) -> bool:
    """True iff the three axis literals form a right-handed orthonormal
    basis under the convention ``forward × lateral == up``.

    This matches the standard robotics right-handed z-up convention
    where (forward, lateral, up) = (x, y, z) and x × y = z.
    """
    if (up not in _AXIS_TO_VEC
            or forward not in _AXIS_TO_VEC
            or lateral not in _AXIS_TO_VEC):
        return False
    # Three distinct absolute axes — rejects e.g. up=x, forward=x.
    abs_axes = {up.lstrip("-"), forward.lstrip("-"), lateral.lstrip("-")}
    if len(abs_axes) != 3:
        return False
    return tuple(_cross_int(_AXIS_TO_VEC[forward], _AXIS_TO_VEC[lateral])) \
        == tuple(_AXIS_TO_VEC[up])


# ---------------------------------------------------------------------------
# Suggest alignment (fuzzy match candidates)
# ---------------------------------------------------------------------------


_COMMON_AFFIXES = ("_joint", "joint_", "j_", "_actuator", "actuator_")


def suggest_alignment(
    source_joint_names: Sequence[str],
    target_sku: str,
) -> Dict[str, str]:
    """Propose a candidate ``{source_name: target_joint_name}`` mapping.

    Pure-string heuristics, applied in order; first match wins:
      1. exact case-insensitive name match
      2. match after stripping common affixes (``_joint``, ``joint_``,
         ``j_``, ``_actuator``, ``actuator_``)
      3. substring containment (source ⊂ target or target ⊂ source)
      4. closest match via :func:`difflib.get_close_matches`
         (cutoff 0.6)

    Returns an empty mapping when ``target_sku`` cannot be resolved or
    declares no joints. Callers treat the output as a hint only — the
    final alignment must still be cross-validated through
    :func:`load_alignment_override`.
    """
    from registers import robots as _robots_registry

    canonical_sku = _robots_registry.resolve_id(target_sku)
    if canonical_sku is None:
        return {}
    robot_spec = _robots_registry.get_robot_spec(canonical_sku)
    if robot_spec is None:
        return {}

    seen: set = set()
    target_names: List[str] = []
    for fmt in robot_spec.available_formats:
        for jn in robot_spec.joint_order_for(fmt):
            if jn not in seen:
                seen.add(jn)
                target_names.append(jn)
    if not target_names:
        return {}

    target_lower_to_orig: Dict[str, str] = {n.lower(): n for n in target_names}

    def _strip_affixes(s: str) -> str:
        out = s
        for a in _COMMON_AFFIXES:
            out = out.replace(a, "")
        return out

    target_stripped_to_orig: Dict[str, str] = {
        _strip_affixes(low): orig for low, orig in target_lower_to_orig.items()
    }

    suggestions: Dict[str, str] = {}
    for src in source_joint_names:
        src_str = str(src)
        src_low = src_str.lower()

        # Layer 1 — exact case-insensitive
        if src_low in target_lower_to_orig:
            suggestions[src_str] = target_lower_to_orig[src_low]
            continue

        # Layer 2 — stripped equivalence
        src_stripped = _strip_affixes(src_low)
        if src_stripped in target_stripped_to_orig:
            suggestions[src_str] = target_stripped_to_orig[src_stripped]
            continue

        # Layer 3 — substring containment
        contains_match: Optional[str] = None
        for tlow, torig in target_lower_to_orig.items():
            if src_low in tlow or tlow in src_low:
                contains_match = torig
                break
        if contains_match is not None:
            suggestions[src_str] = contains_match
            continue

        # Layer 4 — difflib closest match
        close = difflib.get_close_matches(
            src_low, list(target_lower_to_orig), n=1, cutoff=0.6,
        )
        if close:
            suggestions[src_str] = target_lower_to_orig[close[0]]

    return suggestions


# ---------------------------------------------------------------------------
# Contract consistency check
# ---------------------------------------------------------------------------


def validate_alignment_against_contract(
    override: AlignmentOverride,
    contract: "ReferenceMotionContract",
) -> None:
    """Cross-check ``override`` against a loaded
    :class:`ReferenceMotionContract`.

    Compares (source_format ↔ clip.format_id) and the canonical forms
    of (target_sku ↔ contract.target_sku) and (target_family ↔
    contract.target_family). The SKU comparison normalizes both sides
    via ``registers.robots.resolve_id`` so alias differences do not
    cause a false mismatch.

    Raises :exc:`AlignmentOverrideError` on any disagreement.
    """
    from registers import robots as _robots_registry

    if override.source_format != contract.clip.format_id:
        raise AlignmentOverrideError(
            f"alignment_override.source_format={override.source_format!r} "
            f"does not match the contract's clip.format_id="
            f"{contract.clip.format_id!r}. The loader that produced this "
            "contract must use the same source format as the override; "
            "pick a matching override or re-emit the contract from the "
            "matching loader."
        )

    ov_canonical = (
        _robots_registry.resolve_id(override.target_sku)
        or override.target_sku
    )
    ct_canonical = (
        _robots_registry.resolve_id(contract.target_sku)
        or contract.target_sku
    )
    if ov_canonical != ct_canonical:
        raise AlignmentOverrideError(
            f"alignment_override.target_sku={override.target_sku!r} "
            f"(canonical={ov_canonical!r}) does not match the contract's "
            f"target_sku={contract.target_sku!r} "
            f"(canonical={ct_canonical!r}). Re-emit the contract with "
            "the matching target_sku, or pick a different override."
        )

    if override.target_family != contract.target_family:
        raise AlignmentOverrideError(
            f"alignment_override.target_family="
            f"{override.target_family!r} does not match the contract's "
            f"target_family={contract.target_family!r}. Reconcile the "
            "two so a single family value flows through the load."
        )


__all__ = [
    # Schema versions
    "ALIGNMENT_OVERRIDE_SCHEMA_VERSION",
    "AXIS_CONVENTION_SCHEMA_VERSION",
    # Enums
    "ALLOWED_AXIS_LITERALS",
    "ALLOWED_VELOCITY_FRAMES",
    # Defaults
    "DEFAULT_ROOT_UP_AXIS",
    "DEFAULT_ROOT_FORWARD_AXIS",
    "DEFAULT_ROOT_LATERAL_AXIS",
    "DEFAULT_LINEAR_VELOCITY_FRAME",
    "DEFAULT_ANGULAR_VELOCITY_FRAME",
    # Errors
    "AlignmentOverrideError",
    "AxisConventionError",
    # Data
    "AxisConvention",
    "AlignmentOverride",
    # Functions
    "load_alignment_override",
    "suggest_alignment",
    "validate_alignment_against_contract",
]
