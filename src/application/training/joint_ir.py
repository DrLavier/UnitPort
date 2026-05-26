# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.joint_ir — IR-Joint canonical translation core.

The single, authoritative translator between the application layer's IR
joint role names (``hip_FL`` / ``thigh_FR`` / ``calf_RL`` / etc.) and the
substrate layer's physical joint names (``FL_hip_joint`` from a Unitree
USD, ``LF_HAA`` from an ANYmal URDF, ``fl_hx`` from a Boston Dynamics Spot
MJCF, ...).

The contract (Phase 5):

  * **Application layer is IR-only.**  Canvas params, ``TrainingSpec``,
    bundle manifest, deploy contracts — every joint identifier on these
    layers MUST be an IR role from :data:`registers.robots.RobotSpec.joint_ir_roles`.
  * **Substrate layer is physical-only.**  The USD / MJCF / URDF asset and
    the SDK action interface use vendor-specific physical names.
  * **Translation happens at exactly one boundary per direction**, via this
    module.  IR → physical when emitting into env_cfg.py / binding to
    MJCF actuators / forwarding to SDK; physical → IR when reading raw
    asset metadata back into the application layer.

No vendor-string heuristics live here.  Token recognition (e.g.
``fl_hx → hip_FL``) is the job of the one-shot canvas migration script
under ``scripts/``, not the runtime translator.  An unknown token here
is always an error — never a guess.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
)

if TYPE_CHECKING:
    from application.training.training_spec import RobotSpecRef


T = TypeVar("T")


class JointIRResolver:
    """Bind a single :class:`RobotSpecRef` and translate joint names.

    The resolver is a thin view over ``robot.joint_ir_roles`` (IR) ↔
    ``robot.joint_order`` (physical) parallel arrays.  Both arrays must
    be the same length; the *i*\\-th IR role corresponds to the *i*\\-th
    physical joint.

    Example::

        resolver = JointIRResolver(spec.robot)
        physical = resolver.to_physical("hip_FL")
        # → "FL_hip_joint" for unitree.go2

        physical_dict = resolver.to_physical_dict({"hip_FL": 0.1, "thigh_FL": 0.8})
        # → {"FL_hip_joint": 0.1, "FL_thigh_joint": 0.8}

        ir = resolver.to_ir("FL_calf_joint")
        # → "calf_FL"
    """

    def __init__(
        self,
        robot: "RobotSpecRef",
        *,
        active_format: Optional[str] = None,
    ) -> None:
        if robot is None:
            raise ValueError("JointIRResolver requires a RobotSpecRef (got None)")
        self._robot = robot

        # Per-format schema (Stage 3): resolve which format's joint table
        # to read. Caller can pin it explicitly; otherwise we use the
        # spec's ``active_format``. Empty in both → fall back to the
        # legacy flat ``joint_order`` (transitional safety net for
        # callers that haven't been migrated yet).
        fmt = str(active_format or "").strip().upper()
        if not fmt:
            fmt = str(getattr(robot, "active_format", "") or "").upper()
        self._active_format = fmt

        if fmt:
            joints_role_map = robot.joints_role_map_for(fmt)
            ir_roles: List[str] = list(robot.joint_ir_roles_for(fmt))
            physical: List[str] = list(robot.joint_order_for(fmt))
            if not ir_roles:
                # CLAUDE.md §1.8: previously fell back to the flat
                # ``joint_ir_roles`` / ``joint_order`` views (resolved via
                # preferred_format) so a mid-rollout caller with the wrong
                # active_format kept working. That silently substituted a
                # different format's joint table — IL paths that asked for
                # USD got MJCF when joints_per_format["USD"] was null. Fail
                # loud so the data shape that produced this is fixed at the
                # source (Dump <fmt> in the Robot Asset card).
                raise ValueError(
                    f"[JointIRResolver] robot sku={robot.sku!r} declares no "
                    f"joints under joints_per_format[{fmt!r}] — the format "
                    f"the resolver was bound to. Silent fallback to another "
                    f"format's joint table is forbidden: joint orders differ "
                    f"across MJCF / USD / URDF and serving the wrong table "
                    f"corrupts every downstream IR↔physical mapping (PD "
                    f"gains, init-pose, action-joint slot, bundle export). "
                    f"Populate joints_per_format[{fmt!r}] first via the "
                    f"Robot Asset card's Dump {fmt} button."
                )
        else:
            ir_roles = list(getattr(robot, "joint_ir_roles", None) or [])
            physical = list(getattr(robot, "joint_order", None) or [])

        if len(ir_roles) != len(physical):
            raise ValueError(
                f"RobotSpecRef inconsistency: joint_ir_roles ({len(ir_roles)}) "
                f"vs joint_order ({len(physical)}) length mismatch in format "
                f"{fmt!r} — the two parallel arrays MUST be the same length"
            )
        if not ir_roles:
            raise ValueError(
                f"RobotSpecRef sku={robot.sku!r} has empty joint arrays for "
                f"format {fmt!r} — the robot must declare at least one joint "
                f"with an ir_role under joints_per_format[{fmt!r}] in "
                f"registers/data/robots_canonical.json (run Dump {fmt} "
                f"in the Robot Asset card if the format is new)"
            )

        # Build forward + reverse maps for O(1) lookup.
        # Bucket roles ({"misc"} + any "sensor*" prefix) are explicitly
        # allowed to bind multiple physical joints — they're catch-all
        # buckets, not 1:1 identifiers. The auto-suggester routes
        # cosmetic / out-of-scope joints (Go2's Head_upper, base_white,
        # IMU mounts, lidar mounts, ...) into these buckets; demanding
        # uniqueness there would force Dump-time hand-disambiguation of
        # joints the policy never actuates. Forward lookup
        # (``to_physical("misc")``) for bucket roles is intentionally
        # undefined and raises — callers should query specific physical
        # names via ``to_ir`` instead.
        self._ir_to_physical: Dict[str, str] = {}
        self._physical_to_ir: Dict[str, str] = {}
        self._bucket_ir_members: Dict[str, List[str]] = {}
        for ir, phys in zip(ir_roles, physical):
            ir = str(ir)
            phys = str(phys)
            if not ir:
                raise ValueError(
                    f"RobotSpecRef sku={robot.sku!r} has a physical joint "
                    f"{phys!r} with empty ir_role in format {fmt!r} — "
                    f"every joint must declare its ir_role in "
                    f"registers/data/robots_canonical.json"
                )
            is_bucket = ir == "misc" or ir.startswith("sensor")
            if is_bucket:
                # Reverse map stays well-defined (each phys → its bucket).
                # Forward map skipped — to_physical(bucket) is undefined.
                self._physical_to_ir[phys] = ir
                self._bucket_ir_members.setdefault(ir, []).append(phys)
                continue
            if ir in self._ir_to_physical:
                raise ValueError(
                    f"RobotSpecRef sku={robot.sku!r}: IR role {ir!r} is bound "
                    f"to multiple physical joints in format {fmt!r} "
                    f"({self._ir_to_physical[ir]!r} and {phys!r}) — IR roles "
                    f"must be unique (bucket roles 'misc' / 'sensor*' are "
                    f"exempt from this rule)"
                )
            self._ir_to_physical[ir] = phys
            self._physical_to_ir[phys] = ir

    # ------------------------------------------------------------------
    # IR → physical (substrate-emit boundary)
    # ------------------------------------------------------------------

    def to_physical(self, ir_role: str) -> str:
        """Translate one IR role to its physical joint name.

        Raises :class:`KeyError` (with a helpful message) if ``ir_role`` is
        not a registered IR role for this robot, or if it's a bucket role
        (``misc`` / ``sensor*``) which deliberately maps 1:N.
        """
        try:
            return self._ir_to_physical[ir_role]
        except KeyError:
            members = self._bucket_ir_members.get(ir_role)
            if members is not None:
                raise KeyError(
                    f"IR role {ir_role!r} is a multi-bind bucket on robot "
                    f"{self._robot.sku!r} (members: {members!r}). "
                    f"to_physical() is undefined for buckets — query "
                    f"specific physical names via to_ir() instead, or "
                    f"iterate members via bucket_members({ir_role!r})."
                ) from None
            raise KeyError(
                f"IR role {ir_role!r} is not defined for robot "
                f"{self._robot.sku!r} ({self._robot.name!r}). "
                f"Valid IR roles: {sorted(self._ir_to_physical)}"
            ) from None

    def bucket_members(self, ir_role: str) -> List[str]:
        """Return the physical joints bound to a multi-bind bucket role
        (``misc`` / ``sensor*``). Empty list when the role isn't a bucket
        on this robot."""
        return list(self._bucket_ir_members.get(ir_role, []))

    def to_physical_dict(self, ir_dict: Dict[str, T], *, where: str) -> Dict[str, T]:
        """Translate every key of ``ir_dict`` from IR role to physical name.

        Validates the entire key set first via :meth:`validate_ir_keys`,
        so the caller gets one error listing all offending tokens rather
        than failing on the first bad key.

        ``where`` identifies the canvas/spec source for the error message
        (e.g. ``"actor_setting.init_joint_angles"``).
        """
        if not isinstance(ir_dict, dict):
            raise TypeError(
                f"to_physical_dict expected dict, got {type(ir_dict).__name__} "
                f"at {where!r}"
            )
        self.validate_ir_keys(ir_dict.keys(), where=where)
        return {self._ir_to_physical[k]: v for k, v in ir_dict.items()}

    def to_physical_list(self, ir_list: Iterable[str], *, where: str) -> List[str]:
        """Translate an IR-role list to a physical-name list (preserves order)."""
        ir_list = list(ir_list)
        self.validate_ir_keys(ir_list, where=where)
        return [self._ir_to_physical[ir] for ir in ir_list]

    # ------------------------------------------------------------------
    # Physical → IR (asset/SDK reverse boundary)
    # ------------------------------------------------------------------

    def to_ir(self, physical_name: str) -> str:
        """Translate one physical joint name to its IR role."""
        try:
            return self._physical_to_ir[physical_name]
        except KeyError:
            raise KeyError(
                f"Physical joint {physical_name!r} is not registered for robot "
                f"{self._robot.sku!r} ({self._robot.name!r}). "
                f"Valid physical names: {sorted(self._physical_to_ir)}"
            ) from None

    def to_ir_list(self, phys_list: Iterable[str]) -> List[str]:
        """Translate a physical-name list to an IR-role list (preserves order)."""
        return [self.to_ir(p) for p in phys_list]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_ir_keys(self, tokens: Iterable[str], *, where: str) -> None:
        """Assert every token in ``tokens`` is a valid IR role.

        Raises :class:`ValueError` listing all offending tokens + the full
        set of valid IR roles for this robot + a fix hint.  The message is
        designed to land directly in the user's training log so they can
        edit the canvas without diving into source.

        ``where`` is a short human-readable source label such as
        ``"actor_setting.init_joint_angles"`` or
        ``"spec.actor.action_joint_names_expr"``.
        """
        bad = [t for t in tokens if t not in self._ir_to_physical]
        if not bad:
            return
        valid = sorted(self._ir_to_physical)
        raise ValueError(
            f"\n[UnitPort][JointIR] {where} contains non-IR joint name(s): "
            f"{bad!r}\n"
            f"  Robot: {self._robot.sku!r} ({self._robot.name!r})\n"
            f"  Family: {self._robot.families}\n"
            f"  Valid IR roles for this robot ({len(valid)}): {valid}\n"
            f"  Fix: edit the canvas so every joint key is an IR role from "
            f"the list above. RELEASE forbids physical names "
            f"({list(self._physical_to_ir)[:3]}, ...) and vendor abbreviations "
            f"(e.g. fl_hx, LF_HAA) in canvas joint dicts. "
            f"Run `bootstrap/migrate_canvas_joint_names_to_ir.py` to auto-translate "
            f"a legacy canvas, or open the Actor Setting node and re-pick joints "
            f"from the IR-role dropdown."
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def ir_roles(self) -> List[str]:
        """Return the IR role list in joint registration order."""
        return list(self._ir_to_physical)

    @property
    def physical_names(self) -> List[str]:
        """Return the physical joint list in registration order."""
        return list(self._physical_to_ir)

    @property
    def num_joints(self) -> int:
        return len(self._ir_to_physical)

    @property
    def active_format(self) -> str:
        """The asset format whose joint table this resolver was bound to."""
        return self._active_format


# ---------------------------------------------------------------------------
# Top-level convenience: SKU → resolver (deploy / sim review use this)
# ---------------------------------------------------------------------------


def make_resolver_for_sku(
    robot_sku: str,
    *,
    active_format: Optional[str] = None,
) -> JointIRResolver:
    """Build a :class:`JointIRResolver` from a registry SKU lookup.

    Used by the deploy stack (``PolicyRunner`` / ``ActionApplier`` /
    ``PDController`` / ``MujocoReviewTask``) which holds a SKU rather than
    a full ``TrainingSpec``.  Wraps ``registers.robots.get_robot_spec``
    + :meth:`RobotSpecRef.from_registry` so the deploy code path does not
    re-implement the lookup.

    ``active_format`` (Stage 3): the format whose joint table to bind. If
    omitted, falls back to the registry's ``preferred_format``. Deploy
    callers should pin this explicitly when they know which asset their
    sim review / runtime is loading (Isaac Sim → USD, MuJoCo → MJCF, …).

    Raises :class:`ValueError` if the SKU is unknown to the registry.
    """
    from application.training.training_spec import RobotSpecRef
    from registers.robots import get_robot_spec, resolve_id

    if not robot_sku:
        raise ValueError("make_resolver_for_sku: robot_sku must be non-empty")

    canonical_sku = resolve_id(robot_sku) or robot_sku
    rs = get_robot_spec(canonical_sku)
    if rs is None:
        raise ValueError(
            f"make_resolver_for_sku: robot_sku {robot_sku!r} (canonical "
            f"{canonical_sku!r}) is not registered. Available SKUs are listed "
            f"by registers.robots.list_skus()."
        )

    ref = RobotSpecRef.from_registry(rs, active_format=active_format or "")
    return JointIRResolver(ref, active_format=active_format)


__all__ = [
    "JointIRResolver",
    "make_resolver_for_sku",
]
