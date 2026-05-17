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

    def __init__(self, robot: "RobotSpecRef") -> None:
        if robot is None:
            raise ValueError("JointIRResolver requires a RobotSpecRef (got None)")
        self._robot = robot

        ir_roles = list(getattr(robot, "joint_ir_roles", None) or [])
        physical = list(getattr(robot, "joint_order", None) or [])

        if len(ir_roles) != len(physical):
            raise ValueError(
                f"RobotSpecRef inconsistency: joint_ir_roles ({len(ir_roles)}) "
                f"vs joint_order ({len(physical)}) length mismatch — "
                f"the two parallel arrays MUST be the same length"
            )
        if not ir_roles:
            raise ValueError(
                f"RobotSpecRef sku={robot.sku!r} has empty joint arrays — "
                f"the robot must have at least one joint with an ir_role "
                f"declared in registers/data/robots_canonical.json"
            )

        # Build forward + reverse maps for O(1) lookup.
        self._ir_to_physical: Dict[str, str] = {}
        self._physical_to_ir: Dict[str, str] = {}
        for ir, phys in zip(ir_roles, physical):
            ir = str(ir)
            phys = str(phys)
            if not ir:
                raise ValueError(
                    f"RobotSpecRef sku={robot.sku!r} has a physical joint "
                    f"{phys!r} with empty ir_role — every joint must declare "
                    f"its ir_role in registers/data/robots_canonical.json"
                )
            if ir in self._ir_to_physical:
                raise ValueError(
                    f"RobotSpecRef sku={robot.sku!r}: IR role {ir!r} is bound "
                    f"to multiple physical joints "
                    f"({self._ir_to_physical[ir]!r} and {phys!r}) — IR roles "
                    f"must be unique"
                )
            self._ir_to_physical[ir] = phys
            self._physical_to_ir[phys] = ir

    # ------------------------------------------------------------------
    # IR → physical (substrate-emit boundary)
    # ------------------------------------------------------------------

    def to_physical(self, ir_role: str) -> str:
        """Translate one IR role to its physical joint name.

        Raises :class:`KeyError` (with a helpful message) if ``ir_role`` is
        not a registered IR role for this robot.
        """
        try:
            return self._ir_to_physical[ir_role]
        except KeyError:
            raise KeyError(
                f"IR role {ir_role!r} is not defined for robot "
                f"{self._robot.sku!r} ({self._robot.name!r}). "
                f"Valid IR roles: {sorted(self._ir_to_physical)}"
            ) from None

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


# ---------------------------------------------------------------------------
# Top-level convenience: SKU → resolver (deploy / sim review use this)
# ---------------------------------------------------------------------------


def make_resolver_for_sku(robot_sku: str) -> JointIRResolver:
    """Build a :class:`JointIRResolver` from a registry SKU lookup.

    Used by the deploy stack (``PolicyRunner`` / ``ActionApplier`` /
    ``PDController`` / ``MujocoReviewTask``) which holds a SKU rather than
    a full ``TrainingSpec``.  Wraps ``registers.robots.get_robot_spec``
    + :meth:`RobotSpecRef.from_registry` so the deploy code path does not
    re-implement the lookup.

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

    ref = RobotSpecRef.from_registry(rs)
    return JointIRResolver(ref)


__all__ = [
    "JointIRResolver",
    "make_resolver_for_sku",
]
