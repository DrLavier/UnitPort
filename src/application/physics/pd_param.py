# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PD parameterization dataclasses.

The canonical control surface for the sim2sim PD framework. Everything
upstream (canvas, registers, user overlay) produces a :class:`PDParam`;
everything downstream (engine gain solvers, env runtime, bundle
finalizer, deploy contract) consumes one. No raw ``kp``/``kd`` numbers
flow through this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


PD_PARAM_MODE = Literal["natural_freq"]
"""The only supported parameterization today.

A future ``"kp_kd_direct"`` mode would let power users hand-tune raw
gains per joint; not implemented in the initial framework.
"""


GAIN_ENGINE = Literal["mujoco", "physx"]


# Soft physical bounds — see rule R_PD2 in spec_validator. Values outside
# the band almost always mean someone copy-pasted a kp/kd value into the
# omega_n field by mistake. Centralized here so the constants don't drift
# between the validator and the dataclass.
OMEGA_N_MIN = 0.5
OMEGA_N_MAX = 200.0
ZETA_MIN = 0.1
ZETA_MAX = 2.0


@dataclass(frozen=True)
class PDGroup:
    """One joint group's (omega_n, zeta) at a point in time.

    Attributes
    ----------
    name:
        Group identifier (e.g. ``"hip_x"``, ``"knee"``). Stable; matches
        the key in :data:`registers.pd_groups.list_groups`.
    omega_n:
        Undamped natural frequency in rad/s. Physical interpretation:
        the second-order tracking bandwidth (for ``zeta=1``, the
        rise-time-from-rest at 10→90% is ~``2.3 / omega_n`` seconds).
    zeta:
        Damping ratio (dimensionless). ``1.0`` is critically damped;
        ``< 1`` is under-damped (overshoot); ``> 1`` is over-damped.
    ir_role_regex:
        The IR-role pattern that selects which joints this group
        covers. Carried for provenance / debugging only — group→role
        resolution at compile time goes through
        :func:`registers.pd_groups.find_group_for_role`.
    source:
        Where this entry came from. One of:
        ``"family_default"`` (loaded from
        ``registers/data/pd_groups_defaults.json``),
        ``"user_overlay"`` (merged from
        ``<USER_CONFIG_DIR>/registers/pd_groups_custom.json``),
        ``"canvas_override"`` (the canvas ActuatorPDNode set it),
        ``"dr_scaled"`` (domain-randomization perturbation of an
        upstream group at episode reset).
    """

    name: str
    omega_n: float
    zeta: float
    ir_role_regex: str = ""
    source: str = "family_default"

    def __post_init__(self) -> None:
        # Frozen dataclass — call object.__setattr__ to validate without
        # mutating the user-facing fields.
        if not (OMEGA_N_MIN <= float(self.omega_n) <= OMEGA_N_MAX):
            raise ValueError(
                f"PDGroup({self.name!r}): omega_n={self.omega_n} "
                f"out of physical range [{OMEGA_N_MIN}, {OMEGA_N_MAX}] rad/s. "
                f"Did you paste a kp value here? Use omega_n_log_uniform "
                f"if you wanted a DR factor."
            )
        if not (ZETA_MIN <= float(self.zeta) <= ZETA_MAX):
            raise ValueError(
                f"PDGroup({self.name!r}): zeta={self.zeta} "
                f"out of physical range [{ZETA_MIN}, {ZETA_MAX}]. "
                f"Critically damped is 1.0; under-damped < 1.0; "
                f"over-damped > 1.0."
            )


@dataclass(frozen=True)
class PDParam:
    """The full PD parameterization for one training/deploy spec.

    ``family`` is required so a downstream consumer (engine compiler,
    runtime env) can resolve groups against the right slice of the IR
    canonical without having to thread the robot SKU through every
    callsite.
    """

    mode: PD_PARAM_MODE
    family: str
    groups: Dict[str, PDGroup]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_family_defaults(cls, family: str) -> "PDParam":
        """Build a :class:`PDParam` from ``registers.pd_groups`` defaults.

        Fails loud (raises) if the family has no groups defined — the
        consumer would otherwise see an empty parameterization and
        silently produce zero gains for every joint.
        """
        from registers import pd_groups as _pd_groups

        defs = {g["id"]: g for g in _pd_groups.list_groups(family)}
        if not defs:
            raise RuntimeError(
                f"PDParam.from_family_defaults: family={family!r} has no "
                f"groups in pd_groups_definitions.json. Either add groups "
                f"there or declare a different family on the robot."
            )
        defaults = _pd_groups.get_defaults(family)
        groups: Dict[str, PDGroup] = {}
        for gid, gdef in defs.items():
            vals = defaults.get(gid)
            if vals is None:
                raise RuntimeError(
                    f"PDParam.from_family_defaults: family={family!r} group "
                    f"{gid!r} is defined but has no default (omega_n, zeta). "
                    f"Fix pd_groups_defaults.json."
                )
            groups[gid] = PDGroup(
                name=gid,
                omega_n=float(vals["omega_n"]),
                zeta=float(vals["zeta"]),
                ir_role_regex=gdef["ir_role_regex"],
                source="family_default",
            )
        return cls(mode="natural_freq", family=family, groups=groups)

    # ------------------------------------------------------------------
    # Overlay / scaling
    # ------------------------------------------------------------------

    def with_overrides(self, overrides: Dict[str, Dict[str, float]]) -> "PDParam":
        """Return a new :class:`PDParam` with canvas overrides applied.

        ``overrides`` is a ``{group_id: {"omega_n": ..., "zeta": ...}}``
        partial mapping. Unknown groups raise (fail-loud); known groups
        with partial keys keep the unspecified field from the base.
        """
        merged: Dict[str, PDGroup] = dict(self.groups)
        for gid, partial in (overrides or {}).items():
            if gid not in merged:
                raise RuntimeError(
                    f"PDParam.with_overrides: group {gid!r} is not in the "
                    f"base parameterization (known groups: "
                    f"{list(merged.keys())}). Add it to "
                    f"pd_groups_definitions.json or fix the canvas typo."
                )
            base = merged[gid]
            new_omega = float(partial.get("omega_n", base.omega_n))
            new_zeta = float(partial.get("zeta", base.zeta))
            merged[gid] = PDGroup(
                name=base.name,
                omega_n=new_omega,
                zeta=new_zeta,
                ir_role_regex=base.ir_role_regex,
                source="canvas_override",
            )
        return PDParam(mode=self.mode, family=self.family, groups=merged)

    def scaled(
        self,
        *,
        omega_n_factor: float = 1.0,
        zeta_factor: float = 1.0,
    ) -> "PDParam":
        """Return a new :class:`PDParam` with multiplicative DR scaling.

        Used by domain randomization (Stage H) to perturb the
        parameterization at episode reset before re-solving engine
        gains. Factors apply uniformly across every group — per-group
        DR is a future extension if needed.
        """
        new_groups: Dict[str, PDGroup] = {}
        for gid, g in self.groups.items():
            new_groups[gid] = PDGroup(
                name=g.name,
                omega_n=float(g.omega_n) * float(omega_n_factor),
                zeta=float(g.zeta) * float(zeta_factor),
                ir_role_regex=g.ir_role_regex,
                source="dr_scaled",
            )
        return PDParam(mode=self.mode, family=self.family, groups=new_groups)

    # ------------------------------------------------------------------
    # (De)serialization — used by env.yaml sidecar and deploy_contract
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode),
            "family": str(self.family),
            "groups": {
                gid: {
                    "omega_n": float(g.omega_n),
                    "zeta": float(g.zeta),
                    "ir_role_regex": str(g.ir_role_regex),
                    "source": str(g.source),
                }
                for gid, g in self.groups.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PDParam":
        if not isinstance(raw, dict):
            raise ValueError(
                f"PDParam.from_dict: expected dict, got {type(raw).__name__}"
            )
        mode = str(raw.get("mode", "")).strip()
        if mode != "natural_freq":
            raise ValueError(
                f"PDParam.from_dict: unsupported mode={mode!r}; only "
                f"'natural_freq' is implemented today."
            )
        family = str(raw.get("family", "")).strip()
        if not family:
            raise ValueError(
                "PDParam.from_dict: 'family' is missing or empty. The "
                "family is required so consumers can resolve groups "
                "against the right IR canonical slice."
            )
        groups_raw = raw.get("groups", {}) or {}
        if not isinstance(groups_raw, dict) or not groups_raw:
            raise ValueError(
                "PDParam.from_dict: 'groups' is missing or empty — "
                "an empty parameterization would silently produce zero "
                "gains for every joint."
            )
        groups: Dict[str, PDGroup] = {}
        for gid, vals in groups_raw.items():
            if not isinstance(vals, dict):
                raise ValueError(
                    f"PDParam.from_dict: group {gid!r} is not a mapping"
                )
            try:
                groups[gid] = PDGroup(
                    name=str(gid),
                    omega_n=float(vals["omega_n"]),
                    zeta=float(vals["zeta"]),
                    ir_role_regex=str(vals.get("ir_role_regex", "")),
                    source=str(vals.get("source", "family_default")),
                )
            except KeyError as exc:
                raise ValueError(
                    f"PDParam.from_dict: group {gid!r} missing field {exc!s}"
                ) from exc
        return cls(mode="natural_freq", family=family, groups=groups)


@dataclass(frozen=True)
class JointPDGains:
    """Per-joint engine-specific gains, produced by a gain solver.

    ``kp`` and ``kd`` are aligned with ``joint_physical_names`` (the
    engine-side order — MJCF actuator order, USD articulation joint
    order, etc.) and parallel to ``joint_ir_roles`` (which is the role
    each physical joint maps to).
    """

    joint_ir_roles: List[str]
    joint_physical_names: List[str]
    kp: List[float]
    kd: List[float]
    engine: GAIN_ENGINE
    derivation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.joint_physical_names)
        for name, lst in (
            ("joint_ir_roles", self.joint_ir_roles),
            ("kp", self.kp),
            ("kd", self.kd),
        ):
            if len(lst) != n:
                raise ValueError(
                    f"JointPDGains: {name} has length {len(lst)} but "
                    f"joint_physical_names has length {n}"
                )
        for v in self.kp:
            if not (v > 0.0):
                raise ValueError(
                    f"JointPDGains: every kp must be strictly positive; "
                    f"got {self.kp!r}. A zero/negative kp would produce "
                    f"a non-tracking joint silently."
                )
        for v in self.kd:
            if v < 0.0:
                raise ValueError(
                    f"JointPDGains: every kd must be non-negative; got "
                    f"{self.kd!r}."
                )

    def derivation_armature(self) -> List[float]:
        """Per-joint reflected-rotor armature recorded in ``derivation``,
        parallel to ``joint_physical_names``. Empty when the derivation predates
        the armature-aware solver (returns ``0.0`` per joint then). The MuJoCo
        solver folds this armature into ``m_eff`` (``kp = m_eff·ωn²``); it is
        surfaced here so the realized-inertia calibration gate and the PhysX
        ``ImplicitActuatorCfg.armature`` emit path consume the SAME value."""
        per_joint = self.derivation.get("per_joint") or []
        by_name = {
            e.get("physical_name"): float(e.get("armature", 0.0))
            for e in per_joint
            if isinstance(e, dict)
        }
        return [by_name.get(n, 0.0) for n in self.joint_physical_names]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_ir_roles": list(self.joint_ir_roles),
            "joint_physical_names": list(self.joint_physical_names),
            "kp": [float(v) for v in self.kp],
            "kd": [float(v) for v in self.kd],
            "engine": str(self.engine),
            "derivation": dict(self.derivation),
        }
