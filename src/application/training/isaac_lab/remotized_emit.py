# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Remotized-actuator emission helpers for the IsaacLab env_cfg compiler.

Pure functions (no compiler / IsaacLab / file-system state) that:
  1. partition a robot's joints into remotized vs. non-remotized groups by
     matching the robot's manifest patterns against the physical joint names,
  2. render the IsaacLab actuator-cfg lines (``ImplicitActuatorCfg`` for the
     non-remotized joints, one ``RemotizedPDActuatorCfg`` per remotized group),
  3. build the ``remotized_joints`` provenance block embedded into the
     deploy_contract.

Kept separate from ``env_cfg_compiler`` so it is unit-testable without the
heavy compiler/IsaacLab machinery: callers inject the gains, the joint lists,
and a ``load_table`` callable, and assert on the returned strings/dict.

Design invariants (do not regress):
  * Gains (kp/kd) are NOT computed here — they come from the mass-weighted
    solver and are passed in keyed by physical joint NAME (World B: names are
    stable across permutations, indices are not).
  * The remotized group carries NO ``effort_limit`` box cap (``None``); the
    angle-dependent ceiling comes from the lookup table at runtime — matching
    the reference SPOT_CFG ``RemotizedPDActuatorCfg``.
  * A manifest pattern matching ZERO joints, or a joint matched by MORE THAN
    ONE entry, RAISES (CLAUDE.md §8 — fail loud; a silent miss would ship a
    robot with the wrong actuator model and look like it worked).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RemotizedGroup:
    """One remotized actuator group: the joints sharing a single lookup table."""

    name: str               # IsaacLab actuator-dict key (e.g. "remotized_0")
    pattern: str            # the manifest joint_pattern that selected these
    joint_names: List[str]  # matched physical names, in `physical` order
    table: Any              # TorqueLookupTable (duck-typed; injected)


class RemotizedManifestError(ValueError):
    """Raised when a remotized manifest cannot be applied to a robot's joints."""


def match_remotized_groups(
    physical: Sequence[str],
    remotized_entries: Sequence[Dict[str, Any]],
    load_table: Callable[[str], Any],
) -> List[RemotizedGroup]:
    """Resolve manifest entries to concrete joint groups.

    Args:
        physical: the robot's physical joint names, in the canonical order.
        remotized_entries: ``manifest["remotized_joints"]`` — each a dict with
            ``joint_pattern`` and ``lookup_table_path`` (and optional ``group``).
        load_table: ``path -> TorqueLookupTable`` (injected for testability).

    Raises:
        RemotizedManifestError: a pattern matched no joints, or a joint was
            claimed by more than one entry.
    """
    groups: List[RemotizedGroup] = []
    claimed: Dict[str, str] = {}  # joint_name -> pattern that claimed it
    for idx, entry in enumerate(remotized_entries or []):
        pattern = entry.get("joint_pattern")
        table_path = entry.get("lookup_table_path") or entry.get("lookup_table")
        if not pattern or not table_path:
            raise RemotizedManifestError(
                f"remotized_joints[{idx}] missing joint_pattern or "
                f"lookup_table(_path): {entry!r}"
            )
        rx = re.compile(pattern)
        matched = [n for n in physical if rx.fullmatch(n)]
        if not matched:
            raise RemotizedManifestError(
                f"remotized_joints[{idx}] pattern {pattern!r} matched no joints "
                f"in {list(physical)} — fix the pattern (a silent miss would "
                f"ship the robot with the wrong actuator model)."
            )
        for n in matched:
            if n in claimed:
                raise RemotizedManifestError(
                    f"joint {n!r} matched by two remotized patterns "
                    f"({claimed[n]!r} and {pattern!r}); patterns must "
                    f"partition the joints."
                )
            claimed[n] = pattern
        name = str(entry.get("group") or f"remotized_{idx}")
        groups.append(
            RemotizedGroup(
                name=name,
                pattern=pattern,
                joint_names=matched,
                table=load_table(table_path),
            )
        )
    return groups


def _fmt_gain_dict(names: Sequence[str], gains: Dict[str, float]) -> str:
    """Render a ``{"joint": value, ...}`` Python literal, sorted by name.

    Matches the existing env_cfg emit precision (``%.6g``). Raises if a name
    has no gain (a missing gain is a contract violation, not a default-to-0
    situation — §8)."""
    items = []
    for n in sorted(names):
        if n not in gains:
            raise RemotizedManifestError(
                f"no gain for joint {n!r}; cannot emit actuator cfg"
            )
        items.append(f'"{n}": {float(gains[n]):.6g}')
    return "{" + ", ".join(items) + "}"


def _fmt_cap(names: Sequence[str], value: "float | Dict[str, float]") -> str:
    """Render an effort/velocity cap as either a per-joint ``{name: v}`` dict
    (mirrors stiffness/damping) or a bare scalar (legacy / uniform).

    A per-joint dict is the real, per-motor limit (legged_gym parity, from the
    robot's captured USD DriveAPI); a scalar is the uniform fallback used when
    the robot has no captured physics. Per-joint rendering reuses
    ``_fmt_gain_dict`` so a missing name fails loud (§8)."""
    if isinstance(value, dict):
        return _fmt_gain_dict(names, value)
    return f"{value}"


def _fmt_lookup_array(table: Any) -> str:
    """Render the 3-column ``[[angle, 1.0, max_torque], ...]`` literal that
    ``RemotizedPDActuatorCfg.joint_parameter_lookup`` consumes."""
    arr = table.to_isaaclab_array()
    rows = ", ".join(
        f"[{row[0]:.6f}, {row[1]:.1f}, {row[2]:.6f}]" for row in arr
    )
    return "[" + rows + "]"


def build_actuator_lines(
    *,
    physical: Sequence[str],
    kp: Dict[str, float],
    kd: Dict[str, float],
    effort_limit: "float | Dict[str, float]",
    velocity_limit: "float | Dict[str, float]",
    groups: Sequence[RemotizedGroup],
    armature: Optional[Dict[str, float]] = None,
    indent: str = "            ",
    implicit_cls: str = "ImplicitActuatorCfg",
    remotized_cls: str = "RemotizedPDActuatorCfg",
) -> List[str]:
    """Render the actuator-dict entry lines for the emitted env_cfg.

    With no remotized groups, emits the SINGLE legacy ``"legs"`` group with
    ``joint_names_expr=[".*"]`` — byte-for-byte the prior behavior (the
    non-remotized path is unchanged). With groups present, partitions the
    joints: the non-remotized joints stay on ``ImplicitActuatorCfg`` (now with
    an explicit name list, since a joint may belong to only one IsaacLab
    actuator group), and each remotized group emits a ``RemotizedPDActuatorCfg``
    with no effort_limit (lookup-driven ceiling).

    ``armature`` (CLAUDE.md §10): per-joint reflected-rotor inertia keyed by the
    SAME physical NAME as ``kp``/``kd`` (NOT index — IsaacLab resolves the
    emitted ``armature={name: value}`` dict by joint name against the USD
    articulation, so the MJCF→USD joint-order difference cannot mis-pair the
    values; this is the World-B joint-order lesson). ``mj_fullM`` folds this
    armature into the ``m_eff`` that derived ``kp``, but PhysX's USD authors no
    armature — emitting it here makes PhysX realize the SAME joint-space inertia
    diagonal MuJoCo does. ``None`` (legacy) omits the field; PhysX then uses the
    USD prim value (typically 0) and the calibration gate flags the divergence.
    """
    lines: List[str] = []
    arm = dict(armature or {})

    def _arm_kw(names: Sequence[str]) -> str:
        # Emit ``armature={...}`` ONLY when every name has a value (by-name,
        # never default-to-0 — a missing armature is a contract gap, §8). When
        # ``armature`` is absent entirely (legacy callers), emit nothing.
        if not arm:
            return ""
        return f"armature={_fmt_gain_dict(names, arm)}, "

    if not groups:
        lines.append(
            f'{indent}"legs": {implicit_cls}(joint_names_expr=[".*"], '
            f"stiffness={_fmt_gain_dict(physical, kp)}, "
            f"damping={_fmt_gain_dict(physical, kd)}, "
            f"{_arm_kw(physical)}"
            f"effort_limit={_fmt_cap(physical, effort_limit)}, "
            f"velocity_limit={_fmt_cap(physical, velocity_limit)}),"
        )
        return lines

    remotized_names = {n for g in groups for n in g.joint_names}
    non_rem = [n for n in physical if n not in remotized_names]
    if non_rem:
        names_expr = "[" + ", ".join(f'"{n}"' for n in non_rem) + "]"
        lines.append(
            f'{indent}"legs": {implicit_cls}(joint_names_expr={names_expr}, '
            f"stiffness={_fmt_gain_dict(non_rem, kp)}, "
            f"damping={_fmt_gain_dict(non_rem, kd)}, "
            f"{_arm_kw(non_rem)}"
            f"effort_limit={_fmt_cap(non_rem, effort_limit)}, "
            f"velocity_limit={_fmt_cap(non_rem, velocity_limit)}),"
        )
    for g in groups:
        names_expr = "[" + ", ".join(f'"{n}"' for n in g.joint_names) + "]"
        lines.append(
            f'{indent}"{g.name}": {remotized_cls}(joint_names_expr={names_expr}, '
            f"stiffness={_fmt_gain_dict(g.joint_names, kp)}, "
            f"damping={_fmt_gain_dict(g.joint_names, kd)}, "
            f"{_arm_kw(g.joint_names)}"
            # velocity_limit is a real actuator parameter, independent of the
            # torque ceiling — emit it like the non-remotized group (a missing
            # velocity_limit makes the deploy-contract parser reject the group).
            f"velocity_limit={_fmt_cap(g.joint_names, velocity_limit)}, "
            f"joint_parameter_lookup={_fmt_lookup_array(g.table)}, "
            # effort_limit=None: no box cap; the lookup table sets the
            # angle-dependent ceiling at runtime (matches reference SPOT_CFG).
            # The trailing comment makes that intent legible in the generated
            # env_cfg itself, not just here. The DEPLOY contract's per-joint
            # effort_limit for these joints is derived from the lookup PEAK by
            # manifest_parser (the IsaacLab cfg keeps None; IsaacLab forces inf).
            f"effort_limit=None),"
            f"  # IsaacLab forces this to inf in __init__; explicit None makes intent clear"
        )
    return lines


def build_remotized_provenance(
    *,
    physical: Sequence[str],
    ir_roles: Sequence[str],
    groups: Sequence[RemotizedGroup],
) -> Optional[Dict[str, Any]]:
    """Build the ``pd_derivation.remotized_joints`` provenance block.

    Returns ``None`` when no joints are remotized. Otherwise a dict::

        {"array_ordering": "ir_roles",
         "joints": {<physical_name>: {"ir_role", "table_sha256",
                    "peak_torque", "min_torque", "table": <table.to_dict()>}}}

    The embedded ``table`` is the SINGLE copy reused by Step 4.2's
    ``mujoco_torque_lookups`` — both engines read the same payload, so they
    cannot drift (World B lesson). Keyed by physical NAME (stable across
    permutations); ``array_ordering: ir_roles`` documents that the joint
    listing follows the IR-role order, NOT the table's internal
    monotonic-axis order (which lives inside each ``table`` object).
    """
    if not groups:
        return None
    by_name = {n: g for g in groups for n in g.joint_names}
    joints: Dict[str, Any] = {}
    for role, name in zip(ir_roles, physical):
        g = by_name.get(name)
        if g is None:
            continue
        table = g.table
        joints[name] = {
            "ir_role": role,
            "group": g.name,
            "table_sha256": table.sha256(),
            "peak_torque": table.peak_torque,
            "min_torque": table.min_torque,
            "table": table.to_dict(),
        }
    return {"array_ordering": "ir_roles", "joints": joints}
