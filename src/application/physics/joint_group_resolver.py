# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Expand a :class:`PDParam` against a robot's IR-role list.

The compile-time bridge between the abstract group definition
(``hip_x`` → regex ``^hip_(FL|FR|RL|RR)$``) and a concrete per-joint
``(omega_n, zeta)`` lookup. Both gain solvers consume the output of
:func:`expand_groups_to_joints` — they should never re-match regexes
themselves.
"""

from __future__ import annotations

from typing import List, Tuple

from .pd_param import PDGroup, PDParam


class UnmappedRoleError(RuntimeError):
    """Raised when an IR role has no matching group in the parameterization.

    Distinct exception type so callers can catch it specifically and
    re-raise with a UI-targeted message (the canvas can highlight the
    offending joint).
    """


def expand_groups_to_joints(
    pd_param: PDParam,
    joint_ir_roles: List[str],
) -> List[Tuple[str, PDGroup]]:
    """Return ``[(role, matched_group), ...]`` aligned with ``joint_ir_roles``.

    Parameters
    ----------
    pd_param:
        The parameterization to match against.
    joint_ir_roles:
        IR roles for the robot's joints, in the order they appear in
        the engine (MJCF actuator order, USD articulation order, etc.).

    Raises
    ------
    UnmappedRoleError
        If any role has no matching group. The error lists the offending
        roles and the available groups so a user editing the canvas can
        fix the gap (typically by extending
        ``pd_groups_definitions.json`` or — for one-off testing —
        ``<USER_CONFIG_DIR>/registers/pd_groups_custom.json``).
    """
    if not isinstance(joint_ir_roles, list):
        raise TypeError(
            f"expand_groups_to_joints: joint_ir_roles must be a list, "
            f"got {type(joint_ir_roles).__name__}"
        )
    if not joint_ir_roles:
        raise ValueError(
            "expand_groups_to_joints: joint_ir_roles is empty — "
            "an articulation with zero joints would still expose qpos "
            "for a free root; check the JointIRResolver output."
        )

    # Compile once per call rather than per role × per group. The
    # registers cache already holds compiled patterns, but groups can
    # also come from PDParam.with_overrides() / .scaled() which carry
    # the regex as a plain string — so we re-compile here for safety.
    import re

    compiled: List[Tuple[str, "re.Pattern[str]", PDGroup]] = []
    for gid, g in pd_param.groups.items():
        regex = g.ir_role_regex or ""
        if not regex:
            raise RuntimeError(
                f"expand_groups_to_joints: group {gid!r} carries an empty "
                f"ir_role_regex. The canvas-side PDParam.from_family_defaults "
                f"path should always populate it from "
                f"pd_groups_definitions.json; an empty regex here is a bug."
            )
        try:
            compiled.append((gid, re.compile(regex), g))
        except re.error as exc:
            raise RuntimeError(
                f"expand_groups_to_joints: group {gid!r} ir_role_regex "
                f"{regex!r} failed to compile: {exc}"
            ) from exc

    out: List[Tuple[str, PDGroup]] = []
    unmapped: List[str] = []
    ambiguous: List[Tuple[str, List[str]]] = []

    for role in joint_ir_roles:
        matches = [(gid, g) for gid, pat, g in compiled if pat.fullmatch(role)]
        if not matches:
            unmapped.append(role)
            continue
        if len(matches) > 1:
            ambiguous.append((role, [gid for gid, _ in matches]))
            continue
        out.append((role, matches[0][1]))

    if unmapped:
        raise UnmappedRoleError(
            f"PD parameterization (family={pd_param.family!r}) does not "
            f"cover joint role(s) {unmapped}. Available groups: "
            f"{[gid for gid, _, _ in compiled]}. Fix by either adding a "
            f"matching group to pd_groups_definitions.json (family "
            f"defaults) or to <USER_CONFIG_DIR>/registers/pd_groups_custom.json "
            f"(per-user override)."
        )
    if ambiguous:
        details = "; ".join(f"{role!r} matches {gids}" for role, gids in ambiguous)
        raise RuntimeError(
            f"PD parameterization (family={pd_param.family!r}) has "
            f"overlapping group regexes: {details}. Resolve by tightening "
            f"the regexes so each role matches exactly one group. The "
            f"registry validator should have caught this at boot — please "
            f"file a bug if you're seeing it at runtime."
        )

    return out
