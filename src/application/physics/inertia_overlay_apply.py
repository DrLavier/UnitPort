# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Shared application of a USD→MJCF inertia overlay onto a live ``MjModel``.

ONE patcher, used by BOTH halves of the sim2sim inertia contract:

  * the **runtime** (``MjActor`` at deploy / sim load), and
  * the **PD m_eff derivation** (``mujoco_gain_solver.effective_inertia_diag``
    at gain-solve time).

The point (CLAUDE.md §10 / §11): the joint-space inertia the gains are derived
against must be byte-identical to the inertia the robot is actually simulated
with. Before this module existed, ``effective_inertia_diag`` read the **raw**
MJCF (link inertia as authored in the menagerie file) while ``MjActor`` applied
the overlay (link inertia corrected toward the USD the policy trained on) — so
``kp`` was derived on one link inertia and *realized* on another. For H1 that
residual is ~2%; for a Spot-class trunk (10× inertia error) it is catastrophic.
Routing both through this single function closes that "derive ≠ realize" gap
and guarantees a future overlay-schema change cannot silently desync the two.

This function is **pure**: it takes an already-parsed overlay dict (the one
``read_mjcf_inertia_overlay`` returns) plus a ``MjModel`` and patches the four
body-dynamics arrays in place. It reads NOTHING from disk and imports NO
service layer — the *caller* resolves the overlay and decides how to log. Only
``body_mass`` / ``body_inertia`` / ``body_iquat`` / ``body_ipos`` are touched;
joints, actuators (including ``dof_armature``), collision geometry, sites, and
the kinematic tree are never modified. ``dof_armature`` is deliberately left
alone — armature is the MJCF-authoritative half of the joint-space inertia
diagonal (CLAUDE.md §10) and flows to PhysX via ``ImplicitActuatorCfg.armature``,
not through this body-inertia overlay.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def apply_inertia_overlay_to_model(
    mj_model: Any, overlay: Dict[str, Any] | None
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """Patch ``mj_model`` body dynamics arrays from ``overlay``'s body_overrides.

    Parameters
    ----------
    mj_model:
        A live ``mujoco.MjModel`` (mutated in place).
    overlay:
        The overlay dict from ``read_mjcf_inertia_overlay`` (carries a
        ``body_overrides`` block keyed by MJCF body name with per-field
        ``{from, to}`` pairs and a ``fields_applied`` token list), or ``None``.

    Returns
    -------
    (applied, missing):
        ``applied`` is ``[(body_name, [change_str, ...]), ...]`` for every body
        actually patched; ``missing`` is the list of body names named in the
        overlay but absent from the model (the caller may WARN). Both empty when
        the overlay is ``None`` / carries no ``body_overrides``.

    The ``"inertia"`` token writes ``body_inertia`` **and** ``body_iquat``
    together — they jointly define the tensor in the body frame and must never
    be split.
    """
    if not overlay:
        return [], []
    body_overrides = overlay.get("body_overrides") or {}
    if not body_overrides:
        return [], []

    import mujoco
    import numpy as np

    applied: List[Tuple[str, List[str]]] = []
    missing: List[str] = []
    for mjcf_body, ov in body_overrides.items():
        if not isinstance(ov, dict):
            continue
        bid = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, str(mjcf_body))
        )
        if bid < 0:
            missing.append(str(mjcf_body))
            continue
        fields = list(ov.get("fields_applied") or [])
        changes: List[str] = []

        if "mass" in fields and isinstance(ov.get("mass"), dict):
            to = ov["mass"].get("to")
            if to is not None:
                old = float(mj_model.body_mass[bid])
                mj_model.body_mass[bid] = float(to)
                changes.append(f"mass {old:.4f}→{float(to):.4f}")

        if "inertia" in fields:
            diag_to = (ov.get("diaginertia") or {}).get("to")
            iquat_to = (ov.get("iquat") or {}).get("to")
            if diag_to is not None:
                old = [float(v) for v in mj_model.body_inertia[bid]]
                mj_model.body_inertia[bid] = np.asarray(diag_to, dtype=np.float64)
                changes.append(
                    f"diaginertia {[round(v, 4) for v in old]}→"
                    f"{[round(float(v), 4) for v in diag_to]}"
                )
            if iquat_to is not None:
                q = np.asarray(iquat_to, dtype=np.float64)
                n = float(np.linalg.norm(q))
                if n > 1e-9:
                    q = q / n
                mj_model.body_iquat[bid] = q
                changes.append("iquat updated")

        if "com" in fields and isinstance(ov.get("com"), dict):
            to = ov["com"].get("to")
            if to is not None:
                mj_model.body_ipos[bid] = np.asarray(to, dtype=np.float64)
                changes.append("com updated")

        if changes:
            applied.append((str(mjcf_body), changes))

    return applied, missing
