# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Capture the live IsaacLab articulation joint order at training time.

WHY THIS EXISTS
---------------
The exported ONNX policy emits its action vector in IsaacLab's **live PhysX
articulation DOF order**: ``JointPositionActionCfg(joint_names=[".*"])`` resolves
``.*`` against the running articulation and preserves that order, so action slot
``i`` == articulation DOF ``i``. The deploy bundle's per-joint arrays
(``joint_sdk_names`` / ``default_joint_pos`` / PD gains / caps) MUST be ordered by
THIS same order — otherwise every joint-keyed obs/action slot lands on the wrong
joint and the deploy run convulses.

Historically the bundle was ordered by the **Dump-USD prim-traverse order**
(``stage.Traverse()`` authoring order, persisted as
``joints_per_format["USD"]``). For some robots that order differs from the live
PhysX articulation order — e.g. Go2: prim authoring ``[hip×4, (thigh,calf)/leg]``
vs PhysX breadth-first ``[hip×4, thigh×4, calf×4]`` — which silently scrambled
every thigh/calf action slot (the "sim2sim twitch" bug).

This module is the single seam that records the ground truth. It is deliberately
**import-safe in the main venv** (stdlib only, no IsaacLab imports):

  * the launcher (Kit venv) calls :func:`resolve_articulation_joint_order` +
    :func:`write_joint_order_sidecar` right after ``runner.learn()``;
  * ``bundle_finalizer`` (main venv) calls :func:`load_joint_order_sidecar` and
    orders the bundle by the captured order;
  * unit tests feed a duck-typed ``env`` — no real Kit required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SIDECAR_NAME = "articulation_joint_order.json"
SIDECAR_SCHEMA = 1


# ---------------------------------------------------------------------------
# Live-env resolution (duck-typed; no IsaacLab import)
# ---------------------------------------------------------------------------
def _unwrap_managed_env(env: Any) -> Any:
    """Follow ``.unwrapped`` / ``.env`` until we reach the ManagerBasedRLEnv.

    The launcher passes the RSL-RL / AMP vec-env wrapper; the probe passes the
    raw gym env. Both ultimately wrap the IsaacLab ``ManagerBasedRLEnv`` that
    owns ``.scene`` and ``.action_manager``.
    """
    seen = set()
    base = env
    while base is not None and id(base) not in seen:
        seen.add(id(base))
        if hasattr(base, "scene") and hasattr(base, "action_manager"):
            return base
        nxt = getattr(base, "unwrapped", None)
        if nxt is not None and nxt is not base:
            base = nxt
            continue
        nxt = getattr(base, "env", None)
        if nxt is not None and nxt is not base:
            base = nxt
            continue
        break
    return base


def _articulation_joint_names(base: Any) -> Optional[List[str]]:
    try:
        scene = getattr(base, "scene", None)
        if scene is None:
            return None
        robot = scene["robot"]
    except Exception:
        return None
    names = getattr(robot, "joint_names", None)
    if names is None:
        data = getattr(robot, "data", None)
        names = getattr(data, "joint_names", None)
    if not names:
        return None
    try:
        return [str(n) for n in names]
    except Exception:
        return None


def _ids_to_list(joint_ids: Any, n: int) -> List[int]:
    if joint_ids is None:
        return list(range(n))
    if isinstance(joint_ids, slice):
        return list(range(*joint_ids.indices(n)))
    try:
        if hasattr(joint_ids, "tolist"):
            return [int(x) for x in joint_ids.tolist()]
        return [int(x) for x in joint_ids]
    except Exception:
        return list(range(n))


def _get_action_term(action_manager: Any, term_name: str) -> Any:
    fn = getattr(action_manager, "get_term", None)
    if callable(fn):
        try:
            return fn(term_name)
        except Exception:
            pass
    terms = getattr(action_manager, "_terms", None)
    if isinstance(terms, dict):
        return terms.get(term_name)
    return None


def _action_term_joint_order(base: Any) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Resolved joint order of the single JointPosition action term.

    This is the *authoritative* policy action order. For ``joint_names=[".*"]``
    it equals the articulation order; for a partial regex it is the order the
    policy actually emits. Returns ``(None, None)`` when there is not exactly one
    joint-position action term (ambiguous → caller falls back to articulation).
    """
    try:
        am = getattr(base, "action_manager", None)
        if am is None:
            return None, None
        active = list(getattr(am, "active_terms", None) or [])
        joint_terms: List[Tuple[List[str], Any]] = []
        for tname in active:
            term = _get_action_term(am, tname)
            if term is None:
                continue
            jn = getattr(term, "_joint_names", None)
            if jn:
                joint_terms.append(([str(x) for x in jn], getattr(term, "_joint_ids", None)))
        if len(joint_terms) == 1:
            names, ids = joint_terms[0]
            return names, _ids_to_list(ids, len(names))
        return None, None
    except Exception:
        return None, None


def resolve_articulation_joint_order(env: Any) -> Dict[str, Any]:
    """Resolve the policy's true joint order from a live IsaacLab env.

    Returns a JSON-serializable payload::

        {"schema": 1, "source": "action_term" | "articulation",
         "joint_names": [<physical names in policy order>],
         "joint_ids": [...], "articulation_joint_names": [...] | None}

    Raises ``RuntimeError`` when neither the action manager nor the articulation
    can be read — fail loud (§8) rather than ship an unverifiable bundle.
    """
    base = _unwrap_managed_env(env)
    art_names = _articulation_joint_names(base)
    act_names, act_ids = _action_term_joint_order(base)

    if act_names:
        return {
            "schema": SIDECAR_SCHEMA,
            "source": "action_term",
            "joint_names": list(act_names),
            "joint_ids": list(act_ids) if act_ids is not None else list(range(len(act_names))),
            "articulation_joint_names": list(art_names) if art_names else None,
        }
    if art_names:
        return {
            "schema": SIDECAR_SCHEMA,
            "source": "articulation",
            "joint_names": list(art_names),
            "joint_ids": list(range(len(art_names))),
            "articulation_joint_names": list(art_names),
        }
    raise RuntimeError(
        "resolve_articulation_joint_order: could not read joint order from the "
        "live env — neither action_manager nor scene['robot'].joint_names was "
        "available. The bundle's joint order would be unverifiable (§8)."
    )


# ---------------------------------------------------------------------------
# Sidecar I/O (stdlib only — runs in Kit launcher AND main-venv finalizer)
# ---------------------------------------------------------------------------
def write_joint_order_sidecar(log_dir: Any, payload: Dict[str, Any]) -> str:
    """Write the captured order to ``<log_dir>/articulation_joint_order.json``."""
    path = os.path.join(str(log_dir), SIDECAR_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def locate_joint_order_sidecar(exported_dir: Any) -> Optional[Path]:
    """Find the sidecar next to the run output (mirrors deploy_meta lookup)."""
    d = Path(exported_dir)
    for cand in (d / SIDECAR_NAME, d / "params" / SIDECAR_NAME, d.parent / SIDECAR_NAME):
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def load_joint_order_sidecar(exported_dir: Any) -> Optional[Dict[str, Any]]:
    """Load + validate the sidecar. Returns ``None`` when absent/malformed."""
    p = locate_joint_order_sidecar(exported_dir)
    if p is None:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    names = data.get("joint_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        return None
    return data


def capture_and_write(env: Any, log_dir: Any) -> Optional[Dict[str, Any]]:
    """Resolve + persist in one call; never raises (training already finished).

    Returns the payload on success, ``None`` on any failure (logged by caller).
    """
    payload = resolve_articulation_joint_order(env)
    write_joint_order_sidecar(log_dir, payload)
    return payload


# ---------------------------------------------------------------------------
# Bundle-order reconciliation (pure; used by bundle_finalizer + tests)
# ---------------------------------------------------------------------------
def resolve_bundle_joint_permutation(
    canonical_ir_roles: List[str],
    captured_names: List[str],
    name_to_role: Dict[str, str],
) -> Tuple[List[str], List[int]]:
    """Compute the permutation reordering ``canonical_ir_roles`` onto the
    captured LIVE articulation order.

    ``captured_names`` are physical joint names in the policy's true action
    order; ``name_to_role`` maps each physical name → IR role. Returns
    ``(captured_ir_order, permutation)`` where
    ``[canonical_ir_roles[i] for i in permutation] == captured_ir_order`` — so
    applying ``permutation`` to every per-joint array (which ship in
    ``canonical_ir_roles`` order) reorders them onto the live order.

    Identity ``permutation`` (``[0, 1, .., n-1]``) iff the live order already
    equals the canonical order — i.e. a byte-identical, no-regression bundle.

    Raises ``ValueError`` (with an actionable message) when a captured joint has
    no IR role, a captured role isn't in the canonical set, or the counts differ
    — never silently drops/reorders a partial set (§8).
    """
    missing_names = [n for n in captured_names if n not in name_to_role]
    if missing_names:
        raise ValueError(
            f"captured live articulation joints {missing_names} have no IR role "
            f"in the registry joint name->role table (table is stale/incomplete)"
        )
    captured_ir_order = [name_to_role[n] for n in captured_names]
    idx = {r: i for i, r in enumerate(canonical_ir_roles)}
    missing_roles = [r for r in captured_ir_order if r not in idx]
    if missing_roles:
        raise ValueError(
            f"live articulation IR roles {missing_roles} are not in the "
            f"canonical joint_ir_roles set {list(canonical_ir_roles)}"
        )
    if len(captured_ir_order) != len(canonical_ir_roles):
        raise ValueError(
            f"live articulation joint count {len(captured_ir_order)} != "
            f"canonical joint_ir_roles count {len(canonical_ir_roles)}"
        )
    permutation = [idx[r] for r in captured_ir_order]
    return captured_ir_order, permutation
