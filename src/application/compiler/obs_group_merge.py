# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""缺口② — fold the legacy two-node il_observation canvas into one node.

Before: asymmetric actor-critic was modelled as TWO ``il_observation`` nodes —
``group_name="policy"`` (the deployed obs) + a standalone ``group_name="critic"``
node carrying the privileged superset, whose ``obs_vector`` output dangled
(consumed by ``_find_by_type``, not by edge — a core-contract violation).

After: ONE node. ``obs_terms`` = policy; ``critic_privileged_terms`` = the extra
terms the critic sees (= critic.obs_terms − policy.obs_terms, scales preserved).
The compiler consumes it BY EDGE to il_policy_network.

This fold is shared by the one-shot bootstrap migrator
(``bootstrap/migrate_canvas_merge_obs_groups.py``) and the canvas load-time hook
(``CanvasPage.from_workflow_dict``). Idempotent via ``metadata.obs_merge_v1``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

OBS_MERGE_FLAG = "obs_merge_v1"


def _param_value(node: Dict[str, Any], key: str) -> Any:
    spec = (node.get("params") or {}).get(key)
    return spec.get("value") if isinstance(spec, dict) else spec


def _set_json_param(node: Dict[str, Any], key: str, value: Any) -> None:
    params = node.setdefault("params", {})
    params[key] = {"name": key, "value": value, "param_type": "json"}


def _del_param(node: Dict[str, Any], key: str) -> None:
    (node.get("params") or {}).pop(key, None)


def _as_obs_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _edge_touches(edge: Dict[str, Any], node_id: Any) -> bool:
    # Tolerate both the RELEASE (source_node/target_node) and DEMO
    # (src_id/dst_id, source/target) edge key spellings.
    for k in ("source_node", "target_node", "src_id", "dst_id", "source", "target"):
        if edge.get(k) == node_id:
            return True
    return False


def merge_obs_groups(canvas: Dict[str, Any]) -> bool:
    """In-place fold of a canvas dict. Returns True iff it changed.

    * 2 obs nodes (policy + critic) → fold privileged = critic − policy onto the
      policy node's ``critic_privileged_terms``; delete the critic node + edges.
    * Any remaining obs node still carrying ``group_name`` (legacy single-policy)
      → strip ``group_name`` and, if it has no ``critic_privileged_terms``, pin
      it to ``{}`` (symmetric) so it does NOT inherit the new manifest default
      ``{base_lin_vel: 2.0}`` and silently gain an asymmetric critic.
    * Fresh post-merge nodes (no ``group_name``) are left untouched.
    Idempotent: a stamped ``metadata.obs_merge_v1`` short-circuits.
    """
    if not isinstance(canvas, dict):
        return False
    meta = canvas.setdefault("metadata", {})
    if isinstance(meta, dict) and meta.get(OBS_MERGE_FLAG) is True:
        return False

    nodes = canvas.get("nodes") or []
    obs_nodes = [n for n in nodes if n.get("schema_id") == "il_observation"]
    changed = False

    policy: Optional[Dict[str, Any]] = next(
        (n for n in obs_nodes if _param_value(n, "group_name") == "policy"), None)
    critic: Optional[Dict[str, Any]] = next(
        (n for n in obs_nodes if _param_value(n, "group_name") == "critic"), None)
    if policy is not None and critic is not None:
        policy_terms = _as_obs_dict(_param_value(policy, "obs_terms"))
        critic_terms = _as_obs_dict(_param_value(critic, "obs_terms"))
        privileged = {k: v for k, v in critic_terms.items() if k not in policy_terms}
        _set_json_param(policy, "critic_privileged_terms", privileged)
        _del_param(policy, "group_name")
        cid = critic.get("id")
        canvas["nodes"] = [n for n in nodes if n.get("id") != cid]
        canvas["edges"] = [
            e for e in (canvas.get("edges") or []) if not _edge_touches(e, cid)
        ]
        changed = True

    for n in (canvas.get("nodes") or []):
        if n.get("schema_id") != "il_observation":
            continue
        params = n.get("params") or {}
        if "group_name" in params:
            _del_param(n, "group_name")
            if "critic_privileged_terms" not in params:
                _set_json_param(n, "critic_privileged_terms", {})
            changed = True

    if isinstance(meta, dict) and meta.get(OBS_MERGE_FLAG) is not True:
        meta[OBS_MERGE_FLAG] = True
        changed = True
    return changed
