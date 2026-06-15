# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Migrate legacy ``new_item_*`` training-motion item keys to registered ids.

Historically the Training Motion Editor minted opaque keys (``new_item_1``)
and stored the user-typed name in a dead ``title`` field that the compiler
never read. The canvas therefore carried a command id that
``registers.commands`` did not know, and
``env_cfg_compiler._build_training_item_dict`` failed loud at compile time:

    training_item 'new_item_1' not found in registers.commands.

The contract now is "the item id IS its name". This fold:

* renames any *unregistered* training-item key to its intended name — the
  ``title`` override when present, else the key itself;
* registers that name as a user command via
  ``registers.commands.register_named_command`` (inheriting the velocity
  template of a matching builtin, e.g. "Turn" → ``turn``), so the id is valid
  for BOTH backends;
* drops the dead ``title`` field;
* rewrites any ``reward_in__<old>`` edge port on the same node to
  ``reward_in__<new>`` so per-item reward wiring survives the rename.

Shared by the one-shot bootstrap migrator
(``bootstrap/migrate_canvas_training_item_ids.py``) and the canvas load-time
hook (``CanvasPage.from_workflow_dict``). Idempotent via
``metadata.training_item_ids_v1``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

#: ``metadata`` flag that marks a canvas as already folded. Idempotency guard.
TRAINING_ITEM_IDS_FLAG = "training_item_ids_v1"

_REWARD_PORT_PREFIX = "reward_in__"
#: Edge keys that may carry the ``reward_in__<id>`` target port (RELEASE +
#: DEMO spellings — same tolerance as ``obs_group_merge``).
_TARGET_PORT_KEYS = ("target_port", "dst_port", "target_handle")
_TARGET_NODE_KEYS = ("target_node", "dst_id", "target")


def _parse_items(spec: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return ``(items_dict, was_json_string)`` from a ``training_items`` param
    spec. ``items_dict`` is ``None`` when the param is absent / unparseable."""
    value = spec.get("value") if isinstance(spec, dict) else spec
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None, True
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return None, True
        return (parsed if isinstance(parsed, dict) else None), True
    return (dict(value) if isinstance(value, dict) else None), False


def _store_items(node: Dict[str, Any], items: Dict[str, Any], was_str: bool) -> None:
    """Write ``items`` back into the node's ``training_items`` param, preserving
    the original serialization (raw dict vs JSON string)."""
    params = node.setdefault("params", {})
    spec = params.get("training_items")
    payload: Any = json.dumps(items) if was_str else items
    if isinstance(spec, dict):
        spec["value"] = payload
    else:
        params["training_items"] = {
            "name": "training_items", "value": payload, "param_type": "json",
        }


def _node_id(node: Dict[str, Any]) -> Any:
    return node.get("id")


def _edge_targets_node(edge: Dict[str, Any], nid: Any) -> bool:
    return any(edge.get(k) == nid for k in _TARGET_NODE_KEYS)


def migrate_training_item_ids(data: Dict[str, Any]) -> bool:
    """Fold legacy ``new_item_*`` / titled training-item keys in ``data``
    (mutated in place). Returns ``True`` iff anything changed.

    Idempotent: a second call (or a canvas already on registered ids with no
    ``title`` fields) is a no-op once ``metadata.training_item_ids_v1`` is set.
    """
    if not isinstance(data, dict):
        return False
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        data["metadata"] = meta
    if meta.get(TRAINING_ITEM_IDS_FLAG):
        return False

    # Imported lazily so this module stays import-safe in worker venvs that
    # lack the project sys.path (it is only ever *called* from the app process).
    from registers import commands as _commands_reg

    nodes: List[Any] = data.get("nodes") or []
    edges: List[Any] = data.get("edges") or []
    changed = False

    for node in nodes:
        if not isinstance(node, dict):
            continue
        params = node.get("params") or {}
        if "training_items" not in params:
            continue
        items, was_str = _parse_items(params.get("training_items"))
        if items is None:
            continue

        nid = _node_id(node)
        rename: Dict[str, str] = {}
        new_items: Dict[str, Any] = {}
        node_changed = False

        for key, entry in items.items():
            if not isinstance(entry, dict):
                new_items[key] = entry
                continue
            title = str(entry.get("title") or "").strip()
            if "title" in entry:
                entry = {k: v for k, v in entry.items() if k != "title"}
                node_changed = True

            if _commands_reg.get_item(key) is not None:
                # Already a registered id — keep it (title, if any, dropped).
                new_items[key] = entry
                continue

            # Unregistered key (a legacy draft / renamed-but-unregistered item):
            # derive the intended name and register it so the id becomes valid.
            name = title or str(key)
            new_id = _commands_reg.register_named_command(name)
            if new_id != key:
                rename[key] = new_id
                node_changed = True
            new_items[new_id] = entry

        if rename:
            for edge in edges:
                if not isinstance(edge, dict) or not _edge_targets_node(edge, nid):
                    continue
                for pk in _TARGET_PORT_KEYS:
                    pv = edge.get(pk)
                    if not isinstance(pv, str) or not pv.startswith(_REWARD_PORT_PREFIX):
                        continue
                    old = pv[len(_REWARD_PORT_PREFIX):]
                    if old in rename:
                        edge[pk] = _REWARD_PORT_PREFIX + rename[old]

        if node_changed:
            _store_items(node, new_items, was_str)
            changed = True

    meta[TRAINING_ITEM_IDS_FLAG] = True
    return changed


__all__ = ["migrate_training_item_ids", "TRAINING_ITEM_IDS_FLAG"]
