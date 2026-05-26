# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: lower reward fan-out to per-motion-item (v2).

Background — v2 reward fan-out (2026-05):
    * Rewards node loses its IL inputs (actor_pipe / command_pipe).
    * Rewards output ``total_reward`` is renamed ``reward_pipe`` (multi).
    * Training Motion gains ``actor_pipe`` input + dynamic per-item
      ``reward_in__<item_id>`` ports.
    * Trainer (il_ppo_trainer / amp_trainer) renames input ``total_reward``
      → ``command_pipe``.
    * Each reward_term dict drops the obsolete ``applies_to`` field —
      phase fan-out is now expressed by which training item a Rewards
      node is connected to.

This script upgrades a saved canvas.json so it survives the new schema:

1. For every ``rewards`` node:
     - delete every edge with ``target_node == <rewards.id>`` and
       ``target_port in ("actor_pipe", "command_pipe")`` (the dropped
       inputs).
     - in ``params.reward_terms.value`` (dict of term_id → {weight,
       applies_to, ...}) pop the ``applies_to`` key from each entry.
     - remove the now-unused ``phase_layout`` param if present.
2. For each ``rewards → trainer (total_reward)`` edge:
     - drop the edge.
3. For every trainer (``il_ppo_trainer`` / ``amp_trainer``):
     - add an edge ``training_motion → trainer (command_pipe)`` if a
       training_motion node exists (one per trainer, idempotent).
4. For every ``actor_setting → rewards (actor_pipe)`` edge:
     - redirect target to ``training_motion.actor_pipe`` (single
       training_motion expected; idempotent: dedupe).
5. For every rewards node, add one fallback edge
   ``rewards.reward_pipe → training_motion.reward_in__<first_enabled_item>``
   so the rewards data still feeds training. The user is expected to
   re-wire fan-out manually after migration; a log line surfaces this.
6. Stamps ``metadata.canvas_rewards_v2: true`` for idempotency.

Per-file ``.bak_v1`` backups are written alongside the original.

Usage::

    cd D:/Unitport/EXE/RELEASE
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_rewards_v2.py `
        custom_mods/canvas/IsaacLab_Go2_PPO.canvas.json
    # Recursive sweep:
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_rewards_v2.py `
        custom_mods/canvas
    # Dry run — report intended changes without writing:
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_rewards_v2.py `
        <path> --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_FLAG = "canvas_rewards_v2"


# ---------------------------------------------------------------------------
# Edge / node helpers
# ---------------------------------------------------------------------------


def _new_edge_id() -> str:
    return f"e_{uuid.uuid4().hex[:12]}"


def _is_already_migrated(canvas: Dict[str, Any]) -> bool:
    md = canvas.get("metadata") if isinstance(canvas.get("metadata"), dict) else {}
    return bool(md.get(_FLAG))


def _node_ids_by_schema(canvas: Dict[str, Any], schema_id: str) -> List[str]:
    return [
        str(n.get("id"))
        for n in canvas.get("nodes", [])
        if isinstance(n, dict) and n.get("schema_id") == schema_id
    ]


def _get_node(canvas: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    for n in canvas.get("nodes", []):
        if isinstance(n, dict) and str(n.get("id")) == str(node_id):
            return n
    return None


def _first_enabled_item_id(tm_node: Dict[str, Any]) -> Optional[str]:
    params = tm_node.get("params") or {}
    ti = params.get("training_items") if isinstance(params, dict) else None
    if not isinstance(ti, dict):
        return None
    raw = ti.get("value")
    if isinstance(raw, str):
        try:
            d = json.loads(raw) if raw.strip() else {}
        except Exception:
            return None
    elif isinstance(raw, dict):
        d = raw
    else:
        return None
    for item_id, entry in d.items():
        if isinstance(entry, dict) and bool(entry.get("enabled", True)):
            return str(item_id)
    # No enabled item — pick the first key (better than nothing for migration).
    keys = list(d.keys())
    return str(keys[0]) if keys else None


def _strip_applies_to_in_terms(reward_terms_param: Dict[str, Any]) -> int:
    """Mutate ``reward_terms`` param value in place; return count stripped."""
    value = reward_terms_param.get("value")
    if isinstance(value, str):
        try:
            d = json.loads(value) if value.strip() else {}
        except Exception:
            return 0
        was_str = True
    elif isinstance(value, dict):
        d = value
        was_str = False
    else:
        return 0
    stripped = 0
    for term_id, term in list(d.items()):
        if isinstance(term, dict) and "applies_to" in term:
            term.pop("applies_to", None)
            stripped += 1
    if was_str:
        reward_terms_param["value"] = json.dumps(d, ensure_ascii=False)
    else:
        reward_terms_param["value"] = d
    return stripped


# ---------------------------------------------------------------------------
# Main migrate
# ---------------------------------------------------------------------------


def migrate_canvas(canvas: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (mutated_canvas, log_lines). Idempotent."""
    log: List[str] = []
    if _is_already_migrated(canvas):
        log.append("already migrated (metadata.canvas_rewards_v2 = true) — skipped")
        return canvas, log

    rewards_ids = set(_node_ids_by_schema(canvas, "rewards"))
    tm_ids = _node_ids_by_schema(canvas, "training_motion")
    trainer_ids = set(
        _node_ids_by_schema(canvas, "il_ppo_trainer")
        + _node_ids_by_schema(canvas, "amp_trainer")
    )
    actor_setting_ids = set(_node_ids_by_schema(canvas, "actor_setting"))

    # ---- 1. Strip rewards-node IL inputs + applies_to + phase_layout ----
    for rid in rewards_ids:
        rn = _get_node(canvas, rid)
        if rn is None:
            continue
        params = rn.get("params") if isinstance(rn.get("params"), dict) else None
        if isinstance(params, dict):
            rt = params.get("reward_terms")
            if isinstance(rt, dict):
                n = _strip_applies_to_in_terms(rt)
                if n:
                    log.append(
                        f"rewards/{rid}: stripped applies_to from {n} term(s)"
                    )
            if "phase_layout" in params:
                params.pop("phase_layout", None)
                log.append(f"rewards/{rid}: dropped obsolete param 'phase_layout'")

    edges: List[Dict[str, Any]] = list(canvas.get("edges", []))
    new_edges: List[Dict[str, Any]] = []
    # Track redirected actor_pipe sources so we can dedupe new actor edges.
    actor_redirect_sources: Set[str] = set()
    # Track rewards → trainer edges for logging.
    dropped_total_reward = 0
    dropped_rewards_inputs = 0
    redirected_actor = 0

    primary_tm = tm_ids[0] if tm_ids else None

    for e in edges:
        sn = str(e.get("source_node"))
        sp = str(e.get("source_port"))
        tn = str(e.get("target_node"))
        tp = str(e.get("target_port"))

        # 1a. Drop edges into deleted rewards inputs.
        if tn in rewards_ids and tp in ("actor_pipe", "command_pipe"):
            dropped_rewards_inputs += 1
            # If it was actor_pipe and source was actor_setting + we have a
            # training_motion to redirect into, queue the redirect.
            if (
                tp == "actor_pipe"
                and sn in actor_setting_ids
                and primary_tm is not None
            ):
                actor_redirect_sources.add(sn)
            continue

        # 1b. Drop rewards → trainer (total_reward) edges; the migration
        # synthesises new training_motion → trainer (command_pipe) edges below.
        if (
            sn in rewards_ids and sp == "total_reward"
            and tn in trainer_ids and tp == "total_reward"
        ):
            dropped_total_reward += 1
            continue

        # 1c. Drop legacy training_motion → rewards (command_pipe) — already
        # caught above by 1a since tp == "command_pipe" matched.

        # 1d. Drop any remaining edge into a trainer.total_reward port (e.g.
        # if a non-rewards node was somehow feeding it — defensive).
        if tn in trainer_ids and tp == "total_reward":
            dropped_total_reward += 1
            continue

        new_edges.append(e)

    if dropped_rewards_inputs:
        log.append(
            f"edges: dropped {dropped_rewards_inputs} input-edge(s) into "
            f"deleted rewards inputs"
        )
    if dropped_total_reward:
        log.append(
            f"edges: dropped {dropped_total_reward} 'total_reward' edge(s)"
        )

    # ---- 2. Add training_motion → trainer (command_pipe) per trainer ----
    if primary_tm is not None:
        existing_tm_trainer: Set[Tuple[str, str]] = {
            (str(e.get("source_node")), str(e.get("target_node")))
            for e in new_edges
            if str(e.get("source_port")) == "command_pipe"
            and str(e.get("target_port")) == "command_pipe"
            and str(e.get("target_node")) in trainer_ids
        }
        for trid in sorted(trainer_ids):
            if (primary_tm, trid) in existing_tm_trainer:
                continue
            new_edges.append({
                "id": _new_edge_id(),
                "source_node": primary_tm,
                "source_port": "command_pipe",
                "target_node": trid,
                "target_port": "command_pipe",
                "edge_type": "flow",
            })
            log.append(
                f"edges: + training_motion/{primary_tm} → trainer/{trid} "
                f"(command_pipe)"
            )
    elif trainer_ids:
        log.append(
            f"WARN: trainer(s) {sorted(trainer_ids)!r} exist but no "
            f"training_motion node found — cannot wire command_pipe input"
        )

    # ---- 3. Add actor_setting → training_motion (actor_pipe) ----
    if primary_tm is not None and actor_redirect_sources:
        existing_actor_to_tm: Set[Tuple[str, str]] = {
            (str(e.get("source_node")), str(e.get("target_node")))
            for e in new_edges
            if str(e.get("source_port")) == "actor_pipe"
            and str(e.get("target_port")) == "actor_pipe"
            and str(e.get("target_node")) == primary_tm
        }
        for src in sorted(actor_redirect_sources):
            if (src, primary_tm) in existing_actor_to_tm:
                continue
            new_edges.append({
                "id": _new_edge_id(),
                "source_node": src,
                "source_port": "actor_pipe",
                "target_node": primary_tm,
                "target_port": "actor_pipe",
                "edge_type": "flow",
            })
            redirected_actor += 1
        if redirected_actor:
            log.append(
                f"edges: + {redirected_actor} actor_setting → "
                f"training_motion/{primary_tm} (actor_pipe)"
            )

    # ---- 4. Add rewards.reward_pipe → training_motion.reward_in__<first> ----
    if primary_tm is not None and rewards_ids:
        tm_node = _get_node(canvas, primary_tm)
        first_item = _first_enabled_item_id(tm_node) if tm_node is not None else None
        if not first_item:
            log.append(
                f"WARN: training_motion/{primary_tm} has no enabled items — "
                f"rewards fan-out fallback skipped; reconnect by hand."
            )
        else:
            target_port = f"reward_in__{first_item}"
            existing_reward_targets: Set[Tuple[str, str]] = {
                (str(e.get("source_node")), str(e.get("target_node")))
                for e in new_edges
                if str(e.get("source_port")) == "reward_pipe"
                and str(e.get("target_node")) == primary_tm
            }
            for rid in sorted(rewards_ids):
                if (rid, primary_tm) in existing_reward_targets:
                    continue
                new_edges.append({
                    "id": _new_edge_id(),
                    "source_node": rid,
                    "source_port": "reward_pipe",
                    "target_node": primary_tm,
                    "target_port": target_port,
                    "edge_type": "flow",
                })
                log.append(
                    f"edges: + rewards/{rid} → training_motion/{primary_tm}."
                    f"{target_port}  (FALLBACK — confirm per-item routing)"
                )

    canvas["edges"] = new_edges

    # ---- 5. Idempotency stamp ----
    md = canvas.get("metadata")
    if not isinstance(md, dict):
        md = {}
        canvas["metadata"] = md
    md[_FLAG] = True

    return canvas, log


# ---------------------------------------------------------------------------
# File I/O — match migrate_canvas_strict_v1's atomic pattern
# ---------------------------------------------------------------------------


def _migrate_file(path: Path, *, dry_run: bool) -> List[str]:
    log: List[str] = [f"== {path}"]
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        log.append(f"  read failed: {exc}")
        return log
    try:
        canvas = json.loads(raw)
    except Exception as exc:
        log.append(f"  json parse failed: {exc}")
        return log
    if not isinstance(canvas, dict):
        log.append("  not an object — skipped")
        return log

    new_canvas, changes = migrate_canvas(canvas)
    for line in changes:
        log.append(f"  {line}")

    if dry_run:
        log.append("  (dry-run) no write")
        return log
    if not changes:
        return log

    backup = path.with_suffix(path.suffix + ".bak_v1")
    try:
        if not backup.exists():
            backup.write_text(raw, encoding="utf-8")
            log.append(f"  backup → {backup.name}")
    except Exception as exc:
        log.append(f"  backup failed: {exc}; aborting write")
        return log
    try:
        path.write_text(
            json.dumps(new_canvas, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.append("  written")
    except Exception as exc:
        log.append(f"  write failed: {exc}")
    return log


def _iter_canvas_paths(root: Path):
    if root.is_file():
        if root.suffix == ".json" and ".canvas." in root.name:
            yield root
        elif root.suffix == ".json":
            yield root
        return
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*.canvas.json")):
        if p.is_file():
            yield p


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="canvas.json file or directory to walk")
    p.add_argument("--dry-run", action="store_true",
                   help="report intended changes without writing")
    args = p.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"path not found: {root}", file=sys.stderr)
        return 2

    paths = list(_iter_canvas_paths(root))
    if not paths:
        print(f"no canvas.json files under {root}", file=sys.stderr)
        return 1

    for path in paths:
        for line in _migrate_file(path, dry_run=args.dry_run):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
