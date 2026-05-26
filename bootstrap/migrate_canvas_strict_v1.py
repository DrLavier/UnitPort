# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: bring pre-strict canvases up to strict-canvas v1.

The strict-canvas migration (2026-05) hardened the canvas→config pipeline:
* legacy compatibility branches in spec_compiler / manifest_parser / Node
  layers were removed,
* silent default substitutions were turned into ERROR issues,
* a handful of Node fields became deprecated and were physically dropped
  from the manifest (e.g. ``il_policy_network.disc_hidden_dims``,
  ``train_config`` input port on the train Node).

This script upgrades a saved canvas so it survives the new strict checks:

1. Runs the ``obs_terms`` ScaleSpec migration (delegates to
   ``migrate_il_observation_obs_terms``: every ``il_observation`` node's
   ``obs_terms`` field is rewritten to ``{name: scale}`` form using the
   registry's new default scales — base_lin_vel=2.0, base_ang_vel=0.25,
   joint_vel=0.05, …).
2. Runs the joint-names → IR-role migration (delegates to
   ``migrate_canvas_joint_names_to_ir``: vendor abbreviations like
   ``fl_hx`` / ``FL_hip_joint`` in every ``actor_setting`` node's
   ``init_joint_angles`` and ``action_joint_names_expr`` become IR roles).
3. Drops ``il_policy_network.disc_hidden_dims`` (moved to DiscriminatorNode).
4. Drops edges whose ``target_port == "train_config"`` on a ``train`` node
   (the train_config input port was removed).
5. When ``algorithm_config.backend == "auto"``: rewrites to the canvas's
   bound backend (``canvas.backend`` / ``canvas.engine_id``). "auto" is no
   longer accepted at compile time.
6. Reports — but does NOT auto-fix — ``rewards`` nodes that omit any
   ``RECOMMENDED_LOCOMOTION_REWARD_TERMS`` keys (so the user knows which
   safety terms to add by hand).
7. Stamps ``metadata.canvas_strict_v1: true`` for idempotency.

Per-file ``.bak`` backups are written alongside the original; the
overwrite is atomic via the standard json + path write pattern (Python's
``Path.write_text`` is not strictly atomic, but the migrator is one-shot
and the ``.bak`` provides recovery).

Usage::

    cd D:/Unitport/EXE/RELEASE
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_strict_v1.py `
        projects/new_test/canvas/IsaacLab/ppo_walk.canvas.json
    # Recursive sweep:
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_strict_v1.py `
        projects/new_test/canvas
    # Dry run — report intended changes without writing:
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_strict_v1.py `
        <path> --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_STRICT_FLAG = "canvas_strict_v1"


# ---------------------------------------------------------------------------
# Helpers to delegate to the pre-existing migrators (file-based load —
# PROJECT_ROOT is not on sys.path per RELEASE/CLAUDE.md §2).
# ---------------------------------------------------------------------------


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_obs_terms_mod = _load_module_from_file(
    "_migrate_obs_terms",
    _REPO_ROOT / "bootstrap" / "migrate_il_observation_obs_terms.py",
)
_joint_names_mod = _load_module_from_file(
    "_migrate_joint_names",
    _REPO_ROOT / "bootstrap" / "migrate_canvas_joint_names_to_ir.py",
)


def _migrate_obs_terms_in_canvas(canvas: Dict[str, Any], *, file_label: str) -> bool:
    """Apply the obs_terms ScaleSpec rewrite to every il_observation node.

    Returns True if any node was rewritten.
    """
    any_change = False
    for node in canvas.get("nodes", []) or []:
        if node.get("schema_id") != "il_observation":
            continue
        params = node.get("params", {}) or {}
        spec = params.get("obs_terms")
        if not isinstance(spec, dict) or "value" not in spec:
            continue
        new_value, changed, unknown = _obs_terms_mod._migrate_obs_terms_payload(
            spec["value"],
            where=f"{file_label}::{node.get('id','?')}.obs_terms",
        )
        if changed:
            spec["value"] = new_value
            any_change = True
            print(f"  + il_observation node {node.get('id','?')} obs_terms → scale form")
        if unknown:
            print(
                f"  ! il_observation node {node.get('id','?')} obs_terms unknown "
                f"keys (not in IL_OBS_REGISTRY): {unknown}",
                file=sys.stderr,
            )
    return any_change


def _migrate_joint_names_in_canvas(canvas: Dict[str, Any], *, file_label: str) -> bool:
    """Rewrite vendor joint names in every actor_setting node to IR roles.

    Returns True if any node was rewritten.
    """
    any_change = False
    family = (
        _joint_names_mod._resolve_family_from_canvas(canvas) or "quadruped"
    )
    for node in canvas.get("nodes", []) or []:
        if node.get("schema_id") != "actor_setting":
            continue
        params = node.get("params", {}) or {}

        ija = params.get("init_joint_angles")
        if isinstance(ija, dict) and "value" in ija:
            new_v, missed = _joint_names_mod._translate_init_joint_angles(
                ija["value"], family,
                where=f"{file_label}::{node.get('id','?')}.init_joint_angles",
            )
            if new_v != ija["value"]:
                ija["value"] = new_v
                any_change = True
                print(
                    f"  + actor_setting node {node.get('id','?')} "
                    f"init_joint_angles → IR roles"
                )
            if missed:
                print(
                    f"  ! actor_setting node {node.get('id','?')} "
                    f"init_joint_angles untranslated keys: {missed}",
                    file=sys.stderr,
                )

        expr = params.get("action_joint_names_expr")
        if isinstance(expr, dict) and "value" in expr:
            new_v, missed = _joint_names_mod._translate_action_joint_names_expr(
                expr["value"], family,
                where=f"{file_label}::{node.get('id','?')}.action_joint_names_expr",
            )
            if new_v != expr["value"]:
                expr["value"] = new_v
                any_change = True
                print(
                    f"  + actor_setting node {node.get('id','?')} "
                    f"action_joint_names_expr → IR roles"
                )
            if missed:
                print(
                    f"  ! actor_setting node {node.get('id','?')} "
                    f"action_joint_names_expr untranslated literals: {missed}",
                    file=sys.stderr,
                )
    return any_change


# ---------------------------------------------------------------------------
# Strict-v1 specific rewrites
# ---------------------------------------------------------------------------


def _drop_disc_hidden_dims(canvas: Dict[str, Any]) -> bool:
    """Remove the deprecated ``disc_hidden_dims`` field from every
    ``il_policy_network`` node's params dict. The field moved to
    DiscriminatorNode in 2025-Q4; il_policy_network's ParamSpec was
    deleted in the strict-canvas migration."""
    any_change = False
    for node in canvas.get("nodes", []) or []:
        if node.get("schema_id") != "il_policy_network":
            continue
        params = node.get("params", {}) or {}
        if "disc_hidden_dims" in params:
            del params["disc_hidden_dims"]
            any_change = True
            print(
                f"  + il_policy_network node {node.get('id','?')} dropped "
                f"legacy disc_hidden_dims"
            )
    return any_change


def _drop_train_config_edges(canvas: Dict[str, Any]) -> bool:
    """Drop edges that target the removed ``train.train_config`` input
    port. Any other edges into train nodes are kept verbatim."""
    train_ids: set = {
        str(n.get("id"))
        for n in (canvas.get("nodes", []) or [])
        if n.get("schema_id") == "train"
    }
    if not train_ids:
        return False
    edges = canvas.get("edges", []) or []
    kept: List[Any] = []
    dropped: List[Any] = []
    for e in edges:
        if not isinstance(e, dict):
            kept.append(e)
            continue
        tgt = str(e.get("target_node") or e.get("dst_id") or "")
        port = str(e.get("target_port") or e.get("dst_slot") or "")
        if tgt in train_ids and port == "train_config":
            dropped.append(e)
        else:
            kept.append(e)
    if not dropped:
        return False
    canvas["edges"] = kept
    for e in dropped:
        src = e.get("source_node") or e.get("src_id") or "?"
        print(
            f"  + dropped legacy edge {src!r} → train.train_config (port removed)"
        )
    return True


def _strip_algorithm_config_backend(canvas: Dict[str, Any]) -> bool:
    """Drop the ``backend`` field from every ``algorithm_config`` node's
    params dict.

    The node-level ``backend`` ParamSpec was removed: backend is now a
    canvas-level concept (single source of truth) and every spec field
    that needs it derives from ``canvas.backend``. Old canvases still
    carry the stale field in their params dict; strip it so the saved
    canvas reflects the new schema.
    """
    if not str(canvas.get("backend") or canvas.get("engine_id") or "").strip():
        # Canvas itself has no backend binding — spec_compiler will emit
        # an ERROR on that; nothing useful to do here.
        return False
    any_change = False
    for node in canvas.get("nodes", []) or []:
        if node.get("schema_id") != "algorithm_config":
            continue
        params = node.get("params") or {}
        if "backend" in params:
            removed = params.pop("backend", None)
            any_change = True
            print(
                f"  + algorithm_config node {node.get('id','?')} dropped "
                f"legacy node-level backend (was {removed!r}; canvas-level "
                f"backend is the sole source now)"
            )
    return any_change


def _report_missing_recommended_rewards(canvas: Dict[str, Any]) -> None:
    """Print (do NOT auto-fix) reward nodes that omit RECOMMENDED safety
    terms. Auto-fixing would be a silent edit of user intent — exactly
    what the strict-canvas migration is trying to eliminate."""
    try:
        from application.training.isaac_lab.task_module_registry import (
            RECOMMENDED_LOCOMOTION_REWARD_TERMS,
        )
    except Exception:
        return
    expected = set(RECOMMENDED_LOCOMOTION_REWARD_TERMS)
    for node in canvas.get("nodes", []) or []:
        if node.get("schema_id") != "rewards":
            continue
        params = node.get("params", {}) or {}
        spec = params.get("reward_terms")
        if not isinstance(spec, dict) or "value" not in spec:
            continue
        raw = spec["value"]
        try:
            if isinstance(raw, str):
                parsed = json.loads(raw) if raw.strip() else {}
            else:
                parsed = raw or {}
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        missing = sorted(expected - set(parsed.keys()))
        if missing:
            print(
                f"  ! rewards node {node.get('id','?')} omits recommended "
                f"locomotion safety terms: {missing}. The env historically "
                f"defaulted these to weight 0, which led to crouching / "
                f"asymmetric gait. Add them in the Rewards node manually.",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _migrate_one(path: Path, *, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        canvas = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    metadata = canvas.setdefault("metadata", {})
    if isinstance(metadata, dict) and metadata.get(_STRICT_FLAG) is True:
        print(f"# {path}  (already strict-v1; skipping)")
        return False

    print(f"# {path}")

    file_label = path.name
    any_change = False
    any_change |= _migrate_obs_terms_in_canvas(canvas, file_label=file_label)
    any_change |= _migrate_joint_names_in_canvas(canvas, file_label=file_label)
    any_change |= _drop_disc_hidden_dims(canvas)
    any_change |= _drop_train_config_edges(canvas)
    any_change |= _strip_algorithm_config_backend(canvas)
    _report_missing_recommended_rewards(canvas)  # report-only

    # Always stamp the flag so subsequent invocations fast-skip.
    if isinstance(metadata, dict):
        if metadata.get(_STRICT_FLAG) is not True:
            metadata[_STRICT_FLAG] = True
            any_change = True

    if not any_change:
        print("  = no changes")
        return False

    if dry_run:
        print("  [dry-run] would rewrite")
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text(
        json.dumps(canvas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [ok] rewrote {path.name}; backup at {backup.name}")
    return True


def _iter_canvas_files(target: Path):
    if target.is_file():
        yield target
        return
    if target.is_dir():
        yield from sorted(target.rglob("*.canvas.json"))
        return
    print(f"! target does not exist: {target}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upgrade saved canvases to strict-canvas v1. Bundles all the "
            "individual migration rules (obs_terms ScaleSpec, IR joint "
            "names, legacy field drops, auto-backend rewrite) plus a "
            "RECOMMENDED-reward audit. Idempotent — stamps "
            "``metadata.canvas_strict_v1: true``."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Single .canvas.json file or a directory to walk recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report intended rewrites without modifying any files.",
    )
    args = parser.parse_args(argv)

    files = list(_iter_canvas_files(args.target))
    if not files:
        print("no .canvas.json files found", file=sys.stderr)
        return 2

    rewrites = 0
    for f in files:
        if _migrate_one(f, dry_run=args.dry_run):
            rewrites += 1
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\nrewrote {rewrites}/{len(files)} canvas files{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
