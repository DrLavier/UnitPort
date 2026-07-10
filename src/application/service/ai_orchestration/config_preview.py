# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AI Build Configuration-preview projector — the producer behind the panel's
collapsible Robot / Engine / Training Ground / Environment / Rewards sections.

The redesigned :class:`~application.ui.canvas.ai_build_panel.AiBuildPanel` ships a
``set_config_preview(sections)`` seam but no producer (a documented BACKEND TODO),
so the sections showed an all-``—`` skeleton and never followed agent edits. This
is that producer: a **Qt-free** projection of the live canvas workflow dict into
the panel's plain ``[(title, [(label, value), …]), …]`` model, so the page can
push it on open AND after every mutation (real-time) without the widget importing
any business logic.

An **empty canvas** projects to ``[]`` (the panel then shows its "describe a run"
placeholder); a canvas with any config node projects real values — a node the
agent has NOT touched simply reads its own params (never a fabricated value, §8).
Labels reuse the panel's ``tr`` keys so they render identically to the skeleton.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import log_warning, tr

__all__ = ["project_config_sections"]

Section = Tuple[str, List[Tuple[str, str]]]

# Matches the panel's ``banner_none`` default — an honest "unset", not "empty by
# bug": a field with no value on the canvas reads as this dash.
_DASH = "—"

#: Trainer schemas that may carry env-count / iteration / mode knobs.
_TRAINER_SCHEMAS = ("il_ppo_trainer", "amp_trainer", "algorithm_config")


def project_config_sections(canvas: Dict[str, Any]) -> List[Section]:
    """Project ``canvas`` (a ``to_workflow_dict`` shape) into panel sections.

    Returns ``[]`` for an empty / non-dict canvas (→ the panel's placeholder);
    otherwise the five configured sections with values read live from the nodes.
    A projection failure degrades to ``[]`` with a WARN (cosmetic panel — must
    never break the run).
    """
    if not isinstance(canvas, dict) or not (canvas.get("nodes") or []):
        return []
    try:
        return [
            _robot_section(canvas),
            _engine_section(canvas),
            _ground_section(canvas),
            _env_section(canvas),
            _rewards_section(canvas),
        ]
    except Exception as exc:  # cosmetic preview — never propagate
        log_warning(f"[ai.config_preview] projection failed: {exc!r}")
        return []


# =============================================================================
# sections
# =============================================================================

def _robot_section(canvas: Dict[str, Any]) -> Section:
    node = _first(canvas, "robot")
    name = sku = family = _DASH
    asset = str(_param(node, "asset_id", "") or "").strip()
    if asset:
        from registers import robots

        resolved = robots.resolve_to_sku(asset)
        if resolved:
            sku = resolved
            spec = robots.get_robot_spec(resolved)
            if spec is not None:
                name = spec.name or resolved
                family = spec.families[0].title() if spec.families else _DASH
        else:  # unresolved raw id — show it rather than a dash (honest)
            name = asset
    return (
        tr("canvas.ai_build.cfg_sec_robot", "Robot"),
        [
            (tr("canvas.ai_build.cfg_f_name", "Name"), name),
            (tr("canvas.ai_build.cfg_f_sku", "SKU"), sku),
            (tr("canvas.ai_build.cfg_f_family", "Family"), family),
        ],
    )


def _engine_section(canvas: Dict[str, Any]) -> Section:
    backend = str(canvas.get("backend") or "").strip()
    backend_disp = _DASH
    if backend:
        try:
            from registers import backends

            backend_disp = backends.get_display_name(backend) or backend
        except Exception:  # registry not ready — raw id
            backend_disp = backend
    algo = _first(canvas, "algorithm_config")
    device = _fmt(_param(algo, "device")) if algo is not None else _DASH
    return (
        tr("canvas.ai_build.cfg_sec_engine", "Engine"),
        [
            (tr("canvas.ai_build.cfg_f_backend", "Backend"), backend_disp),
            (tr("canvas.ai_build.cfg_f_device", "Device"), device),
        ],
    )


def _ground_section(canvas: Dict[str, Any]) -> Section:
    pg = _first(canvas, "play_ground_setting")
    terrain = _fmt(_param(pg, "scene_type")) if pg is not None else _DASH
    curriculum_raw = _param(pg, "curriculum_enabled") if pg is not None else None
    curriculum = _fmt(bool(curriculum_raw)) if curriculum_raw is not None else _DASH
    difficulty = (
        _fmt(_param(pg, "difficulty_levels"))
        if (pg is not None and curriculum_raw)
        else _DASH
    )
    return (
        tr("canvas.ai_build.cfg_sec_ground", "Training Ground"),
        [
            (tr("canvas.ai_build.cfg_f_terrain", "Terrain"), terrain),
            (tr("canvas.ai_build.cfg_f_difficulty", "Difficulty"), difficulty),
            (tr("canvas.ai_build.cfg_f_curriculum", "Curriculum"), curriculum),
        ],
    )


def _env_section(canvas: Dict[str, Any]) -> Section:
    trainer = _first_of(canvas, _TRAINER_SCHEMAS)
    num_envs = _fmt(_param(trainer, "num_envs")) if trainer is not None else _DASH
    task = _first(canvas, "task_config")
    episode = _fmt(_param(task, "truncation_max_steps")) if task is not None else _DASH
    return (
        tr("canvas.ai_build.cfg_sec_env", "Environment"),
        [
            (tr("canvas.ai_build.cfg_f_num_envs", "Num Envs"), num_envs),
            (tr("canvas.ai_build.cfg_f_episode", "Episode Length"), episode),
            # No dedicated env-spacing param on the canvas today (honest dash).
            (tr("canvas.ai_build.cfg_f_spacing", "Env Spacing"), _DASH),
        ],
    )


def _rewards_section(canvas: Dict[str, Any]) -> Section:
    rows: List[Tuple[str, str]] = []
    for node in _nodes(canvas, "rewards"):
        terms = _coerce_dict(_param(node, "reward_terms"))
        for page_id, page in terms.items():
            if not isinstance(page, dict):
                continue
            for func, payload in page.items():
                weight = _weight_of(payload)
                label = str(func) if str(page_id) == "__global__" else f"{func} [{page_id}]"
                rows.append((label, _fmt(weight) if weight is not None else _DASH))
    if not rows:
        rows = [(tr("canvas.ai_build.cfg_f_reward_terms", "Reward Terms"), _DASH)]
    return (tr("canvas.ai_build.cfg_sec_rewards", "Rewards"), rows)


# =============================================================================
# helpers
# =============================================================================

def _nodes(canvas: Dict[str, Any], schema_id: str) -> List[Dict[str, Any]]:
    return [
        n for n in (canvas.get("nodes") or [])
        if str(n.get("schema_id")) == schema_id and not n.get("disabled")
    ]


def _first(canvas: Dict[str, Any], schema_id: str) -> Optional[Dict[str, Any]]:
    got = _nodes(canvas, schema_id)
    return got[0] if got else None


def _first_of(canvas: Dict[str, Any], schema_ids: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    for sid in schema_ids:
        node = _first(canvas, sid)
        if node is not None:
            return node
    return None


def _param(node: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if node is None:
        return default
    cell = (node.get("params") or {}).get(key)
    if cell is None:
        return default
    val = cell.get("value") if isinstance(cell, dict) else cell
    return default if val is None else val


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _weight_of(payload: Any) -> Optional[float]:
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict) and "weight" in payload:
        w = payload.get("weight")
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            return float(w)
        if isinstance(w, str):
            try:
                return float(w.strip())
            except ValueError:
                return None
    return None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return _DASH
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
