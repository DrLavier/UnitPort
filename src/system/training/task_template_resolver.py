from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.system.training.robot_family import resolve_robot_family
from src.system.training.task_templates import (
    TASK_TEMPLATE_ASSET,
    TASK_TEMPLATE_FAMILY,
    TASK_TEMPLATE_ROBOT,
    TaskTemplate,
)
from src.system.training.training_asset_registry import TrainingAssetRegistry


def _empty_template() -> TaskTemplate:
    return {
        "reward_terms": {},
        "termination_conditions": {},
    }


def _copy_template(template: Optional[TaskTemplate]) -> TaskTemplate:
    if not template:
        return _empty_template()
    return {
        "reward_terms": dict(template.get("reward_terms", {})),
        "termination_conditions": dict(template.get("termination_conditions", {})),
    }


def _merge_task_templates(*templates: Optional[TaskTemplate]) -> TaskTemplate:
    merged = _empty_template()
    for template in templates:
        if not template:
            continue
        merged["reward_terms"].update(template.get("reward_terms", {}))
        merged["termination_conditions"].update(
            template.get("termination_conditions", {})
        )
    return merged


def _normalize_numeric_map(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_task_template(raw: Any) -> TaskTemplate:
    if not isinstance(raw, dict):
        return _empty_template()
    return {
        "reward_terms": _normalize_numeric_map(raw.get("reward_terms")),
        "termination_conditions": _normalize_numeric_map(raw.get("termination_conditions")),
    }


def _asset_template_path(asset_id: str, asset_root: Optional[Path] = None) -> Path:
    registry = TrainingAssetRegistry(root=asset_root)
    return registry.root / str(asset_id or "").strip().lower() / "task_template.json"


def _load_asset_template_file(
    asset_id: str,
    asset_root: Optional[Path] = None,
) -> TaskTemplate:
    template_path = _asset_template_path(asset_id, asset_root)
    if not template_path.exists():
        return _empty_template()
    try:
        return _normalize_task_template(
            json.loads(template_path.read_text(encoding="utf-8"))
        )
    except Exception:
        return _empty_template()


def resolve_task_template(
    *,
    robot_type: str = "",
    robot_family: str = "",
    asset_id: Optional[str] = None,
    asset_root: Optional[Path] = None,
) -> TaskTemplate:
    """
    Resolve effective task defaults using family -> robot -> asset precedence.

    Unknown family / robot / asset inputs are tolerated and simply contribute
    no override layer.
    """
    resolved_family = str(robot_family or "").strip().lower()
    if not resolved_family:
        resolved_family = resolve_robot_family(robot_type)

    resolved_robot = str(robot_type or "").strip().lower()
    resolved_asset = str(asset_id or "").strip().lower()

    family_template = TASK_TEMPLATE_FAMILY.get(resolved_family)
    robot_template = TASK_TEMPLATE_ROBOT.get(resolved_robot)
    asset_template = _merge_task_templates(
        TASK_TEMPLATE_ASSET.get(resolved_asset),
        _load_asset_template_file(resolved_asset, asset_root),
    )

    return _merge_task_templates(
        family_template,
        robot_template,
        asset_template,
    )


def get_family_template(robot_family: str) -> TaskTemplate:
    return _copy_template(TASK_TEMPLATE_FAMILY.get(str(robot_family or "").strip().lower()))


def get_robot_template(robot_type: str) -> TaskTemplate:
    return _copy_template(TASK_TEMPLATE_ROBOT.get(str(robot_type or "").strip().lower()))


def get_asset_template(asset_id: str, asset_root: Optional[Path] = None) -> TaskTemplate:
    resolved_asset = str(asset_id or "").strip().lower()
    return _merge_task_templates(
        TASK_TEMPLATE_ASSET.get(resolved_asset),
        _load_asset_template_file(resolved_asset, asset_root),
    )
