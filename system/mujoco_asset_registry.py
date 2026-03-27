#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical MuJoCo asset registry for all currently registered models.

Resolution order:
1. /models/mujoco_menagerie
2. /brands_sdk/<Brand>/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from system.model_registry import get_model_spec, iter_model_specs


@dataclass(frozen=True)
class MujocoAssetRule:
    brand_id: str
    model_id: str
    menagerie_dirs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MujocoAssetLocation:
    brand_id: str
    model_id: str
    source: str
    scene_path: Path


@dataclass(frozen=True)
class MujocoAssetStatus:
    brand_id: str
    model_id: str
    source: str
    scene_path: Optional[Path]
    found: bool


_REGISTERED_MODEL_RULES: Dict[Tuple[str, str], MujocoAssetRule] = {
    ("unitree", "go2"): MujocoAssetRule("unitree", "go2", ("unitree_go2",)),
    ("unitree", "go2w"): MujocoAssetRule("unitree", "go2w", ()),
    ("unitree", "a1"): MujocoAssetRule("unitree", "a1", ("unitree_a1",)),
    # Temporarily unregistered in system.model_registry: no MuJoCo assets available.
    # ("unitree", "b1"): MujocoAssetRule("unitree", "b1", ()),
    ("unitree", "b2"): MujocoAssetRule("unitree", "b2", ()),
    ("unitree", "b2w"): MujocoAssetRule("unitree", "b2w", ()),
    ("unitree", "g1"): MujocoAssetRule("unitree", "g1", ("unitree_g1",)),
    ("unitree", "h1"): MujocoAssetRule("unitree", "h1", ("unitree_h1",)),
    ("unitree", "h1_2"): MujocoAssetRule("unitree", "h1_2", ()),
    ("bostiondynamics", "spot"): MujocoAssetRule("bostiondynamics", "spot", ("boston_dynamics_spot",)),
    # Temporarily unregistered in system.model_registry: no MuJoCo assets available.
    # ("xiaomi", "cyberdog"): MujocoAssetRule("xiaomi", "cyberdog", ()),
    # ("xiaomi", "cyberdog2"): MujocoAssetRule("xiaomi", "cyberdog2", ()),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _existing_scene_paths(base_dir: Path) -> Iterable[Path]:
    yield base_dir / "scene.xml"
    yield base_dir / "scene_mjx.xml"


def _brand_asset_roots(brand: str) -> Iterable[Path]:
    root = _project_root() / "brands_sdk"
    brand_key = str(brand or "").strip().lower()
    if brand_key == "unitree":
        yield root / "Unitree" / "unitree_mujoco" / "unitree_robots"
        yield root / "unitree" / "unitree_mujoco" / "unitree_robots"
        yield root / "Unitree"
        yield root / "unitree"
    elif brand_key == "bostiondynamics":
        yield root / "BostionDynamics"
        yield root / "bostiondynamics"
        yield root / "BostionDynamics" / "spot-sdk"
    elif brand_key == "xiaomi":
        yield root / "XiaoMi"
        yield root / "xiaomi"


def _menagerie_candidates(rule: MujocoAssetRule) -> Iterable[Path]:
    root = _project_root() / "models" / "mujoco_menagerie"
    for dirname in rule.menagerie_dirs:
        base_dir = root / dirname
        for scene_path in _existing_scene_paths(base_dir):
            yield scene_path


def _fallback_candidates(rule: MujocoAssetRule) -> Iterable[Path]:
    model_id = rule.model_id
    for base in _brand_asset_roots(rule.brand_id):
        yield base / model_id / "scene.xml"
        yield base / "data" / model_id / "scene.xml"
        yield base / "unitree_robots" / model_id / "scene.xml"
        yield base / "unitree_robots" / "data" / model_id / "scene.xml"
        yield base / f"{model_id}.xml"
        yield base / model_id / f"{model_id}.xml"
        if rule.brand_id == "bostiondynamics":
            yield base / "scene.xml"
            yield base / "spot.xml"


def registered_mujoco_asset_rules() -> Tuple[MujocoAssetRule, ...]:
    rules: List[MujocoAssetRule] = []
    for spec in iter_model_specs():
        rule = _REGISTERED_MODEL_RULES.get((spec.brand_id, spec.model_id))
        if rule is None:
            rule = MujocoAssetRule(spec.brand_id, spec.model_id, ())
        rules.append(rule)
    rules.sort(key=lambda item: (item.brand_id, item.model_id))
    return tuple(rules)


def resolve_mujoco_asset(brand: str, model_id: str) -> Optional[MujocoAssetLocation]:
    spec = get_model_spec(brand, model_id)
    if spec is None:
        return None

    rule = _REGISTERED_MODEL_RULES.get((spec.brand_id, spec.model_id))
    if rule is None:
        return None

    for scene_path in _menagerie_candidates(rule):
        if scene_path.exists():
            return MujocoAssetLocation(spec.brand_id, spec.model_id, "mujoco_menagerie", scene_path)

    for scene_path in _fallback_candidates(rule):
        if scene_path.exists():
            return MujocoAssetLocation(spec.brand_id, spec.model_id, "brand_models", scene_path)

    return None


def asset_status_for_registered_models() -> List[MujocoAssetStatus]:
    rows: List[MujocoAssetStatus] = []
    for rule in registered_mujoco_asset_rules():
        location = resolve_mujoco_asset(rule.brand_id, rule.model_id)
        rows.append(
            MujocoAssetStatus(
                brand_id=rule.brand_id,
                model_id=rule.model_id,
                source=(location.source if location is not None else "missing"),
                scene_path=(location.scene_path if location is not None else None),
                found=(location is not None),
            )
        )
    return rows


def list_registered_asset_matches() -> List[MujocoAssetLocation]:
    matches: List[MujocoAssetLocation] = []
    for row in asset_status_for_registered_models():
        if row.found and row.scene_path is not None:
            matches.append(
                MujocoAssetLocation(
                    brand_id=row.brand_id,
                    model_id=row.model_id,
                    source=row.source,
                    scene_path=row.scene_path,
                )
            )
    return matches


def list_menagerie_asset_matches() -> List[MujocoAssetLocation]:
    return [item for item in list_registered_asset_matches() if item.source == "mujoco_menagerie"]
