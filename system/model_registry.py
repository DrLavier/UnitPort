#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical brand/model registry.

This module is the single source of truth for:
- canonical brand ids and legacy aliases
- brand -> model membership
- adapter keys
- required settings groups
- SDK project bindings
- MuJoCo asset/support flags
- Forward / locomotion metadata used by frontend and backend

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ForwardSpec:
    """Canonical Forward/locomotion capability for one model."""

    ir_availability: str
    ir_display_name: str
    ir_raw_action: str
    adapter_available: bool
    adapter_raw_action: str
    max_forward_vx: Optional[float] = None
    gait_type: Optional[str] = None
    robot_category: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    """Canonical metadata for one robot model."""

    brand_id: str
    model_id: str
    display_name: str
    sdk_required: bool
    sdk_only: bool
    has_mujoco_assets: bool
    mujoco_supported: bool
    sdk_project_keys: Tuple[str, ...] = ()
    forward: Optional[ForwardSpec] = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BrandSpec:
    """Canonical metadata for one brand."""

    brand_id: str
    display_name: str
    aliases: Tuple[str, ...]
    adapter_key: str
    sdk_project_keys: Tuple[str, ...]
    sdk_project_urls: Mapping[str, str]
    required_settings: Tuple[str, ...]
    models: Mapping[str, ModelSpec] = field(default_factory=dict)


_BASE_REQUIRED_SETTINGS: Tuple[str, ...] = (
    "brand",
    "robot_type",
    "connection_mode",
    "timeout",
    "stop_policy",
    "safety_enabled",
)


def _forward(
    *,
    ir_availability: str,
    ir_display_name: str,
    ir_raw_action: str = "forward",
    adapter_available: bool,
    adapter_raw_action: str = "walk",
    max_forward_vx: Optional[float] = None,
    gait_type: Optional[str] = None,
    robot_category: Optional[str] = None,
) -> ForwardSpec:
    return ForwardSpec(
        ir_availability=ir_availability,
        ir_display_name=ir_display_name,
        ir_raw_action=ir_raw_action,
        adapter_available=adapter_available,
        adapter_raw_action=adapter_raw_action,
        max_forward_vx=max_forward_vx,
        gait_type=gait_type,
        robot_category=robot_category,
    )


def _model(
    brand_id: str,
    model_id: str,
    *,
    display_name: Optional[str] = None,
    sdk_required: bool,
    sdk_only: bool,
    has_mujoco_assets: bool,
    mujoco_supported: bool,
    sdk_project_keys: Tuple[str, ...] = (),
    forward: Optional[ForwardSpec] = None,
    notes: Tuple[str, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        brand_id=brand_id,
        model_id=model_id,
        display_name=display_name or model_id,
        sdk_required=sdk_required,
        sdk_only=sdk_only,
        has_mujoco_assets=has_mujoco_assets,
        mujoco_supported=mujoco_supported,
        sdk_project_keys=sdk_project_keys,
        forward=forward,
        notes=notes,
    )


_REGISTRY: Dict[str, BrandSpec] = {
    "unitree": BrandSpec(
        brand_id="unitree",
        display_name="Unitree",
        aliases=("Unitree",),
        adapter_key="unitree_sdk2",
        sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
        sdk_project_urls={
            "unitree_sdk2_python": "https://github.com/unitreerobotics/unitree_sdk2_python.git",
            "unitree_mujoco": "https://github.com/unitreerobotics/unitree_mujoco.git",
        },
        required_settings=_BASE_REQUIRED_SETTINGS + (
            "dds_domain_id",
            "network_interface",
            "control_level",
            "mode_switch_policy",
        ),
        models={
            "go2": _model(
                "unitree",
                "go2",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.5,
                    gait_type="trot",
                    robot_category="quadruped",
                ),
            ),
            "go2w": _model(
                "unitree",
                "go2w",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.5,
                    gait_type="wheel",
                    robot_category="quadruped",
                ),
            ),
            "a1": _model(
                "unitree",
                "a1",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python",),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.0,
                    gait_type="trot",
                    robot_category="quadruped",
                ),
            ),
            # Temporarily unregistered: no MuJoCo assets available.
            # "b1": _model(
            #     "unitree",
            #     "b1",
            #     sdk_required=True,
            #     sdk_only=True,
            #     has_mujoco_assets=False,
            #     mujoco_supported=False,
            #     sdk_project_keys=("unitree_sdk2_python",),
            #     forward=_forward(
            #         ir_availability="unsupported",
            #         ir_display_name="Move - Forward",
            #         adapter_available=False,
            #     ),
            # ),
            "b2": _model(
                "unitree",
                "b2",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.2,
                    gait_type="trot",
                    robot_category="quadruped",
                ),
            ),
            "b2w": _model(
                "unitree",
                "b2w",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.2,
                    gait_type="wheel",
                    robot_category="quadruped",
                ),
            ),
            "g1": _model(
                "unitree",
                "g1",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.0,
                    gait_type="walk",
                    robot_category="humanoid",
                ),
            ),
            "h1": _model(
                "unitree",
                "h1",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.2,
                    gait_type="walk",
                    robot_category="humanoid",
                ),
            ),
            "h1_2": _model(
                "unitree",
                "h1_2",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("unitree_sdk2_python", "unitree_mujoco"),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.2,
                    gait_type="walk",
                    robot_category="humanoid",
                ),
            ),
        },
    ),
    "bostiondynamics": BrandSpec(
        brand_id="bostiondynamics",
        display_name="BostionDynamics",
        aliases=("BostionDynamics", "bostondynamics", "BostonDynamics", "boston_dynamics"),
        adapter_key="spot_sdk",
        sdk_project_keys=("spot-sdk",),
        sdk_project_urls={
            "spot-sdk": "https://github.com/boston-dynamics/spot-sdk.git",
        },
        required_settings=_BASE_REQUIRED_SETTINGS + (
            "hostname",
            "auth_token",
            "timesync_required",
            "lease_required",
            "safety_channel",
            "power_policy",
        ),
        models={
            "spot": _model(
                "bostiondynamics",
                "spot",
                sdk_required=True,
                sdk_only=False,
                has_mujoco_assets=True,
                mujoco_supported=True,
                sdk_project_keys=("spot-sdk",),
                forward=_forward(
                    ir_availability="available",
                    ir_display_name="Move - Forward",
                    adapter_available=True,
                    max_forward_vx=1.0,
                    gait_type="trot",
                    robot_category="quadruped",
                ),
            ),
        },
    ),
    "xiaomi": BrandSpec(
        brand_id="xiaomi",
        display_name="XiaoMi",
        aliases=("XiaoMi",),
        adapter_key="cyberdog_sdk",
        sdk_project_keys=("cyberdog_ros2",),
        sdk_project_urls={
            "cyberdog_ros2": "https://github.com/MiRoboticsLab/cyberdog_ros2.git",
        },
        required_settings=_BASE_REQUIRED_SETTINGS + (
            "ros_domain_id",
            "rmw_impl",
            "namespace",
            "action_endpoints",
            "timestamp_policy",
        ),
        models={
            # Temporarily unregistered: no MuJoCo assets available.
            # "cyberdog": _model(
            #     "xiaomi",
            #     "cyberdog",
            #     sdk_required=True,
            #     sdk_only=True,
            #     has_mujoco_assets=False,
            #     mujoco_supported=False,
            #     sdk_project_keys=("cyberdog_ros2",),
            #     forward=_forward(
            #         ir_availability="unsupported",
            #         ir_display_name="Move - Forward",
            #         adapter_available=True,
            #         max_forward_vx=0.8,
            #     ),
            # ),
            # "cyberdog2": _model(
            #     "xiaomi",
            #     "cyberdog2",
            #     sdk_required=True,
            #     sdk_only=True,
            #     has_mujoco_assets=False,
            #     mujoco_supported=False,
            #     sdk_project_keys=("cyberdog_ros2",),
            #     forward=_forward(
            #         ir_availability="unsupported",
            #         ir_display_name="Move - Forward",
            #         adapter_available=True,
            #         max_forward_vx=0.8,
            #     ),
            # ),
        },
    ),
}


_BRAND_ALIAS_TO_ID: Dict[str, str] = {}
for _brand_id, _brand in _REGISTRY.items():
    _BRAND_ALIAS_TO_ID[_brand_id] = _brand_id
    _BRAND_ALIAS_TO_ID[_brand.display_name.lower()] = _brand_id
    for _alias in _brand.aliases:
        _BRAND_ALIAS_TO_ID[_alias.strip().lower()] = _brand_id


def normalize_brand_id(value: str) -> str:
    """Return canonical brand id for *value*.

    Unknown values are normalized in a best-effort way rather than raising.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return _BRAND_ALIAS_TO_ID.get(raw, raw.replace("-", "").replace("_", ""))


def canonical_brand_ids() -> Tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def list_brand_specs() -> Tuple[BrandSpec, ...]:
    return tuple(_REGISTRY.values())


def list_brand_items() -> Tuple[Tuple[str, str], ...]:
    """Return ordered ``(brand_id, display_name)`` pairs for UI pickers."""
    return tuple((spec.brand_id, spec.display_name) for spec in _REGISTRY.values())


def get_brand_spec(brand: str) -> Optional[BrandSpec]:
    return _REGISTRY.get(normalize_brand_id(brand))


def get_model_spec(brand: str, model_id: str) -> Optional[ModelSpec]:
    brand_spec = get_brand_spec(brand)
    if brand_spec is None:
        return None
    return brand_spec.models.get(str(model_id or "").strip().lower())


def iter_model_specs(*, brand: str = "") -> Iterable[ModelSpec]:
    if brand:
        brand_spec = get_brand_spec(brand)
        if brand_spec is None:
            return ()
        return tuple(brand_spec.models.values())
    return tuple(
        model
        for brand_spec in _REGISTRY.values()
        for model in brand_spec.models.values()
    )


def get_model_ids(brand: str) -> List[str]:
    brand_spec = get_brand_spec(brand)
    if brand_spec is None:
        return []
    return list(brand_spec.models.keys())


def get_brand_model_map(*, display_names: bool = False) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for brand_id, brand_spec in _REGISTRY.items():
        key = brand_spec.display_name if display_names else brand_id
        result[key] = list(brand_spec.models.keys())
    return result


def get_robot_brand_map() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for brand_id, brand_spec in _REGISTRY.items():
        for model_id in brand_spec.models:
            result[model_id] = brand_id
    return result


def get_adapter_key_for_brand(brand: str) -> str:
    brand_spec = get_brand_spec(brand)
    return brand_spec.adapter_key if brand_spec is not None else ""


def get_required_settings(brand: str) -> Tuple[str, ...]:
    brand_spec = get_brand_spec(brand)
    return brand_spec.required_settings if brand_spec is not None else _BASE_REQUIRED_SETTINGS


def sdk_project_keys_for_brand(brand: str) -> Tuple[str, ...]:
    brand_spec = get_brand_spec(brand)
    return brand_spec.sdk_project_keys if brand_spec is not None else ()


def sdk_project_url(brand: str, project_key: str) -> str:
    brand_spec = get_brand_spec(brand)
    if brand_spec is None:
        return ""
    return str(brand_spec.sdk_project_urls.get(project_key, "") or "")


def sdk_project_keys_for_model(brand: str, model_id: str) -> Tuple[str, ...]:
    spec = get_model_spec(brand, model_id)
    return spec.sdk_project_keys if spec is not None else ()


def get_ir_action_binding(brand: str, model_id: str, intent_id: str) -> Dict[str, object]:
    """Return model-specific binding metadata for one IR intent.

    The return value is a plain dict so upper layers can consume it without
    importing additional registry-specific types.
    """
    spec = get_model_spec(brand, model_id)
    if spec is None:
        return {}

    if str(intent_id or "").strip().lower() == "locomotion.forward":
        forward = spec.forward
        if forward is None:
            return {}
        max_speed = forward.max_forward_vx if forward.max_forward_vx is not None else 1.0
        default_speed = 0.3 if (forward.robot_category or "") == "humanoid" else 0.5
        if default_speed > max_speed:
            default_speed = max_speed
        parameter_key = "vx" if (spec.brand_id == "unitree" and spec.model_id == "h1") else "speed"
        parameter_label = "Forward velocity vx (m/s)" if parameter_key == "vx" else "Forward speed ratio (0-1)"
        parameter_description = (
            "Forward velocity passed to Unitree H1 LocoClient.Move(vx, 0, 0)."
            if parameter_key == "vx"
            else "Normalized forward speed ratio. Runtime converts it to model-specific vx."
        )
        return {
            "availability": forward.ir_availability,
            "raw_action": forward.adapter_raw_action if forward.adapter_available else "",
            "parameters": {
                parameter_key: {
                    "type": "float",
                    "default": default_speed,
                    "min": 0.0,
                    "max": max_speed,
                    "label": parameter_label,
                    "description": parameter_description,
                },
                "duration": {
                    "type": "float",
                    "default": 1.0,
                    "min": 0.0,
                    "description": "Duration in seconds.",
                },
            },
            "tags": tuple(
                tag for tag in (
                    "locomotion",
                    "ir",
                    forward.robot_category or "",
                    forward.gait_type or "",
                ) if tag
            ),
        }

    return {}


def mujoco_supported_models(brand: str) -> Tuple[str, ...]:
    return tuple(
        spec.model_id
        for spec in iter_model_specs(brand=brand)
        if spec.mujoco_supported
        and spec.forward is not None
        and spec.forward.max_forward_vx is not None
        and spec.forward.gait_type is not None
        and spec.forward.robot_category is not None
    )


def locomotion_profile_models(brand: str) -> Tuple[ModelSpec, ...]:
    return tuple(
        spec
        for spec in iter_model_specs(brand=brand)
        if spec.forward is not None
        and spec.forward.max_forward_vx is not None
        and spec.forward.gait_type is not None
        and spec.forward.robot_category is not None
        and spec.mujoco_supported
    )
