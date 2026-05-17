"""AdapterFactory — single entrypoint resolving SKU + strategy → adapter instance.

Robot identity (brand + model) flows from the RealRobotConnectionCard combos
into ``ConnectionProfile.brand`` / ``.robot``. The connection orchestrator
calls :meth:`AdapterFactory.build` to instantiate the right brand subclass.

Resolution order (highest priority first):

1. **Per-strategy adapter map** — ``RobotSpec.capabilities["adapter_strategies"]``
   may declare ``{strategy_name: dotted.class.path}``. Each robot can offer
   distinct adapter classes for ROS2-Generic vs vendor-SDK paths.

2. **Default adapter key** — ``RobotSpec.adapter`` is the brand-level
   adapter key (e.g. ``"mangdang_ros2"``); resolved via
   :func:`registers.brands.get_adapter_class` to a dotted-path string.
   Used when the strategy is "ros2_generic" or unspecified.

3. **No match** → :class:`AdapterUnavailable` with stable error code.

Adapter instances are cached per (sku, strategy) — a single adapter object
serves multiple connect/disconnect cycles. ``release_cache(sku, strategy)``
drops the cache entry (called when robot is removed from registers, etc.).

UI consumers use :func:`available_strategies` to populate the
"Adapter strategy" combo dynamically: brands with both ROS2-generic and
brand-SDK adapters expose two strategies; brands with only one expose one
(combo disabled but visible).
"""

from __future__ import annotations

import importlib
import threading
from typing import Dict, List, Optional, Tuple, Type

from unitport_sdk import log_info, log_warning

from application.service.adapters.base_adapter import AdapterUnavailable, BaseAdapter
from application.service.connection.profile import ConnectionProfile
from registers import brands as _brands_reg
from registers import resolve_robot_sku as _resolve_robot_sku
from registers import robots as _robots_reg


_CACHE: Dict[Tuple[str, str], BaseAdapter] = {}
_LOCK = threading.RLock()


# Strategy aliases — UI labels translate to internal keys.
_STRATEGY_ALIAS: Dict[str, str] = {
    "ROS2 Generic": "ros2_generic",
    "Brand SDK":    "brand_sdk",
    "ros2_generic": "ros2_generic",
    "brand_sdk":    "brand_sdk",
}


def _normalise_strategy(strategy: str) -> str:
    return _STRATEGY_ALIAS.get(str(strategy or "ros2_generic"), str(strategy))


def _import_class(dotted_path: str) -> Type[BaseAdapter]:
    """Import ``dotted_path`` and return the named class.

    Raises :class:`AdapterUnavailable` with structured detail on any failure.
    """
    if not dotted_path or "." not in dotted_path:
        raise AdapterUnavailable(
            "adapter_path_invalid",
            f"adapter dotted-path is missing or malformed: {dotted_path!r}",
        )
    module_path, _, class_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise AdapterUnavailable(
            "adapter_module_not_found",
            f"failed to import {module_path!r}: {exc}",
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AdapterUnavailable(
            "adapter_class_not_found",
            f"{module_path!r} has no attribute {class_name!r}",
        )
    if not isinstance(cls, type) or not issubclass(cls, BaseAdapter):
        raise AdapterUnavailable(
            "adapter_class_not_baseadapter",
            f"{dotted_path!r} is not a BaseAdapter subclass",
        )
    return cls


def _resolve_dotted_path(sku: str, strategy: str) -> str:
    """Return the dotted-path string for (sku, strategy), or raise."""
    spec = _robots_reg.get_robot_spec(sku)
    if spec is None:
        raise AdapterUnavailable(
            "robot_sku_unknown",
            f"no robot registered for sku {sku!r} — "
            f"check registers/data/robots_canonical.json",
        )

    # Per-strategy override map lives under capabilities.adapter_strategies.
    strategies = spec.capabilities.get("adapter_strategies", {}) or {}
    if isinstance(strategies, dict):
        cls_path = strategies.get(strategy)
        if cls_path:
            return str(cls_path)

    # Fallback: ros2_generic strategy resolves via brands.get_host_adapter_class.
    if strategy == "ros2_generic":
        adapter_key = (spec.adapter or "").strip()
        if not adapter_key:
            raise AdapterUnavailable(
                "adapter_key_missing",
                f"sku {sku!r} has no `adapter` field in robots_canonical.json",
            )
        cls_path = _brands_reg.get_host_adapter_class(adapter_key)
        if cls_path:
            return str(cls_path)
        # Allow direct dotted-paths as a fallback (caller-supplied power user
        # path or test fixture).
        if "." in adapter_key:
            return adapter_key
        raise AdapterUnavailable(
            "adapter_key_unresolved",
            f"adapter_key {adapter_key!r} not in registers/brands.host_built_in",
        )

    raise AdapterUnavailable(
        "strategy_unsupported",
        f"sku {sku!r} does not declare strategy {strategy!r} — "
        f"add an entry under robot.capabilities.adapter_strategies",
    )


class AdapterFactory:
    """Static factory: SKU + strategy → BaseAdapter instance (cached)."""

    @staticmethod
    def build(sku: str, strategy: str = "ros2_generic") -> BaseAdapter:
        strat = _normalise_strategy(strategy)
        key = (str(sku), strat)
        with _LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                return cached
            dotted = _resolve_dotted_path(str(sku), strat)
            cls = _import_class(dotted)
            instance = cls()
            _CACHE[key] = instance
            log_info(
                f"[adapter_factory] built {cls.__name__} for sku={sku!r} "
                f"strategy={strat!r}"
            )
            return instance

    @staticmethod
    def class_for(sku: str, strategy: str = "ros2_generic") -> Type[BaseAdapter]:
        """Return the adapter class without instantiating.

        UI uses this to read class-level attributes (DEFAULT_DOMAIN_ID, etc.)
        for combo defaults without paying the construction cost.
        """
        strat = _normalise_strategy(strategy)
        dotted = _resolve_dotted_path(str(sku), strat)
        return _import_class(dotted)

    @staticmethod
    def build_for_profile(
        profile: ConnectionProfile,
        strategy: str = "ros2_generic",
    ) -> BaseAdapter:
        """Resolve adapter from ``profile.brand`` + ``profile.robot``.

        Connector convenience — avoids forcing every call site to recompute
        ``resolve_robot_sku(profile.brand, profile.robot)`` first.
        """
        sku = _resolve_robot_sku(profile.brand, profile.robot)
        return AdapterFactory.build(sku, strategy)

    @staticmethod
    def class_for_brand_model(
        brand: str,
        model: str,
        strategy: str = "ros2_generic",
    ) -> Type[BaseAdapter]:
        """UI convenience — peek the adapter class from (brand, model)."""
        return AdapterFactory.class_for(_resolve_robot_sku(brand, model), strategy)

    @staticmethod
    def release_cache(sku: str = "", strategy: str = "") -> None:
        """Drop cached instances. With no args, drops everything."""
        with _LOCK:
            if not sku and not strategy:
                _CACHE.clear()
                return
            keys = [k for k in _CACHE.keys() if (not sku or k[0] == sku)
                    and (not strategy or k[1] == _normalise_strategy(strategy))]
            for k in keys:
                _CACHE.pop(k, None)


def available_strategies(sku: str) -> List[str]:
    """Return user-facing strategy labels available for ``sku``.

    Used by RealRobotConnectionCard to populate the Adapter strategy combo —
    options not in this list are disabled (greyed out) but still rendered so
    the user understands the spectrum of choices.
    """
    spec = _robots_reg.get_robot_spec(sku)
    if spec is None:
        return []
    out: List[str] = []
    strategies = spec.capabilities.get("adapter_strategies", {}) or {}
    if isinstance(strategies, dict):
        for key in strategies.keys():
            if key == "ros2_generic":
                out.append("ROS2 Generic")
            elif key == "brand_sdk":
                out.append("Brand SDK")
            else:
                out.append(str(key))
    # ros2_generic is implicitly available whenever the adapter_key resolves.
    if (spec.adapter or "") and "ROS2 Generic" not in out:
        try:
            _ = _resolve_dotted_path(sku, "ros2_generic")
            out.insert(0, "ROS2 Generic")
        except AdapterUnavailable:
            pass
    return out


__all__ = [
    "AdapterFactory",
    "available_strategies",
]
