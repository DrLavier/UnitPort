"""Brand package registry.

Single source of truth for resolving brand-specific robot model classes
from service adapters. All adapters MUST go through ``get_brand_model_class``
— never import brand modules by string path. Adding a new brand requires
exactly one edit: a new entry in ``_BRAND_MODULES`` below.

The canonical, and ONLY supported, package root for brand packages is
``src.system.brand_packages``. No legacy fallbacks are honored.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple


# Brand key → (submodule name under src.system.brand_packages, exported class name).
# To register a new brand: add it here, and make sure the submodule's __init__.py
# exports the class name listed.
_BRAND_MODULES: Dict[str, Tuple[str, str]] = {
    "unitree":          ("unitree",          "UnitreeModel"),
    "boston_dynamics":  ("boston_dynamics",  "SpotModel"),
}

_PACKAGE_ROOT = "src.system.brand_packages"


def list_brands() -> list[str]:
    """Return all registered brand keys."""
    return sorted(_BRAND_MODULES.keys())


def get_brand_model_class(brand: str) -> Any:
    """Resolve a brand key to its model class.

    Parameters
    ----------
    brand:
        Brand key (case-insensitive), e.g. ``"unitree"`` or ``"boston_dynamics"``.

    Returns
    -------
    The model class object (not an instance).

    Raises
    ------
    KeyError
        If the brand is not registered in ``_BRAND_MODULES``.
    ImportError
        If the registered module cannot be imported, or does not export the
        expected class name. The error message names both so misconfiguration
        is obvious at startup rather than failing silently later.
    """
    key = (brand or "").strip().lower()
    if key not in _BRAND_MODULES:
        raise KeyError(
            f"Brand '{brand}' is not registered. "
            f"Known brands: {list_brands()}. "
            f"Register it in {__name__}._BRAND_MODULES."
        )

    submodule, class_name = _BRAND_MODULES[key]
    full_module = f"{_PACKAGE_ROOT}.{submodule}"
    try:
        module = importlib.import_module(full_module)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import brand package '{full_module}' for brand '{key}': {exc}"
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(
            f"Brand package '{full_module}' does not export '{class_name}'. "
            f"Check {full_module}.__init__.py."
        )
    return cls
