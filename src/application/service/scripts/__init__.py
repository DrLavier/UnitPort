"""application.service.scripts — variant resolver for training-function presets.

The factory presets in ``src/scripts/{rewards,terminations,observations,
discriminator}/registry.py`` are read-only, ship-with-the-product Python
modules. Users can *augment* each preset with one or more **variants** —
flavored implementations tagged by robot family — which live under
``Paths.USER_CONFIG_DIR / scripts / <kind> / <key> / <variant>.py`` and a
sibling ``variants.toml`` meta file.

This package owns:

- enumeration of user variants on disk
- per-variant metadata (families, description, base, timestamps)
- resolution of ``(kind, key, variant)`` into runnable Python source —
  falling back to the preset when ``variant in (None, "preset")``
- creation / update / deletion of user variants (atomic writes via
  :class:`DataManager`; ``user_scripts_changed`` signal emit on mutation)

All on-disk I/O is funnelled through :class:`DataManager` and
:class:`Paths`; no direct ``open()`` / ``Path(...)`` / ``ConfigParser``
calls per RELEASE/CLAUDE.md §5.

Cross-process / cross-thread story: the resolver itself is a singleton
of stateless module-level functions (DataManager owns the I/O cache).
Saves emit ``AppSignals.user_scripts_changed(kind, key)`` so sidebar
panels can refresh without polling.

Public surface — see :mod:`application.service.scripts.resolver`.
"""
from __future__ import annotations

from .resolver import (
    ResolvedScript,
    VariantMeta,
    delete_variant,
    family_filter,
    list_keys,
    list_variants,
    map_engine_to_backend,
    resolve,
    save_variant,
)

__all__ = [
    "ResolvedScript",
    "VariantMeta",
    "delete_variant",
    "family_filter",
    "list_keys",
    "list_variants",
    "map_engine_to_backend",
    "resolve",
    "save_variant",
]
