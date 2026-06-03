# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Terrain loader base: ABC + error type + the public dispatcher.

Data-free, mirroring ``application.training.motion.loaders.base``. It
declares the contract every concrete terrain loader implements:

  * :exc:`TerrainLoaderError` — single exception type for any parse,
    structural, or registration failure.
  * :class:`TerrainLoader` — abstract ``(path) → TerrainContract``
    interface plus the ``format_id`` class attribute pinning a loader's
    identity in the registry.
  * :data:`LOADER_REGISTRY` — public ``{format_id: loader_class}`` dict
    populated by this subpackage's ``__init__`` via explicit
    :func:`register_loader` calls (no class side-effects).
  * :func:`register_loader` / :func:`list_loader_formats` /
    :func:`get_loader` — the dispatcher surface.

I/O note (same carve-out as the motion loaders)
-----------------------------------------------
Concrete loaders read user-supplied paths (project tree / user data
dir). ``np.load`` / image decoders are the right tools — wrapping them in
``DataManager`` would round-trip arrays through dtype conversions the SDK
does not model. This is the established precedent in
``motion/loaders/base.py`` (per-extension DataManager dispatch is for
SDK-managed files; arbitrary user paths use the canonical Python
loaders), not a new §4 exception.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Union

from application.training.terrain.contract import TerrainContract


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TerrainLoaderError(ValueError):
    """Raised when a terrain file cannot be parsed by the requested loader,
    or when a registry operation (register / get) is invalid."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TerrainLoader(ABC):
    """Abstract ``(path) → TerrainContract`` contract."""

    #: Stable identifier used by :func:`get_loader` and ``LOADER_REGISTRY``.
    format_id: str = ""

    @abstractmethod
    def load(self, path: Union[Path, str]) -> TerrainContract:
        """Read ``path`` and return a validated :class:`TerrainContract`.

        Geometry that a bare array file cannot carry (physical
        ``size_x`` / ``size_y``, and for normalised image sources the
        elevation scale) is supplied to the concrete loader's
        constructor, mirroring how :class:`NpyLoader` takes ``fps`` —
        never reverse-inferred from the file.

        Raises :exc:`TerrainLoaderError` on any parse or structural
        failure, and
        :exc:`application.training.terrain.contract.TerrainContractError`
        when the produced contract fails validation. Concrete loaders
        MUST call :func:`validate_terrain_contract` before returning so a
        malformed surface fails loud at the boundary (CLAUDE.md §8).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


#: Public ``{format_id: loader_class}`` registry. Populated by
#: ``application.training.terrain.loaders.__init__`` via explicit
#: :func:`register_loader` calls. Mutable so plugins can install
#: themselves without modifying this module.
LOADER_REGISTRY: Dict[str, Type[TerrainLoader]] = {}


def register_loader(loader_cls: Type[TerrainLoader]) -> None:
    """Add a loader class to :data:`LOADER_REGISTRY`.

    Re-registering an existing ``format_id`` raises
    :exc:`TerrainLoaderError` — silently shadowing a built-in loader
    makes debugging downstream schema mismatches unnecessarily hard.
    """
    if not isinstance(loader_cls, type) or not issubclass(loader_cls, TerrainLoader):
        raise TerrainLoaderError(
            f"register_loader: {loader_cls!r} is not a TerrainLoader subclass"
        )
    fid = getattr(loader_cls, "format_id", "")
    if not fid:
        raise TerrainLoaderError(
            f"register_loader: {loader_cls.__name__} has no non-empty "
            f"format_id class attribute"
        )
    if fid in LOADER_REGISTRY:
        raise TerrainLoaderError(
            f"register_loader: format_id {fid!r} is already registered to "
            f"{LOADER_REGISTRY[fid].__name__}; refusing to shadow."
        )
    LOADER_REGISTRY[fid] = loader_cls


def list_loader_formats() -> List[str]:
    """Return the registered ``format_id`` values in sorted order."""
    return sorted(LOADER_REGISTRY)


def get_loader(format_id: str, **kwargs) -> TerrainLoader:
    """Factory: return a loader instance for ``format_id``.

    ``**kwargs`` are forwarded to the concrete loader's constructor.
    Unknown ``format_id`` raises :exc:`TerrainLoaderError`.
    """
    cls = LOADER_REGISTRY.get(format_id)
    if cls is None:
        raise TerrainLoaderError(
            f"Unknown terrain format_id {format_id!r}. "
            f"Known formats: {list_loader_formats()}"
        )
    return cls(**kwargs)


__all__ = [
    "TerrainLoaderError",
    "TerrainLoader",
    "LOADER_REGISTRY",
    "register_loader",
    "list_loader_formats",
    "get_loader",
]
