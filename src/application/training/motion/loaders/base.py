# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Motion loader base: ABC + error type + the public dispatcher.

This module is data-free. It declares the contract every concrete loader
in this subpackage implements:

  * :exc:`LoaderError` — single exception type raised on any parse,
    structural, or registration failure.
  * :class:`MotionLoader` — abstract ``(path) → ReferenceMotionContract``
    interface plus the ``format_id`` and ``default_consumption_mode``
    class attributes that pin a loader's identity in the registry.
  * :data:`LOADER_REGISTRY` — public ``{format_id: loader_class}`` dict
    populated by the subpackage's ``__init__`` via explicit
    :func:`register_loader` calls (no class side-effects).
  * :func:`register_loader` / :func:`list_loader_formats` /
    :func:`get_loader` — the dispatcher surface.

Why no concrete loader lives in this file: this module is imported by
every other loader file, so keeping it free of format-specific logic
prevents cycles when, e.g., the AMP loader needs to import the base
class but the augment module wants helpers from the AMP loader.

I/O note: concrete loaders read user-supplied paths (project tree /
user data dir). ``np.load`` and ``json.load`` are the right tools —
wrapping them in ``DataManager`` would round-trip arrays through dtype
conversions the SDK doesn't model. Stays consistent with DataManager's
intent (per-extension dispatch for SDK-managed files; arbitrary user
paths can use the canonical Python loaders).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Union

from application.training.motion.contract import ReferenceMotionContract


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LoaderError(ValueError):
    """Raised when a motion file cannot be parsed by the requested loader,
    or when a registry operation (register / get) is invalid."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MotionLoader(ABC):
    """Abstract ``(path) → ReferenceMotionContract`` contract."""

    #: Stable identifier used by :func:`get_loader` and ``LOADER_REGISTRY``.
    format_id: str = ""

    #: Default consumption mode the loader stamps onto every contract it
    #: builds. AMP-style clips default to ``"amp_discriminator"``;
    #: loaders producing tracking-only clips override this once that
    #: consumer path lands.
    default_consumption_mode: str = "amp_discriminator"

    @abstractmethod
    def load(
        self,
        path: Union[Path, str],
        *,
        target_sku: str,
        target_family: str,
    ) -> ReferenceMotionContract:
        """Read ``path`` and return a :class:`ReferenceMotionContract`.

        ``target_sku`` and ``target_family`` are keyword-only and
        **required**: the caller passes the canonical robot SKU and
        primary family slug for the robot this clip is being prepared
        for. The loader propagates them onto the returned contract;
        downstream consumers cross-check joint roles against the
        registry's IR slice for that family. The loader does **not**
        reverse-look these from the motion file or the wrapped clip
        (that would re-introduce the asset_id-style implicit identity
        the subprocess boundary tightening removed).

        Raises :exc:`LoaderError` on any parse or structural failure
        and :exc:`application.training.motion.contract.
        ReferenceMotionContractError` when ``target_sku`` /
        ``target_family`` are empty (the contract's ``__post_init__``
        enforces non-empty). Does NOT run :func:`validate_clip` —
        that is the caller's responsibility (separation of concerns:
        loaders produce data, the validator cross-checks against a
        robot spec).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


#: Public ``{format_id: loader_class}`` registry. Populated by
#: ``application.training.motion.loaders.__init__`` via explicit
#: :func:`register_loader` calls. The mapping is mutable so plugins
#: (5A-β alignment workstream, 5A-γ humanoid loader) can install
#: themselves without modifying this module.
LOADER_REGISTRY: Dict[str, Type[MotionLoader]] = {}


def register_loader(loader_cls: Type[MotionLoader]) -> None:
    """Add a loader class to :data:`LOADER_REGISTRY`.

    Re-registering an existing ``format_id`` raises :exc:`LoaderError`
    — silently shadowing a built-in loader makes debugging downstream
    schema mismatches unnecessarily hard.
    """
    if not isinstance(loader_cls, type) or not issubclass(loader_cls, MotionLoader):
        raise LoaderError(
            f"register_loader: {loader_cls!r} is not a MotionLoader subclass"
        )
    fid = getattr(loader_cls, "format_id", "")
    if not fid:
        raise LoaderError(
            f"register_loader: {loader_cls.__name__} has no non-empty "
            f"format_id class attribute"
        )
    if fid in LOADER_REGISTRY:
        raise LoaderError(
            f"register_loader: format_id {fid!r} is already registered to "
            f"{LOADER_REGISTRY[fid].__name__}; refusing to shadow."
        )
    LOADER_REGISTRY[fid] = loader_cls


def list_loader_formats() -> List[str]:
    """Return the registered ``format_id`` values in sorted order."""
    return sorted(LOADER_REGISTRY)


def get_loader(format_id: str, **kwargs) -> MotionLoader:
    """Factory: return a loader instance for ``format_id``.

    ``**kwargs`` are forwarded to the concrete loader's constructor.
    Unknown ``format_id`` raises :exc:`LoaderError`.
    """
    cls = LOADER_REGISTRY.get(format_id)
    if cls is None:
        raise LoaderError(
            f"Unknown motion format_id {format_id!r}. "
            f"Known formats: {list_loader_formats()}"
        )
    return cls(**kwargs)


__all__ = [
    "LoaderError",
    "MotionLoader",
    "LOADER_REGISTRY",
    "register_loader",
    "list_loader_formats",
    "get_loader",
]
