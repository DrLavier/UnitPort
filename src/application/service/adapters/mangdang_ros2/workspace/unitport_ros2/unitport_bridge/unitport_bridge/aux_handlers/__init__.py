"""Auxiliary-actuator handler registry for the robot-side bridge.

Generic adapters under ``unitport_bridge.adapters.*`` cover the kinematic
surfaces (servos, wheels, trajectories). Everything else — LED panels,
LCD screens, speakers, rumble motors, push-button LEDs — goes through
the :class:`BaseAuxHandler` interface here, so brand-specific driver
code can live under ``aux_handlers/*`` without bloating the adapters.

The handler for a given robot is chosen at bridge startup by
:func:`resolve_aux_handler`. Priority order:

    1. Explicit ``aux_handler`` path in the profile (takes an import string
       like ``unitport_bridge.aux_handlers.mini_pupper_handler.MiniPupperAuxHandler``).
    2. Brand-id look-up in ``_BRAND_TO_HANDLER`` below.
    3. Fallback to :class:`NullAuxHandler` (log-only).

Handlers receive the parsed JSON payload and are free to drop, queue, or
synchronously drive a vendor library. They should never raise — the
bridge node catches but will log at WARN, not FATAL.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class BaseAuxHandler:
    """Interface for surface-specific aux handlers.

    Subclasses override :meth:`handle`. Constructors accept ``profile`` +
    ``mapping`` so vendor drivers can key off user-confirmed settings
    (e.g. which of the two RGB LEDs counts as "left").
    """

    def __init__(
        self,
        *,
        profile: Dict[str, Any],
        mapping: Dict[str, Any],
        logger: Any = None,
    ) -> None:
        self._profile = dict(profile or {})
        self._mapping = dict(mapping or {})
        self._logger = logger

    def handle(self, *, surface: str, target: str, payload: Dict[str, Any]) -> None:
        """Dispatch a single AuxCommand. Must not raise.

        Default implementation logs and drops — subclasses SHOULD override.
        """
        self._log_info(
            f"aux[{type(self).__name__}] surface={surface!r} "
            f"target={target!r} payload={payload!r} (no-op)"
        )

    # ------------------------------------------------------------------
    # Helpers — shared by subclasses to keep logging consistent.
    # ------------------------------------------------------------------

    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.info(msg)
                return
            except Exception:
                pass
        print(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.warn(msg)
                return
            except Exception:
                pass
        print(msg)


class NullAuxHandler(BaseAuxHandler):
    """Fallback handler — logs each command and drops it."""

    def handle(self, *, surface: str, target: str, payload: Dict[str, Any]) -> None:
        self._log_info(
            f"aux[null] surface={surface!r} target={target!r} payload={payload!r}"
        )


# Brand-id → dotted import path. The host sets ``profile.brand_id`` via the
# brand overlay step; the robot's copy of the profile carries it through.
# Entries here don't import at module load time — resolution is lazy so a
# missing vendor package doesn't break non-branded robots.
_BRAND_TO_HANDLER: Dict[str, str] = {
    "mangdang_mini_pupper_v2":
        "unitport_bridge.aux_handlers.mini_pupper_handler.MiniPupperAuxHandler",
}


def resolve_aux_handler(
    *,
    brand_id: str,
    profile: Dict[str, Any],
    mapping: Dict[str, Any],
    logger: Any = None,
) -> BaseAuxHandler:
    """Pick and instantiate an aux handler for this robot.

    Never raises — an unresolvable brand falls through to NullAuxHandler,
    with a WARN log so the failure is visible in `ros2 node info`.
    """
    dotted = str((profile or {}).get("aux_handler") or "")
    if not dotted:
        dotted = _BRAND_TO_HANDLER.get(str(brand_id or ""), "")
    if not dotted:
        return NullAuxHandler(profile=profile, mapping=mapping, logger=logger)

    try:
        module_path, _, class_name = dotted.rpartition(".")
        if not module_path:
            raise ValueError(f"bad aux_handler path: {dotted!r}")
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls(profile=profile, mapping=mapping, logger=logger)
    except Exception as exc:
        if logger is not None:
            try:
                logger.warn(
                    f"aux_handler {dotted!r} failed to load: {exc}; "
                    "falling back to NullAuxHandler"
                )
            except Exception:
                pass
        return NullAuxHandler(profile=profile, mapping=mapping, logger=logger)


__all__ = [
    "BaseAuxHandler",
    "NullAuxHandler",
    "resolve_aux_handler",
]
