# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.backend — Training backend abstraction.

Lets ``il_ppo_trainer`` / ``amp_trainer`` nodes route a compiled training
spec to one of several runtimes (Isaac Lab, SB3 + MuJoCo, …) without
hard-coding which one is used.

Stage 1 scope:
    * Define ``TrainingBackend`` ABC.
    * Provide ``register_backend`` / ``get_backend`` / ``list_available_backends``
      / ``select_backend`` global helpers.
    * Register a thin ``IsaacLabBackendAdapter`` that wraps the existing
      ``IsaacLabTrainingTask`` (already shipped under
      ``application.training.isaac_lab``).
    * Reserve ``"sb3_mujoco"`` as a known backend name; the SB3 adapter
      lands in Stage 10 alongside the SB3 launcher / Task pair.

The ``algorithm_config`` node exposes a ``backend`` enum
(``auto`` / ``isaac_lab`` / ``sb3_mujoco``); ``select_backend`` reads that
preference and dispatches accordingly.

Backends are not auto-registered on import; callers (or app bootstrap)
register them explicitly via ``register_backend``. ``ensure_default_backends``
is a one-call helper that registers the built-in adapters in the canonical
order (Isaac Lab first, SB3 second). Either approach is safe with respect
to Stage 1's contract — the registry is idempotent on re-register.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from unitport_sdk import log_debug, log_warning

if TYPE_CHECKING:
    from application.training._sdk.task_runner import TrainingTask


# ---------------------------------------------------------------------------
# Canonical backend names
# ---------------------------------------------------------------------------

BACKEND_ISAAC_LAB = "isaac_lab"
BACKEND_SB3_MUJOCO = "sb3_mujoco"
BACKEND_AUTO = "auto"

# Selector preference order when ``backend == "auto"``. Edit this list to
# change the auto-fallback chain. Isaac Lab is preferred because it has
# already been wired end-to-end (real PPO runs land bundles).
_AUTO_PREFERENCE: tuple = (BACKEND_ISAAC_LAB, BACKEND_SB3_MUJOCO)


# ---------------------------------------------------------------------------
# TrainingBackend ABC
# ---------------------------------------------------------------------------

class TrainingBackend(ABC):
    """Hot-pluggable training runtime.

    A concrete backend wraps one (sim engine, RL framework) pair behind
    the ``build_task`` factory. The Task it returns runs synchronously
    on a SDK ``TasksManager`` worker thread — for compute-heavy backends
    (SB3, AMP-PPO) the Task itself launches a subprocess + tails its
    stdout (mirroring ``IsaacLabBackend``'s subprocess model).

    Concrete subclasses MUST set ``name`` to one of the canonical
    ``BACKEND_*`` constants so registry / selector logic agrees.
    """

    name: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend's runtime dependencies are installed.

        Implementations should be cheap (one ``importlib.util.find_spec``
        call or one ``registers.backends`` lookup) — this is called every
        time ``select_backend("auto")`` resolves a preference.
        """

    @abstractmethod
    def build_task(
        self,
        spec: Dict,
        run_id: str,
    ) -> "TrainingTask":
        """Construct a SDK Task that, when ``run()``, drives one training run.

        Args:
            spec:    Flat training spec dict (Stage 3 ``TrainingSpec.to_dict``
                     output). The exact schema is the contract between Stage 3
                     compiler and each backend; for Stage 1 we accept any dict
                     and pass through to the adapter.
            run_id:  Unique run identifier (UTC-timestamped slug). Used for
                     ``<project>/training/runs/<backend_id>/<run_id>/`` layout.

        Raises:
            RuntimeError: backend not installed (callers should check
                          ``is_available()`` first or use ``select_backend``).
            ValueError:   spec is missing required fields for this backend.
        """

    # Optional metadata
    def description(self) -> str:
        return self.__doc__ or self.name


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, TrainingBackend] = {}


def register_backend(backend: TrainingBackend) -> None:
    """Register a backend under ``backend.name``. Re-registering replaces."""
    if not backend.name:
        raise ValueError(f"backend {backend!r} has empty .name")
    if backend.name in _REGISTRY:
        log_warning(f"[training.backend] re-registering '{backend.name}'")
    _REGISTRY[backend.name] = backend
    log_debug(f"[training.backend] registered: {backend.name}")


def get_backend(name: str) -> Optional[TrainingBackend]:
    """Return the registered backend with that name, or None."""
    return _REGISTRY.get(name)


def list_backends() -> List[TrainingBackend]:
    """Return all registered backends (order = registration order)."""
    return list(_REGISTRY.values())


def list_available_backends() -> List[TrainingBackend]:
    """Return only backends whose ``is_available()`` returns True."""
    out: List[TrainingBackend] = []
    for b in _REGISTRY.values():
        try:
            if b.is_available():
                out.append(b)
        except Exception as exc:
            log_warning(
                f"[training.backend] {b.name}.is_available() raised: {exc}"
            )
    return out


def select_backend(preference: str = BACKEND_AUTO) -> TrainingBackend:
    """Resolve a backend by user preference or ``"auto"`` fallback.

    Selection rules:
        ``"auto"``       — first available backend in ``_AUTO_PREFERENCE`` order.
        ``"isaac_lab"``  — must be registered AND available, else raises.
        ``"sb3_mujoco"`` — same.

    Raises:
        ValueError:   unknown preference string.
        RuntimeError: requested backend not installed, OR ``"auto"`` and
                      no backend is available.
    """
    pref = (preference or BACKEND_AUTO).strip().lower()

    if pref == BACKEND_AUTO:
        for name in _AUTO_PREFERENCE:
            b = _REGISTRY.get(name)
            if b is None:
                continue
            try:
                if b.is_available():
                    return b
            except Exception:
                continue
        installed = [b.name for b in list_available_backends()]
        raise RuntimeError(
            "No training backend available (auto fallback exhausted). "
            f"Registered={list(_REGISTRY)}; installed={installed}. "
            f"Install Isaac Lab or SB3+MuJoCo, or pin a specific backend."
        )

    if pref not in _REGISTRY:
        raise ValueError(
            f"Unknown backend preference {preference!r}; valid options: "
            f"{[BACKEND_AUTO, *_REGISTRY]}"
        )
    b = _REGISTRY[pref]
    if not b.is_available():
        raise RuntimeError(
            f"Backend {pref!r} is registered but not installed. "
            f"Check `registers.backends.refresh_engine_availability()`."
        )
    return b


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------

class IsaacLabBackendAdapter(TrainingBackend):
    """Adapter over ``application.training.isaac_lab.IsaacLabTrainingTask``.

    ``build_task`` parses the incoming ``spec`` (the nested
    ``TrainingSpec.to_dict()`` shape produced by Stage 3
    :func:`compile_training_spec`) and delegates field extraction to
    :meth:`IsaacLabConfig.from_training_spec`. The flat-key ``spec.get(...)``
    lookups the legacy MVP used silently dropped every canvas value because
    ``to_dict()`` nests under ``task`` / ``env`` / ``algorithm`` — this
    adapter no longer participates in that schema parsing.
    """

    name = BACKEND_ISAAC_LAB

    def is_available(self) -> bool:
        # Primary: registers.backends.installed table (subprocess-probed,
        # root resolved via backends_installed.json::isaac_lab.local_root
        # with PROJECT_ROOT/Engines/isaac_lab/ auto-discovery fallback).
        try:
            from registers import backends as registers_backends
            info = registers_backends.get_installed(BACKEND_ISAAC_LAB)
            if info and info.get("available"):
                return True
        except Exception:
            pass
        # Fallback: in-venv import probe — only useful if Isaac Lab is
        # installed into the same Python that runs Studio (rare; mostly
        # unit-test convenience).
        try:
            import importlib.util
            return importlib.util.find_spec("isaaclab") is not None
        except Exception:
            return False

    def build_task(self, spec: Dict, run_id: str) -> "TrainingTask":
        from application.service.projects import current_project_info
        from application.training.isaac_lab import (
            IsaacLabConfig,
            IsaacLabTrainingTask,
        )

        cfg = IsaacLabConfig.from_training_spec(spec)
        # Phase 3: pass spec + project explicitly so the IL task's
        # finally-block can call ``finalize_isaac_lab_bundle`` with the
        # original TrainingSpec (for robot.brand / joint_names /
        # decimation that IsaacLabConfig does not preserve) and a
        # project reference that does not depend on the AppSignals
        # singleton at finalize time.
        proj = current_project_info()
        return IsaacLabTrainingTask(cfg, run_id=run_id, spec=spec, project=proj)


class SB3MujocoBackendAdapter(TrainingBackend):
    """SB3 + MuJoCo backend — Stage 10 wires this to the subprocess launcher.

    ``is_available`` probes the runtime trio (``stable_baselines3``,
    ``mujoco``, ``gymnasium``) so the auto selector / UI dropdown can
    advertise the option even before refresh_engine_availability has
    run. ``build_task`` returns an :class:`SB3TrainingTask` that the
    SDK ``TasksManager`` drives — Popen + stdin spec + line protocol.
    """

    name = BACKEND_SB3_MUJOCO

    def is_available(self) -> bool:
        try:
            import importlib.util
            return all(
                importlib.util.find_spec(m) is not None
                for m in ("stable_baselines3", "mujoco", "gymnasium")
            )
        except Exception:
            return False

    def build_task(self, spec: Dict, run_id: str) -> "TrainingTask":
        """Build an :class:`SB3TrainingTask`.

        ``spec`` is the :meth:`TrainingSpec.to_dict` output produced by
        Stage 3 :func:`compile_training_spec` — a nested dict keyed by
        ``algorithm`` / ``robot`` / ``env`` / ``rewards`` / etc. The
        launcher round-trips it through :meth:`TrainingSpec.from_dict` so
        the subprocess receives a fully-typed spec.

        Earlier builds also accepted a pre-wrapped ``{"spec": {...}, ...}``
        payload from a transitional caller (now removed). The single wrap
        path here is the only contract — pass-through has been retired.
        """
        from application.training.launcher.sb3_task import SB3TrainingTask

        spec_dict: Dict[str, Any] = dict(spec or {})
        algo = spec_dict.get("algorithm")
        if not isinstance(algo, dict):
            log_warning(
                "[training.backend] sb3_mujoco: spec['algorithm'] is not a dict — "
                "looks like a legacy flat payload from a pre-Stage-12 caller; "
                "the launcher will fall back to TrainingSpec defaults"
            )
        total_timesteps = None
        if isinstance(algo, dict):
            total_timesteps = algo.get("total_timesteps")
        payload = {
            "spec": spec_dict,
            "run_id": run_id,
            "total_timesteps": total_timesteps,
            "export_bundle": True,
        }
        return SB3TrainingTask(payload, run_id=run_id)


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def ensure_default_backends() -> None:
    """Register the built-in adapters in canonical order.

    Idempotent — safe to call from app bootstrap, tests, and node
    execute() paths. Most callers should invoke this once during
    ``UnitPortMain.data_load`` after ``RegistryHub.load_all`` so the
    registers.backends availability table is already populated.
    """
    if BACKEND_ISAAC_LAB not in _REGISTRY:
        register_backend(IsaacLabBackendAdapter())
    if BACKEND_SB3_MUJOCO not in _REGISTRY:
        register_backend(SB3MujocoBackendAdapter())


__all__ = [
    "BACKEND_AUTO",
    "BACKEND_ISAAC_LAB",
    "BACKEND_SB3_MUJOCO",
    "TrainingBackend",
    "IsaacLabBackendAdapter",
    "SB3MujocoBackendAdapter",
    "register_backend",
    "get_backend",
    "list_backends",
    "list_available_backends",
    "select_backend",
    "ensure_default_backends",
]
