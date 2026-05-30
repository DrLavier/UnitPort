# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.remote_backend — Cloud (remote) training backends.

Parallel to :mod:`application.training.backend` (the LOCAL runtime registry),
kept as a SEPARATE registry on purpose: "where it runs" (local vs cloud) is
orthogonal to "what runs it" (the engine: isaac_lab / sb3_mujoco). Folding cloud
into the local ``select_backend("auto")`` chain would poison its
``is_available()`` semantics (a local availability probe says nothing about
whether a remote server can run the job).

A ``RemoteTrainingBackend`` is a thin factory: ``build_task`` returns a
:class:`application.training.remote.submit_task.RemoteSubmitTask` that performs
the SSH submission. The engine the spec targets is carried as ``engine_name`` so
one Task class handles every engine (the differences are remote-launch flags,
not separate Python paths).

Registry is keyed by ``"remote:<engine_name>"``. ``select_remote_backend`` maps
a resolved engine name to its remote backend. ``ensure_remote_backends``
registers the built-ins (Isaac Lab active; SB3 reserved for parity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from unitport_sdk import log_debug, log_warning

if TYPE_CHECKING:
    from application.training._sdk.task_runner import TrainingTask


# Engine names this maps onto (mirror application.training.backend constants).
ENGINE_ISAAC_LAB = "isaac_lab"
ENGINE_SB3_MUJOCO = "sb3_mujoco"


def remote_name_for_engine(engine_name: str) -> str:
    """Canonical remote-backend registry key for an engine name."""
    return f"remote:{engine_name}"


# ---------------------------------------------------------------------------
# RemoteTrainingBackend ABC
# ---------------------------------------------------------------------------


class RemoteTrainingBackend(ABC):
    """Hot-pluggable cloud submission backend for one engine.

    Concrete subclasses MUST set ``name`` to ``"remote:<engine_name>"`` and
    ``engine_name`` to the matching engine constant.
    """

    name: str = ""
    engine_name: str = ""

    @abstractmethod
    def build_task(self, spec: Dict, run_id: str) -> "TrainingTask":
        """Construct the SSH-submit Task for ``spec``.

        ``spec`` is the ``TrainingSpec.to_dict()`` payload. Raises
        ``RemoteSubmitConfigError`` synchronously (no project / no server /
        missing credential) so the Play button surfaces it as a dialog before
        the Task is ever submitted.
        """

    def description(self) -> str:
        return self.__doc__ or self.name


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, RemoteTrainingBackend] = {}


def register_remote_backend(backend: RemoteTrainingBackend) -> None:
    """Register under ``backend.name``. Re-registering replaces."""
    if not backend.name:
        raise ValueError(f"remote backend {backend!r} has empty .name")
    if backend.name in _REGISTRY:
        log_warning(f"[training.remote_backend] re-registering '{backend.name}'")
    _REGISTRY[backend.name] = backend
    log_debug(f"[training.remote_backend] registered: {backend.name}")


def get_remote_backend(name: str) -> Optional[RemoteTrainingBackend]:
    return _REGISTRY.get(name)


def list_remote_backends() -> List[RemoteTrainingBackend]:
    return list(_REGISTRY.values())


def select_remote_backend(engine_name: str) -> RemoteTrainingBackend:
    """Resolve the remote backend for a concrete engine name.

    Raises:
        ValueError: no remote backend registered for ``engine_name``.
    """
    key = remote_name_for_engine((engine_name or "").strip())
    backend = _REGISTRY.get(key)
    if backend is None:
        raise ValueError(
            f"No remote backend for engine {engine_name!r} "
            f"(key {key!r}); registered={list(_REGISTRY)}. "
            "Call ensure_remote_backends() first."
        )
    return backend


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


class RemoteIsaacLabAdapter(RemoteTrainingBackend):
    """Submit an Isaac Lab spec to a cloud SSH server."""

    name = remote_name_for_engine(ENGINE_ISAAC_LAB)
    engine_name = ENGINE_ISAAC_LAB

    def build_task(self, spec: Dict, run_id: str) -> "TrainingTask":
        from application.training.remote.submit_task import RemoteSubmitTask

        return RemoteSubmitTask(
            spec, engine_name=self.engine_name, run_id=run_id,
        )


class RemoteSB3Adapter(RemoteTrainingBackend):
    """Submit an SB3+MuJoCo spec to a cloud SSH server (parity placeholder).

    Wired through the same RemoteSubmitTask; the remote entrypoint dispatches
    on ``--engine sb3_mujoco``. Registered for symmetry so a future SB3 cloud
    path needs no new backend plumbing.
    """

    name = remote_name_for_engine(ENGINE_SB3_MUJOCO)
    engine_name = ENGINE_SB3_MUJOCO

    def build_task(self, spec: Dict, run_id: str) -> "TrainingTask":
        from application.training.remote.submit_task import RemoteSubmitTask

        return RemoteSubmitTask(
            spec, engine_name=self.engine_name, run_id=run_id,
        )


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------


def ensure_remote_backends() -> None:
    """Register the built-in remote adapters. Idempotent."""
    if remote_name_for_engine(ENGINE_ISAAC_LAB) not in _REGISTRY:
        register_remote_backend(RemoteIsaacLabAdapter())
    if remote_name_for_engine(ENGINE_SB3_MUJOCO) not in _REGISTRY:
        register_remote_backend(RemoteSB3Adapter())


__all__ = [
    "ENGINE_ISAAC_LAB",
    "ENGINE_SB3_MUJOCO",
    "RemoteTrainingBackend",
    "RemoteIsaacLabAdapter",
    "RemoteSB3Adapter",
    "register_remote_backend",
    "get_remote_backend",
    "list_remote_backends",
    "select_remote_backend",
    "ensure_remote_backends",
    "remote_name_for_engine",
]
