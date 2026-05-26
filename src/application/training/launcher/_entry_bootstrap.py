# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Shared subprocess bootstrap — used by sb3_entry / il_train_launcher.

Both training backends spawn a child process that needs to recover the
parent's project context + backend partition before it can route artifacts
under ``<project>/training/{runs,exported}/<backend_id>/`` — the shape
``BundleExporter.export_from_artifacts`` and ``TrainingStore`` expect.

Originally this 7-step ritual was inlined in ``launcher/sb3_entry.py``
lines 161–223. Pulling it here per RELEASE plan §B5 prevents the IL
launcher from re-implementing it (concept drift = silently wrong
``backend_id`` partitioning at write time).

Steps performed by :func:`bootstrap_subprocess`:

    1. ``RegistryHub.load_all()`` — registries (robots / IR / brands /
       backends) so anything calling into them later does not race a
       lazy load.
    2. Read ``project_path`` and ``backend_id`` from the payload.
    3. ``ProjectStore.refresh_snapshot()`` — child has its own singleton.
    4. ``find_by_path`` → ``set_current_project_info(info)`` so
       ``current_project_info()`` returns the right ProjectInfo.
    5. ``set_current_backend(backend_id)`` so ``current_backend()`` and
       any defaults that read from AppSignals see the right value.
    6. Resolve ``run_dir = <project>/training/runs/<backend_id>/<run_id>/``
       and create it (plus ``run_dir/checkpoints``).
    7. Return the resolved triple to caller.

Failure handling is the caller's concern — sb3_entry emits an
``[UnitPort][ERROR]`` line to stdout; the IL launcher logs to stderr.
This module never raises into the caller's process unless the payload
is structurally malformed (TypeError on dict access, etc.).

Reentrance: idempotent. Calling twice with the same payload sets the
same project / backend / run_dir again — useful when a launcher needs
to re-bootstrap after a hot-reload.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class BootstrapResult:
    """What :func:`bootstrap_subprocess` resolved for the child process.

    ``bound_project`` is None when no ``project_path`` was supplied (a
    pre-condition violation in the new flow — main-process tasks reject
    unbound submissions before reaching the subprocess). ``run_dir`` is
    None whenever ``bound_project`` is None; the caller MUST treat that
    as an error rather than synthesising a fallback path.
    """
    bound_project: Optional[Any]   # ProjectInfo when bound
    backend_id: str                # always non-empty (defaulted)
    run_dir: Optional[Path]        # project-scoped run dir or None


def bootstrap_subprocess(
    payload: Dict[str, Any],
    *,
    default_backend: str,
    log_line: Optional[Callable[[str], None]] = None,
) -> BootstrapResult:
    """Run the 7-step subprocess bootstrap shared by SB3 / IL launchers.

    Args:
        payload: parsed stdin / argv payload. Reads ``run_id``,
            ``project_path``, ``backend_id`` keys (all string,
            optional except ``run_id``).
        default_backend: backend_id to use when payload lacks one (e.g.
            ``"sb3_mujoco"`` or ``"isaac_lab"``).
        log_line: optional one-line log sink — sb3_entry passes its
            ``_emit_line`` so the parent's MSG_LOG path captures the
            bootstrap trace verbatim. None = silent.

    Returns:
        :class:`BootstrapResult` with whatever could be resolved. None
        fields signal "skipped this step because input was missing"
        (e.g. no project_path → bound_project None → run_dir None).
    """
    def _log(msg: str) -> None:
        if log_line is not None:
            try:
                log_line(msg)
            except Exception:
                pass

    # Step 1: RegistryHub load — needed before any robots/ir/backends
    # lookup. Failure is fatal because BundleExporter et al. depend on it.
    try:
        from registers import RegistryHub
        RegistryHub.load_all()
    except Exception as exc:
        _log(f"[UnitPort] _entry_bootstrap: RegistryHub.load_all failed: {exc}")
        # Re-raise — caller decides how to surface (sb3_entry returns 3,
        # IL launcher lets traceback propagate to its top-level except).
        raise

    # Step 2: extract identifiers from payload
    run_id = str(payload.get("run_id") or "").strip()
    project_path_str = str(payload.get("project_path") or "").strip()
    backend_id = str(payload.get("backend_id") or "").strip() or default_backend

    bound_project: Optional[Any] = None
    run_dir: Optional[Path] = None

    if project_path_str:
        # Step 3 + 4: rehydrate ProjectStore singleton in this process
        # and bind the project so current_project_info() returns it.
        try:
            from application.service.projects import (
                get_project_store,
                set_current_project_info,
            )
            ps = get_project_store()
            ps.refresh_snapshot()
            info = ps.find_by_path(Path(project_path_str))
            if info is not None:
                set_current_project_info(info)
                bound_project = info
                _log(
                    f"[UnitPort] _entry_bootstrap: project={info.name} "
                    f"backend={backend_id}"
                )
            else:
                _log(
                    f"[UnitPort] _entry_bootstrap: project_path "
                    f"{project_path_str!r} not found in ProjectStore "
                    f"snapshot — running unbound"
                )
        except Exception as exc:
            _log(f"[UnitPort] _entry_bootstrap: project bind skipped: {exc}")

    # Step 5: AppSignals.current_backend() — BundleExporter and similar
    # callees read this when callers forget to pass backend_id explicitly.
    # We always set it so cross-process drift is impossible.
    try:
        from application.service.signals import set_current_backend
        set_current_backend(backend_id)
    except Exception as exc:
        _log(f"[UnitPort] _entry_bootstrap: set_current_backend skipped: {exc}")

    # Step 6 + 7: resolve project-scoped run_dir + mkdir.
    if bound_project is not None and run_id:
        try:
            run_dir = (
                bound_project.path / "training" / "runs" / backend_id / run_id
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            _log(f"[UnitPort] _entry_bootstrap: run_dir={run_dir}")
        except Exception as exc:
            _log(
                f"[UnitPort] _entry_bootstrap: run_dir resolve skipped: {exc}"
            )
            run_dir = None

    return BootstrapResult(
        bound_project=bound_project,
        backend_id=backend_id,
        run_dir=run_dir,
    )


__all__ = ["BootstrapResult", "bootstrap_subprocess"]
