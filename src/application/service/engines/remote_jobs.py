# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Remote-training job records — local registry of cloud-submitted runs.

When the user trains with the Cloud target (``EngineService.get_link_target()
== "cloud"``), :class:`RemoteSubmitTask` pushes the run to an SSH server and
gets back a job handle (remote PID + remote run directory). That handle is the
ONLY way to later poll status / pull artifacts, so it is persisted here rather
than living only in the canvas or a transient Task object.

Storage: ``<USER_CONFIG_DIR>/engines/remote_jobs.json`` — a versioned envelope::

    {"schema_version": 1, "jobs": [ {RemoteJobRecord as dict}, ... ]}

Per-user (rides the workspace), and included in the cloud-sync include set so a
user who switches machines keeps the handle list.

This module is the single source of truth for the remote-job schema; the submit
Task and any future poll / pull Task read and write through these helpers only.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from unitport_sdk import Paths, log_warning, push_data, read_data

SCHEMA_VERSION = 1
_REL = "engines/remote_jobs.json"

# Allowed status values. ``submitted`` is the only one written today; the rest
# are the vocabulary the deferred poll/pull Task will transition through.
STATUS_SUBMITTED = "submitted"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"


@dataclass
class RemoteJobRecord:
    """One cloud-submitted training run's handle. Mirrors the JSON envelope row."""

    run_id: str
    engine_id: str              # registry engine id, e.g. "isaac_lab"
    engine_name: str            # spec engine, "isaac_lab" | "sb3_mujoco"
    server_name: str            # EngineService server name — reverse-lookup creds
    host: str                   # redundant copy so UI renders even if server deleted
    remote_run_dir: str         # absolute path of the run dir ON the remote host
    remote_pid: int             # job handle (poll / kill)
    status: str                 # one of the STATUS_* constants
    submitted_at: str           # ISO8601 UTC
    updated_at: str = ""        # set by the deferred poll Task
    last_error: str = ""        # populated on a failed transition


def _path() -> Path:
    return Paths.USER_CONFIG_DIR / "engines" / "remote_jobs.json"


def now_iso() -> str:
    """ISO8601 UTC timestamp (seconds precision, ``Z`` suffix)."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _empty_envelope() -> dict:
    return {"schema_version": SCHEMA_VERSION, "jobs": []}


def _load_envelope() -> dict:
    p = _path()
    if not p.exists():
        return _empty_envelope()
    try:
        data = read_data(p)
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[remote-jobs] read failed ({p}): {exc}")
        return _empty_envelope()
    if not isinstance(data, dict):
        return _empty_envelope()
    data.setdefault("schema_version", SCHEMA_VERSION)
    if not isinstance(data.get("jobs"), list):
        data["jobs"] = []
    return data


def _record_from_dict(d: dict) -> Optional[RemoteJobRecord]:
    try:
        return RemoteJobRecord(
            run_id=str(d.get("run_id") or ""),
            engine_id=str(d.get("engine_id") or ""),
            engine_name=str(d.get("engine_name") or ""),
            server_name=str(d.get("server_name") or ""),
            host=str(d.get("host") or ""),
            remote_run_dir=str(d.get("remote_run_dir") or ""),
            remote_pid=int(d.get("remote_pid") or 0),
            status=str(d.get("status") or STATUS_UNKNOWN),
            submitted_at=str(d.get("submitted_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
            last_error=str(d.get("last_error") or ""),
        )
    except (TypeError, ValueError):
        return None


def list_remote_jobs() -> List[RemoteJobRecord]:
    """Return all persisted job records (newest-appended last)."""
    out: List[RemoteJobRecord] = []
    for d in _load_envelope().get("jobs", []):
        if isinstance(d, dict):
            rec = _record_from_dict(d)
            if rec is not None:
                out.append(rec)
    return out


def get_remote_job(run_id: str) -> Optional[RemoteJobRecord]:
    """Return the record for ``run_id`` or None."""
    for rec in list_remote_jobs():
        if rec.run_id == run_id:
            return rec
    return None


def append_remote_job(rec: RemoteJobRecord) -> bool:
    """Append ``rec`` and persist. Returns False (loud-warned) on write failure.

    The caller (RemoteSubmitTask) has ALREADY launched the remote job by the
    time this runs; a persist failure loses the handle but must NOT be raised
    as a submit failure — the job is genuinely running. Hence False, not raise.
    """
    env = _load_envelope()
    env["jobs"].append(asdict(rec))
    env["schema_version"] = SCHEMA_VERSION
    if not push_data(_REL, env):
        log_warning(
            f"[remote-jobs] persist failed for run_id={rec.run_id!r} — remote "
            f"job is running (pid={rec.remote_pid}) but its handle was lost"
        )
        return False
    return True


def update_remote_job(run_id: str, **fields) -> bool:
    """Patch ``fields`` onto the record for ``run_id``; stamps ``updated_at``."""
    env = _load_envelope()
    found = False
    for d in env["jobs"]:
        if isinstance(d, dict) and d.get("run_id") == run_id:
            d.update(fields)
            d["updated_at"] = now_iso()
            found = True
            break
    if not found:
        log_warning(f"[remote-jobs] update: run_id={run_id!r} not found")
        return False
    if not push_data(_REL, env):
        log_warning(f"[remote-jobs] update persist failed for run_id={run_id!r}")
        return False
    return True


__all__ = [
    "RemoteJobRecord",
    "SCHEMA_VERSION",
    "STATUS_SUBMITTED",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
    "STATUS_UNKNOWN",
    "now_iso",
    "list_remote_jobs",
    "get_remote_job",
    "append_remote_job",
    "update_remote_job",
]
