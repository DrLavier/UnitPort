"""Run-id-prefixed logging helpers so CmdLogWidget can filter by run.

Use these from inside ``TrainingTask.run`` and from the Isaac Lab
subprocess log tail. They marshal to the main thread via the SDK's
``LogSignal`` like every other ``log_*`` call — safe from any thread.
"""

from __future__ import annotations

from unitport_sdk import log_debug, log_error, log_info, log_success, log_warning


def _fmt(run_id: str, msg: str) -> str:
    return f"[{run_id}] {msg}" if run_id else msg


def train_log_info(run_id: str, msg: str) -> None:
    log_info(_fmt(run_id, msg))


def train_log_warning(run_id: str, msg: str) -> None:
    log_warning(_fmt(run_id, msg))


def train_log_error(run_id: str, msg: str) -> None:
    log_error(_fmt(run_id, msg))


def train_log_success(run_id: str, msg: str) -> None:
    log_success(_fmt(run_id, msg))


def train_log_debug(run_id: str, msg: str) -> None:
    log_debug(_fmt(run_id, msg))


__all__ = [
    "train_log_info",
    "train_log_warning",
    "train_log_error",
    "train_log_success",
    "train_log_debug",
]
