# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Hardware-adaptive SB3 vectorisation planning.

The SB3 backend runs ``n_envs`` MuJoCo environments in parallel. Too few
wastes the machine; too many oversubscribes the CPU or exhausts RAM (each
``SubprocVecEnv`` worker is a full Python process). The right number is a
function of the *training machine's* hardware, which is not knowable when the
canvas is authored — especially for cloud runs where the worker differs from
the editor.

So the canvas carries a **mode** (``auto`` / ``manual``), and this module
resolves it into a concrete ``(n_envs, vec_type)`` plan at launch time, on the
box that actually trains. ``manual`` passes the user's canvas values through
unchanged; ``auto`` derives them from CPU core count + available RAM.

This is the single source of truth for the plan — both the launcher
(``sb3_trainer.train_sb3``) and the UI BAR1 preflight estimate call
``resolve_parallelism`` so the displayed budget matches what will run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from unitport_sdk import log_info, log_warning


# ---------------------------------------------------------------------------
# AUTO heuristic constants — tunable in ONE place (no magic numbers scattered
# across the trainer). Changing these changes the machine-adaptive policy.
# ---------------------------------------------------------------------------

# Cores held back from worker envs for the main trainer process (PPO gradient
# update + rollout collection runs there) and OS responsiveness. Adaptive:
# a 2/4-core laptop can only spare 1; a workstation spares 2.
_RESERVE_CORES_SMALL = 1   # logical cores <= _SMALL_CORE_THRESHOLD
_RESERVE_CORES_LARGE = 2   # logical cores >  _SMALL_CORE_THRESHOLD
_SMALL_CORE_THRESHOLD = 4

# Estimated peak RAM per SubprocVecEnv worker (GB). A spawn worker re-imports
# numpy + mujoco + the env modules and holds one MjModel/MjData — NOT torch
# (the policy lives only in the main process). Conservative so we under- rather
# than over-subscribe RAM on tight machines.
_PER_WORKER_RAM_GB = 0.5

# Fraction of *available* (not total) RAM we are willing to commit to workers.
# The remainder cushions the main process (torch + replay/rollout buffers) and
# the OS file cache.
_RAM_SAFETY_FRACTION = 0.7

# Upper bound on auto-selected envs. SB3's PPO gradient step is serial on the
# main process, so CPU-MuJoCo throughput sees sharply diminishing returns past
# this; the cap also avoids spawning a pathological number of processes on
# many-core servers. Manual mode is NOT capped (the user owns that choice).
_MAX_AUTO_ENVS = 32

_MIN_ENVS = 1


@dataclass(frozen=True)
class HardwareInfo:
    """Detected training-machine capacity."""

    logical_cores: int
    physical_cores: int
    total_ram_gb: float
    avail_ram_gb: float
    ram_detected: bool  # False ⇒ psutil unavailable, RAM cap skipped

    def describe(self) -> str:
        ram = (
            f"{self.avail_ram_gb:.1f}/{self.total_ram_gb:.1f} GB avail/total"
            if self.ram_detected else "RAM: unknown (psutil unavailable)"
        )
        return (
            f"{self.logical_cores} logical / {self.physical_cores} physical "
            f"cores, {ram}"
        )


@dataclass(frozen=True)
class ParallelismPlan:
    """Resolved vectorisation plan for one training run."""

    n_envs: int
    vec_type: str            # "dummy" | "subproc"
    mode: str                # "auto" | "manual"
    rationale: str           # human-readable, logged at launch
    hardware: HardwareInfo


def detect_hardware() -> HardwareInfo:
    """Probe CPU core count + RAM via psutil (with an os-only fallback).

    psutil is a hard dependency (requirements.txt), but we treat it as an
    optional import (CLAUDE.md §8(a)): if it is somehow absent we still report
    core count via ``os.cpu_count()`` and skip the RAM cap, logging a WARN.
    """
    logical = os.cpu_count() or 1
    physical = logical
    total_gb = 0.0
    avail_gb = 0.0
    ram_detected = False
    try:
        import psutil  # noqa: WPS433

        logical = int(psutil.cpu_count(logical=True) or logical)
        physical = int(psutil.cpu_count(logical=False) or logical)
        vm = psutil.virtual_memory()
        total_gb = float(vm.total) / 1e9
        avail_gb = float(vm.available) / 1e9
        ram_detected = True
    except Exception as exc:  # noqa: BLE001
        # WHY KEPT: §8(a) optional-import branch. psutil is in requirements, so
        # this is a defensive degrade (cpu-only auto), not a silent correctness
        # fallback — it is loudly logged and only widens the plan's RAM headroom
        # assumption (cores still cap it).
        log_warning(
            f"[auto_parallelism] psutil unavailable ({exc!r}); auto n_envs will "
            f"use CPU cores only (no RAM cap). Install psutil for RAM-aware "
            f"tuning."
        )
    return HardwareInfo(
        logical_cores=max(1, logical),
        physical_cores=max(1, physical),
        total_ram_gb=total_gb,
        avail_ram_gb=avail_gb,
        ram_detected=ram_detected,
    )


def _auto_n_envs(hw: HardwareInfo) -> tuple[int, str]:
    """Return ``(n_envs, rationale)`` for the auto policy on ``hw``."""
    reserve = (
        _RESERVE_CORES_SMALL
        if hw.logical_cores <= _SMALL_CORE_THRESHOLD
        else _RESERVE_CORES_LARGE
    )
    cpu_cap = max(_MIN_ENVS, hw.logical_cores - reserve)

    parts = [f"cpu_cap={cpu_cap} ({hw.logical_cores} cores - {reserve} reserved)"]
    n = cpu_cap
    if hw.ram_detected:
        ram_cap = max(
            _MIN_ENVS,
            int((hw.avail_ram_gb * _RAM_SAFETY_FRACTION) / _PER_WORKER_RAM_GB),
        )
        parts.append(
            f"ram_cap={ram_cap} ({hw.avail_ram_gb:.1f}GB x {_RAM_SAFETY_FRACTION}"
            f"/{_PER_WORKER_RAM_GB}GB per worker)"
        )
        n = min(n, ram_cap)
    n = min(n, _MAX_AUTO_ENVS)
    if n == _MAX_AUTO_ENVS:
        parts.append(f"clamped to max_auto={_MAX_AUTO_ENVS}")
    n = max(_MIN_ENVS, n)
    return n, "; ".join(parts)


def resolve_parallelism(
    *,
    mode: str,
    manual_n_envs: int,
    manual_vec_type: str,
    hardware: Optional[HardwareInfo] = None,
) -> ParallelismPlan:
    """Resolve the canvas parallelism settings into a concrete plan.

    Parameters
    ----------
    mode:
        ``"auto"`` (hardware-derived) or ``"manual"`` (use the canvas values).
    manual_n_envs / manual_vec_type:
        The canvas ``env_assembler.n_envs`` / ``.vec_type`` — used verbatim in
        manual mode, and (n_envs) as a sanity floor reference in auto.
    hardware:
        Pre-detected :class:`HardwareInfo`; detected here when omitted.

    Notes
    -----
    ``vec_type`` is forced to ``"dummy"`` whenever ``n_envs == 1`` — a single
    worker never benefits from the SubprocVecEnv spawn + IPC overhead, and
    in-process running keeps eval/video rendering simple.
    """
    hw = hardware or detect_hardware()
    mode_norm = (mode or "auto").strip().lower()

    if mode_norm == "manual":
        n = max(_MIN_ENVS, int(manual_n_envs))
        vt = (manual_vec_type or "subproc").strip().lower()
        if vt not in ("dummy", "subproc"):
            raise ValueError(
                f"[auto_parallelism] manual vec_type={manual_vec_type!r} is not "
                f"'dummy' or 'subproc'. Fix env_assembler.vec_type (CLAUDE.md §8)."
            )
        if n == 1:
            vt = "dummy"
        rationale = f"manual: n_envs={n}, vec_type={vt}"
        return ParallelismPlan(
            n_envs=n, vec_type=vt, mode="manual", rationale=rationale, hardware=hw,
        )

    if mode_norm != "auto":
        raise ValueError(
            f"[auto_parallelism] parallelism_mode={mode!r} is not 'auto' or "
            f"'manual'. Fix env_assembler.parallelism_mode (CLAUDE.md §8)."
        )

    n, why = _auto_n_envs(hw)
    vt = "dummy" if n <= 1 else "subproc"
    rationale = f"auto -> n_envs={n}, vec_type={vt} [{why}] on {hw.describe()}"
    if n == 1:
        log_warning(
            "[auto_parallelism] auto resolved to n_envs=1 on an underpowered "
            "machine — robot RL trains very slowly with a single env. Consider "
            "a machine with more cores/RAM, or cloud training."
        )
    else:
        log_info(f"[auto_parallelism] {rationale}")
    return ParallelismPlan(
        n_envs=n, vec_type=vt, mode="auto", rationale=rationale, hardware=hw,
    )


__all__ = [
    "HardwareInfo",
    "ParallelismPlan",
    "detect_hardware",
    "resolve_parallelism",
]
