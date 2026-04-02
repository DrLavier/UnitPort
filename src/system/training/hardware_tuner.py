#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware auto-tuner for UnitPort Training Ground.

Detects available CPU cores, RAM, and GPU VRAM at the start of a training
run and patches the TrainingJobSpec's env_config / algorithm_config in-place
for maximum throughput — without exceeding the machine's physical limits.

Only overrides values that are still at their compiled defaults — any
parameter the user has explicitly changed is left untouched.

Usage (inside the training subprocess)::

    from src.system.training.hardware_tuner import auto_tune_for_hardware
    auto_tune_for_hardware(spec, log_fn=_log)
    trainer = SB3Trainer(spec, ...)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from src.system.training.training_spec import TrainingJobSpec

LogFn = Callable[[str], None]

# ── Default values from AlgorithmConfig / EnvConfig ──────────────────────────
# Auto-tune only fires when the user-facing value equals these defaults,
# meaning the user has not explicitly customised the parameter.
_DEFAULT_N_ENVS          = 8
_DEFAULT_BATCH_SIZE      = 256
_DEFAULT_LEARNING_STARTS = 10_000
_DEFAULT_N_STEPS         = 2048   # PPO only

# Estimated memory per MuJoCo subprocess (Python overhead + MJCF model data).
# Conservative: Go2 model is ~50 MB data; Python cold-start is ~80–120 MB.
_RAM_PER_ENV_GB = 0.20   # 200 MB per env

# Memory to reserve for the OS, training process, GPU driver, and SB3 model.
_RAM_RESERVE_GB = 3.5


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_system_ram_gb() -> float:
    """
    Return total physical RAM in GB.

    Tries (in order): psutil, Windows GlobalMemoryStatusEx, /proc/meminfo.
    Falls back to a conservative 8.0 GB on any error — never raises.
    """
    # psutil is the cleanest path if present
    try:
        import psutil  # type: ignore[import]
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Windows fallback — ctypes only, no extra deps
    try:
        if os.name == "nt":
            import ctypes
            class _MEMSTATUS(ctypes.Structure):  # noqa: N801
                _fields_ = [
                    ("dwLength",                  ctypes.c_ulong),
                    ("dwMemoryLoad",               ctypes.c_ulong),
                    ("ullTotalPhys",               ctypes.c_ulonglong),
                    ("ullAvailPhys",               ctypes.c_ulonglong),
                    ("ullTotalPageFile",           ctypes.c_ulonglong),
                    ("ullAvailPageFile",           ctypes.c_ulonglong),
                    ("ullTotalVirtual",            ctypes.c_ulonglong),
                    ("ullAvailVirtual",            ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual",    ctypes.c_ulonglong),
                ]
            stat = _MEMSTATUS()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    # Linux / macOS fallback
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0   # conservative fallback


def detect_hardware() -> Tuple[int, float, bool, float, str]:
    """
    Return ``(cpu_count, ram_gb, cuda_available, vram_gb, device_name)``.

    Never raises — every field has a conservative fallback.
    """
    cpu_count = max(1, os.cpu_count() or 4)
    ram_gb    = detect_system_ram_gb()

    cuda_available = False
    vram_gb        = 0.0
    device_name    = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            props       = torch.cuda.get_device_properties(0)
            vram_gb     = props.total_memory / (1024 ** 3)
            device_name = props.name
    except Exception:
        pass

    return cpu_count, ram_gb, cuda_available, vram_gb, device_name


# ---------------------------------------------------------------------------
# Optimal-value calculators
# ---------------------------------------------------------------------------

def _optimal_n_envs(cpu_count: int, ram_gb: float) -> int:
    """
    Map (cpu_count, ram_gb) to an optimal SubprocVecEnv worker count.

    Two independent caps, both applied:

    CPU cap
        ``cpu_count`` — use all available cores for MuJoCo simulation.
        The training thread dispatches CUDA kernels (microsecond bursts) so
        it does not need a dedicated core.  Minimum 2.

    RAM cap
        ``(ram_gb - _RAM_RESERVE_GB) / _RAM_PER_ENV_GB`` — ensures enough
        physical RAM remains for the OS, the training process, and the GPU
        driver after all env subprocesses are spawned.
        _RAM_RESERVE_GB = 3.5 GB covers: Python trainer (~1 GB), CUDA driver
        and model (~1 GB), OS and browser/IDE (~1.5 GB).

    Final result is ``min(cpu_cap, ram_cap)`` clipped to [2, 32].
    """
    cpu_cap = max(2, cpu_count)
    ram_cap = max(2, int((ram_gb - _RAM_RESERVE_GB) / _RAM_PER_ENV_GB))
    return max(2, min(32, cpu_cap, ram_cap))


def _optimal_batch_size(cuda_available: bool, vram_gb: float) -> int:
    """
    Map GPU VRAM to a batch size that keeps the GPU saturated.

    For the tiny policy nets used in legged-robot RL (2×256 MLP, 45-dim obs)
    the GPU is almost always compute-starved at default batch_size=256.
    Larger batches amortise dispatch overhead and fill CUDA SMs properly.

    The batch data itself is tiny (<<1 MB per sample) so VRAM overflow is
    not a concern — this purely targets GPU compute utilisation.

    Tier table:
      ≥ 20 GB  →  8192   (RTX 4090 / A100)
      ≥ 12 GB  →  4096   (RTX 3090 / 4080)
      ≥  6 GB  →  2048   (RTX 3060 Ti / 4060)
      ≥  2 GB  →  1024   (entry CUDA)
      CPU only →   512   (moderate; avoids GIL thrashing)
    """
    if not cuda_available:
        return 512
    if vram_gb >= 20:
        return 8192
    if vram_gb >= 12:
        return 4096
    if vram_gb >= 6:
        return 2048
    return 1024


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def auto_tune_for_hardware(
    spec: "TrainingJobSpec",
    log_fn: Optional[LogFn] = None,
) -> None:
    """
    Detect hardware and patch *spec* in-place for maximum training throughput.

    Safe on any machine
    -------------------
    * ``n_envs`` is bounded by *both* CPU core count and available RAM so the
      auto-tuner can never cause memory exhaustion on a weaker machine.
    * On machines with fewer resources the tuner typically *reduces* n_envs
      from the canvas default of 8 to keep the system stable.
    * All ``batch_size`` values target GPU *compute* utilisation only; the
      actual tensor data is tiny (<< 1 MB per sample) so there is no VRAM
      overflow risk across any supported GPU tier.

    Parameters
    ----------
    spec:
        Fully compiled ``TrainingJobSpec`` — mutated in-place.
    log_fn:
        Optional log-line emitter; receives one string per change.

    Changes applied (only when still at defaults)
    ----------------------------------------------
    ``env_config.n_envs``
        Set to ``min(cpu_cap, ram_cap)`` so every CPU core is busy
        running a parallel MuJoCo simulation without starving RAM.

    ``algorithm_config.batch_size``
        Set according to detected VRAM tier (256 → 8192 on a 4090) so the
        GPU stays saturated between rollout collections.

    ``algorithm_config.n_steps``  (PPO only)
        Raised if the total rollout buffer ``n_steps × n_envs`` is less than
        ``2 × batch_size`` — prevents SB3 from truncating minibatches.

    ``algorithm_config.learning_starts``  (SAC only)
        Raised to ``4 × batch_size`` so the replay buffer has enough diverse
        transitions before the first gradient update.
    """
    env_cfg  = getattr(spec, "env_config", None)
    algo_cfg = getattr(spec, "algorithm_config", None)
    if env_cfg is None or algo_cfg is None:
        return

    cpu_count, ram_gb, cuda_avail, vram_gb, device_name = detect_hardware()
    algo_name = (getattr(algo_cfg, "algorithm", "") or "PPO").upper()
    changes: list[str] = []

    # ── n_envs ────────────────────────────────────────────────────────────
    # Respect the user's n_envs setting.  SubprocVecEnv (when available)
    # handles per-env parallelism via multiprocessing — the auto-tuner
    # should NOT override n_envs, as it controls simulation diversity and
    # sample throughput, both of which are deliberate user choices.
    #
    # If the user left n_envs at the default AND the machine has fewer
    # resources than the default assumes, reduce to prevent OOM.
    current_n_envs = int(getattr(env_cfg, "n_envs", _DEFAULT_N_ENVS) or _DEFAULT_N_ENVS)
    if current_n_envs == _DEFAULT_N_ENVS:
        max_safe = _optimal_n_envs(cpu_count, ram_gb)
        if max_safe < current_n_envs:
            env_cfg.n_envs = max_safe
            changes.append(
                f"n_envs  {current_n_envs} -> {max_safe}  "
                f"(RAM safety: {ram_gb:.0f} GB, capped down)"
            )
    final_n_envs = int(getattr(env_cfg, "n_envs", current_n_envs))

    # ── batch_size ────────────────────────────────────────────────────────
    current_batch = int(
        getattr(algo_cfg, "batch_size", _DEFAULT_BATCH_SIZE) or _DEFAULT_BATCH_SIZE
    )
    if current_batch == _DEFAULT_BATCH_SIZE:
        optimal_b = _optimal_batch_size(cuda_avail, vram_gb)
        if optimal_b != current_batch:
            algo_cfg.batch_size = optimal_b
            vram_tag = f"{vram_gb:.0f} GB VRAM" if cuda_avail else "CPU"
            changes.append(
                f"batch_size  {current_batch} -> {optimal_b}  ({vram_tag})"
            )
    final_batch = int(getattr(algo_cfg, "batch_size", current_batch))

    # ── PPO: rollout buffer must be >= 2 x batch_size ─────────────────────
    # SB3 silently truncates minibatches when n_steps * n_envs < batch_size,
    # wasting GPU capacity.  Raise n_steps until buffer >= 2 x batch_size.
    if algo_name == "PPO":
        n_steps = int(getattr(algo_cfg, "n_steps", _DEFAULT_N_STEPS) or _DEFAULT_N_STEPS)
        if n_steps == _DEFAULT_N_STEPS:
            rollout_total = n_steps * final_n_envs
            min_rollout = final_batch * 2
            if rollout_total < min_rollout:
                new_n_steps = min(
                    8192,
                    (min_rollout + final_n_envs - 1) // final_n_envs,
                )
                if new_n_steps != n_steps:
                    algo_cfg.n_steps = new_n_steps
                    changes.append(
                        f"n_steps  {n_steps} -> {new_n_steps}  "
                        f"(rollout {n_steps * final_n_envs:,} -> "
                        f"{new_n_steps * final_n_envs:,}, >= 2x batch)"
                    )

    # ── SAC: learning_starts >= 4 x batch_size ────────────────────────────
    # With a large batch the replay buffer needs more diverse samples before
    # the first gradient update is meaningful.
    if algo_name == "SAC":
        ls = int(
            getattr(algo_cfg, "learning_starts", _DEFAULT_LEARNING_STARTS)
            or _DEFAULT_LEARNING_STARTS
        )
        if ls == _DEFAULT_LEARNING_STARTS:
            min_ls = final_batch * 4
            if min_ls > ls:
                algo_cfg.learning_starts = min_ls
                changes.append(
                    f"learning_starts  {ls} -> {min_ls}  (= 4 x batch_size)"
                )

    # ── Log summary ───────────────────────────────────────────────────────
    if log_fn:
        hw_str = f"{cpu_count}-core CPU, {ram_gb:.0f} GB RAM"
        if cuda_avail:
            hw_str += f", {device_name} ({vram_gb:.0f} GB)"
        else:
            hw_str += ", no CUDA"
        log_fn(f"[auto-tune] Hardware: {hw_str}")
        if changes:
            log_fn(f"[auto-tune] {len(changes)} param(s) tuned for this hardware:")
            for c in changes:
                log_fn(f"[auto-tune]   {c}")
        else:
            log_fn("[auto-tune] All params already customised -- no changes applied")
