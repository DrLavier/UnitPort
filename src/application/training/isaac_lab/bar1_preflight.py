# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BAR1 (PCIe aperture) preflight for *headed* Isaac Lab training.

Headed training renders the Isaac Sim viewport, which maps GPU buffers
through the PCIe **BAR1** aperture. When that aperture is exhausted the
NVIDIA driver aborts the process with an opaque Xid / device-lost error
deep inside Kit — the user sees a cryptic crash with no hint that BAR1
was the cause. This module turns the canvas-derived run config
(``headless`` + ``num_envs``) plus a BAR1 capacity probe into a clear
OK / WARN / BLOCK verdict **before** the ~30 s Kit bootstrap, so a doomed
run fails loud with an actionable message instead of a confusing crash.

What actually predicts the crash (and what does not):

* **Reliable, config-driven:** ``headless`` (no viewport → BAR1 is not the
  bottleneck) and ``num_envs`` (headed rendering maps every env's render
  buffers through BAR1, so a large env count is the dominant driver).
* **Reliable, hardware:** BAR1 *total* — a tiny aperture (~256 MiB) means
  Resizable BAR is OFF; headed Isaac Sim cannot fit a non-trivial scene
  through it and crashes for all but a handful of envs.
* **NOT reliable on Windows:** BAR1 *free*. Under the WDDM driver model the
  whole aperture is typically pre-mapped, so ``nvidia-smi`` reports free ≈ 0
  even when only a couple GiB of VRAM is in use (observed: RTX 4090, BAR1
  32768 MiB total / 29 MiB free at 2 GiB VRAM use). We therefore use free
  only to *escalate a warning message*, never to hard-BLOCK — blocking on a
  WDDM artifact would make headed training impossible on healthy machines.

No silent fallback (CLAUDE.md §8): if BAR1 cannot be queried at all we
surface that as a WARN ("could not verify — if it crashes, suspect BAR1"),
never as a silent pass.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional

# --- thresholds (empirical heuristics; tune here, all documented) ----------
# Below this BAR1 total we treat Resizable BAR as effectively OFF (the legacy
# 256 MiB aperture). 1 GiB cleanly separates "no ReBAR" from any ReBAR-on /
# datacenter GPU (whose BAR1 is GiB-scale, usually ≈ VRAM).
_REBAR_ON_MIN_MB = 1024
# Headed envs that a no-ReBAR (256 MiB) aperture can plausibly survive. Above
# this a headed run is a near-certain BAR1 crash → BLOCK; at or below it the
# scene may squeak through → WARN.
_NO_REBAR_MAX_ENVS = 16
# Headed env count above which BAR1 pressure is a real risk even with ReBAR on
# — surfaced as a WARN, because the safe ceiling depends on scene complexity
# and we must not false-BLOCK a capable GPU.
_HEADED_ENVS_WARN = 256
# BAR1 free (MiB) below which we *mention* imminent exhaustion in the warning.
# Advisory only (see the WDDM caveat above) — never a BLOCK trigger.
_CRITICAL_FREE_MB = 256

_FIX_HINT = (
    "Fixes: (1) train headless (uncheck the headed/viewport option) — "
    "large-scale RL should run headless; (2) drop num_envs to a handful "
    "(e.g. 16-64) for a headed/GUI run; (3) ensure Resizable BAR / "
    "'Above 4G Decoding' is enabled in BIOS; (4) close other GPU apps."
)

Level = Literal["ok", "warn", "block"]


@dataclass(frozen=True)
class Bar1Info:
    """BAR1 aperture totals for one GPU, in MiB."""

    total_mb: int
    used_mb: int
    free_mb: int
    gpu_name: str
    source: str  # "pynvml" | "nvidia-smi"


@dataclass(frozen=True)
class Bar1Verdict:
    level: Level
    message: str
    info: Optional[Bar1Info]


def _query_pynvml(gpu_id: int) -> Optional[Bar1Info]:
    """Fast in-process BAR1 probe. Returns None if pynvml is absent or the
    NVML call fails — the caller falls back to nvidia-smi, then surfaces a
    WARN if both miss (so None here never masks a failure; §8 (a))."""
    try:
        import pynvml  # optional third-party (nvidia-ml-py); §8 (a)
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        bar1 = pynvml.nvmlDeviceGetBAR1MemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        mib = 1024 * 1024
        return Bar1Info(
            total_mb=int(bar1.bar1Total // mib),
            used_mb=int(bar1.bar1Used // mib),
            free_mb=int(bar1.bar1Free // mib),
            gpu_name=str(name),
            source="pynvml",
        )
    except Exception:
        # NVML present but this query failed (bad index, driver mismatch).
        # Return None so nvidia-smi gets a turn; a total miss is WARNed.
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _query_nvidia_smi(gpu_id: int) -> Optional[Bar1Info]:
    """BAR1 probe via the ``nvidia-smi`` CLI (always present on NVIDIA hosts;
    the path SysMonitorWidget already relies on). Parses the ``BAR1 Memory
    Usage`` block of ``nvidia-smi -q -d MEMORY``."""
    try:
        mem = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "-q", "-d", "MEMORY"],
            capture_output=True, text=True, timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if mem.returncode != 0:
        return None

    block_m = re.search(
        r"BAR1 Memory Usage(.*?)(?:\n\s*\n|\Z)", mem.stdout, re.S
    )
    block = block_m.group(1) if block_m else ""

    def _grab(label: str) -> Optional[int]:
        m = re.search(rf"{label}\s*:\s*([0-9]+)\s*MiB", block)
        return int(m.group(1)) if m else None

    total, used, free = _grab("Total"), _grab("Used"), _grab("Free")
    if total is None or free is None:
        return None
    if used is None:
        used = max(total - free, 0)

    name = f"GPU {gpu_id}"
    try:
        nq = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id),
             "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
        if nq.returncode == 0 and nq.stdout.strip():
            name = nq.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass  # name is cosmetic; the verdict does not depend on it

    return Bar1Info(
        total_mb=int(total), used_mb=int(used), free_mb=int(free),
        gpu_name=name, source="nvidia-smi",
    )


def query_bar1(gpu_id: int = 0) -> Optional[Bar1Info]:
    """Probe the target GPU's BAR1 aperture (pynvml fast-path, nvidia-smi
    fallback). None ⇒ neither method could read it; the caller surfaces that
    as a WARN rather than a silent pass (§8)."""
    return _query_pynvml(gpu_id) or _query_nvidia_smi(gpu_id)


def assess_bar1_risk(
    *, headless: bool, num_envs: int, gpu_id: int = 0
) -> Bar1Verdict:
    """Decide whether a headed run is likely to crash on BAR1 exhaustion.

    Headless runs always return ``ok`` (no viewport → BAR1 is not the
    bottleneck). For headed runs see the module docstring for the signal
    reliability rationale: ReBAR-off is a hard BLOCK, everything else is at
    most a WARN so a capable GPU is never falsely blocked.
    """
    if headless:
        return Bar1Verdict("ok", "", None)

    info = query_bar1(gpu_id)

    if info is None:
        return Bar1Verdict(
            "warn",
            "Could not query GPU BAR1 aperture (nvidia-smi / pynvml "
            "unavailable). Headed training renders through BAR1; if it "
            "crashes with a GPU device-lost / Xid error, suspect BAR1 "
            f"exhaustion. {_FIX_HINT}",
            None,
        )

    gpu = info.gpu_name

    # --- Hard BLOCK: Resizable BAR OFF + a non-trivial headed run ----------
    if info.total_mb < _REBAR_ON_MIN_MB and num_envs > _NO_REBAR_MAX_ENVS:
        return Bar1Verdict(
            "block",
            f"Resizable BAR appears OFF on {gpu}: BAR1 aperture is only "
            f"{info.total_mb} MiB. A headed run of {num_envs} environments "
            f"renders through this tiny aperture and will exhaust it, "
            f"crashing the GPU driver before training starts. {_FIX_HINT}",
            info,
        )

    # --- WARN: collect the risk reasons (never BLOCK on a ReBAR-on GPU) ----
    reasons = []
    if num_envs > _HEADED_ENVS_WARN:
        reasons.append(
            f"{num_envs} environments is high for a headed run — each env's "
            f"render buffers are mapped through BAR1"
        )
    if info.free_mb < _CRITICAL_FREE_MB:
        # Advisory: Windows WDDM under-reports free, so this is a hint, not
        # proof. Still worth surfacing — on the reported RTX 4090 the BAR1
        # was 32 GiB total / 29 MiB free when headed training crashed.
        reasons.append(
            f"BAR1 currently shows only {info.free_mb} MiB free of "
            f"{info.total_mb} MiB (note: Windows/WDDM can under-report free)"
        )

    if reasons:
        return Bar1Verdict(
            "warn",
            f"Headed training on {gpu} may crash on BAR1 exhaustion: "
            + "; ".join(reasons)
            + f". {_FIX_HINT}",
            info,
        )

    return Bar1Verdict(
        "ok",
        f"BAR1 preflight OK on {gpu} "
        f"(total {info.total_mb} MiB, free {info.free_mb} MiB, "
        f"num_envs {num_envs}).",
        info,
    )
