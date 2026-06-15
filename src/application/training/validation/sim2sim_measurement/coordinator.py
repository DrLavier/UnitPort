# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 coordinator (.venv311) — orchestrates the batch-file contract.

    generate_probes ──► sim2sim_probes.json
    run_mujoco      ──► mujoco_results.jsonl           (this process, MuJoCo)
    [ user runs il_sim2sim_launcher in the Kit venv ──► physx_results.jsonl ]
    analyze         ◄── both results → range table + residual report

No live IPC: the PhysX side is a separate manual Kit run reading the same probe
file. ``demo_end_to_end_with_mock`` substitutes a synthetic PhysX result so the
whole pipeline is verifiable in ``.venv311``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from unitport_sdk import log_debug, log_info

from .discriminator import MeasurementAnalysis, analyze as _analyze
from .mock_physx import synthesize_physx_results
from .protocol import (
    AlignedPlant,
    EngineResult,
    ProbeSet,
    read_results_jsonl,
)
from .range_table import RangeTable, build_range_table
from .residual_report import ResidualReport, build_residual_report


# ---------------------------------------------------------------------------
# probe generation + MuJoCo side
# ---------------------------------------------------------------------------

def resolve_usd_source(sku: str) -> str:
    """Resolve the PhysX-side USD source for ``sku`` HERE (in .venv311, which
    has the registry/asset service) so it can be baked into the probe file —
    the Kit launcher then needs no ``registers``/``unitport_sdk`` import (no
    PyQt6 in the headless Kit venv). Mirrors ``env_cfg_compiler``: prefer a
    local ``usd_path`` on disk, else the ``usd_url`` (which carries a
    ``nucleus:`` marker the Kit launcher resolves against ISAAC_NUCLEUS_DIR).
    Returns "" when the SKU has no USD (MuJoCo-only run / tests)."""
    if not sku:
        return ""
    try:
        from application.service.robot_assets.service import get_robot_asset_service

        asset = get_robot_asset_service().resolve(sku)
    except Exception as exc:  # noqa: BLE001
        log_debug(f"[sim2sim] usd resolve failed for sku={sku!r}: {exc}")
        return ""
    if asset is None:
        return ""
    usd_path = getattr(asset, "usd_path", None)
    if usd_path is not None and Path(usd_path).exists():
        return str(usd_path)
    usd_url = getattr(asset, "usd_url", "")
    return str(usd_url or "")


def generate_probes(
    *,
    sku: str = "",
    mjcf_path: str = "",
    friction_static: Optional[float] = None,
    n_repeats: int = 5,
    rollout_torques: Optional[Sequence[Sequence[float]]] = None,
) -> ProbeSet:
    """Build the standard probe battery for a plant (drop + slip + trajectory)."""
    from . import scenarios
    from .mujoco_driver import load_mujoco_plant

    plant = AlignedPlant(
        sku=sku, mjcf_path=mjcf_path, friction_static=friction_static,
        usd_source=resolve_usd_source(sku))
    actor = load_mujoco_plant(plant)
    ps = scenarios.build_default_probe_set(
        actor, plant, n_repeats=n_repeats, rollout_torques=rollout_torques)
    log_debug(
        f"[sim2sim] generated {len(ps.probes)} probes "
        f"(sku={sku!r} mjcf={mjcf_path!r} nu={ps.nu} n_repeats={n_repeats}) "
        f"dims={sorted({p.dimension for p in ps.probes})}")
    return ps


def run_mujoco(probe_set: ProbeSet, out_path: Path | str) -> List[EngineResult]:
    """Run the MuJoCo side open-loop and write ``mujoco_results.jsonl``."""
    from .mujoco_driver import run_probe_set

    results = run_probe_set(probe_set, out_path)
    log_debug(
        f"[sim2sim] MuJoCo side: {len(results)} engine-results "
        f"({len(probe_set.probes)} probes) → {out_path}")
    return results


# ---------------------------------------------------------------------------
# Kit-side command (the user runs this manually in the Kit venv)
# ---------------------------------------------------------------------------

def _launcher_path() -> Path:
    from unitport_sdk import Paths

    return (
        Path(Paths.SRC_ROOT)
        / "application" / "training" / "isaac_lab" / "launcher"
        / "il_sim2sim_launcher.py"
    )


def _resolve_isaac_python() -> Optional[str]:
    """Best-effort Kit venv python (returns None if no install is registered —
    the caller then prints the launcher path for the user to run by hand)."""
    try:
        from registers import backends as _backends

        for inst in _backends.list_isaac_installations(ensure_base=True):
            root = inst.get("root")
            if not root:
                continue
            py = _backends._find_isaac_python(Path(root))  # noqa: SLF001
            if py:
                return str(py)
    except Exception:
        pass
    return None


def kit_command(probes_path: Path | str, out_path: Path | str) -> List[str]:
    """Build the Kit-side measurement command. When the Kit python cannot be
    resolved, the first element is a ``<KIT_PYTHON>`` placeholder the user
    fills in (e.g. ``Engines/isaac_lab/.venv/Scripts/python.exe``)."""
    py = _resolve_isaac_python() or "<KIT_PYTHON>"
    return [
        py, str(_launcher_path()),
        "--probes", str(probes_path),
        "--out", str(out_path),
        "--headless",
    ]


def print_kit_instructions(probes_path: Path | str, out_path: Path | str) -> str:
    """Human-readable instruction block for the manual Kit run."""
    cmd = " ".join(kit_command(probes_path, out_path))
    msg = (
        "[sim2sim] PhysX side is Kit-gated — run this in the Isaac Lab venv:\n"
        f"  {cmd}\n"
        f"It reads {probes_path} and writes {out_path}; then re-run analyze()."
    )
    return msg


# PhysX probe replay (open-loop, no policy) over the standard battery: a few
# seconds of physics per probe × repeats. Generous ceiling — first Kit boot +
# USD fetch dominates.
_KIT_MEASURE_TIMEOUT_S = 900


class Sim2SimMeasurementError(RuntimeError):
    """Raised when the PhysX-side Kit measurement subprocess fails."""


def _terminate_kit_tree(proc: Any) -> None:
    """Kill the Isaac Lab child (and the renderer/physics helpers it may have
    spawned) and reap it. Isaac Sim's ``simulation_app.close()`` can hang on
    Windows teardown, so even after the launcher has written its result file the
    process may linger eating CPU — we tear the whole tree down rather than wait
    on it. Best-effort: a kill failure is logged, never raised."""
    import os
    import subprocess

    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            # taskkill /T kills the whole process tree (Kit spawns helpers);
            # proc.kill() alone would orphan them.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=30,
            )
        else:
            proc.terminate()
    except Exception as exc:  # noqa: BLE001
        log_debug(f"[sim2sim] PhysX child kill (pid={proc.pid}) failed: {exc}")
    try:
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001 — TimeoutExpired or already-reaped
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def run_kit_subprocess(
    probes_path: Path | str,
    out_path: Path | str,
    *,
    timeout_s: int = _KIT_MEASURE_TIMEOUT_S,
) -> List[EngineResult]:
    """Spawn ``il_sim2sim_launcher.py`` in the Isaac Lab venv and return the
    parsed PhysX results once it has written ``out_path``.

    The PhysX/MuJoCo handshake is a FILE contract ("No live IPC", see module
    docstring): the launcher writes ``physx_results.jsonl`` atomically (temp +
    ``os.replace``), so the instant that file exists the measurement is complete
    and parseable. We therefore wait on the FILE, not on the process exiting —
    Isaac Sim's ``simulation_app.close()`` routinely hangs for minutes (or never
    returns) on Windows AFTER the launcher has already written its results.
    Blocking on ``subprocess.run`` (process exit) would burn the full
    ``timeout_s`` on a hung teardown and then DISCARD a complete, expensive
    measurement — which left the self-check forever "skipped" and uncached, so
    every launch re-measured and re-hung. Once the result file appears we kill
    the (possibly still-hanging) child tree and return. (§8 / §11.)

    Mirrors the USD body-dump spawn (``discovery_subprocess``): same EULA/UTF-8
    child env. Raises :class:`Sim2SimMeasurementError` on any failure so the
    caller fails loud rather than analysing a stale/absent result (§8)."""
    import collections
    import os
    import subprocess
    import threading
    import time

    py = _resolve_isaac_python()
    if py is None:
        raise Sim2SimMeasurementError(
            "Isaac Lab Python interpreter not found — register Isaac Lab "
            "(Settings -> Engines) before the cross-engine measurement can run."
        )
    cmd = [
        py, str(_launcher_path()),
        "--probes", str(probes_path),
        "--out", str(out_path),
        "--headless",
    ]
    log_debug(f"[sim2sim] spawning PhysX measurement: {' '.join(cmd[:2])} ...")
    # First-time-per-plant this boots Isaac Sim and replays the probe battery
    # (~1-2 min); the verdict is then cached by plant fingerprint so subsequent
    # launches of the same robot are instant. Tell the user so the wait does not
    # read as a freeze — and stream the child's per-probe progress below.
    log_info(
        "[sim2sim] cross-engine self-check: booting Isaac Sim + replaying probes "
        "(first run for this robot ~1-2 min; cached afterwards). Progress streams "
        "below — no need to press Stop."
    )

    out = Path(out_path)
    # Remove any stale result from a previous (possibly different-fingerprint)
    # run so "the file appeared" unambiguously means THIS run wrote it — never
    # return a prior plant's measurement (§8).
    try:
        out.unlink()
    except FileNotFoundError:
        pass

    child_env = os.environ.copy()
    child_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env.setdefault("PYTHONUTF8", "1")

    log_path = out.with_name("physx_stdout.log")
    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        # Own process group so taskkill /T can reach the whole Kit tree.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    # A reader THREAD drains the child's stdout pipe (so chatty Kit can never
    # deadlock on a full OS buffer), forwards its per-probe progress to the app
    # log so the GUI shows liveness exactly like the MuJoCo side, mirrors every
    # line to ``physx_stdout.log`` for post-mortem, and keeps a rolling tail for
    # the failure message. We still WAIT ON THE RESULT FILE (atomic os.replace),
    # not on process exit — Isaac's teardown can hang long after the results are
    # written (see this function's docstring).
    tail: "collections.deque[str]" = collections.deque(maxlen=40)

    def _pump(stream: Any, log_fh: Any) -> None:
        try:
            for raw in stream:
                line = raw.rstrip("\n")
                if not line:
                    continue
                try:
                    log_fh.write(line + "\n")
                    log_fh.flush()
                except Exception:  # noqa: BLE001
                    pass
                tail.append(line)
                # Forward milestone lines (scene build, per-probe, per-repeat)
                # but skip the high-frequency "... step N/250" spam so the GUI
                # log stays readable while still ticking ~once/sec.
                if " step " in line:
                    continue
                msg = line.replace("[UnitPort][SIM2SIM]", "").strip()
                if msg:
                    log_debug(f"[sim2sim/physx] {msg}")
        except Exception:  # noqa: BLE001 — pipe closed on kill / decode glitch
            pass

    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=child_env, **popen_kwargs,
        )
        reader = threading.Thread(
            target=_pump, args=(proc.stdout, log_fh), daemon=True)
        reader.start()

        start = time.monotonic()
        last_beat = start
        result_ready = False
        exited_rc: Optional[int] = None
        while True:
            if out.exists():
                result_ready = True
                break
            exited_rc = proc.poll()
            if exited_rc is not None:
                # Child exited (hard-exit after writing) without a file yet — one
                # last check covers an os.replace landing between the two reads.
                if out.exists():
                    result_ready = True
                break
            now = time.monotonic()
            if now - start > timeout_s:
                break
            if now - last_beat >= 20.0:
                last_beat = now
                log_info(
                    f"[sim2sim] PhysX self-check still running "
                    f"({int(now - start)}s elapsed)…")
            time.sleep(1.0)

        # Whether we got the file, the child died, or we timed out, do not leave
        # a hung Kit process behind. Killing it also closes the pipe → the reader
        # thread ends.
        _terminate_kit_tree(proc)
        reader.join(timeout=5.0)
    finally:
        try:
            log_fh.close()
        except Exception:  # noqa: BLE001
            pass

    if not result_ready:
        tail_txt = "\n".join(tail) or "(no Kit log captured)"
        if exited_rc is None:
            raise Sim2SimMeasurementError(
                f"PhysX measurement timed out after {timeout_s}s without "
                f"writing results (probes={probes_path}). Kit log tail:\n{tail_txt}"
            )
        raise Sim2SimMeasurementError(
            f"PhysX measurement subprocess exited {exited_rc} without writing "
            f"results to {out_path}. Kit log tail:\n{tail_txt}"
        )

    results = read_results_jsonl(out)
    if not results:
        raise Sim2SimMeasurementError(
            f"PhysX measurement wrote no results to {out_path}")
    log_debug(f"[sim2sim] PhysX side: {len(results)} engine-results <- {out_path}")
    return results


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze(
    probe_set: ProbeSet,
    mujoco_results: List[EngineResult],
    physx_results: List[EngineResult],
    *,
    timestamp: str = "",
    clean_s: Optional[float] = None,
    chaos_s: Optional[float] = None,
    mujoco_lean: bool = True,
) -> Tuple[RangeTable, ResidualReport, MeasurementAnalysis]:
    """Run the discriminator and build both Stage-1 outputs (range table +
    residual report). ``timestamp`` is passed in (no Date.now() in this layer);
    callers stamp it."""
    kwargs: Dict[str, Any] = {}
    if clean_s is not None:
        kwargs["clean_s"] = clean_s
    if chaos_s is not None:
        kwargs["chaos_s"] = chaos_s
    ana = _analyze(probe_set, mujoco_results, physx_results, **kwargs)
    for d in ana.per_dimension:
        span = (
            f"factor[{d.factor_low:.3f},{d.factor_high:.3f}]"
            if d.factor_low is not None else "no-range(chaotic)")
        log_debug(
            f"[sim2sim] dim={d.dimension} verdict={d.verdict} "
            f"n_probes={d.n_probes} {span}")
    table = build_range_table(ana, timestamp=timestamp, mujoco_lean=mujoco_lean)
    report = build_residual_report(ana, timestamp=timestamp)
    return table, report, ana


def analyze_from_files(
    probes_path: Path | str,
    mujoco_results_path: Path | str,
    physx_results_path: Path | str,
    *,
    timestamp: str = "",
) -> Tuple[RangeTable, ResidualReport, MeasurementAnalysis]:
    """Offline analysis from the three on-disk artifacts (the real flow once the
    user has produced ``physx_results.jsonl`` in the Kit venv)."""
    probe_set = ProbeSet.read_json(probes_path)
    mj = read_results_jsonl(mujoco_results_path)
    px = read_results_jsonl(physx_results_path)
    return analyze(probe_set, mj, px, timestamp=timestamp)


# ---------------------------------------------------------------------------
# end-to-end demo (mock PhysX) — the .venv311 verification entrypoint
# ---------------------------------------------------------------------------

def demo_end_to_end_with_mock(
    sku_or_path: str,
    *,
    out_dir: Optional[Path | str] = None,
    n_repeats: int = 5,
    timestamp: str = "demo",
) -> Dict[str, Any]:
    """Full pipeline with a synthetic PhysX side — generate → MuJoCo → mock →
    analyze → write artifacts. Returns a summary dict. Writes the probe set +
    both result streams + the range table + residual report under ``out_dir``
    (a temp dir if omitted). Used by the Stage-1 verification test/CLI."""
    import tempfile

    is_path = sku_or_path.endswith(".xml") or "/" in sku_or_path or "\\" in sku_or_path
    probe_set = generate_probes(
        sku=("" if is_path else sku_or_path),
        mjcf_path=(sku_or_path if is_path else ""),
        friction_static=0.8,
        n_repeats=n_repeats,
    )
    base = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    base.mkdir(parents=True, exist_ok=True)

    probes_path = base / "sim2sim_probes.json"
    probe_set.write_json(probes_path)
    mj_path = base / "mujoco_results.jsonl"
    mj_results = run_mujoco(probe_set, mj_path)

    # Mock the Kit side (real flow: user runs il_sim2sim_launcher here).
    px_results = synthesize_physx_results(probe_set, mj_results)

    table, report, ana = analyze(
        probe_set, mj_results, px_results, timestamp=timestamp)

    table_path = base / "range_table.json"
    table.write_json(table_path)
    report_base = base / "residual_report"
    report.write_artifacts(report_base)

    return {
        "out_dir": str(base),
        "n_probes": len(probe_set.probes),
        "dimensions": {d.dimension: d.verdict for d in ana.per_dimension},
        "range_table": str(table_path),
        "residual_report_md": str(report_base.with_suffix(".md")),
        "kit_command": kit_command(probes_path, base / "physx_results.jsonl"),
    }


__all__ = [
    "resolve_usd_source",
    "generate_probes",
    "run_mujoco",
    "kit_command",
    "print_kit_instructions",
    "Sim2SimMeasurementError",
    "run_kit_subprocess",
    "analyze",
    "analyze_from_files",
    "demo_end_to_end_with_mock",
]
