# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Automatic cross-engine plant-residual self-check (train-launch pipeline).

The boot-time ``RobotAssetSelfCheckTask`` auto-dumps an empty per-format table;
this is its sim2sim sibling. On an Isaac Lab training launch it measures the
robot's cross-engine PLANT residual once per plant identity (``plant_fingerprint``
= sha256 over MJCF bytes + USD source), caches the verdict into the
``registers.sim2sim_ranges`` registry, and:

  * Layer 1 — the realized==designed alignment gates already run at export.
  * Layer 2 — the measured factor envelope feeds the cross-engine domain
    randomization coverage (consumed by the bundle DR-coverage provenance).
  * Layer 3 — a ``chaotic`` (non-convergent) dimension is recorded and surfaced
    so the launch can soft-block on a user acknowledgement (deduped by
    fingerprint, so an already-acknowledged plant never re-prompts).

``measure`` runs the MuJoCo side in-process and spawns the PhysX side in the
Isaac Lab venv (``coordinator.run_kit_subprocess``). Robot-specific dimensions
(contact stiffness, trajectory) are measured against the robot articulation;
the friction cone is solver-level (robot-agnostic) and injected from the
campaign block-slide reference rather than the collapse-confounded legged slip.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from unitport_sdk import Task, log_info, log_warning

from . import coordinator
from .protocol import AlignedPlant, ProbeSet
from .range_table import (
    DimensionRange,
    FRICTION_RESIDUAL_REFERENCE,
    RangeTable,
    build_range_table,
    persist_to_user_config,
)

_ACK_SECTION = "Sim2Sim"
_ACK_KEY = "acknowledged_chaotic"


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

@dataclass
class Sim2SimVerdict:
    sku: str
    fingerprint: str
    dimensions: Dict[str, str]      # dimension -> grade
    chaotic_dims: List[str]
    from_cache: bool
    measured: bool                  # a Kit measurement just ran
    needs_ack: bool                 # has chaotic dims AND fingerprint not acked

    @property
    def clean(self) -> bool:
        return not self.chaotic_dims

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "fingerprint": self.fingerprint,
            "dimensions": dict(self.dimensions),
            "chaotic_dims": list(self.chaotic_dims),
            "from_cache": self.from_cache,
            "measured": self.measured,
            "needs_ack": self.needs_ack,
            "clean": self.clean,
        }


# ---------------------------------------------------------------------------
# plant identity + acknowledgement state
# ---------------------------------------------------------------------------

def _resolve_asset(sku: str) -> Any:
    from application.service.robot_assets.service import get_robot_asset_service

    return get_robot_asset_service().resolve(sku)


def plant_fingerprint(sku: str) -> str:
    """Stable id for the plant Stage-2/3 cares about: sha256 over the MJCF bytes
    + the USD source string. A change re-arms the measurement; an unchanged
    plant is a cache hit. Truncated to 16 hex chars (collision-irrelevant here)."""
    h = hashlib.sha256()
    asset = _resolve_asset(sku)
    mjcf = getattr(asset, "mjcf_path", None) if asset is not None else None
    if mjcf is not None and Path(mjcf).exists():
        h.update(Path(mjcf).read_bytes())
    h.update(coordinator.resolve_usd_source(sku).encode("utf-8"))
    return h.hexdigest()[:16]


def _ack_set() -> set:
    from unitport_sdk import Config

    raw = Config.get_value(_ACK_SECTION, _ACK_KEY, fallback="") or ""
    return {tok for tok in str(raw).split(",") if tok}


def is_acknowledged(sku: str, fingerprint: str) -> bool:
    return f"{sku}:{fingerprint}" in _ack_set()


def acknowledge(sku: str, fingerprint: str) -> None:
    """Persist the user's "proceed despite the non-convergent dimension" choice,
    keyed by plant fingerprint so the same plant never re-prompts (user.ini)."""
    from unitport_sdk import Config

    s = _ack_set()
    s.add(f"{sku}:{fingerprint}")
    Config.set_value(_ACK_SECTION, _ACK_KEY, ",".join(sorted(s)))


# ---------------------------------------------------------------------------
# verdict assembly + cache
# ---------------------------------------------------------------------------

def _verdict_from_table(
    tbl: Dict[str, Any], sku: str, fingerprint: str, *,
    from_cache: bool, measured: bool,
) -> Sim2SimVerdict:
    dims: Dict[str, str] = {}
    chaotic: List[str] = []
    for d in (tbl.get("dimensions") or []):
        name = str(d.get("dimension"))
        grade = str(d.get("verdict"))
        dims[name] = grade
        if grade == "chaotic":
            chaotic.append(name)
    needs_ack = bool(chaotic) and not is_acknowledged(sku, fingerprint)
    return Sim2SimVerdict(
        sku=sku, fingerprint=fingerprint, dimensions=dims,
        chaotic_dims=chaotic, from_cache=from_cache, measured=measured,
        needs_ack=needs_ack,
    )


def cached_verdict(sku: str, fingerprint: str) -> Optional[Sim2SimVerdict]:
    """Return the cached verdict iff a table exists AND its recorded plant
    fingerprint matches (a mismatch = plant changed → stale → re-measure)."""
    from registers import sim2sim_ranges

    sim2sim_ranges.reload()
    if not sim2sim_ranges.has(sku):
        return None
    tbl = sim2sim_ranges.get_ranges(sku)
    if str((tbl.get("provenance") or {}).get("plant_fingerprint", "")) != fingerprint:
        return None
    return _verdict_from_table(
        tbl, sku, fingerprint, from_cache=True, measured=False)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _inject_friction_reference(table: RangeTable) -> None:
    """The friction cone is a solver-level (robot-agnostic) cross-engine
    property — record the campaign block-slide reference rather than the
    collapse-confounded legged slip, which is excluded from the battery."""
    if table.get("friction_cone") is not None:
        return
    ref = FRICTION_RESIDUAL_REFERENCE
    table.dimensions.append(DimensionRange(
        dimension="friction_cone", verdict="parameterizable",
        channel="box_displacement",
        factor_low=float(ref["factor_low"]), factor_high=float(ref["factor_high"]),
        n_probes=0,
        notes=["robot-agnostic solver-level friction-cone reference "
               "(block-slide campaign); covered by existing ground-friction DR"],
    ))


def measure(
    sku: str, *, run_dir: Path | str, timestamp: str,
    rollout_torques: Optional[Any] = None,
) -> Sim2SimVerdict:
    """Run the cross-engine plant measurement and persist the verdict.

    MuJoCo side in-process; PhysX side via the Isaac Lab venv subprocess. The
    battery is the robot-specific plant dims (contact stiffness via drop,
    trajectory replay); the friction cone is injected from the campaign
    reference. Persists the range table to ``sim2sim_ranges`` keyed by plant
    fingerprint, and writes the residual report alongside the run for trace."""
    from . import scenarios
    from .mujoco_driver import (
        actuator_joint_names,
        load_mujoco_plant,
        nominal_actuator_joint_pos,
    )

    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    asset = _resolve_asset(sku)
    mjcf = str(getattr(asset, "mjcf_path", "") or "") if asset is not None else ""
    fingerprint = plant_fingerprint(sku)

    plant = AlignedPlant(
        sku=sku, mjcf_path=mjcf, usd_source=coordinator.resolve_usd_source(sku))
    actor = load_mujoco_plant(plant)
    # Populate the actuator joint-name order + nominal stance, exactly like
    # scenarios.build_default_probe_set. This hand-rolled probe path (drop +
    # trajectory) previously left both empty, so the PhysX launcher received an
    # empty default_joint_pos and crashed building the ArticulationCfg
    # (resolve_matching_names_values got None) — the residual self-check then
    # silently downgraded to "skipped" (CLAUDE.md §8: a measurement gate that
    # never actually measures). joint_names also lets the Kit side map torque
    # columns to USD joints by name instead of falling back to USD order.
    plant.joint_names = actuator_joint_names(actor)
    plant.default_joint_pos = nominal_actuator_joint_pos(actor)
    probes = (scenarios.drop_to_contact_probes(actor)
              + scenarios.trajectory_replay_probes(
                  actor, rollout_torques=rollout_torques))
    probe_set = ProbeSet(
        aligned_plant=plant, probes=probes,
        n_qpos=int(actor.mj_model.nq), n_qvel=int(actor.mj_model.nv),
        nu=int(actor.mj_model.nu))

    probes_path = run / "sim2sim_probes.json"
    probe_set.write_json(probes_path)
    mj = coordinator.run_mujoco(probe_set, run / "mujoco_results.jsonl")
    px = coordinator.run_kit_subprocess(probes_path, run / "physx_results.jsonl")

    table, report, _ana = coordinator.analyze(
        probe_set, mj, px, timestamp=timestamp)
    table.sku = sku
    _inject_friction_reference(table)
    table.provenance["plant_fingerprint"] = fingerprint
    persist_to_user_config(table)
    try:
        report.write_artifacts(run / "residual_report")
    except Exception as exc:  # noqa: BLE001
        log_warning(f"[sim2sim] residual report write skipped: {exc}")

    verdict = _verdict_from_table(
        table.to_dict(), sku, fingerprint, from_cache=False, measured=True)
    log_info(
        f"[sim2sim] measured sku={sku} fp={fingerprint} "
        f"dims={verdict.dimensions} chaotic={verdict.chaotic_dims}")
    return verdict


def selfcheck(sku: str, *, run_dir: Path | str, timestamp: str = "") -> Sim2SimVerdict:
    """Measure-if-stale gate — the auto-pipeline the train launch calls.

    Cache hit (fingerprint matches a persisted table) returns instantly; a miss
    or a stale fingerprint runs the cross-engine measurement. Never raises on a
    clean/parameterizable result; a ``chaotic`` dimension comes back as
    ``needs_ack`` for the launch to soft-block on."""
    if not sku:
        raise ValueError("sim2sim selfcheck: empty sku (§8)")
    fingerprint = plant_fingerprint(sku)
    cached = cached_verdict(sku, fingerprint)
    if cached is not None:
        log_info(
            f"[sim2sim] cache hit sku={sku} fp={fingerprint} "
            f"clean={cached.clean} needs_ack={cached.needs_ack}")
        return cached
    log_info(f"[sim2sim] no fresh verdict sku={sku} fp={fingerprint} — measuring")
    return measure(sku, run_dir=run_dir, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Task wrapper (mirror RobotAssetSelfCheckTask)
# ---------------------------------------------------------------------------

class Sim2SimResidualSelfCheckTask(Task):
    """Train-launch self-check Task — measure-if-stale for one robot, return the
    verdict. The UI seam opens a soft-block acknowledgement dialog iff
    ``verdict.needs_ack`` (the sim2sim analog of the boot IR-assignment dialog)."""

    def __init__(self, sku: str, run_dir: Path | str) -> None:
        super().__init__(name="Sim2Sim Residual Self-Check")
        self._sku = str(sku)
        self._run_dir = run_dir

    def run(self) -> Sim2SimVerdict:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.log_info(f"sim2sim self-check for sku={self._sku!r}")
        verdict = selfcheck(self._sku, run_dir=self._run_dir, timestamp=ts)
        if verdict.chaotic_dims:
            self.log_info(
                f"non-convergent dimension(s): {verdict.chaotic_dims} "
                f"(needs_ack={verdict.needs_ack})")
        else:
            self.log_info("plant residual clean / parameterizable — no block")
        return verdict


__all__ = [
    "Sim2SimVerdict",
    "Sim2SimResidualSelfCheckTask",
    "plant_fingerprint",
    "is_acknowledged",
    "acknowledge",
    "cached_verdict",
    "measure",
    "selfcheck",
]
