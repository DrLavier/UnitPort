# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol for the Stage-1 cross-engine measurement (the batch-file
contract). One ``ProbeSet`` (written by the .venv311 coordinator) drives both
engines; each side writes a stream of ``EngineResult`` (JSONL). All numbers are
plain floats/lists so the file is portable across the two venvs.

Fail-loud (§8): every ``from_dict`` raises on a missing/malformed field rather
than defaulting — a silently-truncated probe set would corrupt the residual
spectrum without any signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROTOCOL_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# small helpers (fail-loud field access)
# ---------------------------------------------------------------------------

def _req(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"{ctx}: required field {key!r} missing")
    return d[key]


def _flist(v: Any, ctx: str) -> List[float]:
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{ctx}: expected a list of floats, got {type(v).__name__}")
    return [float(x) for x in v]


def _fmat(v: Any, ctx: str) -> List[List[float]]:
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{ctx}: expected a list of rows, got {type(v).__name__}")
    return [_flist(row, ctx) for row in v]


# ---------------------------------------------------------------------------
# aligned plant + per-probe contact knobs
# ---------------------------------------------------------------------------

@dataclass
class AlignedPlant:
    """The Layer-1 aligned plant BOTH engines reproduce, so the measured
    residual is pure Type-II/III. ``sku`` lets each engine rebuild its realized
    plant (MuJoCo: MjActor overlay; PhysX: ArticulationCfg armature). When
    ``mjcf_path`` is given it overrides registry resolution (self-contained
    tests). ``friction_static`` is the Stage-0 single-source ground μ applied to
    both engines (method A)."""

    # "robot" (the SKU's articulation) or "box" (a single free rigid body on a
    # plane — the clean foot-ground friction-cone probe with NO articulation, so
    # no multi-body collapse confounds the slip). Both engines build the same
    # box from this flag (no sku/usd needed for a box).
    plant_kind: str = "robot"
    sku: str = ""
    mjcf_path: str = ""
    # The USD source the PhysX side spawns, RESOLVED BY THE COORDINATOR (which
    # has the registry/asset service) and baked into the probe file so the Kit
    # launcher needs ZERO app-stack imports (no ``registers``/``unitport_sdk`` →
    # no PyQt6 in the headless Kit venv). May carry a ``nucleus:`` marker the Kit
    # launcher resolves against ISAAC_NUCLEUS_DIR (§9 self-contained artifact).
    usd_source: str = ""
    friction_static: Optional[float] = None
    # Canonical joint order the torque_seq + qpos/qvel joint-part are expressed
    # in (= the MuJoCo actor's joint order). The Kit launcher maps these names
    # onto the USD articulation order (USD order ≠ MJCF order), so a torque is
    # applied to the SAME physical joint on both engines (joint-order parity).
    joint_names: List[str] = field(default_factory=list)
    # Nominal stance joint angles (actuator joint name → radians) from the
    # MuJoCo keyframe-0 / qpos0. The Kit launcher uses these as the
    # ArticulationCfg.InitialStateCfg.joint_pos so the spawned articulation
    # passes IsaacLab's joint-limit validation (a default of 0 fails for joints
    # like the Go2 calf whose limits exclude 0). Per-probe initial states still
    # override at runtime.
    default_joint_pos: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plant_kind": str(self.plant_kind),
            "sku": str(self.sku),
            "mjcf_path": str(self.mjcf_path),
            "usd_source": str(self.usd_source),
            "friction_static": (
                None if self.friction_static is None else float(self.friction_static)
            ),
            "joint_names": list(self.joint_names),
            "default_joint_pos": {
                str(k): float(v) for k, v in self.default_joint_pos.items()
            },
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlignedPlant":
        if not isinstance(d, dict):
            raise ValueError("AlignedPlant.from_dict: expected dict")
        mu = d.get("friction_static")
        return cls(
            plant_kind=str(d.get("plant_kind", "robot")),
            sku=str(d.get("sku", "")),
            mjcf_path=str(d.get("mjcf_path", "")),
            usd_source=str(d.get("usd_source", "")),
            friction_static=(None if mu is None else float(mu)),
            joint_names=[str(n) for n in (d.get("joint_names", []) or [])],
            default_joint_pos={
                str(k): float(v)
                for k, v in (d.get("default_joint_pos", {}) or {}).items()
            },
            notes=list(d.get("notes", []) or []),
        )


@dataclass
class ContactCfg:
    """Per-probe contact-parameter overrides — the Type-II knobs a probe sweeps.
    ``solref``/``solimp`` are applied to ALL geoms (MuJoCo ``geom_solref`` (2) /
    ``geom_solimp`` (5)); ``friction`` overrides the sliding μ for this probe.
    All None ⇒ use the aligned-plant defaults (no override)."""

    solref: Optional[List[float]] = None
    solimp: Optional[List[float]] = None
    friction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solref": (None if self.solref is None else [float(x) for x in self.solref]),
            "solimp": (None if self.solimp is None else [float(x) for x in self.solimp]),
            "friction": (None if self.friction is None else float(self.friction)),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ContactCfg":
        if not d:
            return cls()
        sr = d.get("solref")
        si = d.get("solimp")
        fr = d.get("friction")
        return cls(
            solref=(None if sr is None else _flist(sr, "ContactCfg.solref")),
            solimp=(None if si is None else _flist(si, "ContactCfg.solimp")),
            friction=(None if fr is None else float(fr)),
        )


# ---------------------------------------------------------------------------
# probe + probe set
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    """One open-loop measurement: from ``(init_qpos, init_qvel)`` replay the
    ``torque_seq`` (per-step joint torques, shape ``(T, nu)``) and record the
    per-step next state + contact. ``n_repeats`` perturbed-initial-condition
    runs (seeded by ``ic_perturb_seeds``, magnitude ``ic_perturb_scale``) feed
    the systematic-vs-chaotic discriminator. ``dimension`` tags which Type-II
    residual dimension this probe targets (so the discriminator can aggregate
    per dimension)."""

    probe_id: str
    scenario: str
    dimension: str
    init_qpos: List[float]
    init_qvel: List[float]
    torque_seq: List[List[float]]
    contact_cfg: ContactCfg = field(default_factory=ContactCfg)
    n_repeats: int = 5
    ic_perturb_seeds: List[int] = field(default_factory=list)
    ic_perturb_scale: float = 1e-4

    def __post_init__(self) -> None:
        if self.n_repeats < 1:
            raise ValueError(f"Probe {self.probe_id!r}: n_repeats must be >= 1")
        if not self.ic_perturb_seeds:
            # Deterministic default seeds (repeat 0 is the UNPERTURBED baseline).
            self.ic_perturb_seeds = list(range(self.n_repeats))
        if len(self.ic_perturb_seeds) != self.n_repeats:
            raise ValueError(
                f"Probe {self.probe_id!r}: ic_perturb_seeds has "
                f"{len(self.ic_perturb_seeds)} entries but n_repeats={self.n_repeats}"
            )
        if not self.torque_seq:
            raise ValueError(f"Probe {self.probe_id!r}: torque_seq is empty")

    @property
    def n_steps(self) -> int:
        return len(self.torque_seq)

    @property
    def nu(self) -> int:
        return len(self.torque_seq[0]) if self.torque_seq else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": str(self.probe_id),
            "scenario": str(self.scenario),
            "dimension": str(self.dimension),
            "init_qpos": [float(x) for x in self.init_qpos],
            "init_qvel": [float(x) for x in self.init_qvel],
            "torque_seq": [[float(x) for x in row] for row in self.torque_seq],
            "contact_cfg": self.contact_cfg.to_dict(),
            "n_repeats": int(self.n_repeats),
            "ic_perturb_seeds": [int(s) for s in self.ic_perturb_seeds],
            "ic_perturb_scale": float(self.ic_perturb_scale),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Probe":
        ctx = "Probe.from_dict"
        pid = str(_req(d, "probe_id", ctx))
        return cls(
            probe_id=pid,
            scenario=str(_req(d, "scenario", ctx)),
            dimension=str(_req(d, "dimension", ctx)),
            init_qpos=_flist(_req(d, "init_qpos", ctx), f"{ctx}[{pid}].init_qpos"),
            init_qvel=_flist(_req(d, "init_qvel", ctx), f"{ctx}[{pid}].init_qvel"),
            torque_seq=_fmat(_req(d, "torque_seq", ctx), f"{ctx}[{pid}].torque_seq"),
            contact_cfg=ContactCfg.from_dict(d.get("contact_cfg")),
            n_repeats=int(d.get("n_repeats", 5)),
            ic_perturb_seeds=[int(s) for s in (d.get("ic_perturb_seeds") or [])],
            ic_perturb_scale=float(d.get("ic_perturb_scale", 1e-4)),
        )


@dataclass
class ProbeSet:
    """The full measurement batch the coordinator writes once and both engines
    consume. ``n_qpos``/``n_qvel``/``nu`` are recorded for cross-engine shape
    validation (a probe whose state width doesn't match the plant is a §8
    error, not a silent reshape)."""

    aligned_plant: AlignedPlant
    probes: List[Probe]
    n_qpos: int
    n_qvel: int
    nu: int
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "aligned_plant": self.aligned_plant.to_dict(),
            "n_qpos": int(self.n_qpos),
            "n_qvel": int(self.n_qvel),
            "nu": int(self.nu),
            "probes": [p.to_dict() for p in self.probes],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProbeSet":
        ctx = "ProbeSet.from_dict"
        ver = int(d.get("schema_version", 0))
        if ver != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"{ctx}: schema_version {ver} != {PROTOCOL_SCHEMA_VERSION} "
                "(regenerate the probe set with the current coordinator)"
            )
        probes_raw = _req(d, "probes", ctx)
        if not isinstance(probes_raw, list) or not probes_raw:
            raise ValueError(f"{ctx}: 'probes' is missing or empty")
        return cls(
            aligned_plant=AlignedPlant.from_dict(_req(d, "aligned_plant", ctx)),
            probes=[Probe.from_dict(p) for p in probes_raw],
            n_qpos=int(_req(d, "n_qpos", ctx)),
            n_qvel=int(_req(d, "n_qvel", ctx)),
            nu=int(_req(d, "nu", ctx)),
            schema_version=ver,
        )

    def write_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        import os
        os.replace(str(tmp), str(path))

    @classmethod
    def read_json(cls, path: Path | str) -> "ProbeSet":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# per-step record + per (probe, repeat, engine) result
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One post-step observation of the realized plant."""

    next_qpos: List[float]
    next_qvel: List[float]
    next_qacc: List[float]
    contact_force_norm: float   # summed |contact normal+friction force| this step
    n_contacts: int
    slip: float                 # tangential slip proxy (m/s) at the foot contacts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_qpos": [float(x) for x in self.next_qpos],
            "next_qvel": [float(x) for x in self.next_qvel],
            "next_qacc": [float(x) for x in self.next_qacc],
            "contact_force_norm": float(self.contact_force_norm),
            "n_contacts": int(self.n_contacts),
            "slip": float(self.slip),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        ctx = "StepRecord.from_dict"
        return cls(
            next_qpos=_flist(_req(d, "next_qpos", ctx), f"{ctx}.next_qpos"),
            next_qvel=_flist(_req(d, "next_qvel", ctx), f"{ctx}.next_qvel"),
            next_qacc=_flist(_req(d, "next_qacc", ctx), f"{ctx}.next_qacc"),
            contact_force_norm=float(_req(d, "contact_force_norm", ctx)),
            n_contacts=int(_req(d, "n_contacts", ctx)),
            slip=float(_req(d, "slip", ctx)),
        )


@dataclass
class EngineResult:
    """One engine's open-loop rollout of one ``(probe, repeat)``. Streamed one
    per JSONL line so a long measurement is restartable and the file can be
    read incrementally."""

    probe_id: str
    repeat: int
    engine: str                 # "mujoco" | "physx"
    steps: List[StepRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": str(self.probe_id),
            "repeat": int(self.repeat),
            "engine": str(self.engine),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineResult":
        ctx = "EngineResult.from_dict"
        steps_raw = _req(d, "steps", ctx)
        if not isinstance(steps_raw, list):
            raise ValueError(f"{ctx}: 'steps' must be a list")
        return cls(
            probe_id=str(_req(d, "probe_id", ctx)),
            repeat=int(_req(d, "repeat", ctx)),
            engine=str(_req(d, "engine", ctx)),
            steps=[StepRecord.from_dict(s) for s in steps_raw],
        )


# ---------------------------------------------------------------------------
# JSONL stream helpers
# ---------------------------------------------------------------------------

def write_results_jsonl(path: Path | str, results: List[EngineResult]) -> None:
    """Atomic write of a list of EngineResults as JSONL (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")
    import os
    os.replace(str(tmp), str(path))


def read_results_jsonl(path: Path | str) -> List[EngineResult]:
    """Read a JSONL EngineResult stream, fail-loud on any malformed line."""
    out: List[EngineResult] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(EngineResult.from_dict(json.loads(line)))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"read_results_jsonl: malformed EngineResult at {path}:{i + 1}: {exc}"
                ) from exc
    if not out:
        raise ValueError(f"read_results_jsonl: {path} contained no results (§8)")
    return out


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "AlignedPlant",
    "ContactCfg",
    "Probe",
    "ProbeSet",
    "StepRecord",
    "EngineResult",
    "write_results_jsonl",
    "read_results_jsonl",
]
