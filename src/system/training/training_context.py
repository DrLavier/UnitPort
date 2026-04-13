#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainingContext — single-entry refresh orchestrator for the standardized
training backend.

Six sections (absolute, single-source):

    §1 robot       — robot asset (RobotSpec)
    §2 scene       — terrain / arena / task / init pose / domain rand
    §3 aux_assets  — reference motions & other training auxiliary data
    §4 algorithm   — PPO / SAC / AMP-PPO / IL hyperparameters + rewards
    §5 export      — ONNX / TorchScript / manifest / eval / vis-check
    §6 sim         — MuJoCo / Isaac Lab physics + env + obs/action

Contract
--------
``TrainingContext.refresh(graph)`` is the ONLY sanctioned entry point for
re-reading canvas state. It:

    1. Snapshots the previous good state.
    2. Calls ``TrainingSpecCompiler().compile(graph, ...)``.
    3. Partitions the resulting TrainingJobSpec into 6 sections.
    4. Runs cross-section validators.
    5. Commits atomically on success; rolls back on error.

Downstream consumers (trainers, exporters, UI panels) MUST read from
``context.robot / .scene / .aux / .algorithm / .export / .sim`` rather
than re-parsing the graph or reaching into ``TrainingJobSpec`` directly.
This is the mechanism that replaces the ad-hoc multi-source override
pattern currently spread across ``training_config.py``,
``isaac_lab_config.py``, ``il_presets.py`` and the UI.

No Qt imports — pure data layer. Signal wiring lives in
``TrainingWorkspaceWindow``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.system.training.training_config import (
    ActorConfig,
    AlgorithmConfig,
    AMPConfig,
    DomainRandConfig,
    EnvConfig,
    EvalConfig,
    ExportConfig,
    GatedRewardConfig,
    ImitationLearningConfig,
    InitPoseConfig,
    ObsActionConfig,
    PhysicsConfig,
    ReferenceMotionConfig,
    RewardConfig,
    RobotSpec,
    SceneConfig,
    TaskConfig,
    TerminationConfig,
    VisCheckConfig,
)
from src.system.training.training_spec import (
    TrainingJobSpec,
    TrainingSpecCompiler,
    TrainingSpecError,
)


# ---------------------------------------------------------------------------
# Section dataclasses — thin, read-only views onto TrainingJobSpec slices.
# Step 1 keeps them as containers; Step 2 will migrate the authoritative
# fields off of TrainingJobSpec / IsaacLabConfig / il_presets into these.
# ---------------------------------------------------------------------------

@dataclass
class RobotSection:
    """§1 — Robot asset + Actor setting.

    Both the unified Robot node and its partner ActorSetting node live
    in this section — together they are the complete Actor canvas block.
    Downstream consumers read ``spec`` for asset identity / structural
    metadata / sensor topology and ``actor`` for init joint posture +
    actuator gain overrides.
    """
    spec: RobotSpec = field(default_factory=RobotSpec)
    actor: ActorConfig = field(default_factory=ActorConfig)


@dataclass
class SceneSection:
    """§2 — Terrain / arena / task / init pose / domain rand."""
    task: TaskConfig = field(default_factory=TaskConfig)
    scene: Optional[SceneConfig] = None
    init_pose: Optional[InitPoseConfig] = None
    domain_rand: Optional[DomainRandConfig] = None


@dataclass
class AuxAssetSection:
    """§3 — Training auxiliary assets (reference motions, future: demos)."""
    reference_motion: Optional[ReferenceMotionConfig] = None


@dataclass
class AlgorithmSection:
    """§4 — Learning algorithm + rewards + termination."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    amp: Optional[AMPConfig] = None
    imitation: Optional[ImitationLearningConfig] = None
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    gated_reward: Optional[GatedRewardConfig] = None


@dataclass
class ExportSection:
    """§5 — Export / eval / vis-check."""
    export: Optional[ExportConfig] = None
    eval_cfg: Optional[EvalConfig] = None
    vis_check: Optional[VisCheckConfig] = None


@dataclass
class SimSection:
    """§6 — Simulation (physics + env + obs/action)."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    obs_action: ObsActionConfig = field(default_factory=ObsActionConfig)


_Sections = Tuple[
    RobotSection, SceneSection, AuxAssetSection,
    AlgorithmSection, ExportSection, SimSection,
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str                       # "error" | "warning"
    section: str                        # "robot"|"scene"|"commands"|...|"cross"|"safety"
    code: str                           # machine-readable id (e.g. "S3_start_point_action_dim_mismatch")
    message: str                        # human-readable, goes to the log
    node_id: Optional[str] = None       # canvas node to highlight (one id only)
    node_type: Optional[str] = None     # used by the UI to resolve node_id when missing
    fix_hint: Optional[str] = None      # optional "here is how to fix it" suggestion

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.severity}:{self.section}:{self.code}] {self.message}"


# ---------------------------------------------------------------------------
# StartPointCompatReport — structured output for the Export node's §2 panel
# ---------------------------------------------------------------------------
# Same source-of-truth as SafetyReview's Start Point checks (S2-S5) but
# shaped for visual rendering rather than error logging. The Export
# node's ``start_point_compat_panel`` widget consumes this directly.
#
# The widget cares about:
#   * whether a Start Point is connected at all
#   * the current ``load_mode`` the Start Point node is in
#   * whether the referenced bundle manifest loaded successfully
#   * per-field old vs new comparison (framework, action dim, decimation,
#     joint order) with a per-field status tag
#   * a single overall_status for the header dot (ok/warning/error/none)
#
# SafetyReview still emits issues for the log path; this struct is a
# parallel read-only view into the same data.


@dataclass
class CompatField:
    """Single row in :class:`StartPointCompatReport`.

    Each entry describes one aspect of the compatibility diff —
    framework, action dim, decimation, joint order — with both the
    old bundle value and the new canvas value side by side.
    """

    label: str                    # display name: "Framework" / "Action dim" / ...
    old_value: str                # stringified value from the manifest
    new_value: str                # stringified value from the current canvas
    status: str                   # "ok" | "warning" | "error"
    detail: str = ""              # optional per-row note for the UI tooltip


@dataclass
class StartPointCompatReport:
    """Structured Start Point compatibility snapshot.

    Produced by :meth:`TrainingContext.compat_report`. Consumed by the
    Export node's ``start_point_compat_panel`` widget for the §2 panel
    and by anything else that wants a human-readable Start Point diff
    outside of the log stream.
    """

    has_start_point: bool = False         # Start Point node exists in the graph
    start_point_label: str = ""           # asset_id / checkpoint_file short label
    load_mode: str = "scratch"            # scratch | resume | warm_start_actor
    manifest_loaded: bool = False         # manifest.yaml parsed successfully
    manifest_error: str = ""              # when manifest_loaded is False, why
    fields: List[CompatField] = field(default_factory=list)
    overall_status: str = "none"          # ok | warning | error | none
    summary: str = ""                     # one-liner for the panel footer

    def has_blocker(self) -> bool:
        """True iff the report contains any error-severity entry."""
        return self.overall_status == "error"


@dataclass
class RefreshResult:
    ok: bool
    issues: List[ValidationIssue]
    revision: int

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# TrainingContext
# ---------------------------------------------------------------------------

class TrainingContext:
    """Single source of truth for the 6 training sections.

    The object is long-lived (one per TrainingWorkspaceWindow). Call
    :meth:`refresh` whenever the canvas graph changes — on node param
    writes and on edge create/remove. Reads go through the section
    properties, never back into ``spec`` directly.
    """

    def __init__(self) -> None:
        self._spec: Optional[TrainingJobSpec] = None
        self._robot: RobotSection = RobotSection()
        self._scene: SceneSection = SceneSection()
        self._aux: AuxAssetSection = AuxAssetSection()
        self._algo: AlgorithmSection = AlgorithmSection()
        self._export: ExportSection = ExportSection()
        self._sim: SimSection = SimSection()
        self._issues: List[ValidationIssue] = []
        self._revision: int = 0
        self._has_committed: bool = False

    # ------------------------------------------------------------------
    # Read-only accessors (downstream consumers use these)
    # ------------------------------------------------------------------

    @property
    def spec(self) -> Optional[TrainingJobSpec]:
        """Most-recently-committed TrainingJobSpec (read-only snapshot)."""
        return self._spec

    @property
    def robot(self) -> RobotSection:
        return self._robot

    @property
    def scene(self) -> SceneSection:
        return self._scene

    @property
    def aux(self) -> AuxAssetSection:
        return self._aux

    @property
    def algorithm(self) -> AlgorithmSection:
        return self._algo

    @property
    def export(self) -> ExportSection:
        return self._export

    @property
    def sim(self) -> SimSection:
        return self._sim

    @property
    def issues(self) -> List[ValidationIssue]:
        return list(self._issues)

    @property
    def revision(self) -> int:
        """Monotonic counter incremented on every successful commit."""
        return self._revision

    @property
    def ready(self) -> bool:
        """True iff the last refresh committed and has no error-severity issues."""
        return self._has_committed and not any(
            i.severity == "error" for i in self._issues
        )

    # ------------------------------------------------------------------
    # Refresh (the only mutator)
    # ------------------------------------------------------------------

    def refresh(
        self,
        graph: Dict[str, Any],
        *,
        policy_id: str = "",
        experiment_id: str = "",
    ) -> RefreshResult:
        """Re-read *graph* and rebuild all 6 sections atomically.

        On compile failure or any error-severity cross-section issue, the
        previous good state is left untouched so that downstream consumers
        never observe a half-updated view.
        """
        snapshot = self._snapshot()

        try:
            spec = TrainingSpecCompiler().compile(
                graph, policy_id=policy_id, experiment_id=experiment_id,
            )
        except TrainingSpecError as exc:
            # Compile errors do not point at a specific node by design
            # (missing required node has no node to highlight). We set
            # node_type="train" as a best-effort anchor so the Train
            # node is highlighted as the "everything downstream is
            # blocked" focal point — the message text spells out which
            # nodes are missing.
            issue = ValidationIssue(
                severity="error",
                section="compile",
                code="compile_failed",
                message=str(exc),
                node_type="train",
                node_id=self._find_node_id_by_type(graph, "train"),
                fix_hint="Add the missing nodes listed in the message above.",
            )
            self._issues = [issue]
            return RefreshResult(
                ok=False, issues=[issue], revision=self._revision,
            )
        except Exception as exc:  # defensive — compiler should only raise TrainingSpecError
            issue = ValidationIssue(
                severity="error",
                section="compile",
                code="compile_crash",
                message=f"{type(exc).__name__}: {exc}",
                node_type="train",
                node_id=self._find_node_id_by_type(graph, "train"),
            )
            self._issues = [issue]
            return RefreshResult(
                ok=False, issues=[issue], revision=self._revision,
            )

        sections = self._partition(spec)
        issues = self._cross_validate(spec, sections, graph)

        if any(i.severity == "error" for i in issues):
            self._restore(snapshot)
            self._issues = issues
            return RefreshResult(
                ok=False, issues=issues, revision=self._revision,
            )

        # Commit
        self._spec = spec
        (self._robot, self._scene, self._aux,
         self._algo, self._export, self._sim) = sections
        self._issues = issues
        self._revision += 1
        self._has_committed = True
        return RefreshResult(ok=True, issues=issues, revision=self._revision)

    # ------------------------------------------------------------------
    # SafetyReview — pre-training full sweep
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Start Point compatibility — structured view for the Export §2 panel
    # ------------------------------------------------------------------

    def compat_report(self, graph: Dict[str, Any]) -> "StartPointCompatReport":
        """Build a :class:`StartPointCompatReport` for visual rendering.

        Walks the graph for the ``base_asset`` (Start Point) node, loads
        the referenced bundle's manifest, and produces one
        :class:`CompatField` per invariant. Reuses the manifest-loading
        and comparison helpers that :meth:`_check_start_point` uses for
        SafetyReview — this method is the **display-side** twin of that
        method, not a duplicate.

        Safe to call when the canvas has no Start Point node at all
        (returns ``has_start_point=False`` with an explanatory summary)
        or when the current spec failed to compile (fields that depend
        on the committed spec are skipped with a neutral status).
        """
        report = StartPointCompatReport()

        sp_id = self._find_node_id_by_type(graph, "base_asset")
        if sp_id is None:
            report.summary = (
                "No Start Point node on the canvas — training will start "
                "from randomly initialised weights."
            )
            return report

        sp_params: Dict[str, Any] = {}
        for n in (graph.get("nodes") or []):
            if str(n.get("id")) == sp_id:
                sp_params = n.get("parameters", {}) or {}
                break

        report.has_start_point = True
        load_mode = str(sp_params.get("load_mode", "scratch") or "scratch").strip().lower()
        report.load_mode = load_mode

        ckpt = str(sp_params.get("checkpoint_file", "") or "").strip()
        asset_id = str(sp_params.get("asset_id", "") or "").strip()
        report.start_point_label = asset_id or (
            Path(ckpt).stem if ckpt else "(unconfigured)"
        )

        if load_mode == "scratch":
            report.summary = (
                "Start Point load_mode=scratch — the bundle is ignored "
                "and training starts from fresh weights."
            )
            report.overall_status = "ok"
            return report

        # Try to load the manifest.
        manifest = self._load_start_point_manifest(ckpt, asset_id)
        if manifest is None:
            report.manifest_loaded = False
            report.manifest_error = (
                f"Manifest load failed (checkpoint_file={ckpt!r}, "
                f"asset_id={asset_id!r})."
            )
            report.overall_status = "error"
            report.summary = (
                "Cannot read the referenced bundle manifest. Pick a "
                "different Start Point or set load_mode=scratch."
            )
            return report

        report.manifest_loaded = True

        # --- Per-field compatibility rows ---
        spec = self._spec   # may be None when refresh hasn't committed

        # Framework
        old_source = str((manifest.get("skill") or {}).get("source_type", "") or "").lower()
        new_source = self._current_framework(graph)
        if old_source and new_source:
            framework_ok = self._frameworks_compatible(old_source, new_source)
            report.fields.append(CompatField(
                label="Framework",
                old_value=old_source or "(unknown)",
                new_value=new_source,
                status="ok" if framework_ok else "error",
                detail=("" if framework_ok else "Cross-framework warm-start is not supported."),
            ))

        # Action dim
        old_action_dim = self._manifest_action_dim(manifest)
        new_action_dim: Optional[int] = None
        if spec is not None:
            if spec.robot_spec.joint_names:
                new_action_dim = len(spec.robot_spec.joint_names)
            elif spec.robot_spec.action_dim:
                new_action_dim = int(spec.robot_spec.action_dim)
        if old_action_dim is not None:
            if new_action_dim is None:
                status = "warning"
                detail = "Current canvas did not compile — cannot verify action dim."
                new_str = "(n/a)"
            elif old_action_dim == new_action_dim:
                status = "ok"
                detail = ""
                new_str = str(new_action_dim)
            else:
                status = "error"
                detail = "Warm-start requires matching action spaces."
                new_str = str(new_action_dim)
            report.fields.append(CompatField(
                label="Action dim",
                old_value=str(old_action_dim),
                new_value=new_str,
                status=status,
                detail=detail,
            ))

        # Decimation
        old_decim = self._manifest_decimation(manifest)
        new_decim: Optional[int] = None
        if spec is not None:
            try:
                sim_dt = float(getattr(spec.physics_config, "sim_dt", 0) or 0)
                control_dt = float(getattr(spec.physics_config, "control_dt", 0) or 0)
                if sim_dt > 0 and control_dt > 0:
                    new_decim = int(round(control_dt / sim_dt))
            except Exception:
                new_decim = None
        if old_decim is not None:
            if new_decim is None:
                status = "warning"
                detail = "Physics Config did not compile."
                new_str = "(n/a)"
            elif old_decim == new_decim:
                status = "ok"
                detail = ""
                new_str = str(new_decim)
            else:
                status = "warning"
                detail = "Policy will step at a different control rate than it was trained at."
                new_str = str(new_decim)
            report.fields.append(CompatField(
                label="Decimation",
                old_value=str(old_decim),
                new_value=new_str,
                status=status,
                detail=detail,
            ))

        # Joint order
        old_joints = self._manifest_joint_names(manifest)
        new_joints = list(spec.robot_spec.joint_names) if spec else []
        if old_joints:
            if not new_joints:
                status = "warning"
                detail = "Robot asset did not resolve — cannot compare."
                old_str = f"{len(old_joints)} joints"
                new_str = "(n/a)"
            elif old_joints == new_joints:
                status = "ok"
                detail = ""
                old_str = f"{len(old_joints)} joints"
                new_str = f"{len(new_joints)} joints"
            else:
                # warm_start_actor tolerates reorder (re-indexes weights),
                # resume does not (strict byte-for-byte replay).
                status = "error" if load_mode == "resume" else "warning"
                if status == "error":
                    detail = (
                        "'resume' requires identical ordering. Switch to "
                        "'warm_start_actor' or pick a bundle trained on the "
                        "same robot family."
                    )
                else:
                    detail = (
                        "warm_start_actor will reindex weights to match the "
                        "new order; critic is discarded."
                    )
                old_str = f"{old_joints[:3]}..."
                new_str = f"{new_joints[:3]}..."
            report.fields.append(CompatField(
                label="Joint order",
                old_value=old_str,
                new_value=new_str,
                status=status,
                detail=detail,
            ))

        # --- Roll up overall status + summary ---
        if any(f.status == "error" for f in report.fields):
            report.overall_status = "error"
            report.summary = (
                "Start Point is not compatible with the current canvas. "
                "See the error rows above and switch load_mode=scratch if "
                "you want to start fresh."
            )
        elif any(f.status == "warning" for f in report.fields):
            report.overall_status = "warning"
            report.summary = (
                f"{load_mode!r} will run with reduced compatibility — see "
                f"warning rows above."
            )
        else:
            report.overall_status = "ok"
            report.summary = (
                f"{load_mode!r} is compatible with the current canvas."
            )

        return report

    def safety_review(
        self,
        graph: Dict[str, Any],
        *,
        policy_id: str = "",
        experiment_id: str = "",
    ) -> RefreshResult:
        """Pre-flight validation sweep. Core gate for training / review.

        Pipeline:

          1. :meth:`refresh` — atomically rebuild the 6 sections.
             Every cross-section issue emitted here already carries
             ``node_id`` / ``node_type`` / ``fix_hint`` so the UI can
             highlight the offending node directly.

          2. Extra safety invariants that need the full graph and
             external IO (Start Point manifest loading, bundle conflict
             checks, review scene ↔ backend compatibility, …). These
             are run even when refresh failed to compile, as long as
             they can derive any meaningful answer from the graph
             alone — compile-failed canvases still benefit from
             "you connected Start Point but the bundle is missing".

          3. Issues from (1) and (2) are merged and ``ok`` is
             recomputed so the caller only needs to check one flag.

        The workspace layer calls this with ``blocking=True`` on the
        Train / Launch Review buttons and with ``blocking=False`` on
        edge-change auto-checks.
        """
        result = self.refresh(
            graph, policy_id=policy_id, experiment_id=experiment_id,
        )

        extra: List[ValidationIssue] = []
        extra.extend(self._check_start_point(graph, result))

        if not extra:
            return result

        merged = list(result.issues) + extra
        has_error = any(i.severity == "error" for i in merged)
        return RefreshResult(
            ok=result.ok and not has_error,
            issues=merged,
            revision=result.revision,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _partition(spec: TrainingJobSpec) -> _Sections:
        robot = RobotSection(spec=spec.robot_spec, actor=spec.actor_config)
        scene = SceneSection(
            task=spec.task_config,
            scene=spec.scene_config,
            init_pose=spec.init_pose_config,
            domain_rand=spec.domain_rand_config,
        )
        aux = AuxAssetSection(reference_motion=spec.reference_motion_config)
        algo = AlgorithmSection(
            algorithm=spec.algorithm_config,
            amp=spec.amp,
            imitation=spec.imitation_config,
            reward=spec.reward_config,
            termination=spec.termination_config,
            gated_reward=spec.gated_reward_config,
        )
        export = ExportSection(
            export=spec.export_config,
            eval_cfg=spec.eval_config,
            vis_check=spec.vis_check_config,
        )
        sim = SimSection(
            physics=spec.physics_config,
            env=spec.env_config,
            obs_action=spec.obs_action_config,
        )
        return robot, scene, aux, algo, export, sim

    @staticmethod
    def _find_node_id_by_type(
        graph: Optional[Dict[str, Any]], node_type: str,
    ) -> Optional[str]:
        """Return the first node id matching ``node_type`` in *graph*,
        or ``None`` if the graph is missing / has no such node."""
        if not graph:
            return None
        for n in (graph.get("nodes") or []):
            if n.get("node_type") == node_type:
                nid = n.get("id")
                return str(nid) if nid is not None else None
        return None

    @classmethod
    def _cross_validate(
        cls,
        spec: TrainingJobSpec,
        sections: _Sections,
        graph: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Run invariants that span more than one section.

        Step 1 ships a small, high-value set. Add more as the
        section-level single-source migration progresses.
        """
        robot, scene, aux, algo, export, sim = sections
        issues: List[ValidationIssue] = []

        # Helper — resolve a node_id from node_type via the graph.
        def _nid(node_type: str) -> Optional[str]:
            return cls._find_node_id_by_type(graph, node_type)

        # --- C1: decimation consistency (sim_dt vs control_dt) -------------
        sim_dt = _safe_float(getattr(sim.physics, "sim_dt", None))
        control_dt = _safe_float(getattr(sim.physics, "control_dt", None))
        if sim_dt is not None and control_dt is not None:
            if sim_dt <= 0 or control_dt <= 0:
                issues.append(ValidationIssue(
                    severity="error",
                    section="sim",
                    code="C1_dt_non_positive",
                    message=(
                        f"sim_dt={sim_dt} and control_dt={control_dt} must both be > 0"
                    ),
                    node_type="physics_config",
                    node_id=_nid("physics_config"),
                    fix_hint="Open Physics Config and set sim_dt / control_dt > 0.",
                ))
            else:
                ratio = control_dt / sim_dt
                decim = round(ratio)
                if decim < 1 or abs(decim - ratio) > 1e-6:
                    issues.append(ValidationIssue(
                        severity="error",
                        section="sim",
                        code="C1_decimation_non_integer",
                        message=(
                            f"control_dt / sim_dt = {ratio:.6f} must be a positive integer "
                            f"(control_dt={control_dt}, sim_dt={sim_dt})"
                        ),
                        node_type="physics_config",
                        node_id=_nid("physics_config"),
                        fix_hint=(
                            f"Adjust control_dt so it is an integer multiple of sim_dt. "
                            f"E.g. sim_dt=0.002, control_dt=0.02 gives decimation=10."
                        ),
                    ))

        # --- C2: AMP algorithm requires a reference-motion asset -----------
        algo_name = str(getattr(algo.algorithm, "algorithm", "") or "").upper().replace("-", "_")
        il_mode = ""
        if graph is not None:
            for _n in (graph.get("nodes") or []):
                if str(_n.get("node_type", "")) in ("il_ppo_trainer", "amp_ppo_trainer"):
                    il_mode = str((_n.get("parameters") or {}).get("training_mode", "") or "").strip().upper()
                    if str(_n.get("node_type", "")) == "amp_ppo_trainer" and not il_mode:
                        il_mode = "AMP_PPO"
                    break
        is_amp = algo_name in ("AMP_PPO", "AMPPPO", "AMP") or il_mode == "AMP_PPO"
        if is_amp:
            if aux.reference_motion is None:
                issues.append(ValidationIssue(
                    severity="error",
                    section="aux_assets",
                    code="C2_amp_requires_motions",
                    message=(
                        "Training Mode AMP_PPO requires a ReferenceMotion node; "
                        "none found in the canvas graph."
                    ),
                    node_type="il_ppo_trainer",
                    node_id=_nid("il_ppo_trainer") or _nid("amp_ppo_trainer"),
                    fix_hint="Add a ReferenceMotion node with an AMP motion file, or switch Training Mode to PPO.",
                ))

        # --- C3: robot action_dim consistency (warning only) ---------------
        robot_action_dim = _safe_int(getattr(robot.spec, "action_dim", None))
        obs_action_dim = _safe_int(getattr(sim.obs_action, "action_dim", None))
        if robot_action_dim and obs_action_dim and robot_action_dim != obs_action_dim:
            issues.append(ValidationIssue(
                severity="warning",
                section="robot",
                code="C3_action_dim_mismatch",
                message=(
                    f"RobotSpec.action_dim={robot_action_dim} differs from "
                    f"ObsActionConfig.action_dim={obs_action_dim}"
                ),
                node_type="robot",
                node_id=_nid("robot"),
                fix_hint=(
                    "Verify the Robot asset and Obs & Action node describe the "
                    "same robot family (same joint count)."
                ),
            ))

        # --- C4: export presence requires an algorithm --------------------
        if export.export is not None and algo.algorithm is None:
            issues.append(ValidationIssue(
                severity="error",
                section="export",
                code="C4_export_without_algorithm",
                message="ExportNode is wired but no AlgorithmConfig was compiled.",
                node_type="export",
                node_id=_nid("export"),
                fix_hint="Add an Algorithm Config node and connect it to the Train node.",
            ))

        # --- S6: export bundle_name must be set to a real value ----------
        # Fresh ExportNode defaults to the <NEW> sentinel so SafetyReview
        # can force the user to pick a bundle name at least once per new
        # task. Empty strings and the literal legacy default
        # "trained_policy" are also blocked — they are almost always the
        # result of "forgot to rename" and overwrite something unrelated.
        if export.export is not None:
            bundle = str(getattr(export.export, "bundle_name", "") or "").strip()
            _blocked_names = {"", "<NEW>", "trained_policy"}
            if bundle in _blocked_names:
                issues.append(ValidationIssue(
                    severity="error",
                    section="export",
                    code="S6_export_name_unset",
                    message=(
                        f"Export node Checkpoint={bundle!r} has not been renamed "
                        f"from its default. Pick a real bundle name before "
                        f"launching training."
                    ),
                    node_type="export",
                    node_id=_nid("export"),
                    fix_hint="Open the Export node and set Checkpoint to a project-specific name (e.g. go2_flat_v1).",
                ))

        # --- C5: gait-enabled commands need phase-tracking rewards ---------
        try:
            gait_on = bool(getattr(spec.commands, "gait_enabled", False))
        except Exception:
            gait_on = False
        if gait_on:
            reward_terms = set()
            try:
                reward_terms.update(getattr(algo.reward, "reward_terms", {}) or {})
                reward_terms.update(getattr(spec, "reward_terms", {}) or {})
            except Exception:
                pass
            _hints = ("phase", "air_time", "feet_", "swing", "contact_force")
            has_phase_reward = any(
                any(h in str(name).lower() for h in _hints)
                for name in reward_terms
            )
            if not has_phase_reward:
                issues.append(ValidationIssue(
                    severity="warning",
                    section="commands",
                    code="C5_gait_without_phase_reward",
                    message=(
                        "Training Commands has gait_enabled but no "
                        "phase/foot-tracking reward term was found in "
                        "Rewards. The policy will see the 7 gait command "
                        "dimensions but nothing will push it to follow them."
                    ),
                    node_type="training_commands",
                    node_id=_nid("training_commands"),
                    fix_hint=(
                        "Enable at least one of track_gait_phase / "
                        "track_body_height_cmd / track_swing_height_cmd / "
                        "feet_air_time on the Rewards node."
                    ),
                ))

        # --- C6: Play Ground → aggregation point must be wired -----------
        # PlayGroundSetting is the single source of truth for §2 Scene.
        # Its scene_pipe must terminate at either:
        #   - an IL trainer's scene_pipe input  (Isaac Lab path), or
        #   - EnvAssembler.scene_pipe           (SB3 path).
        # When the edge is absent, the compiled run silently falls back
        # to defaults — exactly the drift the merge was meant to kill.
        _IL_TRAINER_TYPES = ("il_ppo_trainer", "amp_ppo_trainer")
        if graph is not None:
            pg_id = _nid("play_ground_setting")
            il_trainer_ids: List[str] = []
            for _t in _IL_TRAINER_TYPES:
                _tid = _nid(_t)
                if _tid is not None:
                    il_trainer_ids.append(_tid)
            ea_id = _nid("env_assembler")
            has_any_terminus = bool(il_trainer_ids) or (ea_id is not None)
            if pg_id is not None and has_any_terminus:
                edges = list(graph.get("edges") or [])
                def _pg_terminus(e: Dict[str, Any]) -> bool:
                    if str(e.get("src_id")) != pg_id:
                        return False
                    if str(e.get("src_slot")) != "scene_pipe":
                        return False
                    dst_id = str(e.get("dst_id"))
                    dst_slot = str(e.get("dst_slot"))
                    if dst_id in il_trainer_ids and dst_slot == "scene_pipe":
                        return True
                    if ea_id is not None and dst_id == ea_id and dst_slot == "scene_pipe":
                        return True
                    return False
                if not any(_pg_terminus(e) for e in edges):
                    issues.append(ValidationIssue(
                        severity="error",
                        section="scene",
                        code="C6_play_ground_not_wired",
                        message=(
                            "Play Ground Setting is not connected to the "
                            "compile pipeline. Wire its scene_pipe into "
                            "the IL Trainer (Isaac Lab) or Env Assembler "
                            "(SB3); otherwise the compiled run falls "
                            "back to defaults instead of the canvas scene."
                        ),
                        node_type="play_ground_setting",
                        node_id=pg_id,
                        fix_hint=(
                            "Drag Play Ground Setting's scene_pipe out "
                            "onto either IL Trainer's scene_pipe input "
                            "or Env Assembler's scene_pipe input."
                        ),
                    ))

        # Alias for the C7 / trainer-list check below so the Trainer →
        # Export port validation still runs for both backends.
        trainer_ids = list(il_trainer_ids) if graph is not None else []
        if graph is not None:
            _train_id = _nid("train")
            if _train_id is not None:
                trainer_ids.append(_train_id)

            # --- C7: Trainer → Export port names must match --------------
            # Canonical port is ``train_pipe`` on both endpoints. Historical
            # canvases may carry ``trained_policy`` or ``train_result`` —
            # both are rejected here so the user reconnects them.
            exp_id = _nid("export")
            if exp_id is not None and trainer_ids:
                edges = list(graph.get("edges") or [])
                trainer_to_export = [
                    e for e in edges
                    if str(e.get("src_id")) in trainer_ids
                    and str(e.get("dst_id")) == exp_id
                ]
                bad_edges = [
                    e for e in trainer_to_export
                    if str(e.get("src_slot")) != "train_pipe"
                    or str(e.get("dst_slot")) != "train_pipe"
                ]
                if bad_edges:
                    issues.append(ValidationIssue(
                        severity="error",
                        section="export",
                        code="C7_trainer_export_port_mismatch",
                        message=(
                            "Trainer → Export edge uses a stale port name. "
                            "Both endpoints must be 'train_pipe'."
                        ),
                        node_type="export",
                        node_id=exp_id,
                        fix_hint=(
                            "Delete the existing Trainer→Export edge and "
                            "reconnect it — the trainer now exposes a "
                            "'train_pipe' output that lines up 1:1 with "
                            "the Export node's input."
                        ),
                    ))

        return issues

    # ------------------------------------------------------------------
    # Start Point compatibility (S2–S5)
    # ------------------------------------------------------------------

    def _check_start_point(
        self,
        graph: Dict[str, Any],
        refresh_result: "RefreshResult",
    ) -> List[ValidationIssue]:
        """Load the Start Point node's referenced bundle and compare
        its deploy_contract to the current canvas.

        Runs regardless of whether refresh succeeded — Start Point
        problems are independent of the canvas topology.
        """
        issues: List[ValidationIssue] = []

        sp_id = self._find_node_id_by_type(graph, "base_asset")
        if sp_id is None:
            return issues  # No Start Point node → nothing to check.

        # Pull Start Point parameters directly from the graph.
        sp_params: Dict[str, Any] = {}
        for n in (graph.get("nodes") or []):
            if str(n.get("id")) == sp_id:
                sp_params = n.get("parameters", {}) or {}
                break

        load_mode = str(sp_params.get("load_mode", "scratch") or "scratch").strip().lower()
        if load_mode == "scratch":
            return issues  # Scratch = no warm-start, nothing to validate.

        # Locate the referenced bundle / checkpoint. UnitPort's Start
        # Point node stores the target either as an explicit
        # ``checkpoint_file`` (abs path, run:* + asset:* resolve here)
        # or as an ``asset_id`` that the TrainingAssetRegistry knows
        # how to look up. We prefer the explicit path when both are
        # set so there is no ambiguity.
        ckpt = str(sp_params.get("checkpoint_file", "") or "").strip()
        asset_id = str(sp_params.get("asset_id", "") or "").strip()

        manifest = self._load_start_point_manifest(ckpt, asset_id)
        if manifest is None:
            issues.append(ValidationIssue(
                severity="error",
                section="safety",
                code="S2_start_point_manifest_missing",
                message=(
                    f"Start Point load_mode={load_mode!r} but the referenced "
                    f"bundle manifest could not be loaded "
                    f"(checkpoint_file={ckpt!r}, asset_id={asset_id!r})."
                ),
                node_type="base_asset",
                node_id=sp_id,
                fix_hint=(
                    "Pick a different start point or set load_mode=scratch."
                ),
            ))
            return issues

        # --- S2 framework match (sb3 vs isaac) -------------------------
        old_source = str(
            (manifest.get("skill") or {}).get("source_type", "") or ""
        ).lower()
        new_source = self._current_framework(graph)
        if old_source and new_source and not self._frameworks_compatible(old_source, new_source):
            issues.append(ValidationIssue(
                severity="error",
                section="safety",
                code="S2_start_point_framework_mismatch",
                message=(
                    f"Start Point bundle was trained with {old_source!r} but "
                    f"the current canvas trainer is {new_source!r}. "
                    f"Cross-framework warm-start is not supported."
                ),
                node_type="base_asset",
                node_id=sp_id,
                fix_hint="Pick a start-point bundle from the matching framework, or set load_mode=scratch.",
            ))

        # S3–S5 need a committed spec to compare against. If refresh
        # failed to compile, we cannot compute the "new side" of the
        # diff and skip those checks silently (S2 already reported).
        spec = self._spec if refresh_result.ok else None

        # --- S3 action_dim match ---------------------------------------
        old_action_dim = self._manifest_action_dim(manifest)
        new_action_dim = (
            len(spec.robot_spec.joint_names) if spec and spec.robot_spec.joint_names
            else (spec.robot_spec.action_dim if spec and spec.robot_spec.action_dim else None)
        )
        if old_action_dim is not None and new_action_dim:
            if old_action_dim != new_action_dim:
                issues.append(ValidationIssue(
                    severity="error",
                    section="safety",
                    code="S3_start_point_action_dim_mismatch",
                    message=(
                        f"Start Point bundle has action_dim={old_action_dim}, "
                        f"but the current Robot node has {new_action_dim}. "
                        f"Warm-start requires matching action spaces."
                    ),
                    node_type="base_asset",
                    node_id=sp_id,
                    fix_hint=(
                        "Use a start-point bundle trained on the same robot "
                        "family, or set load_mode=scratch."
                    ),
                ))

        # --- S4 decimation match ---------------------------------------
        old_decim = self._manifest_decimation(manifest)
        new_decim = None
        if spec is not None:
            try:
                sim_dt = float(getattr(spec.physics_config, "sim_dt", 0) or 0)
                control_dt = float(getattr(spec.physics_config, "control_dt", 0) or 0)
                if sim_dt > 0 and control_dt > 0:
                    new_decim = int(round(control_dt / sim_dt))
            except Exception:
                new_decim = None
        if old_decim is not None and new_decim is not None and old_decim != new_decim:
            issues.append(ValidationIssue(
                severity="warning",
                section="safety",
                code="S4_start_point_decimation_mismatch",
                message=(
                    f"Start Point bundle has decimation={old_decim}, current "
                    f"canvas resolves to {new_decim}. The policy will step at "
                    f"a different control rate than it was trained at."
                ),
                node_type="base_asset",
                node_id=sp_id,
                fix_hint=(
                    "Open Physics Config and match sim_dt / control_dt to the "
                    "start-point bundle's contract."
                ),
            ))

        # --- S5 joint order match (warning; error on 'resume') ---------
        old_joints = self._manifest_joint_names(manifest)
        new_joints = list(spec.robot_spec.joint_names) if spec else []
        if old_joints and new_joints and old_joints != new_joints:
            severity = "error" if load_mode == "resume" else "warning"
            issues.append(ValidationIssue(
                severity=severity,
                section="safety",
                code="S5_start_point_joint_order_mismatch",
                message=(
                    f"Start Point bundle joint order differs from the "
                    f"current Robot asset. Old first 4: "
                    f"{old_joints[:4]}, new first 4: {new_joints[:4]}. "
                    + ("'resume' requires identical ordering — switch to "
                       "'warm_start_actor' to keep the actor weights "
                       "despite the reorder." if severity == "error"
                       else "warm_start_actor will reindex weights but the "
                            "critic is discarded.")
                ),
                node_type="base_asset",
                node_id=sp_id,
            ))

        return issues

    # ------------------------------------------------------------------
    # Start Point helpers (self-contained so Qt / filesystem access
    # stays out of the ValidationIssue data path).
    # ------------------------------------------------------------------

    @staticmethod
    def _load_start_point_manifest(
        checkpoint_file: str, asset_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort manifest load. Returns ``None`` on any failure —
        the caller surfaces the failure as ``S2_start_point_manifest_missing``."""
        import yaml
        from pathlib import Path

        candidates: List[Path] = []

        if checkpoint_file:
            ck = Path(checkpoint_file)
            if ck.exists():
                # When the path points at policy.onnx / policy.pt, walk
                # up to the bundle root and look for manifest.yaml.
                for parent in (ck.parent, ck.parent.parent):
                    candidates.append(parent / "manifest.yaml")

        if asset_id:
            try:
                from src.system.training.training_asset_registry import TrainingAssetRegistry
                entry = TrainingAssetRegistry().get(asset_id)
                root = getattr(entry, "root_dir", None) or getattr(entry, "bundle_dir", None)
                if root is not None:
                    candidates.append(Path(root) / "manifest.yaml")
            except Exception:
                pass

        for p in candidates:
            try:
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            return data
            except Exception:
                continue
        return None

    @staticmethod
    def _current_framework(graph: Dict[str, Any]) -> str:
        """Sniff the current canvas's training framework by scanning
        for the authoritative trainer node. The unified IL trainer's
        training_mode parameter picks PPO vs AMP."""
        for n in (graph.get("nodes") or []):
            nt = n.get("node_type", "")
            if nt == "train":
                return "sb3_ppo"
            if nt in ("il_ppo_trainer", "amp_ppo_trainer"):
                p = n.get("parameters", {}) or {}
                mode = str(p.get("training_mode", "") or "").strip()
                if mode == "AMP_PPO" or nt == "amp_ppo_trainer":
                    return "isaac_amp"
                return "isaac_ppo"
        return ""

    @staticmethod
    def _frameworks_compatible(old_source: str, new_source: str) -> bool:
        old_is_sb3 = old_source.startswith("sb3")
        new_is_sb3 = new_source.startswith("sb3")
        old_is_isaac = "isaac" in old_source or "il" in old_source or "amp" in old_source
        new_is_isaac = "isaac" in new_source or "il" in new_source or "amp" in new_source
        return (old_is_sb3 and new_is_sb3) or (old_is_isaac and new_is_isaac)

    @staticmethod
    def _manifest_action_dim(manifest: Dict[str, Any]) -> Optional[int]:
        dc = manifest.get("deploy_contract") or {}
        ids = dc.get("joint_sdk_names")
        if isinstance(ids, list):
            return len(ids)
        # SkillManifest v2 fallback
        skill = manifest.get("skill") or {}
        val = skill.get("action_dim")
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _manifest_decimation(manifest: Dict[str, Any]) -> Optional[int]:
        dc = manifest.get("deploy_contract") or {}
        val = dc.get("decimation")
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _manifest_joint_names(manifest: Dict[str, Any]) -> List[str]:
        dc = manifest.get("deploy_contract") or {}
        raw = dc.get("joint_sdk_names") or []
        return [str(x) for x in raw] if isinstance(raw, list) else []

    # -- snapshot / rollback -------------------------------------------------

    def _snapshot(self) -> Tuple[Any, ...]:
        return (
            self._spec,
            self._robot, self._scene, self._aux,
            self._algo, self._export, self._sim,
            list(self._issues),
            self._revision,
            self._has_committed,
        )

    def _restore(self, snap: Tuple[Any, ...]) -> None:
        (self._spec,
         self._robot, self._scene, self._aux,
         self._algo, self._export, self._sim,
         issues, revision, has_committed) = snap
        self._issues = list(issues)
        self._revision = revision
        self._has_committed = has_committed


# ---------------------------------------------------------------------------
# Small helpers (stay private — don't export)
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def issues_by_node(result: "RefreshResult") -> Dict[str, List[ValidationIssue]]:
    """Group issues by their target node_id for UI highlight dispatch.

    Issues without a ``node_id`` are collected under the empty-string
    key so callers can still surface them in an "unscoped" log bucket.
    """
    out: Dict[str, List[ValidationIssue]] = defaultdict(list)
    for issue in result.issues:
        out[issue.node_id or ""].append(issue)
    return dict(out)


def worst_severity(issues: List[ValidationIssue]) -> Optional[str]:
    """Return 'error' if any error present, else 'warning' if any
    warning present, else None. Used by the UI to pick a border colour."""
    has_error = any(i.severity == "error" for i in issues)
    if has_error:
        return "error"
    has_warning = any(i.severity == "warning" for i in issues)
    if has_warning:
        return "warning"
    return None


__all__ = [
    "RobotSection",
    "SceneSection",
    "AuxAssetSection",
    "AlgorithmSection",
    "ExportSection",
    "SimSection",
    "ValidationIssue",
    "CompatField",
    "StartPointCompatReport",
    "RefreshResult",
    "TrainingContext",
    "issues_by_node",
    "worst_severity",
]
