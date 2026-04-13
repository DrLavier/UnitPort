"""Bundle Robustness Matrix — same bundle × N fields → success table.

Per knowledge_base/mission_design.yaml §8 P5:

    "Bundle Robustness Matrix" tool (reuse vis_check_runner pattern):
    same bundle × N fields → success rate / mean survival steps

This module is the consumer-facing tool for sim2real validation: take
a single bundle, run it through every field in a list, and report a
table of {field_id, status, steps_survived, fall_time, …}. Bundles
trained only on flat ground will fall on stair_climb within seconds;
well-trained bundles will run the full episode in every field. The
matrix makes that visible.

Design notes
------------
- The tool does NOT depend on a real ONNX bundle being on disk for
  callers (e.g. tests, smoke harnesses) that just want to exercise
  the field × actor composition. The optional ``runner_fn`` parameter
  lets callers inject any callable ``(sim_env) -> EpisodeStats`` —
  default is a "no-policy idle run" that just steps physics.
- For real bundle evaluation, the production runner_fn wraps
  PolicyRunner.load + run_episode against the same MjSimEnv that
  BehaviorNode uses (P3+).
- Reports are returned as plain dataclasses (not formatted strings)
  so callers can render them as tables, CSVs, or feed into RL
  hyper-param sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Per-episode summary returned by a runner_fn."""
    success: bool
    steps_run: int
    terminated: bool
    termination_reason: str = ""
    extra: Dict[str, Any] = _field(default_factory=dict)


@dataclass
class FieldRow:
    """One row of the robustness matrix."""
    field_id: str
    status: str                # "success" | "failed" | "errored"
    steps_run: int
    terminated: bool
    termination_reason: str = ""
    error: str = ""            # populated when status == "errored"


@dataclass
class RobustnessReport:
    bundle_path: str
    actor_id: str
    rows: List[FieldRow] = _field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.rows)

    def success_rate(self) -> float:
        if not self.rows:
            return 0.0
        wins = sum(1 for r in self.rows if r.status == "success")
        return wins / len(self.rows)

    def mean_steps(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.steps_run for r in self.rows) / len(self.rows)

    def to_table(self) -> List[Dict[str, Any]]:
        """Render as a list of dicts (CSV / JSON friendly)."""
        return [
            {
                "field_id": r.field_id,
                "status": r.status,
                "steps_run": r.steps_run,
                "terminated": r.terminated,
                "termination_reason": r.termination_reason,
                "error": r.error,
            }
            for r in self.rows
        ]


# ---------------------------------------------------------------------------
# Default runner_fn — physics idle (no policy)
# ---------------------------------------------------------------------------


def idle_runner(max_steps: int = 100) -> Callable[[Any], EpisodeStats]:
    """Build a runner_fn that just steps physics with zero ctrl.

    Useful for tests and for "is the field even survivable on its own"
    smoke checks. ``success`` is True iff the actor stayed standing
    for the full ``max_steps`` (i.e. is_terminated() never fired).
    """

    def _run(sim_env: Any) -> EpisodeStats:
        sim_env.reset()
        steps = 0
        terminated = False
        for _ in range(max_steps):
            sim_env.sim_step()
            steps += 1
            try:
                if sim_env.is_terminated():
                    terminated = True
                    break
            except Exception:
                pass
        return EpisodeStats(
            success=not terminated,
            steps_run=steps,
            terminated=terminated,
            termination_reason="fall" if terminated else "max_steps",
        )

    return _run


# ---------------------------------------------------------------------------
# Production runner_fn — PolicyRunner against the real bundle
# ---------------------------------------------------------------------------


def policy_runner_factory(
    bundle_path: Path,
    max_steps: int = 200,
    command: Optional[List[float]] = None,
) -> Callable[[Any], EpisodeStats]:
    """Build a runner_fn that loads ``bundle_path`` through PolicyRunner
    and runs an episode against the supplied sim_env.

    Failures during ``load`` or ``run_episode`` are caught and converted
    into ``EpisodeStats(success=False)`` with the exception message in
    ``extra["error"]`` so the matrix never crashes mid-row.
    """

    def _run(sim_env: Any) -> EpisodeStats:
        from src.system.policy.policy_runner import PolicyRunner

        try:
            runner = PolicyRunner()
            runner.load(Path(bundle_path), env=sim_env)
            episode = runner.run_episode(
                sim_env,
                max_steps=max_steps,
                command=command or [0.0, 0.0, 0.0],
                render=False,
            )
            return EpisodeStats(
                success=bool(getattr(episode, "success", False)),
                steps_run=int(getattr(episode, "steps_run", 0) or 0),
                terminated=bool(getattr(episode, "terminated", False)),
                termination_reason=str(getattr(episode, "termination_reason", "") or ""),
                extra={
                    "engine_type": getattr(episode, "engine_type", ""),
                },
            )
        except Exception as exc:
            return EpisodeStats(
                success=False,
                steps_run=0,
                terminated=False,
                termination_reason=f"runner_error: {exc}",
                extra={"error": str(exc)},
            )

    return _run


# ---------------------------------------------------------------------------
# Matrix orchestrator
# ---------------------------------------------------------------------------


@dataclass
class RobustnessMatrix:
    bundle_path: str
    field_ids: List[str]
    actor_id: str = "go2"
    project_root: Optional[Path] = None
    runner_fn: Optional[Callable[[Any], EpisodeStats]] = None

    def run(self, max_steps: int = 100) -> RobustnessReport:
        """Iterate over each field, build a fresh MjSimEnv, run the
        ``runner_fn`` (or the idle default), and aggregate per-field
        rows into a :class:`RobustnessReport`.
        """
        from src.system.runtime.simulation.mujoco import (
            MjActor,
            MjSimEnv,
            load_field,
        )

        runner = self.runner_fn or idle_runner(max_steps=max_steps)
        report = RobustnessReport(
            bundle_path=str(self.bundle_path),
            actor_id=self.actor_id,
        )

        for fid in self.field_ids:
            try:
                field = load_field(fid, project_root=self.project_root)
                actor = MjActor.from_menagerie(self.actor_id)
                env = MjSimEnv.build(actor, field)
            except Exception as exc:
                report.rows.append(FieldRow(
                    field_id=fid,
                    status="errored",
                    steps_run=0,
                    terminated=False,
                    error=f"setup_failed: {exc}",
                ))
                continue

            try:
                stats = runner(env)
            except Exception as exc:
                report.rows.append(FieldRow(
                    field_id=fid,
                    status="errored",
                    steps_run=0,
                    terminated=False,
                    error=f"runner_raised: {exc}",
                ))
                continue

            report.rows.append(FieldRow(
                field_id=fid,
                status="success" if stats.success else "failed",
                steps_run=stats.steps_run,
                terminated=stats.terminated,
                termination_reason=stats.termination_reason,
            ))

        return report
