# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scheduler — the §6 control flow, block-structured (v2).

Every run opens with a **silent intent gate** (:mod:`.intent_gate`): the raw
prompt plus the thread's in-chat conversation history — but no registry
projection, no canvas snapshot, no templates — is classified as "canvas
build/modify needing the knowledge base" or not. Only a ``True`` verdict
enters the orchestration below (and the sink is told via
``notify_build_engaged`` so the GUI can surface build-run affordances); a
``False`` verdict is answered directly from the model's own knowledge
(:meth:`Scheduler._answer_directly`, history included) with no knowledge base,
no tools and no canvas mutation. The gate leaves no UI trace either way. The
chat history also feeds the Understanding phase, whose brief then carries the
resolved context into the per-block passes.

Drives the fixed three-phase shape the user sees:

  1. **Understanding** — the Design agent reads the request and emits a short
     natural-language brief (surfaced as the first bubble).
  2. **Configuration** — the Orchestrator builds the canvas **block by block**
     (Robot → Actor&Env → Rewards → Training → Export). Each block is a small,
     verifiable scope (fewer tokens, higher accuracy) with its node-type tool
     enum scoped to that block; the block's nodes are highlighted while it runs.
     Bounded repair re-dispatches each blocking issue to the block that owns it.
  3. **Summary** — one small read-only completion reports what was built.

User-supplied "cheap" decisions (robot SKU, training length, compute intensity,
reference motion) arrive as :class:`~application.service.ai_orchestration.
contracts.Requirements` and are applied **deterministically** (via the C1
Mutation API, attributed to ``SYSTEM``) before any agent runs, then locked in
the agent prompts so no tokens are spent re-deciding them.

Scheduling has **no judgment**: the commit decision is "compile gate reports
zero ERROR within the repair budget", never delegated to a model. On exhaustion
the scheduler fails loud, reports the remaining blocking issues, and signals NOT
to persist. Per-phase token usage + node highlights are reported through the
:class:`MutationSink` (best-effort UI feedback; a minimal/headless sink no-ops
them). The scheduler is Qt-free.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from unitport_sdk import log_warning, tr

from ..blocks import (
    BLOCKS,
    CanvasBlock,
    block_for_node_id,
    resolve_intensity,
    resolve_length,
)
from ..contracts import (
    AgentRole,
    LlmClient,
    LlmTransportError,
    MutationSink,
    Requirements,
    Severity,
    Usage,
    ValidationIssue,
)
from ..layout import compute_layout
from ..mutation_api import MutationApi
from ..node_ref import present_snapshot, render_repair_message
from ..registry_projection import RegistryProjection, build_projection
from ..validation.compile_gate import compile_gate, errors_of
from ..validation.critic import run_critic
from ..validation.invariants import run_integrator
from .agent import Agent
from .intent_gate import run_intent_gate
from .roles import (
    V1_ACTIVE_ROLES,
    chat_role,
    design_role,
    orchestrator_role,
    report_role,
)
from . import templates

__all__ = ["Scheduler", "RunResult"]

LogFn = Callable[[str], None]
PauseGate = Callable[[], None]

#: Trainer nodes that may carry a length / env-count knob (backend-specific).
_TRAINER_SCHEMAS = ("il_ppo_trainer", "algorithm_config", "amp_trainer")


@dataclass
class RunResult:
    """The outcome of one orchestration run."""

    ok: bool
    should_commit: bool
    rounds: int = 0
    blocking: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    backend: str = ""
    sku: str = ""
    tokens: int = 0
    #: True when the silent intent gate declined the prompt and the run was a
    #: direct answer from the model's own knowledge (no canvas orchestration).
    chat_only: bool = False


class Scheduler:
    """The B2 deterministic harness (block-structured)."""

    def __init__(
        self,
        sink: MutationSink,
        llm: LlmClient,
        *,
        max_repair_rounds: int = 3,
        on_log: Optional[LogFn] = None,
        validate: Optional[
            Callable[[Dict[str, Any], "RegistryProjection"], List[ValidationIssue]]
        ] = None,
        requirements: Optional[Requirements] = None,
        pause_gate: Optional[PauseGate] = None,
    ) -> None:
        self._sink = sink
        self._llm = llm
        self._max_repair = max_repair_rounds
        self._log = on_log or sink.log_markup
        # Optional validation override for headless tests that cannot resolve a
        # real robot asset (dumped USD/MJCF tables live under USER_CONFIG_DIR).
        # Production leaves this None → the real B4 tiers run.
        self._validate_override = validate
        self._requirements = requirements
        # Called at every phase/block boundary; blocks while the user has paused
        # the run (and returns on resume / cancel). No-op by default.
        self._pause_gate = pause_gate or (lambda: None)
        self._cum_tokens = 0

    # =====================================================================

    def run(
        self,
        prompt: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RunResult:
        """Run one orchestration turn.

        ``history`` is the thread's in-chat conversation so far (OpenAI-style
        entries from ``conversation_store.to_llm_history``) — it rides along
        with the gate, the direct answer and the Understanding phase so
        follow-ups keep their referents. It is NOT the project knowledge base;
        None/empty is the legitimate first-message state.
        """
        history = list(history or [])
        # -----------------------------------------------------------------
        # Silent intent gate (knowledge-free — chat history yes, project
        # knowledge no; no UI trace). Decide whether this message is a canvas
        # build/modify request that needs the project knowledge base BEFORE
        # attaching any of it. False → answer directly from the model's own
        # knowledge; True → the normal orchestration below, with the sink told
        # first so the GUI can surface build-run affordances (e.g. the
        # checkpoint notice) only for real builds.
        # -----------------------------------------------------------------
        decision = run_intent_gate(self._llm, prompt, history=history)
        self._cum_tokens += int(decision.usage.total or 0)
        if not decision.status:
            return self._answer_directly(prompt, decision.msg, history)
        self._notify_build_engaged()

        snap = self._sink.snapshot()
        req = self._requirements

        # --- lock backend + robot deterministically (card SKU wins) ----------
        backend = str(snap.get("backend") or "").strip()
        current_sku = (
            (req.robot_sku.strip() if req and req.robot_sku else "")
            or _robot_sku(snap)
        )
        if not backend or not current_sku:
            msg = ("AI Build needs a robot and a backend on the canvas first. "
                   "Add a robot node (pick a SKU) and bind a backend, then retry.")
            issue = ValidationIssue(
                code=_generic(), severity=Severity.ERROR, message=msg, field="robot"
            )
            self._emit("canvas.ai_build.run_need_robot", f"[[error:{msg}]]")
            return RunResult(
                ok=False, should_commit=False, blocking=[issue], summary=msg
            )

        # A prompt may ask to build/convert the canvas for a DIFFERENT robot
        # ("convert this Go2 canvas to A1"). Resolve that target deterministically
        # from the prompt against the registry and lock to it — the robot is a
        # pre-decision, not an agent guess (§8). Two named targets fail loud
        # rather than silently picking one.
        try:
            retarget_sku = _resolve_target_robot(prompt, current_sku)
        except _AmbiguousRobotError as exc:
            msg = (
                "The request names more than one target robot "
                f"({', '.join(_robot_display(s) for s in exc.skus)}); name exactly "
                "one so AI Build knows which robot to build for."
            )
            issue = ValidationIssue(
                code=_generic(), severity=Severity.ERROR, message=msg, field="robot"
            )
            self._emit("canvas.ai_build.run_robot_ambiguous", f"[[error:{msg}]]")
            return RunResult(
                ok=False, should_commit=False, blocking=[issue], summary=msg,
                backend=backend, sku=current_sku,
            )

        sku = retarget_sku or current_sku
        if retarget_sku is not None:
            self._emit(
                "canvas.ai_build.run_retarget_robot",
                "[[info:Retargeting the canvas robot: {frm} -> {to}]]",
                frm=_robot_display(current_sku), to=_robot_display(sku),
            )

        families = _families_for_sku(sku)
        projection = build_projection(backend, families)

        # =================================================================
        # Phase 1 — Understanding
        # =================================================================
        self._pause_gate()
        self._begin_phase(
            "understanding", tr("canvas.ai_build.phase_understanding", "Understanding")
        )
        self._emit(
            "canvas.ai_build.run_design",
            "[[info:Design — locked {sku} / {backend} ({fam})]]",
            sku=sku, backend=backend,
            fam="/".join(sorted(families)) or "unknown",
        )
        self._progress(0.08, "design")
        design = Agent(design_role(projection), self._llm)
        # The Understanding phase carries the chat history so a follow-up
        # request resolves its referents; the brief it produces is the
        # compressed carrier of that context into the per-block passes (the
        # blocks do NOT get the raw history again — token frugality).
        brief = design.ask(_design_user(prompt, snap), history=history)
        self._phase_tokens(design.last_usage)
        if brief.strip():
            self._log_text(brief)

        # --- deterministic, no-token requirements (SYSTEM) -------------------
        # Always lock the robot node to the resolved sku (this is where a
        # Go2 -> A1 retarget actually lands on the canvas), then the card's
        # length / intensity / motion when a Requirements card was supplied.
        locked = self._apply_requirements(projection, snap, req, backend, sku)
        snap = self._sink.snapshot()

        # §8: if the prompt asked to switch robots, the switch MUST have landed —
        # never proceed (and later report success) on a canvas still showing the
        # old robot. That "illusion of success" is exactly what this run must not
        # produce.
        if retarget_sku is not None and _robot_node_asset_sku(snap) != sku:
            msg = (
                f"Could not switch the canvas robot to {_robot_display(sku)}; the "
                "robot node still holds the previous asset. Aborting rather than "
                "reporting a change that did not happen."
            )
            issue = ValidationIssue(
                code=_generic(), severity=Severity.ERROR, message=msg,
                node_id="robot", field="asset_id",
            )
            self._emit("canvas.ai_build.run_retarget_failed", f"[[error:{msg}]]")
            return RunResult(
                ok=False, should_commit=False, blocking=[issue], summary=msg,
                backend=backend, sku=sku,
            )

        # --- force-run AUTO PD (deterministic, no tokens) --------------------
        # Derive (ωn,ζ) + effort/velocity caps from the robot ASSET so the robot
        # never ships with wrong family-default gains (§10 — the Go2-Nucleus
        # class of bug). Writes ONLY asset-specific (brand/USD) values; a
        # family-only source leaves pd_groups={} (the compiler uses family
        # defaults there anyway). This mirrors pressing the RobotNode AUTO button.
        pd_locked = self._apply_pd_auto(projection, sku)
        if pd_locked:
            locked = (locked + "\n" + pd_locked).strip() if locked else pd_locked
        snap = self._sink.snapshot()

        # =================================================================
        # Phase 2 — Configuration (block by block)
        # =================================================================
        baseline = templates.find_same_family_template(backend, families)
        if baseline is None:
            self._emit(
                "canvas.ai_build.run_no_reference",
                "[[warning:No same-family reference for this task; building from "
                "registry semantics.]]",
            )
        n_blocks = len(BLOCKS)
        for index, block in enumerate(BLOCKS):
            self._pause_gate()
            title = tr(block.title_key, block.default_title)
            self._begin_phase("block:" + block.id, title)
            self._highlight(block.schema_ids, True)
            self._emit(
                "canvas.ai_build.run_block",
                "[[info:Block {i}/{n} — {title}]]",
                i=index + 1, n=n_blocks, title=title,
            )
            agent = self._block_agent(projection, block, baseline, locked)
            turn = agent.run(_block_user(prompt, brief, snap, baseline, block, projection))
            self._phase_tokens(turn.usage)
            # Reward self-review (局部解审视): teach principle #3 by making the
            # rewards agent audit its OWN config for degenerate optima and revise
            # where the task warrants — the agent won't self-audit unprompted.
            if block.id == "rewards":
                self._reward_self_review(projection, block, baseline, locked)
            # D: catch nodes THIS block created with an unwired required input
            # right away — node-addressable, near the cause — rather than
            # deferring the whole diagnosis to the final gate.
            self._repair_block_orphans(projection, block, baseline, locked)
            self._highlight(block.schema_ids, False)
            self._progress(0.15 + 0.6 * (index + 1) / n_blocks, "building")
            snap = self._sink.snapshot()

        # =================================================================
        # bounded repair loop over the validation tiers (routed to blocks)
        # =================================================================
        rounds = 0
        blocking: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []
        while True:
            rounds += 1
            self._pause_gate()
            snap = self._sink.snapshot()
            issues = (self._validate_override or self._validate)(snap, projection)
            blocking = errors_of(issues)
            warnings = [i for i in issues if i.severity == Severity.WARNING]
            if not blocking:
                break
            if rounds > self._max_repair:
                break
            self._begin_phase(
                "repair", tr("canvas.ai_build.phase_repair", "Repair")
            )
            self._emit(
                "canvas.ai_build.run_repairing",
                "[[warning:{n} blocking issue(s); repairing (round {r})…]]",
                n=len(blocking), r=rounds,
            )
            self._repair_round(projection, snap, blocking, baseline, locked)

        # Deterministic structural completeness (user directive: never ship a
        # structurally-incomplete canvas). If the LLM repair budget left blocking
        # STRUCTURAL holes, close them with a no-token SYSTEM pass and re-validate
        # — the workflow delivers a trainable skeleton rather than aborting.
        if blocking:
            self._begin_phase(
                "complete", tr("canvas.ai_build.phase_complete", "Completing structure")
            )
            self._emit(
                "canvas.ai_build.run_completing",
                "[[info:Repair budget spent with {n} blocking issue(s); completing "
                "the required structure deterministically…]]", n=len(blocking),
            )
            if self._ensure_structural_completeness(projection):
                snap = self._sink.snapshot()
                issues = (self._validate_override or self._validate)(snap, projection)
                blocking = errors_of(issues)
                warnings = [i for i in issues if i.severity == Severity.WARNING]

        # Tidy-layout ONCE, after every node exists (blocks + repair): position all
        # nodes into left-to-right block columns centred on the view, THEN draw the
        # region boxes around the final geometry. Doing it once (not per-block) is
        # what keeps boxes wrapping their nodes instead of scattered placeholders.
        self._apply_layout()
        self._progress(0.85, "validating")

        # =================================================================
        # deterministic commit decision + Phase 3 report
        # =================================================================
        if not blocking:
            self._emit("canvas.ai_build.run_compiled",
                       "[[success:Canvas compiled — no blocking issues.]]")
            self._report_phase(projection, prompt, brief)
            return RunResult(
                ok=True, should_commit=True, rounds=rounds, warnings=warnings,
                summary="compiled clean", backend=backend, sku=sku,
                tokens=self._cum_tokens,
            )
        self._emit(
            "canvas.ai_build.run_exhausted",
            "[[error:Repair exhausted — {n} blocking issue(s); nothing was "
            "persisted.]]",
            n=len(blocking),
        )
        for i in blocking:
            self._emit("canvas.ai_build.run_issue", "[[error:{message}]]",
                       message=i.message)
        return RunResult(
            ok=False, should_commit=False, rounds=rounds, blocking=blocking,
            warnings=warnings, summary="blocking issues remain",
            backend=backend, sku=sku, tokens=self._cum_tokens,
        )

    # =====================================================================
    # intent gate (silent pre-step) + gate-declined direct answer
    # =====================================================================

    def _answer_directly(
        self,
        prompt: str,
        gate_msg: str,
        history: List[Dict[str, str]],
    ) -> RunResult:
        """Gate-declined path: answer the user from the model's OWN knowledge.

        No knowledge base is attached, no tools are offered, and the canvas is
        never touched — the reply renders as one plain NL bubble in its own
        card. The chat history rides along so follow-up questions keep their
        referents. Nothing is committed; a run that answers a question is
        ``ok`` but never persists anything. A transport failure raises (§8)
        exactly like any other phase.
        """
        self._pause_gate()
        self._begin_phase("chat", tr("canvas.ai_build.phase_chat", "Assistant"))
        responder = Agent(chat_role(), self._llm)
        text = responder.ask(prompt, history=history)
        self._phase_tokens(responder.last_usage)
        if text.strip():
            self._log_text(text)
        return RunResult(
            ok=True, should_commit=False, chat_only=True,
            summary=gate_msg or "answered from the model's own knowledge",
            tokens=self._cum_tokens,
        )

    def _notify_build_engaged(self) -> None:
        """Tell the sink the gate admitted this prompt as a real build run.

        Fired once per run, after the silent gate and before any snapshot or
        mutation, so the GUI shows build-run affordances (checkpoint notice)
        only when the canvas will actually be acted on. Cosmetic — a minimal /
        headless sink simply lacks the hook."""
        fn = getattr(self._sink, "notify_build_engaged", None)
        if callable(fn):
            fn()

    # =====================================================================
    # agents
    # =====================================================================

    def _block_agent(
        self,
        projection: RegistryProjection,
        block: CanvasBlock,
        baseline: Optional[Dict[str, Any]],
        locked: str,
    ) -> Agent:
        mapi = MutationApi(self._sink, projection, agent=AgentRole.ORCHESTRATOR)
        return Agent(
            orchestrator_role(
                projection,
                baseline_available=baseline is not None,
                block=block,
                locked=locked,
            ),
            self._llm,
            mutation_api=mapi,
            tools=projection.to_tool_schemas(
                AgentRole.ORCHESTRATOR, allowed_schema_ids=block.schema_ids
            ),
        )

    def _generic_agent(
        self,
        projection: RegistryProjection,
        baseline: Optional[Dict[str, Any]],
        locked: str,
    ) -> Agent:
        """Full-tool orchestrator for repairs that don't resolve to a block."""
        mapi = MutationApi(self._sink, projection, agent=AgentRole.ORCHESTRATOR)
        return Agent(
            orchestrator_role(
                projection, baseline_available=baseline is not None, locked=locked
            ),
            self._llm,
            mutation_api=mapi,
            tools=projection.to_tool_schemas(AgentRole.ORCHESTRATOR),
        )

    def _repair_round(
        self,
        projection: RegistryProjection,
        snap: Dict[str, Any],
        blocking: List[ValidationIssue],
        baseline: Optional[Dict[str, Any]],
        locked: str,
    ) -> None:
        """Re-dispatch each blocking issue to the block that owns its node."""
        by_block: Dict[str, "tuple[CanvasBlock, List[ValidationIssue]]"] = {}
        unrouted: List[ValidationIssue] = []
        for issue in blocking:
            block = block_for_node_id(snap, issue.node_id) if issue.node_id else None
            if block is None:
                unrouted.append(issue)
            else:
                by_block.setdefault(block.id, (block, []))[1].append(issue)

        for block, issues in by_block.values():
            self._pause_gate()
            self._highlight(block.schema_ids, True)
            turn = self._block_agent(projection, block, baseline, locked).run(
                render_repair_message(issues, snap, projection)
            )
            self._phase_tokens(turn.usage)
            self._highlight(block.schema_ids, False)

        # KEEP the generic fallback (Range 0.8): node-less issues (I4 backend
        # mismatch, canvas lowering failure, missing-required-node, ambiguous
        # algorithm) carry no node_id and must stay routable.
        if unrouted:
            self._pause_gate()
            turn = self._generic_agent(projection, baseline, locked).run(
                render_repair_message(unrouted, snap, projection)
            )
            self._phase_tokens(turn.usage)

    def _repair_block_orphans(
        self,
        projection: RegistryProjection,
        block: CanvasBlock,
        baseline: Optional[Dict[str, Any]],
        locked: str,
    ) -> None:
        """Block-exit orphan check (D): if a node this block owns has an unwired
        required input, emit the node-addressable issue and run ONE immediate
        repair pass scoped to the block — near the cause, loud, not deferred.

        Cross-block wiring is fine: ``connect`` is not block-scoped (Range 0.6),
        so this block's Orchestrator can wire e.g. ``robot_pipe`` from the robot
        node. Reuses ``check_topology`` (single source; respects conditional_on)
        — never reimplements the gate."""
        snap = self._sink.snapshot()
        orphans = _block_orphan_port_issues(snap, block, projection)
        if not orphans:
            return
        self._emit(
            "canvas.ai_build.block_orphan",
            "[[warning:Block {b} left {n} required input(s) unwired; wiring now.]]",
            b=block.default_title, n=len(orphans),
        )
        self._pause_gate()
        self._highlight(block.schema_ids, True)
        turn = self._block_agent(projection, block, baseline, locked).run(
            render_repair_message(orphans, snap, projection)
        )
        self._phase_tokens(turn.usage)

    def _reward_self_review(
        self,
        projection: RegistryProjection,
        block: CanvasBlock,
        baseline: Optional[Dict[str, Any]],
        locked: str,
    ) -> None:
        """Operationalize principle #3 (局部解审视): the rewards agent audits its
        OWN reward config against the degenerate-optima checklist and revises
        where THIS task is exposed.

        This is teaching-by-doing, NOT a gate — the agent uses its own judgment
        (it may make no change) and nothing blocks. It exists because prompt +
        baseline alone don't make the agent self-audit (the field-observed A1
        canvas had the full reference backbone in-context and still shipped a
        reward-hacking config). Reuses the rewards block agent, so it already
        carries the doctrine guardrails + the per-term category/hint vocabulary."""
        # No _highlight here: the enclosing block loop already has this block
        # highlighted ON for the whole iteration (self-review runs before the
        # block's highlight-off), so an extra toggle would unbalance the pair.
        self._pause_gate()
        snap = self._sink.snapshot()
        self._emit(
            "canvas.ai_build.reward_review",
            "[[info:Reward self-review — checking for degenerate optima "
            "(standing / single-leg-lift / survive-to-wall)…]]",
        )
        turn = self._block_agent(projection, block, baseline, locked).run(
            _reward_review_user(snap, projection)
        )
        self._phase_tokens(turn.usage)

    def _report_phase(
        self, projection: RegistryProjection, prompt: str, brief: str
    ) -> None:
        """Phase 3 — one small NL summary. Failure is loud-but-non-fatal (§8)."""
        self._pause_gate()
        self._begin_phase("report", tr("canvas.ai_build.phase_report", "Summary"))
        self._progress(0.88, "summarizing")
        snap = self._sink.snapshot()
        reporter = Agent(report_role(projection), self._llm)
        try:
            text = reporter.ask(_report_user(prompt, brief, snap))
        except LlmTransportError as exc:
            # The canvas is already built and validated; only the cosmetic
            # summary failed. Surface it loudly, do NOT roll back the build.
            self._emit(
                "canvas.ai_build.report_failed",
                "[[warning:Could not generate the summary ({detail}); the canvas "
                "is built and valid.]]",
                detail=str(exc),
            )
            return
        self._phase_tokens(reporter.last_usage)
        if text.strip():
            self._log_text(text)

    # =====================================================================
    # deterministic requirements (no agent tokens)
    # =====================================================================

    def _apply_requirements(
        self,
        projection: RegistryProjection,
        snap: Dict[str, Any],
        req: Optional[Requirements],
        backend: str,
        sku: str,
    ) -> str:
        """Apply the deterministic (no-token) decisions via the C1 API; return a
        locked description for the agent prompts. Failures are surfaced loudly
        (§8).

        ``sku`` is the resolved target robot (possibly a prompt-driven retarget);
        the robot node is ALWAYS locked to it here — this is where a Go2 -> A1
        switch actually lands on the canvas. The card's length / intensity /
        motion decisions only apply when a Requirements card was supplied.
        """
        mapi = MutationApi(self._sink, projection, agent=AgentRole.SYSTEM)
        locked: List[str] = []

        # robot → robot.asset_id (place a robot node if none). Always ensure the
        # canvas robot matches the locked target sku, whether it came from a
        # Requirements card, the existing canvas, or a prompt-driven retarget.
        nid = _find_node(snap, "robot")
        if nid is None:
            placed = mapi.place_node("robot")
            nid = placed.node_id or None
        if nid and sku:
            self._apply_one(mapi, nid, "asset_id", sku)
        locked.append(f"- robot = {_robot_display(sku)}")

        if req is None:
            self._emit(
                "canvas.ai_build.req_applied",
                "[[info:Applied your locked requirements:]]",
            )
            for line in locked:
                self._log(line)
            return "\n".join(locked)

        # training length → trainer.max_iterations
        iters = resolve_length(backend, req.length_tier, req.max_iterations)
        if iters is not None:
            if self._set_trainer_param(mapi, snap, projection, "max_iterations", iters):
                locked.append(f"- max_iterations = {iters}")
            else:
                self._emit(
                    "canvas.ai_build.req_no_trainer",
                    "[[warning:No trainer node yet for the training-length "
                    "requirement; it will be applied by the agent.]]",
                )
                locked.append(f"- max_iterations = {iters} (apply on the trainer)")

        # compute intensity → trainer.num_envs (IsaacLab only; SB3 auto-parallel)
        envs = resolve_intensity(backend, req.intensity_tier, req.num_envs)
        if envs is not None:
            if self._set_trainer_param(mapi, snap, projection, "num_envs", envs):
                locked.append(f"- num_envs = {envs}")
            else:
                self._emit(
                    "canvas.ai_build.req_no_envs",
                    "[[warning:Compute intensity has no parallel-env knob for this "
                    "backend's trainer; skipped.]]",
                )

        # reference motion → locked hint (clip binding is per-training-item work
        # the Training-block agent does; not a single deterministic param).
        if req.motion.strip():
            locked.append(
                f"- reference motion = {req.motion.strip()} "
                "(bind this clip onto the relevant training items)"
            )

        if locked:
            self._emit(
                "canvas.ai_build.req_applied",
                "[[info:Applied your locked requirements:]]",
            )
            for line in locked:
                self._log(line)
        return "\n".join(locked)

    def _apply_pd_auto(self, projection: RegistryProjection, sku: str) -> str:
        """Force-run the RobotNode AUTO PD resolution deterministically (§10).

        Resolves the recommended ``(ωn, ζ)`` + effort/velocity caps for ``sku``
        (brand manifest → USD-drive reverse-derive → family; the resolver already
        rejects out-of-band USD placeholder gains, §8). Per the ruling, writes
        ``pd_groups`` ONLY when the source is asset-specific (brand/USD) — a
        family-only source leaves ``pd_groups={}`` (the compiler applies family
        defaults there anyway). Caps are written whenever the resolver supplies
        them. Returns locked-lines for the agent prompt (so the robot-block
        Orchestrator leaves the AUTO gains alone). Never raises — a soft failure
        logs and leaves family defaults."""
        from application.physics.pd_preview import resolve_recommended_actuators

        nid = _find_node(self._sink.snapshot(), "robot")
        if nid is None:
            return ""
        try:
            rec = resolve_recommended_actuators(sku)
        except Exception as exc:  # soft: AUTO is best-effort; family default holds
            self._emit(
                "canvas.ai_build.pd_auto_failed",
                "[[warning:AUTO PD could not resolve ({e}); leaving family "
                "defaults for this robot.]]",
                e=str(exc),
            )
            return ""
        if rec is None:
            self._emit(
                "canvas.ai_build.pd_auto_none",
                "[[warning:AUTO PD found no recommendation for this robot; "
                "leaving family defaults.]]",
            )
            return ""

        mapi = MutationApi(self._sink, projection, agent=AgentRole.SYSTEM)
        locked: List[str] = []
        # (ωn,ζ): only asset-specific sources are worth pinning; family stays {}.
        if rec.source_pd in ("brand", "usd") and rec.groups:
            pd_groups = {
                str(gid): {"omega_n": float(w), "zeta": float(z)}
                for gid, (w, z) in rec.groups.items()
            }
            if self._apply_one(mapi, nid, "pd_groups", pd_groups):
                locked.append(
                    f"- PD (ωn,ζ) = AUTO from {rec.source_pd} — do NOT change"
                )
        # Effort / velocity caps whenever a source supplied them.
        if rec.effort_limit is not None:
            if self._apply_one(mapi, nid, "effort_limit", float(rec.effort_limit)):
                locked.append(f"- effort_limit = {float(rec.effort_limit):g} (AUTO)")
        if rec.velocity_limit is not None:
            if self._apply_one(mapi, nid, "velocity_limit", float(rec.velocity_limit)):
                locked.append(
                    f"- velocity_limit = {float(rec.velocity_limit):g} (AUTO)"
                )

        # Always report the source/status (transparency), even when family-only.
        self._emit("canvas.ai_build.pd_auto", "[[info:AUTO PD — {notes}]]", notes=rec.notes)
        if rec.status == "need_dump_usd":
            self._emit(
                "canvas.ai_build.pd_auto_dump",
                "[[warning:A richer PD recommendation is available after "
                "re-dumping USD for this robot.]]",
            )
        return "\n".join(locked)

    def _ensure_structural_completeness(
        self, projection: RegistryProjection
    ) -> bool:
        """Deterministically guarantee the family's structural skeleton.

        The agent must NEVER leave the canvas structurally incomplete (user
        directive: "画布没做完就不应该中止agent工作流，agent必须至少交付结构完整的
        画布"). After the LLM repair budget is spent, this SYSTEM step (no tokens)
        closes the remaining STRUCTURAL holes so the canvas is trainable:

          1. place any missing family-required node (manifest defaults → valid
             content the agent can still refine);
          2. wire any unwired REQUIRED input port from a type-compatible source
             on the canvas (e.g. ``il_observation.obs_vector`` → the trainer /
             policy-net ``obs_vector`` inputs) — reuses the gate's own
             ``_check_required_ports`` + ``source_candidates``, looped so a
             newly-wired provider can satisfy the next port;
          3. give every enabled command item a ``reward_in__<item>`` edge (from a
             Rewards node, placed if none) so no item trains on a constant-0
             reward.

        Content (obs terms, reward weights, per-item differentiation) stays the
        agent's job; this only makes the graph structurally valid. Fail-loud
        (§8): an op it cannot complete is surfaced, never silently skipped.
        Returns True if it changed the canvas."""
        from application.compiler.lowering import canvas_to_ir
        from application.training.spec_validator import (
            _check_required_ports,
            _check_training_item_reward_coverage,
        )
        from ..contracts import IssueCode
        from ..node_ref import source_candidates

        mapi = MutationApi(self._sink, projection, agent=AgentRole.SYSTEM)
        changed = False

        # 1) place missing required nodes.
        snap = self._sink.snapshot()
        for schema in projection.required_nodes:
            if _find_node(snap, schema) is None:
                if mapi.place_node(schema).ok:
                    changed = True
                    self._emit(
                        "canvas.ai_build.scaffold_node",
                        "[[info:Completing structure — added required {s} node.]]",
                        s=schema,
                    )
        snap = self._sink.snapshot()

        # 2) wire unwired REQUIRED input ports from a compatible source (looped:
        #    wiring a provider can unblock the next port that consumes it).
        for _ in range(4):
            try:
                port_issues = [
                    i for i in _check_required_ports(canvas_to_ir(snap))
                    if i.code == IssueCode.MISSING_REQUIRED_PORT and i.node_id and i.field
                ]
            except (ValueError, KeyError, TypeError) as exc:
                self._emit(
                    "canvas.ai_build.scaffold_defer",
                    "[[warning:Structure completion deferred (canvas not lowerable "
                    "yet): {e}]]", e=str(exc),
                )
                port_issues = []
            if not port_issues:
                break
            progressed = False
            for issue in port_issues:
                if self._wire_required_port(mapi, snap, projection, issue, source_candidates):
                    progressed = changed = True
            if not progressed:
                break
            snap = self._sink.snapshot()

        # 3) cover enabled command items lacking a reward edge.
        try:
            reward_gaps = _check_training_item_reward_coverage(canvas_to_ir(snap))
        except (ValueError, KeyError, TypeError):
            reward_gaps = []
        if reward_gaps:
            rewards_id = _find_node(snap, "rewards")
            if rewards_id is None:
                placed = mapi.place_node("rewards")
                rewards_id = placed.node_id if placed.ok else None
                changed = changed or bool(rewards_id)
            motion_id = _find_node(snap, "training_motion")
            for issue in reward_gaps:
                if rewards_id is None or motion_id is None:
                    break
                if not str(issue.field or "").startswith("training_items."):
                    continue
                iid = str(issue.field)[len("training_items."):]
                res = mapi.connect(
                    rewards_id, "reward_pipe", motion_id, f"reward_in__{iid}"
                )
                if res.ok:
                    changed = True
                    self._emit(
                        "canvas.ai_build.scaffold_reward",
                        "[[info:Completing structure — wired reward for item {i}.]]",
                        i=iid,
                    )
        return changed

    def _wire_required_port(
        self, mapi, snap, projection, issue, source_candidates
    ) -> bool:
        """Wire one unwired required input port from a type-compatible source.
        Returns True if an edge was created."""
        tgt_schema = _schema_of(snap, issue.node_id)
        port = projection.port(tgt_schema, issue.field) if tgt_schema else None
        if port is None:
            return False
        for src_id in source_candidates(snap, projection, port.type):
            src_port = _first_output_of_type(projection, _schema_of(snap, src_id), port.type)
            if src_port is None:
                continue
            if mapi.connect(src_id, src_port, issue.node_id, issue.field).ok:
                self._emit(
                    "canvas.ai_build.scaffold_port",
                    "[[info:Completing structure — wired {p} on {n}.]]",
                    p=issue.field, n=issue.node_id,
                )
                return True
        self._emit(
            "canvas.ai_build.scaffold_port_fail",
            "[[warning:Could not auto-wire required port {p} on {n} — no compatible "
            "source on the canvas.]]", p=issue.field, n=issue.node_id,
        )
        return False

    def _apply_layout(self) -> None:
        """Deterministic tidy-layout SYSTEM step (§11): arrange nodes into block
        columns (left-to-right dataflow order) + draw the titled region boxes.

        No tokens, purely cosmetic. A sink that can't lay out (a headless/minimal
        one) simply lacks ``apply_layout`` → no-op. Never raises: a layout hiccup
        must not abort an otherwise-valid build, but it is surfaced loudly (a
        logged warning, never a silent swallow) — layout is non-authoritative UI
        feedback, like ``begin_phase`` / ``highlight_block``."""
        fn = getattr(self._sink, "apply_layout", None)
        if not callable(fn):
            return
        try:
            fn(compute_layout(self._sink.snapshot()))
        except Exception as exc:  # cosmetic feedback must not break a valid build
            log_warning(f"[ai.scheduler] auto-layout skipped: {exc}")

    def _set_trainer_param(
        self,
        mapi: MutationApi,
        snap: Dict[str, Any],
        projection: RegistryProjection,
        key: str,
        value: Any,
    ) -> bool:
        """Set ``key`` on whichever trainer node carries it; False if none does."""
        for schema in _TRAINER_SCHEMAS:
            nid = _find_node(snap, schema)
            if nid is None:
                continue
            if key in projection.param_schema(schema):
                return self._apply_one(mapi, nid, key, value)
        return False

    def _apply_one(
        self, mapi: MutationApi, node_id: str, key: str, value: Any
    ) -> bool:
        """Set one param via the C1 API; surface a tier-1 rejection loudly."""
        result = mapi.set_param(node_id, key, value)
        if not result.ok:
            for issue in result.issues:
                self._emit(
                    "canvas.ai_build.req_failed",
                    "[[warning:Could not apply requirement {key}: {message}]]",
                    key=key, message=str(issue),
                )
        return result.ok

    # =====================================================================
    # validation + sink/UI helpers
    # =====================================================================

    def _validate(
        self, snap: Dict[str, Any], projection: RegistryProjection
    ) -> List[ValidationIssue]:
        """Run tier-2 compile gate + tier-3 integrator + tier-4 critic; dedupe."""
        issues: List[ValidationIssue] = []
        issues.extend(compile_gate(snap))
        issues.extend(r.issue for r in run_integrator(snap, V1_ACTIVE_ROLES))
        issues.extend(run_critic(snap, projection, self._llm))
        return _dedupe(issues)

    def _begin_phase(self, phase_id: str, title: str) -> None:
        fn = getattr(self._sink, "begin_phase", None)
        if callable(fn):
            fn(phase_id, title)

    def _phase_tokens(self, usage: Usage) -> None:
        turn_total = int(getattr(usage, "total", 0) or 0)
        self._cum_tokens += turn_total
        fn = getattr(self._sink, "set_phase_tokens", None)
        if callable(fn):
            fn(turn_total, self._cum_tokens)

    def _highlight(self, schema_ids, on: bool) -> None:
        fn = getattr(self._sink, "highlight_block", None)
        if callable(fn):
            fn(frozenset(schema_ids), bool(on))

    def _progress(self, fraction: float, text: str = "") -> None:
        fn = getattr(self._sink, "set_progress", None)
        if callable(fn):
            fn(float(fraction), text)

    def _log_text(self, text: str) -> None:
        """Log plain natural-language text, one line per source line.

        Lines carry no ``[[role:..]]`` markup → they render in the default role
        (an ordinary NL bubble), which is what the Understanding / Summary
        phases want.
        """
        for line in str(text).splitlines():
            stripped = line.rstrip()
            if stripped:
                self._log(stripped)

    def _emit(self, key: str, default: str, **fields: Any) -> None:
        """Render a tr template (with [[role:..]] markup) and log it (§3)."""
        template = tr(key, default)
        try:
            rendered = template.format(**fields) if fields else template
        except (KeyError, IndexError):
            rendered = default.format(**fields) if fields else default
        self._log(rendered)


# =============================================================================
# prompt builders
# =============================================================================

def _design_user(prompt: str, snap: Dict[str, Any]) -> str:
    return (
        f"User request:\n{prompt}\n\nCurrent canvas node types: "
        f"{sorted({str(n.get('schema_id')) for n in snap.get('nodes', [])})}"
    )


def _block_user(
    prompt: str,
    brief: str,
    snap: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    block: CanvasBlock,
    projection: RegistryProjection,
) -> str:
    parts = [f"User request:\n{prompt}"]
    if brief:
        parts.append(f"Design brief:\n{brief}")
    # Backend-scoped: only offer this block's node types that are valid for the
    # locked backend (never mention an SB3-only node on an isaac_lab run).
    parts.append(
        f"Active block: {block.default_title} "
        f"(only node types {projection.block_schema_ids(block.schema_ids)})"
    )
    parts.append(
        "Current canvas (the blackboard you mutate — nodes keyed by node_id, "
        "the exact string to pass to set_param/connect; 'type' is the schema, "
        "never a reference id):\n"
        + json.dumps(present_snapshot(snap), ensure_ascii=False)
    )
    if baseline is not None:
        parts.append("Same-family reference canvas (mirror its wiring for this "
                     "block, adapt params):\n"
                     + json.dumps(_condense(baseline), ensure_ascii=False))
    return "\n\n".join(parts)


def _reward_review_user(
    snap: Dict[str, Any], projection: RegistryProjection
) -> str:
    """Self-review message for the rewards block (principle #3, 局部解审视).

    Hands the agent its current reward config + the degenerate-optima checklist
    and asks it to audit and revise WHERE THE TASK WARRANTS — explicitly NOT a
    checklist to satisfy (a risk that doesn't apply to this task → no change)."""
    from registers import reward_doctrine

    checklist = reward_doctrine.degenerate_optima()
    caveat = reward_doctrine.caveat()
    parts = [
        "Reward self-review (局部解审视). You have just configured the rewards "
        "block. Abstract each reward into a boundary constraint and ask what "
        "actually maximises the reward sum, then check the current config "
        "against the degenerate optima below. WHERE YOUR JUDGMENT SAYS THIS "
        "TASK is exposed, add or adjust the counter-term (assign_reward_bundle / "
        "set_param). If the config is already sound, or a listed risk does not "
        "apply to this task, make NO change — never add a term just to satisfy "
        "the list. When done, stop calling tools.",
        "Degenerate optima to check:\n"
        + json.dumps(checklist, ensure_ascii=False),
    ]
    if caveat:
        parts.append(caveat)
    parts.append(
        "Current reward configuration (nodes keyed by node_id):\n"
        + json.dumps(present_snapshot(snap), ensure_ascii=False)
    )
    return "\n\n".join(parts)


def _report_user(prompt: str, brief: str, snap: Dict[str, Any]) -> str:
    parts = [f"Original user request:\n{prompt}"]
    if brief:
        parts.append(f"Design brief:\n{brief}")
    parts.append("Final built canvas:\n"
                 + json.dumps(_condense(snap), ensure_ascii=False))
    return "\n\n".join(parts)


def _condense(canvas: Dict[str, Any]) -> Dict[str, Any]:
    """A token-frugal view of a canvas: nodes (id, schema_id, params) + edges."""
    return {
        "backend": canvas.get("backend"),
        "robot_id": canvas.get("robot_id"),
        "nodes": [
            {"id": n.get("id"), "schema_id": n.get("schema_id"),
             "params": n.get("params", {})}
            for n in canvas.get("nodes", [])
        ],
        "edges": canvas.get("edges", []),
    }


# =============================================================================
# helpers
# =============================================================================

class _AmbiguousRobotError(Exception):
    """The prompt named more than one candidate target robot (fail-loud §8)."""

    def __init__(self, skus: List[str]) -> None:
        super().__init__("ambiguous target robot")
        self.skus = list(skus)


def _resolve_target_robot(prompt: str, current_sku: str) -> Optional[str]:
    """Resolve a robot the PROMPT asks to build for, if different from current.

    Deterministic + registry-backed (§8): every maximal ASCII-alphanumeric run in
    the prompt is looked up as a robot alias / model / SKU (``resolve_to_sku``),
    so CJK text around a token is a natural boundary. The referenced robots minus
    the current one are the requested targets:

      * exactly one   -> that SKU (a Go2 -> A1 style retarget);
      * none          -> ``None`` (keep the current robot — the common case);
      * more than one -> :class:`_AmbiguousRobotError` (never guess which).
    """
    from registers import robots

    referenced: set = set()
    for token in set(re.findall(r"[a-z0-9]+", str(prompt or "").lower())):
        matched = robots.resolve_to_sku(token)
        if matched:
            referenced.add(matched)
    targets = referenced - {current_sku}
    if not targets:
        return None
    if len(targets) > 1:
        raise _AmbiguousRobotError(sorted(targets))
    return next(iter(targets))


def _robot_display(sku: str) -> str:
    """Human-facing name for a SKU (falls back to the raw sku on a miss)."""
    from registers import robots

    spec = robots.get_robot_spec(sku)
    return (spec.name if spec is not None and spec.name else sku) or sku


def _robot_node_asset_sku(snap: Dict[str, Any]) -> str:
    """Resolve the robot NODE's ``asset_id`` to a SKU.

    Reads the node param directly (NOT the top-level ``robot_id``, which may lag
    a just-applied change) so a retarget can be verified as actually landed (§8).
    """
    from registers import robots

    for n in snap.get("nodes", []) or []:
        if str(n.get("schema_id")) == "robot":
            cell = (n.get("params") or {}).get("asset_id")
            raw = cell.get("value") if isinstance(cell, dict) else cell
            return robots.resolve_to_sku(str(raw or "")) or ""
    return ""


def _robot_sku(snap: Dict[str, Any]) -> str:
    sku = str(snap.get("robot_id") or "").strip()
    if sku:
        return sku
    for n in snap.get("nodes", []):
        if str(n.get("schema_id")) == "robot":
            cell = (n.get("params") or {}).get("asset_id")
            return str((cell or {}).get("value") if isinstance(cell, dict) else (cell or ""))
    return ""


def _find_node(snap: Dict[str, Any], schema_id: str) -> Optional[str]:
    """Return the instance id of the first node with ``schema_id``, or None."""
    for n in snap.get("nodes", []) or []:
        if str(n.get("schema_id")) == schema_id:
            nid = str(n.get("id") or "")
            return nid or None
    return None


def _schema_of(snap: Dict[str, Any], node_id: str) -> str:
    """schema_id of a node instance on the snapshot, or '' if not found."""
    for n in snap.get("nodes", []) or []:
        if str(n.get("id")) == str(node_id):
            return str(n.get("schema_id") or "")
    return ""


def _first_output_of_type(
    projection: RegistryProjection, schema_id: str, pipe_type: str
) -> Optional[str]:
    """Name of the first OUTPUT port on ``schema_id`` able to feed ``pipe_type``
    (exact match or ``any`` either side; mirrors ``source_candidates``)."""
    if not schema_id:
        return None
    for p in projection.ports(schema_id).get("outputs", []):
        if p.type == pipe_type or p.type == "any" or pipe_type == "any":
            return p.name
    return None


def _families_for_sku(sku: str) -> frozenset:
    from registers import robots

    spec = robots.get_robot_spec(sku)
    return frozenset(spec.families) if spec is not None else frozenset()


def _block_orphan_port_issues(
    snap: Dict[str, Any], block: CanvasBlock, projection: RegistryProjection
) -> List[ValidationIssue]:
    """``MISSING_REQUIRED_PORT`` issues for nodes owned by ``block`` whose
    provider ALREADY exists on the canvas.

    Reuses the single-source ``_check_required_ports`` **directly** — not
    ``check_topology``, which early-returns before the port walk while the
    trainer/family is still UNKNOWN mid-build (so it would never see a robot_pipe
    orphan at the actor block's exit). Respects ``conditional_on`` gating (no
    reimplementation). Filters to (a) this block's nodes and (b) inputs a
    compatible source node already provides — an input whose provider comes in a
    LATER block is not THIS block's orphan (the final gate owns that). Returns
    ``[]`` when the partial canvas can't be lowered yet."""
    from application.compiler.lowering import canvas_to_ir
    from application.training.spec_validator import _check_required_ports
    from ..contracts import IssueCode
    from ..node_ref import source_candidates

    try:
        ir = canvas_to_ir(snap)
    except (ValueError, KeyError, TypeError) as exc:
        # WHY KEPT (§8c): a partial mid-build canvas may not lower yet (e.g. a
        # node placed before its params are complete). The AUTHORITATIVE final
        # compile gate still catches this orphan — this early check just defers.
        # Narrow catch (mirrors validation.compile_gate); a bug in the port walk
        # below is NOT swallowed (it propagates fail-loud, like the final gate).
        log_warning(
            f"[ai.scheduler] block-exit orphan check deferred "
            f"(canvas not lowerable yet): {exc}"
        )
        return []
    issues = _check_required_ports(ir)
    out: List[ValidationIssue] = []
    for issue in issues:
        if issue.code != IssueCode.MISSING_REQUIRED_PORT or not issue.node_id or not issue.field:
            continue
        owner = block_for_node_id(snap, issue.node_id)
        if owner is None or owner.id != block.id:
            continue
        schema = next(
            (str(n.get("schema_id")) for n in snap.get("nodes", [])
             if str(n.get("id")) == issue.node_id),
            "",
        )
        port = projection.port(schema, issue.field) if schema else None
        if port is None:
            continue
        # Only THIS block's orphan if a provider already exists (earlier/same
        # block); otherwise the source is a later block → defer to the final gate.
        if source_candidates(snap, projection, port.type):
            out.append(issue)
    return out


def _dedupe(issues: List[ValidationIssue]) -> List[ValidationIssue]:
    seen = set()
    out: List[ValidationIssue] = []
    for i in issues:
        key = str(i)
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def _generic():
    from ..contracts import IssueCode
    return IssueCode.GENERIC
