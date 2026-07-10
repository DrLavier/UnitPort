# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Foundation contracts for natural-language canvas orchestration (AI Build).

This module freezes the cross-block alignment surface from
``knowledge_base/nl_canvas_orchestration_blueprint.md`` §3. Every other block
in ``application.service.ai_orchestration`` codes against these types, not
against another block's internals:

- **C4** :class:`MutationEvent` — the structured "Thinking" event a primitive
  emits. The :class:`~application.ui.canvas.agent_bridge.CanvasAgentBridge`
  already renders the *human-readable* themed line into the panel; the harness
  captures :class:`MutationEvent` in parallel for attribution / the action
  trace. The two faces coexist; the harness does NOT re-render.
- **C5** :class:`ValidationIssue` (reused verbatim from the training package —
  the single source) wrapped by :class:`RoutedIssue` which adds the harness-side
  ``responsible`` routing target. We deliberately do NOT fork ``ValidationIssue``:
  the compiler/validator own it and adding a field there would couple training
  code to the harness. Routing is derived by :mod:`.routing` from
  ``(code, node_id, field)``.
- :class:`MutationResult` — the C1 primitive return (blueprint §3.C1).
- :class:`MutationSink` (Protocol) — the live-canvas write seam the service
  layer composes against. The GUI marshaller implements it over the bridge;
  tests implement an in-memory fake. **The service layer never imports Qt
  widgets** — it only knows this Protocol.
- :class:`LlmClient` (Protocol) — the provider-agnostic completion surface
  (blueprint §4.B3). One OpenAI-compatible provider is implemented for v1.

``AgentRole`` enumerates the full blueprint roster so the scheduler interfaces
are stable; v1 only instantiates ``DESIGN`` / ``ORCHESTRATOR`` / ``CRITIC`` /
``SYSTEM`` (see the approved plan), the rest are reserved for later expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

# C5 is reused verbatim — the training package owns the single source.
from application.training.spec_validator import IssueCode, Severity, ValidationIssue

if TYPE_CHECKING:  # annotation only — avoids a runtime import cycle
    from .layout import LayoutPlan

__all__ = [
    "AgentRole",
    "MutationEvent",
    "MutationResult",
    "RoutedIssue",
    "MutationSink",
    "LlmClient",
    "ToolCall",
    "Usage",
    "Completion",
    "Requirements",
    "LlmTransportError",
    # re-exported for convenience so blocks import C5 from one place
    "ValidationIssue",
    "IssueCode",
    "Severity",
]


# =============================================================================
# Agent roster (blueprint §5)
# =============================================================================

class AgentRole(Enum):
    """Roles in the orchestration team.

    Full blueprint roster is enumerated so :class:`RoleSpec` / scheduler
    signatures stay stable across the v1 → v7-role expansion. v1 only
    *instantiates* ``DESIGN`` / ``ORCHESTRATOR`` / ``CRITIC`` / ``SYSTEM``.
    """

    DESIGN = "design"
    ORCHESTRATOR = "orchestrator"
    # Reserved param-expert roles — folded into ORCHESTRATOR for v1.
    MOTION = "motion"
    REWARDS = "rewards"
    OBS_ROBUSTNESS = "obs_robustness"
    TRAINER = "trainer"
    # Reserved read-only arbiter — its C3 checks run deterministically in v1.
    INTEGRATOR = "integrator"
    CRITIC = "critic"
    # The deterministic scaffolding (scheduler / compile gate / tier-1).
    SYSTEM = "system"

    def __str__(self) -> str:  # human-facing role label in logs/reports
        return self.value


# =============================================================================
# C4 — Mutation Event (the "Thinking" stream, structured face)
# =============================================================================

@dataclass(frozen=True)
class MutationEvent:
    """One structured mutation event (blueprint §3.C4).

    ``op`` is one of ``place_node`` / ``remove_node`` / ``connect`` /
    ``disconnect`` / ``set_param``. ``agent`` attributes the write to a role.
    ``timestamp`` is wall-clock seconds (``time.time()``), stamped by the
    caller — kept as a plain field so the dataclass stays pure/serializable.
    """

    op: str
    agent: AgentRole
    timestamp: float
    target_node: str = ""
    target_port: str = ""
    value: Any = None


# =============================================================================
# C5 — ValidationIssue routing wrapper
# =============================================================================

@dataclass(frozen=True)
class RoutedIssue:
    """A :class:`ValidationIssue` tagged with the agent role responsible for
    repairing it (blueprint §3.C5 ``responsible`` field).

    ``ValidationIssue`` itself is frozen and owned by the training package, so
    the routing target lives here instead of mutating that type. The scheduler
    turns every issue (from the compile gate, tier-1, the C3 integrator, or the
    critic) into a ``RoutedIssue`` via :func:`application.service.ai_orchestration.routing.route`.
    """

    issue: ValidationIssue
    responsible: AgentRole

    @property
    def is_error(self) -> bool:
        return self.issue.severity == Severity.ERROR


# =============================================================================
# C1 — Mutation primitive result (blueprint §3.C1)
# =============================================================================

@dataclass(frozen=True)
class MutationResult:
    """The result of one C1 primitive / semantic op.

    A rejected mutation (``ok == False``) does NOT alter the blackboard — the
    in-scope tier-1 check ran before any sink call and produced ``issues``.
    ``event`` is populated only for an applied (``ok``) primitive.
    """

    ok: bool
    event: Optional[MutationEvent] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    node_id: str = ""
    edge_id: str = ""


# =============================================================================
# MutationSink — the live-canvas write seam (blueprint §2 / §6 persistence)
# =============================================================================

@runtime_checkable
class MutationSink(Protocol):
    """The seam the service layer writes the canvas through.

    Mirrors :class:`~application.ui.canvas.agent_bridge.CanvasAgentBridge`'s
    five primitives plus a snapshot reader and a log passthrough. The GUI
    marshaller implements this over the bridge (and marshals every call to the
    GUI thread); headless tests implement an in-memory fake.

    Return contract matches the bridge: ``Optional[str]`` id on success / None
    on rejection for ``place_node`` / ``connect``; ``bool`` for the rest.
    """

    def place_node(
        self,
        schema_id: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        x: float = 0.0,
        y: float = 0.0,
        instance_id: Optional[str] = None,
    ) -> Optional[str]: ...

    def remove_node(self, instance_id: str) -> bool: ...

    def connect(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str
    ) -> Optional[str]: ...

    def disconnect(self, edge_id: str) -> bool: ...

    def set_param(self, instance_id: str, key: str, value: Any) -> bool: ...

    def snapshot(self) -> Dict[str, Any]:
        """Return the whole canvas as an IR-shape dict (``to_workflow_dict``)."""
        ...

    def log_markup(self, markup: str) -> None:
        """Append one ``[[role:text]]``-tagged line to the panel."""
        ...

    def set_progress(self, fraction: float, text: str = "") -> None:
        """Report run progress (``0.0``–``1.0``) to the panel's progress bar.

        Non-authoritative UI feedback (the authoritative outcome is the run's
        returned result + the logged lines); a sink may no-op it.
        """
        ...

    def begin_phase(self, phase_id: str, title: str) -> None:
        """Open a new structured phase (Understanding / a block / Report).

        Closes the current agent bubble and opens a fresh one headed by
        ``title``, so the run reads as the fixed three-phase format. Cosmetic UI
        feedback — a minimal/headless sink no-ops it.
        """
        ...

    def set_phase_tokens(self, phase_total: int, cumulative: int) -> None:
        """Report this phase's token total + the running cumulative for the run.

        Rendered as a small line above the current bubble. Cosmetic — no-opable.
        """
        ...

    def highlight_block(self, schema_ids: FrozenSet[str], on: bool) -> None:
        """Highlight (or clear) every node whose schema_id is in ``schema_ids``.

        Visual feedback for "the agent is editing this block now". Cosmetic.
        """
        ...

    def highlight_node(self, instance_id: str, on: bool) -> None:
        """Highlight (or clear) one node by instance id (the just-mutated node).

        Finer-grained companion to :meth:`highlight_block`. Cosmetic — no-opable.
        """
        ...

    def clear_highlights(self) -> None:
        """Clear all agent-editing highlights (run end / cancel / error)."""
        ...

    def notify_build_engaged(self) -> None:
        """The silent intent gate admitted this prompt as a canvas build run.

        Fired once per run, after the gate and before any mutation, so the GUI
        can surface build-run affordances (e.g. the checkpoint notice) only
        for runs that will actually act on the canvas — a gate-declined direct
        answer never shows them. Cosmetic UI feedback — a minimal/headless
        sink no-ops it.
        """
        ...

    def apply_layout(self, plan: "LayoutPlan") -> None:
        """Arrange nodes into block columns + draw the titled region boxes.

        The deterministic tidy-layout SYSTEM step: repositions every node per
        ``plan.positions`` (non-overlapping, left-to-right by block) and rebuilds
        the AI-Build-owned region partition boxes. Cosmetic and non-authoritative
        — a headless/minimal sink may no-op it.
        """
        ...


# =============================================================================
# LlmClient — provider-agnostic completion surface (blueprint §4.B3)
# =============================================================================

@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class Usage:
    """Token usage of one completion (the provider's ``usage`` block).

    Non-authoritative UI feedback (the authoritative outcome is the run result),
    surfaced per phase/bubble in the AI Build overlay. A provider that does not
    report usage yields ``Usage(0, 0, 0)`` — the run is unaffected (it is a
    cosmetic counter, not a required input, so absence is not an §8 violation).
    """

    prompt: int = 0
    completion: int = 0
    total: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
            total=self.total + other.total,
        )


@dataclass(frozen=True)
class Completion:
    """A single model completion: free text plus any tool calls."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None


@dataclass(frozen=True)
class Requirements:
    """User-supplied, no-token decisions from the AI Build Requirements card.

    These are applied **deterministically** by the scheduler (via the C1
    Mutation API) before any agent runs, and then locked in the agent prompts so
    the model never spends tokens re-deciding them (blueprint v2). Tiers
    (``length_tier`` / ``intensity_tier``) resolve to concrete numbers via
    :mod:`application.service.ai_orchestration.blocks`; a non-None numeric
    override wins over its tier. All fields blank/None = "card unused" → nothing
    is pre-applied and the agent decides everything as before.
    """

    robot_sku: str = ""
    motion: str = ""
    length_tier: str = ""
    max_iterations: Optional[int] = None
    intensity_tier: str = ""
    num_envs: Optional[int] = None


class LlmTransportError(RuntimeError):
    """Raised on any provider transport / HTTP / decode failure.

    Fail-loud (CLAUDE.md §8 / blueprint §4.B3): an API failure raises and is
    reported to the user; it is NEVER substituted with a degraded canvas.
    """


@runtime_checkable
class LlmClient(Protocol):
    """Provider-agnostic completion/tool-call surface.

    ``messages`` is an OpenAI-style chat list (``{"role", "content", ...}``);
    ``tools`` is a list of JSON-schema tool definitions (from
    :meth:`RegistryProjection.to_tool_schemas`). ``on_text_delta`` receives
    streamed text chunks when ``stream`` is True. Raises
    :class:`LlmTransportError` on failure.
    """

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ) -> Completion: ...
