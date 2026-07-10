# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Generic tool-calling Agent (blueprint §5).

An :class:`Agent` wraps one :class:`RoleSpec` + the B3 :class:`LlmClient`. A
producer agent (Orchestrator) runs a bounded tool-calling loop, dispatching each
tool call through the C1 :class:`MutationApi` and feeding the structured result
back to the model so it can self-correct. A read-only agent (Design) makes a
single completion and returns its text. The agent NEVER renders the C4 stream —
the bridge does that for applied primitives; the agent only drives mutations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import (
    Completion,
    IssueCode,
    LlmClient,
    MutationResult,
    Severity,
    Usage,
    ValidationIssue,
)
from ..mutation_api import MutationApi
from .roles import RoleSpec

__all__ = ["Agent", "AgentTurn"]


@dataclass
class AgentTurn:
    """The outcome of one agent run."""

    text: str = ""
    results: List[MutationResult] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    rounds: int = 0
    usage: Usage = field(default_factory=Usage)


class Agent:
    """One role's LLM driver."""

    def __init__(
        self,
        role: RoleSpec,
        llm: LlmClient,
        *,
        mutation_api: Optional[MutationApi] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tool_rounds: int = 8,
    ) -> None:
        self._role = role
        self._llm = llm
        self._mapi = mutation_api
        self._tools = tools or []
        self._max_rounds = max_tool_rounds
        #: Token usage of the most recent ``ask()`` (read by the scheduler so the
        #: read-only Design/Report turns also feed the per-phase token counter).
        self.last_usage: Usage = Usage()
        if role.can_write and mutation_api is None:
            raise ValueError(
                f"Agent {role.role}: a writing role needs a MutationApi (§8)"
            )

    # -- read-only single completion (Design) --------------------------------

    def ask(
        self,
        user_content: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """One completion with no tools; returns the assistant text.

        ``history`` (optional) is the thread's in-chat conversation so far,
        spliced between the system prompt and this turn so conversational
        referents survive across runs. None/empty = first message of a thread.
        """
        messages = [
            {"role": "system", "content": self._role.system_prompt},
            *(history or []),
            {"role": "user", "content": user_content},
        ]
        completion = self._llm.complete(messages, tools=None)
        self.last_usage = completion.usage or Usage()
        return completion.text

    # -- producer tool-calling loop (Orchestrator) ---------------------------

    def run(self, user_content: str) -> AgentTurn:
        """Run the bounded tool-calling loop; dispatch writes via MutationApi."""
        if self._mapi is None:
            raise ValueError(f"Agent {self._role.role}: run() needs a MutationApi")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._role.system_prompt},
            {"role": "user", "content": user_content},
        ]
        turn = AgentTurn()
        for _ in range(self._max_rounds):
            turn.rounds += 1
            completion = self._llm.complete(messages, tools=self._tools)
            turn.usage = turn.usage + (completion.usage or Usage())
            if completion.text:
                turn.text = (turn.text + "\n" + completion.text).strip()
            if not completion.tool_calls:
                break
            messages.append(_assistant_message(completion))
            for call in completion.tool_calls:
                result = self._apply(call)
                turn.results.append(result)
                turn.issues.extend(result.issues)
                messages.append(_tool_message(call.id, result))
        return turn

    def feed_repair(self, repair_messages: List[Dict[str, Any]]) -> AgentTurn:
        """Continue a producer with explicit repair context prepended to a fresh
        user turn (used by the scheduler's bounded repair loop)."""
        if self._mapi is None:
            raise ValueError(f"Agent {self._role.role}: feed_repair needs a MutationApi")
        content = "\n".join(m.get("content", "") for m in repair_messages)
        return self.run(content)

    # -- tool dispatch -------------------------------------------------------

    def _apply(self, call: Any) -> MutationResult:
        assert self._mapi is not None
        name = call.name
        args = call.arguments or {}
        try:
            if name == "place_node":
                return self._mapi.place_node(
                    args["schema_id"], instance_id=args.get("instance_id")
                )
            if name == "remove_node":
                return self._mapi.remove_node(args["instance_id"])
            if name == "connect":
                return self._mapi.connect(
                    args["src_node"], args["src_port"],
                    args["dst_node"], args["dst_port"],
                )
            if name == "disconnect":
                return self._mapi.disconnect(args["edge_id"])
            if name == "set_param":
                return self._mapi.set_param(args["node"], args["key"], args["value"])
            if name == "assign_reward_bundle":
                # ``page`` is optional (defaults to the global page in the API);
                # pass it only when the model supplied one so the default holds.
                kwargs: Dict[str, Any] = {}
                page = args.get("page")
                if page:
                    kwargs["page"] = page
                return self._mapi.assign_reward_bundle(
                    args["item_id"], args.get("terms") or {}, **kwargs
                )
        except KeyError as exc:
            return MutationResult(
                ok=False,
                issues=[ValidationIssue(
                    code=IssueCode.GENERIC, severity=Severity.ERROR,
                    message=f"tool {name!r} missing required argument {exc}",
                )],
            )
        return MutationResult(
            ok=False,
            issues=[ValidationIssue(
                code=IssueCode.GENERIC, severity=Severity.ERROR,
                message=f"unknown tool {name!r}",
            )],
        )


# =============================================================================
# OpenAI message helpers
# =============================================================================

def _assistant_message(completion: Completion) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": completion.text or None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in completion.tool_calls
        ],
    }


def _tool_message(tool_call_id: str, result: MutationResult) -> Dict[str, Any]:
    payload = {
        "ok": result.ok,
        "node_id": result.node_id,
        "edge_id": result.edge_id,
        "issues": [str(i) for i in result.issues],
    }
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }
