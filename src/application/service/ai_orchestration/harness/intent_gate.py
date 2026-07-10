# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Silent intent gate — the knowledge-free pre-step of an AI Build run.

Before ANY project knowledge (registry projection, canvas snapshot, reference
templates) is attached, the raw user message is sent to the model with a tiny
classifier prompt. "Knowledge-free" ≠ history-free: the thread's in-chat
conversation history IS spliced in (a follow-up like "make that penalty
stronger" is only judgeable with its referents) — what stays out is the
project knowledge base. The model judges, in this strict order:

  (a) is the message about building / modifying a robot-training canvas?
  (b) if yes — does acting on it require the project knowledge base?

and replies with exactly one JSON object ``{"status": true|false, "msg": "…"}``.

* ``status == True``  → the scheduler proceeds with the normal vibe-building
  orchestration (knowledge base attached from this point on).
* ``status == False`` → the scheduler answers the user directly from the
  model's own knowledge (no knowledge base, no tools, no canvas mutation).

The gate itself is SILENT: it emits nothing to the panel — no phase card, no
log line. Its token usage is still counted into the run total (honest
accounting; the meter never under-reports what the run cost).

Fail-loud (§8): a transport failure raises through; a reply that is not the
required JSON gets exactly one corrective retry, then raises
:class:`LlmTransportError` — the gate never silently guesses a verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..contracts import LlmClient, LlmTransportError, Usage

__all__ = ["GateDecision", "run_intent_gate", "INTENT_GATE_SYSTEM"]


#: The context-free classifier prompt. Deliberately tiny: no vocabulary, no
#: canvas snapshot, no templates — the whole point is to decide whether that
#: knowledge base should be loaded at all. The phrase "intent gate" in the
#: opening sentence is a stable marker that test stubs key on.
INTENT_GATE_SYSTEM = (
    "You are the silent intent gate for UnitPort Studio's AI Build (a node "
    "canvas that configures legged-robot RL training and deployment). The "
    "conversation so far may precede the user's LATEST message; judge the "
    "LATEST message, in the context of that conversation, in this strict "
    "order:\n"
    "(a) Is it asking to BUILD or MODIFY a robot-training canvas — nodes, "
    "wiring, rewards, observations, command items, terminations, domain "
    "randomization, trainer settings, robot selection/switching, or any other "
    "change to the training configuration? If NO, the verdict is false; stop.\n"
    "(b) If yes: does acting on it require the project knowledge base (the "
    "node/registry catalog, the current canvas, reference templates)?\n"
    "status is true only when BOTH (a) and (b) hold. A short follow-up that "
    "plausibly instructs a change to the training setup (e.g. 'make the "
    "penalty stronger', 'undo that') counts as (a) = yes.\n"
    "Reply with ONLY one JSON object — no code fences, no extra text:\n"
    '{"status": true|false, "msg": "<one short sentence explaining the verdict>"}'
)

_RETRY_USER = (
    "Your previous reply was not the required JSON. Reply with ONLY "
    '{"status": true|false, "msg": "<short reason>"} and nothing else.'
)


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict for one user message."""

    status: bool
    msg: str = ""
    usage: Usage = field(default_factory=Usage)


def run_intent_gate(
    llm: LlmClient,
    prompt: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> GateDecision:
    """Classify ``prompt`` knowledge-free; see the module docstring.

    ``history`` is the thread's in-chat conversation so far (OpenAI-style
    ``{"role", "content"}`` entries from
    :func:`~application.service.ai_orchestration.conversation_store.to_llm_history`),
    spliced between the classifier prompt and the new message so follow-ups
    keep their referents. Empty/None is the legitimate first-message state.
    """
    messages = [
        {"role": "system", "content": INTENT_GATE_SYSTEM},
        *(history or []),
        {"role": "user", "content": str(prompt or "")},
    ]
    completion = llm.complete(messages, tools=None)
    usage = completion.usage or Usage()
    try:
        status, msg = _parse_verdict(completion.text)
    except ValueError:
        # One corrective retry, then fail loud — never guess the verdict (§8).
        messages.append({"role": "assistant", "content": completion.text})
        messages.append({"role": "user", "content": _RETRY_USER})
        retry = llm.complete(messages, tools=None)
        usage = usage + (retry.usage or Usage())
        try:
            status, msg = _parse_verdict(retry.text)
        except ValueError as exc:
            raise LlmTransportError(
                "intent gate: the model did not return the required "
                '{"status", "msg"} JSON after a retry '
                f"({exc}); raw reply: {str(retry.text)[:200]!r}"
            ) from exc
    return GateDecision(status=status, msg=msg, usage=usage)


def _parse_verdict(text: str) -> Tuple[bool, str]:
    """Extract ``(status, msg)`` from a gate reply; raises ValueError."""
    payload = _extract_json_object(str(text or ""))
    raw_status = payload.get("status")
    if isinstance(raw_status, bool):
        status = raw_status
    elif isinstance(raw_status, str) and raw_status.strip().lower() in ("true", "false"):
        status = raw_status.strip().lower() == "true"
    else:
        raise ValueError(f"'status' is not a boolean: {raw_status!r}")
    return status, str(payload.get("msg") or "").strip()


def _extract_json_object(text: str) -> dict:
    """Parse the first top-level JSON object in ``text``.

    Tolerates code fences and prose AROUND the object (models add them despite
    instructions) — but tolerates nothing about the object itself: a candidate
    that does not ``json.loads`` to a dict raises ValueError.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(stripped[start : i + 1])
                if isinstance(obj, dict):
                    return obj
                break
    raise ValueError("no parsable JSON object in the reply")
