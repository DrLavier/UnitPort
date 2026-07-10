# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible LLM provider (blueprint §4.B3).

One concrete implementation of the frozen :class:`LlmClient` protocol
(:mod:`application.service.ai_orchestration.contracts`). It speaks the
``POST /chat/completions`` dialect with function/tool calling, reading its
connection settings from :class:`ApiConfig`.

Fail-loud contract (CLAUDE.md §8 / blueprint §4.B3)
---------------------------------------------------
Every transport, HTTP, or decode failure raises :class:`LlmTransportError`.
This module NEVER returns an empty/degraded :class:`Completion` to paper over a
failure: a non-2xx status, a connection/timeout error, a non-JSON body, or a
model-emitted tool call whose ``arguments`` are not valid JSON all raise with a
message that names the offending URL and the status/reason. The caller surfaces
that to the user instead of silently mutating a wrong canvas.

Transport
---------
Implemented on the Python standard library (:mod:`urllib.request`) so the
feature adds **no** new third-party dependency (``requests`` is not in
``requirements.txt``).

URL normalization (see :func:`_resolve_chat_url`)
-------------------------------------------------
The user types a *base* URL in the settings dialog; different providers expect
different suffixes. We normalize once, deterministically:

* trailing slashes are stripped;
* if it already ends with ``/chat/completions`` -> use as-is;
* elif it ends with ``/v1`` (e.g. ``http://host/v1`` or ``.../v1/``)
  -> append ``/chat/completions``;
* else (a bare host/root) -> append ``/v1/chat/completions``.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from application.service.ai_orchestration.api_config import ApiConfig, load_api_config
from application.service.ai_orchestration.contracts import (
    Completion,
    LlmTransportError,
    ToolCall,
    Usage,
)

__all__ = ["OpenAICompatClient", "build_client_from_config", "_resolve_chat_url"]

#: Suffix appended to a fully-qualified API root.
_CHAT_PATH = "chat/completions"


def _resolve_chat_url(base_url: str) -> str:
    """Normalize a user-supplied base URL to a concrete chat-completions URL.

    See the module docstring for the exact rule. Raises
    :class:`LlmTransportError` (fail-loud, §8) when ``base_url`` is blank — a
    request can never be built without an endpoint.
    """
    trimmed = (base_url or "").strip().rstrip("/")
    if not trimmed:
        raise LlmTransportError(
            "LLM base_url is empty; cannot build a request URL. "
            "Configure the AI Build connection first."
        )
    if trimmed.endswith("/" + _CHAT_PATH) or trimmed.endswith(_CHAT_PATH):
        # Already a full chat-completions endpoint.
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/" + _CHAT_PATH
    return trimmed + "/v1/" + _CHAT_PATH


class OpenAICompatClient:
    """An :class:`LlmClient` over any OpenAI ``/chat/completions`` endpoint.

    Structurally conforms to the frozen ``LlmClient`` protocol (it is
    ``runtime_checkable``, so ``isinstance(client, LlmClient)`` holds). Holds the
    resolved endpoint, bearer key, and model id taken from :class:`ApiConfig`.
    """

    def __init__(self, config: ApiConfig, *, timeout: float = 60.0) -> None:
        self.base_url = config.base_url
        # Resolve the token from a named env var when configured that way, else
        # the literal key (api_config.ApiConfig.resolve_api_key). Resolved at
        # construction; a fresh client is built per run so it tracks the env.
        self.api_key = config.resolve_api_key()
        self.model = config.model
        self._timeout = timeout
        #: Resolved once at construction so every call hits the same endpoint.
        self._chat_url = _resolve_chat_url(config.base_url)

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ) -> Completion:
        """Run one completion. Raises :class:`LlmTransportError` on any failure."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": bool(stream),
        }
        # tools + tool_choice only when the caller actually offered tools.
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            self._chat_url, data=body, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                # Parse inside the context so the streaming body stays open.
                if stream:
                    return self._parse_streaming(response, on_text_delta)
                return self._parse_non_streaming(response.read())
        except urllib.error.HTTPError as exc:
            # A non-2xx response is fail-loud — include status + reason + body.
            detail = self._read_error_body(exc)
            suffix = f": {detail}" if detail else ""
            raise LlmTransportError(
                f"LLM request to {self._chat_url} failed with HTTP "
                f"{exc.code} {exc.reason}{suffix}"
            ) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LlmTransportError(
                f"LLM request to {self._chat_url} failed (transport error): {reason}"
            ) from exc

    # -- non-streaming parse ------------------------------------------------

    def _parse_non_streaming(self, raw: bytes) -> Completion:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LlmTransportError(
                f"LLM response from {self._chat_url} was not valid JSON: {exc}"
            ) from exc
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmTransportError(
                f"LLM response from {self._chat_url} missing choices[0].message "
                f"(got keys {list(data) if isinstance(data, dict) else type(data).__name__})"
            ) from exc

        text = message.get("content") or ""
        tool_calls = [
            self._build_tool_call(tc) for tc in (message.get("tool_calls") or [])
        ]
        return Completion(
            text=text, tool_calls=tool_calls, usage=_usage_from(data.get("usage"))
        )

    def _build_tool_call(self, tc: Dict[str, Any]) -> ToolCall:
        """Build one :class:`ToolCall`; fail loud on a malformed entry."""
        try:
            call_id = tc["id"]
            function = tc["function"]
            name = function["name"]
        except (KeyError, TypeError) as exc:
            raise LlmTransportError(
                f"LLM tool_call is malformed (missing id/function/name): {tc!r}"
            ) from exc
        raw_args = function.get("arguments")
        arguments = self._decode_arguments(name, raw_args)
        return ToolCall(id=str(call_id), name=str(name), arguments=arguments)

    def _decode_arguments(self, name: str, raw_args: Any) -> Dict[str, Any]:
        """Decode the JSON-string ``arguments`` of a tool call (fail loud).

        OpenAI emits ``arguments`` as a JSON *string*. An empty/absent value
        means "no arguments" -> ``{}``. Malformed JSON is an unusable tool call
        and raises (§8) rather than being dropped.
        """
        if isinstance(raw_args, dict):
            # Tolerate a provider that pre-parsed the object.
            return raw_args
        try:
            arguments = json.loads(raw_args or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            raise LlmTransportError(
                f"LLM tool_call '{name}' arguments were not valid JSON: {raw_args!r}"
            ) from exc
        if not isinstance(arguments, dict):
            raise LlmTransportError(
                f"LLM tool_call '{name}' arguments did not decode to an object: "
                f"{arguments!r}"
            )
        return arguments

    # -- streaming parse ----------------------------------------------------

    def _parse_streaming(
        self, response: Any, on_text_delta: Optional[Callable[[str], None]]
    ) -> Completion:
        """Assemble a :class:`Completion` from a Server-Sent-Events stream.

        Each ``data: {json}`` line carries a ``choices[0].delta``. Content
        fragments are accumulated (and pushed to ``on_text_delta`` as they
        arrive); tool-call fragments are accumulated *by index* — the ``id`` /
        ``function.name`` appear once and the ``function.arguments`` string is
        delivered in pieces that we concatenate, then ``json.loads`` at the end.
        ``data: [DONE]`` terminates the stream.
        """
        text_parts: List[str] = []
        # index -> {"id": str, "name": str, "args": str}
        tool_acc: Dict[int, Dict[str, str]] = {}
        # Final usage chunk (OpenAI sends it last when stream_options.include_usage
        # is set; absent otherwise → stays None → Usage(0,0,0)).
        usage: Optional[Usage] = None

        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError as exc:
                raise LlmTransportError(
                    f"LLM stream chunk from {self._chat_url} was not valid JSON: "
                    f"{data_str!r}"
                ) from exc

            if chunk.get("usage") is not None:
                usage = _usage_from(chunk.get("usage"))

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            content = delta.get("content")
            if content:
                text_parts.append(content)
                if on_text_delta is not None:
                    on_text_delta(content)

            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0)
                slot = tool_acc.setdefault(index, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                fragment = function.get("arguments")
                if fragment:
                    slot["args"] += fragment

        tool_calls: List[ToolCall] = []
        for index in sorted(tool_acc):
            slot = tool_acc[index]
            arguments = self._decode_arguments(slot["name"], slot["args"])
            tool_calls.append(
                ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)
            )
        return Completion(
            text="".join(text_parts), tool_calls=tool_calls, usage=usage
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _read_error_body(exc: urllib.error.HTTPError) -> str:
        """Best-effort read of an error response body for the raised message.

        Enriches an error we are *already* raising loud; not an §8 fallback for
        a required input. A failed read just yields an empty detail string.
        """
        try:
            body = exc.read()
        except OSError:
            return ""
        if not body:
            return ""
        return body.decode("utf-8", "replace")[:500]


def _usage_from(raw: Any) -> Usage:
    """Build a :class:`Usage` from a provider ``usage`` block (best-effort).

    A missing/malformed block yields ``Usage(0, 0, 0)`` — token counts are a
    cosmetic UI counter, not a required input, so their absence never aborts a
    run (this is NOT an §8 fallback for a required value).
    """
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        prompt=_as_int(raw.get("prompt_tokens")),
        completion=_as_int(raw.get("completion_tokens")),
        total=_as_int(raw.get("total_tokens")),
    )


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_client_from_config(*, timeout: float = 60.0) -> OpenAICompatClient:
    """Build a client from the saved :class:`ApiConfig` (fail loud if absent).

    Reads the per-user connection via :func:`load_api_config`. If it is not
    fully configured (``base_url`` / ``api_key`` / ``model``), raises
    :class:`LlmTransportError` (§8) rather than returning a client that would
    fail cryptically on first use.
    """
    config = load_api_config()
    if not config.is_configured:
        hint = ""
        if config.api_key_env.strip():
            hint = (
                f" The token is configured to come from env var "
                f"'{config.api_key_env.strip()}', which is not set in this "
                f"process."
            )
        raise LlmTransportError(
            "AI Build LLM connection is not configured "
            "(need base_url, a resolvable token, and model). "
            "Open the AI Build settings and fill in the connection before use."
            + hint
        )
    return OpenAICompatClient(config, timeout=timeout)
