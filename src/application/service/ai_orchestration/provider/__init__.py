# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""LLM provider implementations for AI Build (blueprint §4.B3).

The orchestration harness codes against the provider-agnostic
:class:`~application.service.ai_orchestration.contracts.LlmClient` protocol; the
concrete transports live here. v1 ships a single OpenAI-compatible provider
(:class:`OpenAICompatClient`) that speaks ``POST /chat/completions`` with
tool-calling and is built from the saved :class:`ApiConfig` via
:func:`build_client_from_config`. Every transport/decode failure is fail-loud
(CLAUDE.md §8) — see :mod:`.openai_compat`.
"""

from __future__ import annotations

from .openai_compat import OpenAICompatClient, build_client_from_config

__all__ = ["OpenAICompatClient", "build_client_from_config"]
