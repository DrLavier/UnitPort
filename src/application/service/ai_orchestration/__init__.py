# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Natural-language canvas orchestration — app-side services.

Implements the backend of the "AI Build" feature
(``knowledge_base/nl_canvas_orchestration_blueprint.md``). The frontend (panel,
bridge, settings) lives under ``application.ui.canvas``; this package holds the
headless, Qt-free service layer:

- :mod:`api_config`         — B3 credentials (base URL / key / model + cloud flag).
- :mod:`contracts`          — frozen C4/C5 types + AgentRole + MutationSink /
  LlmClient protocols (blueprint §3).
- :mod:`registry_projection` — C2 vocabulary projection scoped to (backend, family).
- :mod:`mutation_api`       — C1 single write entry + tier-1 alignment + semantic ops.
- :mod:`routing`            — C5 issue → responsible AgentRole.
- :mod:`validation`         — B4 tiers (C3 invariants; compile gate + critic land Wave 1).

The conversational UI and the live-canvas mutation seam stay in
``application.ui.canvas``; this layer never imports Qt widgets.

NOTE: imports here are kept lazy. ``api_config`` is imported by ``cloud_sync``
during startup and must stay lightweight; the contract/harness modules
(``contracts`` → ``spec_validator`` → training stack) are imported directly by
their consumers (``from application.service.ai_orchestration.contracts import …``),
never eagerly from this package ``__init__``.
"""
