# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""LLM API connection settings for natural-language canvas orchestration.

The "AI Build" feature drives a model API to mutate the training canvas. A
connection (base URL, API key, model id) is a per-user **agent** — the user may
register several (one per provider/model) and pick one as the *default*. The
whole set is per-user state, so it lives as a JSON file under
``Paths.USER_CONFIG_DIR`` — NOT in ``src/`` and NOT in ``system.ini``
(CLAUDE.md §4). Each agent carries an explicit ``upload_to_cloud`` marker (the
dialog's switch); it defaults to **off** and the cloud-sync service honours it
(``cloud_sync._cloud_upload_allowed``): an API key is never uploaded unless the
user opts in.

Which agent is the default is mirrored to ``user.ini[AiBuild].active_agent_id``
(``Config.set_value``) so the choice takes effect on every launch; the file's
``active_id`` is the self-contained fallback when the overlay is absent.

The token may be given two ways per agent:

* **Literal** — stored in ``api_key`` (the historical form).
* **Environment variable** — the user stores only the variable NAME in
  ``api_key_env``; the token itself is read from the process environment at run
  time (:meth:`ApiConfig.resolve_api_key`) and is **never written to disk**.
  This suits users who keep secrets in env vars and is strictly more secure
  (nothing secret persists, so the cloud-upload concern is moot for that field).

Mirrors the persistence idiom of ``auth.profile_cache`` (``push_data`` to write,
``read_data`` to read), resolved against the LIVE ``USER_CONFIG_DIR`` at call
time so a hot workspace switch is picked up without a restart.

Backward compatibility: :func:`load_api_config` returns the **active agent** and
:func:`save_api_config` writes the **active agent** in place, so every existing
consumer (``openai_compat.build_client_from_config``, ``cloud_sync``, the entry
gate) keeps working unchanged; the multi-agent registry is layered on top via
:func:`load_agent_registry` / :func:`save_agent_registry`.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from unitport_sdk import Config, Paths, log_warning, push_data, read_data

#: Path of the config file relative to ``USER_CONFIG_DIR``. This is also the key
#: the cloud-sync planner gates on (see ``cloud_sync._cloud_upload_allowed``).
API_CONFIG_REL = "ai_orchestration/api_config.json"

#: user.ini overlay (section, key) mirroring the default agent id so the choice
#: takes effect every launch (CLAUDE.md §4 — write via ``Config.set_value``).
ACTIVE_INI_SECTION = "AiBuild"
ACTIVE_INI_KEY = "active_agent_id"

#: Bumped if the on-disk schema changes; load migrates older files (§8(c)).
#: v1               — flat single connection.
#: v2 (2026-06)     — flat single connection + ``api_key_env``.
#: v3 (2026-07)     — multi-agent registry (``agents`` list + ``active_id``).
_SCHEMA_VERSION = 3

#: A resolved token must be longer than this to be accepted from an env var —
#: guards against pointing at an empty / wrong variable. Single source of the
#: ">10" rule shared by the settings dialog's validation + preview.
ENV_TOKEN_MIN_LEN = 10


def _config_file() -> Path:
    """Resolve the config path inside the LIVE ``USER_CONFIG_DIR`` (call time)."""
    return Paths.USER_CONFIG_DIR / API_CONFIG_REL


def new_agent_id() -> str:
    """A fresh, collision-resistant agent id (stable key for a registration)."""
    return "agent-" + uuid.uuid4().hex[:12]


def is_env_token_valid(env_name: str) -> bool:
    """True if ``env_name`` names an env var holding a token longer than the
    minimum (:data:`ENV_TOKEN_MIN_LEN`). The single ">10" check used by the
    settings dialog for both save-validation and the live preview."""
    value = os.environ.get((env_name or "").strip(), "")
    return len(value) > ENV_TOKEN_MIN_LEN


@dataclass
class ApiConfig:
    """One registered agent: the LLM connection the AI-Build harness calls.

    ``id`` is the stable registry key; ``name`` is a user-facing label. Both are
    empty on a freshly-constructed connection and filled by the registry when
    the agent is persisted."""

    base_url: str = ""
    api_key: str = ""
    #: Name of the environment variable holding the token. When non-empty it
    #: takes precedence over ``api_key`` and the literal ``api_key`` is NOT
    #: persisted (the secret never touches disk). See :meth:`resolve_api_key`.
    api_key_env: str = ""
    model: str = ""
    #: File marker: may this file be uploaded by the cloud-sync feature?
    #: Defaults off — a secret is never synced unless the user flips the switch.
    upload_to_cloud: bool = False
    #: Stable registry key + user-facing label (registry-managed; a bare
    #: connection carries them empty).
    id: str = ""
    name: str = ""
    schema_version: int = _SCHEMA_VERSION

    def resolve_api_key(self) -> str:
        """The effective token: from the named env var when set, else literal.

        Resolved at call time from the live process environment, so a token kept
        in an env var is picked up without re-saving. A named-but-unset variable
        resolves to "" (→ ``is_configured`` False → the entry gate re-prompts)."""
        name = self.api_key_env.strip()
        if name:
            return os.environ.get(name, "")
        return self.api_key

    @property
    def is_configured(self) -> bool:
        """True once base URL, model, and a resolvable token are all present."""
        return bool(
            self.base_url.strip() and self.model.strip() and self.resolve_api_key().strip()
        )

    def display_name(self) -> str:
        """Label for the agent list / title button: the name, else the model."""
        return (self.name or "").strip() or (self.model or "").strip()

    @classmethod
    def from_dict(cls, data: dict) -> "ApiConfig":
        """Parse one persisted agent; missing fields take dataclass defaults."""
        return cls(
            base_url=str(data.get("base_url", "")),
            api_key=str(data.get("api_key", "")),
            api_key_env=str(data.get("api_key_env", "")),  # pre-v2 files lack this
            model=str(data.get("model", "")),
            upload_to_cloud=bool(data.get("upload_to_cloud", False)),
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
        )


@dataclass
class AgentRegistry:
    """The set of registered agents plus which one is the default.

    ``active_id`` names the default agent; :meth:`active` returns it (or a blank
    :class:`ApiConfig` when the set is empty — the documented "not configured
    yet" state, checked explicitly by callers)."""

    agents: List[ApiConfig] = field(default_factory=list)
    active_id: str = ""
    schema_version: int = _SCHEMA_VERSION

    def get(self, agent_id: str) -> Optional[ApiConfig]:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def active(self) -> ApiConfig:
        """The default agent, or a blank connection when none is registered."""
        found = self.get(self.active_id)
        if found is not None:
            return found
        return self.agents[0] if self.agents else ApiConfig()

    def normalized(self) -> "AgentRegistry":
        """Ensure every agent has an id and ``active_id`` points at a real agent.

        Fills any blank id (freshly-added agent) and repoints ``active_id`` at
        the first agent when it dangles. Returns ``self`` for chaining."""
        for agent in self.agents:
            if not agent.id.strip():
                agent.id = new_agent_id()
        if self.agents and self.get(self.active_id) is None:
            self.active_id = self.agents[0].id
        if not self.agents:
            self.active_id = ""
        self.schema_version = _SCHEMA_VERSION
        return self


def _migrate_flat_to_registry(data: dict) -> AgentRegistry:
    """Wrap a pre-v3 flat single-connection file into a one-agent registry.

    §8(c) on-disk legacy compatibility: loudly WARN so the user re-saves in the
    current schema; removed once no v1/v2 files remain in the wild."""
    agent = ApiConfig.from_dict(data)
    agent.id = new_agent_id()
    if not agent.name.strip():
        agent.name = agent.model.strip() or "Agent 1"
    log_warning(
        "[ai.api_config] migrated a legacy single-connection api_config.json to "
        "the multi-agent registry (schema v3) — re-save it in AI Build settings "
        "to drop this notice."
    )
    return AgentRegistry(agents=[agent], active_id=agent.id).normalized()


def load_agent_registry() -> AgentRegistry:
    """Read the registered agents, or an empty registry when none exists.

    Never raises on a missing / malformed file: an empty registry is a valid
    "not configured yet" state (its :meth:`~AgentRegistry.active` is blank →
    ``is_configured`` False), which callers check explicitly. Legacy flat files
    are migrated (§8(c)). The default agent honours the user.ini pointer first
    so the choice persists across launches, falling back to the file's own
    ``active_id`` and finally the first agent."""
    path = _config_file()
    reg: AgentRegistry
    if not path.exists():
        reg = AgentRegistry()
    else:
        data = read_data(path)
        if not isinstance(data, dict):
            log_warning(f"[ai.api_config] malformed config at {path}; using defaults")
            reg = AgentRegistry()
        elif "agents" in data:
            agents = [
                ApiConfig.from_dict(a) for a in data.get("agents", []) if isinstance(a, dict)
            ]
            reg = AgentRegistry(
                agents=agents,
                active_id=str(data.get("active_id", "")),
                schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
            ).normalized()
        else:
            reg = _migrate_flat_to_registry(data)

    # The user.ini pointer wins when it names a real agent (persisted default).
    pinned = str(Config.get_value(ACTIVE_INI_SECTION, ACTIVE_INI_KEY, "") or "").strip()
    if pinned and reg.get(pinned) is not None:
        reg.active_id = pinned
    return reg


def save_agent_registry(registry: AgentRegistry) -> bool:
    """Persist the registry to ``USER_CONFIG_DIR`` and mirror the default agent
    id to ``user.ini`` so the choice takes effect every launch. Returns False on
    write failure (the file is the source of truth; the user.ini mirror is a
    best-effort pointer written only after a successful file write)."""
    registry.normalized()
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "active_id": registry.active_id,
        "agents": [asdict(a) for a in registry.agents],
    }
    if not push_data(API_CONFIG_REL, payload):
        log_warning(f"[ai.api_config] failed to write {_config_file()}")
        return False
    # Mirror the default-agent pointer to the user.ini overlay (§4). Pre-wizard
    # writes queue in memory and flush once the workspace is established.
    Config.set_value(ACTIVE_INI_SECTION, ACTIVE_INI_KEY, registry.active_id)
    return True


def load_api_config() -> ApiConfig:
    """The **active agent's** connection, or blank defaults when none exists.

    Single seam every consumer uses (provider client, cloud-sync gate, entry
    gate). Never raises on a missing / malformed file — a blank
    :class:`ApiConfig` is the documented "not configured yet" state
    (``is_configured`` False), which the caller checks explicitly."""
    return load_agent_registry().active()


def save_api_config(config: ApiConfig) -> bool:
    """Persist ``config`` as the **active agent** (backward-compatible seam).

    Updates the active agent's connection fields in place (preserving its id +
    name), or registers it as a new default when the set is empty. The
    multi-agent dialog uses :func:`save_agent_registry` directly; this keeps the
    single-connection callers working unchanged."""
    registry = load_agent_registry()
    active = registry.get(registry.active_id)
    if active is None and registry.agents:
        active = registry.agents[0]
    if active is not None:
        # Preserve the registry identity; overwrite only the connection fields.
        active.base_url = config.base_url
        active.api_key = config.api_key
        active.api_key_env = config.api_key_env
        active.model = config.model
        active.upload_to_cloud = config.upload_to_cloud
        if config.name.strip():
            active.name = config.name
        registry.active_id = active.id
    else:
        agent = ApiConfig.from_dict(asdict(config))
        agent.id = new_agent_id()
        if not agent.name.strip():
            agent.name = agent.model.strip() or "Agent 1"
        registry.agents.append(agent)
        registry.active_id = agent.id
    return save_agent_registry(registry)


def is_api_configured() -> bool:
    """True if the default agent holds a usable connection (URL + key + model)."""
    return load_api_config().is_configured


def is_api_cloud_upload_enabled() -> bool:
    """True only if **every** registered agent allows cloud upload.

    The config file is a single blob holding all agents' secrets, so the
    cloud-sync gate must be unanimous: uploading it is allowed only when every
    agent has flipped its switch on. Fail-safe — an empty registry, a single
    dissenting agent, or any absence / malformed state resolves to False (don't
    sync a secret you're unsure about); each agent's marker already defaults
    off.
    """
    agents = load_agent_registry().agents
    return bool(agents) and all(a.upload_to_cloud for a in agents)


def active_model_name() -> str:
    """The default agent's model id (empty when unconfigured) — for the AI Build
    title button label. Empty is the documented "not set up yet" state."""
    return (load_api_config().model or "").strip()


__all__ = [
    "API_CONFIG_REL",
    "ACTIVE_INI_SECTION",
    "ACTIVE_INI_KEY",
    "ENV_TOKEN_MIN_LEN",
    "ApiConfig",
    "AgentRegistry",
    "new_agent_id",
    "is_env_token_valid",
    "load_agent_registry",
    "save_agent_registry",
    "load_api_config",
    "save_api_config",
    "is_api_configured",
    "is_api_cloud_upload_enabled",
    "active_model_name",
]
