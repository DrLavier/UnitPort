"""User workspace (``USER_CONFIG_DIR``) management.

Ground rules (per product owner):

1. The bootstrap shim ``~/.unitport_paths.ini`` is **NOT allowed to exist**.
   It is deleted on every boot. All workspace configuration lives in
   ``system.ini`` (the SDK's authoritative on-disk config).

2. ``user_config_dir`` is written **only** into
   ``system.ini[Resources].user_config_dir``. Business code that needs to
   resolve "the current account's USER_CONFIG_DIR" reads ``Paths.USER_CONFIG_DIR``
   (which the SDK derives from ``system.ini[Resources]`` after the shim
   is gone) — and ``Paths.USER_CONFIG_DIR`` always equals
   ``<workspace_root>/<slug>`` or ``<workspace_root>/_guest``, never the
   workspace root bare.

3. The WORKSPACE root (the install-time location, e.g. ``A:\\UnitPort\\``)
   lives in ``system.ini[Workspace].root``. It is set once by the install
   wizard and changed only by the explicit "relocate" UI.

4. The active session (signed-in user id + username) lives in
   ``system.ini[Session]``. It is updated by ``AuthManager`` on sign-in /
   sign-out / silent restore.

Public API:

* ``read_workspace_root()`` — current WORKSPACE root from ``system.ini[Workspace].root``.
* ``read_active_user()`` / ``read_active_username()`` — current account.
* ``set_user_workspace(username, user_id=...)`` — hot-switch to per-account
  ``<root>/<slug>/`` (or ``<root>/_guest/`` when ``username`` is empty),
  rename / merge ``_guest/`` into the slug on first sign-in, rewrite
  ``system.ini`` and ``Config.reload`` so ``Paths.USER_CONFIG_DIR`` follows.
* ``reload_workspace_data()`` — DataManager + user.ini overlay + registries
  + project snapshot. Called after a hot-switch.
* ``set_workspace_root(new_root)`` — wizard's "user picked install dir" hook;
  rewrites ``system.ini[Workspace].root`` and re-anchors USER_CONFIG_DIR.
* ``ensure_default_workspace_config()`` — boot-time housekeeping: delete
  any legacy shim, materialise ``_guest/`` if absent, normalise
  ``system.ini`` entries.
* ``migrate_legacy_root_to_guest()`` — pre-isolation installs that kept
  data directly at the WORKSPACE root.
* ``migrate_user_state_to_machine_level()`` — pre-elevation installs that
  kept ``setup_state.json`` / ``sdk/install_state.json`` under
  USER_CONFIG_DIR.
* ``machine_config_dir()`` — alias for the WORKSPACE root, used by
  business code that writes machine-level state (wizard / SDK install).

The ``write_shim`` / ``clear_shim`` / ``migrate_user_config_dir`` names
are kept as thin compatibility wrappers around the new system.ini-driven
implementation so older callers (install_config_wizard imports
``write_shim`` directly) keep working.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, Optional, Tuple

from unitport_sdk import Config, Paths, log_info, log_warning


# Whitelist of pre-isolation legacy file / directory names that
# ``migrate_legacy_root_to_guest`` is allowed to move into ``_guest/``.
# CRITICAL: this is an EXPLICIT WHITELIST, not a blacklist. Per-account
# slug directories (e.g. ``alpharius-unitport.ai/``) live as direct
# children of the WORKSPACE root and MUST NEVER be touched by the legacy
# migrator — moving them would silently destroy a signed-in user's data.
_LEGACY_FILE_NAMES: frozenset[str] = frozenset({
    "user.ini",
    "session.json",
})
_LEGACY_DIR_NAMES: frozenset[str] = frozenset({
    "projects",
    "avatars",
    "registers",
    "engines",
    "robot_assets",
    "runtime",
})

_MACHINE_STATE_FILES: tuple[tuple[str, str], ...] = (
    # (relative path inside USER_CONFIG_DIR, relative path inside BASE)
    ("setup_state.json", "setup_state.json"),
    ("sdk/install_state.json", "sdk/install_state.json"),
)


def default_user_config_dir() -> Path:
    """**DEPRECATED — there is no default USER_CONFIG_DIR.**

    Kept ONLY so older external callers don't crash on import; calling
    this function always raises. The install wizard's
    :class:`DataDirectoryPage` is the sole authoritative establisher
    of ``USER_CONFIG_DIR``; before it runs the value is unset and any
    code that would need a default is a boot-order bug to fix at the
    source, not paper over.
    """
    raise RuntimeError(
        "default_user_config_dir() has no value to return. "
        "USER_CONFIG_DIR is established by the install wizard's "
        "DataDirectoryPage and persisted into system.ini[Workspace].root + "
        "[Resources].user_config_dir. Code reaching this function before "
        "the wizard ran is a boot-order bug — fix the call site, do not "
        "introduce a fallback."
    )


# ===========================================================================
# system.ini patcher — comment-preserving line-based update.
# ===========================================================================

_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")
_KEY_RE = re.compile(r"^(?P<lead>\s*)(?P<key>[A-Za-z0-9_]+)\s*=.*$")


def _patch_system_ini(updates: Dict[str, Dict[str, str]]) -> bool:
    """Write key/value updates into ``system.ini``, preserving comments and order.

    ``updates`` is a ``{section_name: {key: value, ...}}`` mapping.
    Existing keys are rewritten in place; missing keys are appended at
    the end of the section; missing sections are appended at the end of
    the file.

    Returns ``True`` if the file was actually modified.
    """
    path = Paths.SYSTEM_INI
    if not path.exists():
        log_warning(f"[workspace] system.ini missing: {path}")
        return False

    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        log_warning(f"[workspace] cannot read {path}: {exc}")
        return False

    lines = original.splitlines(keepends=True)
    pending: Dict[str, Dict[str, str]] = {
        sec: dict(kvs) for sec, kvs in updates.items()
    }

    # Map every section header to its line range.
    section_ranges: list[tuple[str, int, int]] = []   # (name, start, end_exclusive)
    last_header_idx: Optional[int] = None
    last_header_name: Optional[str] = None
    for i, raw in enumerate(lines):
        m = _SECTION_RE.match(raw.rstrip("\r\n"))
        if m:
            if last_header_idx is not None and last_header_name is not None:
                section_ranges.append((last_header_name, last_header_idx + 1, i))
            last_header_idx = i
            last_header_name = m.group("name").strip()
    if last_header_idx is not None and last_header_name is not None:
        section_ranges.append((last_header_name, last_header_idx + 1, len(lines)))

    # Pass 1: rewrite existing keys in place.
    for sec_name, sec_start, sec_end in section_ranges:
        if sec_name not in pending:
            continue
        kvs = pending[sec_name]
        for i in range(sec_start, sec_end):
            stripped = lines[i].rstrip("\r\n")
            if not stripped or stripped.lstrip().startswith((";", "#")):
                continue
            m = _KEY_RE.match(stripped)
            if not m:
                continue
            key = m.group("key")
            if key in kvs:
                value = kvs.pop(key)
                lead = m.group("lead") or ""
                # Preserve trailing newline of the original line.
                eol = "\n" if not lines[i].endswith(("\r\n", "\n")) else lines[i][len(lines[i].rstrip("\r\n")):]
                lines[i] = f"{lead}{key} = {value}{eol}"

    # Pass 2: append missing keys to existing sections.
    if any(kvs for kvs in pending.values()):
        # Recompute section ranges since we only rewrote lines in place, but
        # let's recompute defensively anyway.
        new_section_ranges: list[tuple[str, int, int]] = []
        last_h_idx: Optional[int] = None
        last_h_name: Optional[str] = None
        for i, raw in enumerate(lines):
            m = _SECTION_RE.match(raw.rstrip("\r\n"))
            if m:
                if last_h_idx is not None and last_h_name is not None:
                    new_section_ranges.append((last_h_name, last_h_idx + 1, i))
                last_h_idx = i
                last_h_name = m.group("name").strip()
        if last_h_idx is not None and last_h_name is not None:
            new_section_ranges.append((last_h_name, last_h_idx + 1, len(lines)))

        # Build a fast lookup from section name → (insert_at, last_content_at).
        sec_lookup: Dict[str, tuple[int, int]] = {
            name: (start, end) for name, start, end in new_section_ranges
        }

        # Insert remaining keys at the end of their section. Walk from last
        # section to first so indices stay valid.
        insertions: list[tuple[int, list[str]]] = []
        for sec_name in list(pending.keys()):
            kvs = pending[sec_name]
            if not kvs:
                continue
            if sec_name not in sec_lookup:
                continue  # missing section handled below
            start, end = sec_lookup[sec_name]
            # Insert just after the last non-blank content line of the section.
            insert_at = end
            for i in range(end - 1, start - 1, -1):
                if lines[i].strip():
                    insert_at = i + 1
                    break
            new_lines = [f"{k} = {v}\n" for k, v in kvs.items()]
            insertions.append((insert_at, new_lines))
            pending[sec_name] = {}
        for insert_at, new_lines in sorted(insertions, key=lambda t: -t[0]):
            lines[insert_at:insert_at] = new_lines

    # Pass 3: append missing sections at the end of the file.
    appended_lines: list[str] = []
    for sec_name, kvs in pending.items():
        if not kvs:
            continue
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        appended_lines.append("\n")
        appended_lines.append(f"[{sec_name}]\n")
        for k, v in kvs.items():
            appended_lines.append(f"{k} = {v}\n")
    if appended_lines:
        lines.extend(appended_lines)

    new_text = "".join(lines)
    if new_text == original:
        return False

    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        log_warning(f"[workspace] cannot write {path}: {exc}")
        return False
    return True


def _read_system_ini_value(section: str, key: str) -> str:
    """Read a single value out of system.ini. Returns '' on any miss."""
    path = Paths.SYSTEM_INI
    if not path.exists():
        return ""
    cp = ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except Exception:                                            # noqa: BLE001
        return ""
    return (cp.get(section, key, fallback="") or "").strip()


def read_system_ini_value(section: str, key: str) -> str:
    """Public wrapper for :func:`_read_system_ini_value`.

    Exposed so sibling services (notably the updater) can read raw
    system.ini values without reaching into the underscore-prefixed
    internal helper.
    """
    return _read_system_ini_value(section, key)


def patch_system_ini(updates: Dict[str, Dict[str, str]]) -> bool:
    """Public wrapper for :func:`_patch_system_ini`.

    Exposed so sibling services can rewrite system.ini entries while
    preserving comments and section ordering. Same return contract as
    the underscore version: ``True`` iff the file was actually modified.
    """
    return _patch_system_ini(updates)


# ===========================================================================
# Shim deletion — the bootstrap shim is illegal under the new ground rules.
# ===========================================================================


def _rescue_legacy_shim_into_system_ini() -> Dict[str, str]:
    """Before deleting the shim, salvage any meaningful state from it.

    Returns a flat dict of recovered fields (empty when there is no
    shim or it's unreadable / empty). The caller decides what to write
    into ``system.ini`` based on what is missing from system.ini today.
    """
    shim = Paths._BOOTSTRAP_SHIM
    if not shim.exists():
        return {}
    cp = ConfigParser()
    try:
        cp.read(shim, encoding="utf-8")
    except Exception:                                            # noqa: BLE001
        return {}
    out: Dict[str, str] = {}
    for sec, key in (
        ("Resources", "user_config_dir"),
        ("Workspace", "root"),
        ("Session", "active_user"),
        ("Session", "active_username"),
    ):
        val = (cp.get(sec, key, fallback="") or "").strip()
        if val:
            out[f"{sec}.{key}"] = val
    return out


def _delete_bootstrap_shim() -> bool:
    """Delete ``~/.unitport_paths.ini`` if present.

    The shim is illegal under the new design — all workspace configuration
    lives in system.ini. Boot-time housekeeping calls this unconditionally
    AFTER rescuing any state it carries into system.ini (see
    :func:`_rescue_legacy_shim_into_system_ini` and the boot path in
    :func:`ensure_default_workspace_config`).

    Returns ``True`` if a file was actually removed.
    """
    shim = Paths._BOOTSTRAP_SHIM
    if not shim.exists():
        return False
    try:
        shim.unlink()
        log_info(f"[workspace] removed illegal bootstrap shim {shim}")
        return True
    except OSError as exc:
        log_warning(f"[workspace] could not remove shim {shim}: {exc}")
        return False


# ===========================================================================
# Workspace root + active session — system.ini-backed.
# ===========================================================================


def read_workspace_root() -> Optional[Path]:
    """Resolve the live WORKSPACE root from ``system.ini[Workspace].root``.

    Returns ``None`` when neither ``[Workspace].root`` nor
    ``[Resources].user_config_dir`` is configured — that is the
    correct state on the first launch BEFORE the install wizard's
    DataDirectoryPage runs. Callers that need a Path must check for
    ``None`` and either wait for the wizard or raise; there is no
    fallback default.

    Resolution order:
      1. ``system.ini[Workspace].root`` if set;
      2. Else derived from ``system.ini[Resources].user_config_dir``:
         if its tail is ``_guest`` or sits next to a ``_guest`` sibling,
         the parent; else the path itself.
      3. None — wizard has not yet been completed.
    """
    explicit = _read_system_ini_value("Workspace", "root")
    if explicit:
        try:
            return Path(explicit).expanduser()
        except OSError:
            pass

    recorded = _read_system_ini_value("Resources", "user_config_dir")
    if recorded:
        try:
            p = Path(recorded).expanduser()
            if p.name == "_guest":
                return p.parent
            if p.parent.exists():
                try:
                    if any(
                        sib.name == "_guest"
                        for sib in p.parent.iterdir()
                        if sib.is_dir()
                    ):
                        return p.parent
                except OSError:
                    pass
            return p
        except OSError:
            pass

    return None


def read_active_user() -> Optional[str]:
    """Return ``system.ini[Session].active_user`` (Supabase uid) or ``None``."""
    val = _read_system_ini_value("Session", "active_user")
    return val or None


def read_active_username() -> Optional[str]:
    """Return ``system.ini[Session].active_username`` (human-readable) or ``None``."""
    val = _read_system_ini_value("Session", "active_username")
    return val or None


def machine_config_dir() -> Path:
    """Machine-level shared root — same as the WORKSPACE root.

    Raises ``RuntimeError`` if the workspace root is not yet configured
    (wizard's DataDirectoryPage hasn't run). Machine-level state has
    no business existing before the user has chosen WHERE.
    """
    root = read_workspace_root()
    if root is None:
        raise RuntimeError(
            "machine_config_dir() called before WORKSPACE.root is "
            "configured. The install wizard's DataDirectoryPage must "
            "run first; fix the call site rather than introducing a "
            "fallback."
        )
    return root


# ===========================================================================
# Machine-level language preference.
#
# The user's UI language is a **device** preference, not an account
# preference. It must survive sign-in / sign-out / account switches so
# that switching accounts never silently changes what the user is
# looking at. We therefore persist it at the WORKSPACE root in
# ``locale.ini`` — alongside ``setup_state.json`` and
# ``sdk/install_state.json`` — and treat that file as the source of
# truth. The SDK's ``user.ini[Localisation].lang`` is kept in sync only
# as a side-effect of ``I18n.set_language`` writing through Config.
# ===========================================================================


_MACHINE_LOCALE_FILENAME = "locale.ini"


def _machine_locale_path() -> Optional[Path]:
    """Return ``WORKSPACE/locale.ini`` or ``None`` when workspace unset.

    A fresh install (wizard not yet run) has no workspace root and
    therefore no place to store the device-level locale; the in-memory
    SDK value carries the pick until the wizard runs.
    """
    root = read_workspace_root()
    if root is None:
        return None
    return root / _MACHINE_LOCALE_FILENAME


def read_machine_locale() -> Optional[str]:
    """Return the device-level language code, or ``None`` if unset.

    Also returns ``None`` if the workspace root is not yet configured —
    the wizard hasn't run, so there's no file to read.
    """
    path = _machine_locale_path()
    if path is None or not path.exists():
        return None
    try:
        cp = ConfigParser()
        cp.read(path, encoding="utf-8")
        if cp.has_option("Localisation", "lang"):
            val = cp.get("Localisation", "lang").strip()
            return val or None
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] read_machine_locale failed: {exc}")
    return None


def write_machine_locale(lang: str) -> None:
    """Persist the device-level language code to ``WORKSPACE/locale.ini``.

    No-op when the workspace root is not yet configured (fresh install
    pre-wizard). The in-memory SDK value holds the pick until the wizard
    establishes a workspace.
    """
    lang = (lang or "").strip()
    if not lang:
        return
    path = _machine_locale_path()
    if path is None:
        return
    cp = ConfigParser()
    if path.exists():
        try:
            cp.read(path, encoding="utf-8")
        except Exception:                                          # pragma: no cover
            cp = ConfigParser()
    if not cp.has_section("Localisation"):
        cp.add_section("Localisation")
    if cp.get("Localisation", "lang", fallback="") == lang:
        return
    cp.set("Localisation", "lang", lang)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with io.StringIO() as buf:
            cp.write(buf)
            path.write_text(buf.getvalue(), encoding="utf-8")
        log_info(f"[workspace] persisted device-level language preference: {lang}")
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] write_machine_locale failed: {exc}")


def apply_machine_locale_preference() -> None:
    """Reconcile in-memory ``I18n`` state with the device-level preference.

    Called at boot (after the SDK has loaded its default from
    ``system.ini``) and after every workspace hot-switch (so a new
    account's empty ``user.ini`` cannot silently revert the language).

    Semantics:

    * If ``WORKSPACE/locale.ini`` carries a ``[Localisation].lang`` value
      and it differs from ``I18n.get_language()``, call
      ``I18n.set_language(saved)``. This writes the value back into the
      now-current ``user.ini`` (so the slug-scoped overlay agrees with
      the machine-level truth) and emits ``language_changed`` so every
      bound widget re-translates.
    * If the file is missing, seed it from the current in-memory
      language so that subsequent account switches have something to
      restore from. This handles the first-run case where the SDK's
      ``I18n.init()`` populated ``_current_lang`` from
      ``system.ini[Localisation].lang`` but nothing has yet written the
      machine-level file.
    """
    try:
        from unitport_sdk import I18n
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] apply_machine_locale_preference: I18n unavailable ({exc})")
        return

    saved = read_machine_locale()
    if saved:
        current = I18n.get_language()
        if current != saved:
            log_info(
                f"[workspace] restoring device language preference "
                f"{saved!r} (was {current!r})"
            )
            try:
                I18n.set_language(saved)
            except Exception as exc:                              # pragma: no cover
                log_warning(f"[workspace] I18n.set_language failed: {exc}")
        return

    # No machine-level file yet — seed it from whatever the SDK loaded.
    current = I18n.get_language()
    if current:
        write_machine_locale(current)


# ===========================================================================
# Reload helpers.
# ===========================================================================


def reload_paths() -> None:
    """Re-resolve every Tier-B ``Paths`` slot and flush ``Config``'s cache.

    After mutating ``system.ini`` we must call this so the SDK re-reads
    ``[Resources]`` and ``Paths.USER_CONFIG_DIR`` reflects the new value.
    """
    # Ensure the SDK does not pick up a stale shim — earlier code paths
    # may have written it; the new design forbids it. Removing is cheap
    # and guarantees Paths._resolve_dynamic falls through to system.ini.
    _delete_bootstrap_shim()
    Paths._resolve_dynamic()
    try:
        Config.reload()
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] Config.reload failed: {exc}")


def reload_workspace_data() -> None:
    """Re-hydrate cached state from the (possibly new) ``USER_CONFIG_DIR``.

    Called after a hot workspace switch (login / logout / account swap)
    so business modules see the new directory's contents instead of the
    previous account's cached values.
    """
    try:
        from unitport_sdk import DataManager
        DataManager.clear()
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] DataManager.clear failed: {exc}")

    try:
        from unitport_sdk import load_data
        if Paths.USER_INI.exists():
            load_data(Paths.USER_INI, force_reload=True)
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] user.ini reload failed: {exc}")

    try:
        from registers import RegistryHub
        RegistryHub.load_all()
        RegistryHub.validate()
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] registry reload failed: {exc}")

    try:
        from application.service.projects import get_project_store
        get_project_store().refresh_snapshot()
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] project snapshot refresh failed: {exc}")

    # Language preference is device-level, not account-level. After the
    # workspace switch the new ``user.ini`` may not carry a
    # ``[Localisation]`` section; re-apply the machine-level value so the
    # user never sees their language flip just because they signed in /
    # out / swapped accounts.
    try:
        apply_machine_locale_preference()
    except Exception as exc:                                      # pragma: no cover
        log_warning(f"[workspace] apply_machine_locale_preference failed: {exc}")

    log_info("[workspace] workspace data re-hydrated for new USER_CONFIG_DIR")


# ===========================================================================
# Slug helpers.
# ===========================================================================


_SAFE_SLUG = re.compile(r"[^a-z0-9._-]")
_DASH_RUNS = re.compile(r"-+")


def _slug_username(username: str) -> str:
    """Convert a user-visible name into a safe directory name.

    Rules:
      * lowercase + ASCII-only
      * keep ``[a-z0-9._-]``, replace other chars with ``-``
      * collapse ``-`` runs and strip leading / trailing ``-``
      * if the result is empty, longer than 40 chars, starts with ``_``
        or ``.``, fall back to ``sha256(raw)[:16]``

    Returns an empty string when ``username`` is empty.

    NOTE: as of the UUID-keyed workspace design this helper is only used
    by the legacy-compat migrator (:func:`_migrate_email_slug_to_uuid`).
    New code routes through the UUID directly — see :func:`set_user_workspace`.
    """
    if not username:
        return ""
    s = _DASH_RUNS.sub("-", _SAFE_SLUG.sub("-", username.strip().lower())).strip("-")
    if not s or len(s) > 40 or s.startswith("_") or s.startswith("."):
        return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
    return s


# ---------------------------------------------------------------------------
# Legacy email-slug → UUID-slug silent migrator.
#
# Pre-cloud-storage builds keyed per-user workspaces by an email-derived
# slug (e.g. ``alice-example.com``). The new design keys by Supabase UUID
# so cloud-side prefixes line up with local FS layout. When a returning
# user has the legacy slug on disk we silently rename it to the UUID slug
# the first time we notice — no UI prompt, no data movement beyond the
# rename, idempotent on subsequent boots.
# ---------------------------------------------------------------------------


def _migrate_email_slug_to_uuid(
    base: Path, user_id: str, username: str
) -> Optional[Path]:
    """Rename ``<base>/<email_slug>/`` to ``<base>/<user_id>/`` in place.

    Returns the resulting target path when the rename happened (or when a
    merge ran), ``None`` when there was nothing to do.

    Behaviour:
      * No-op if ``user_id`` is falsy.
      * No-op if the legacy email-slug dir does not exist.
      * If the target UUID dir does not exist → ``shutil.move`` rename.
      * If both exist → non-destructive merge (prefers existing target,
        the canonical post-migration copy). Empty legacy dir is then
        removed.

    Safe to call multiple times; the second call will find no legacy dir
    and return ``None`` without touching disk.
    """
    if not user_id or not username:
        return None
    legacy_slug = _slug_username(username)
    if not legacy_slug or legacy_slug == user_id:
        return None
    legacy_dir = base / legacy_slug
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return None
    target_dir = base / user_id
    try:
        if target_dir.exists():
            moved = _merge_guest_into_slug(
                legacy_dir, target_dir, prefer_existing=True,
            )
            if moved:
                log_info(
                    f"[workspace] legacy-compat: merged {moved} item(s) "
                    f"from email-slug {legacy_dir.name!r} into "
                    f"UUID-slug {target_dir.name!r}"
                )
            try:
                if not any(legacy_dir.iterdir()):
                    legacy_dir.rmdir()
            except OSError:
                pass
            return target_dir
        shutil.move(str(legacy_dir), str(target_dir))
        log_info(
            f"[workspace] legacy-compat: renamed email-slug workspace "
            f"{legacy_dir.name!r} -> UUID-slug {target_dir.name!r}"
        )
        return target_dir
    except Exception as exc:                                  # noqa: BLE001
        log_warning(
            f"[workspace] legacy email-slug migration failed "
            f"({legacy_dir} -> {target_dir}): {exc}"
        )
        return None


# ===========================================================================
# system.ini writers — the new source of truth.
# ===========================================================================


def _apply_workspace_config(
    *,
    workspace_root: Path,
    user_config_dir: Path,
    active_user: str = "",
    active_username: str = "",
) -> None:
    """Persist workspace state into ``system.ini`` and remove the shim.

    Writes (idempotently, preserving comments):
      ``[Resources] user_config_dir`` — current per-account dir
      ``[Workspace] root``           — install-time WORKSPACE root
      ``[Session] active_user``      — Supabase uid (or empty for guest)
      ``[Session] active_username``  — human-readable name

    Then deletes ``~/.unitport_paths.ini`` (illegal under the new design)
    so the SDK reads exclusively from ``system.ini``.
    """
    _delete_bootstrap_shim()
    _patch_system_ini({
        "Resources": {"user_config_dir": str(user_config_dir)},
        "Workspace": {"root": str(workspace_root)},
        "Session": {
            "active_user": active_user or "",
            "active_username": active_username or "",
        },
    })


# Compatibility wrapper — ``install_config_wizard`` imports this symbol
# directly under the old name. New code should call set_workspace_root.
def write_shim(
    user_config_dir: Path,
    *,
    active_user: Optional[str] = None,
    active_username: Optional[str] = None,
    workspace_root: Optional[Path] = None,
) -> None:
    """**Compat shim** for legacy callers — writes the same data into
    ``system.ini`` (NOT the bootstrap shim, which is now forbidden).

    The old shim file is deleted as a side-effect.
    """
    if workspace_root is None:
        # Infer: if user_config_dir tail is _guest or sits next to one,
        # parent is the workspace root; else treat it as the root itself.
        if user_config_dir.name == "_guest":
            workspace_root = user_config_dir.parent
        elif user_config_dir.parent.exists() and any(
            sib.name == "_guest"
            for sib in user_config_dir.parent.iterdir()
            if sib.is_dir()
        ):
            workspace_root = user_config_dir.parent
        else:
            workspace_root = user_config_dir
    _apply_workspace_config(
        workspace_root=Path(workspace_root),
        user_config_dir=Path(user_config_dir),
        active_user=active_user or "",
        active_username=active_username or "",
    )


def clear_shim() -> None:
    """Remove the legacy bootstrap shim AND clear ``[Resources].user_config_dir``.

    After this, the SDK falls back to its built-in default for
    USER_CONFIG_DIR (``PROJECT_ROOT/runtime/user_config``; never ``~/UnitPort``).
    """
    _delete_bootstrap_shim()
    _patch_system_ini({
        "Resources": {"user_config_dir": ""},
        "Workspace": {"root": ""},
        "Session": {"active_user": "", "active_username": ""},
    })


def set_workspace_root(new_root: Path) -> None:
    """User chose a new WORKSPACE root (e.g. wizard data-dir picker, or
    ``UserPanel`` relocate). Writes the root to ``system.ini[Workspace].root``
    and re-anchors USER_CONFIG_DIR to ``<new_root>/<slug or _guest>/``.
    """
    new_root = Path(new_root).expanduser()
    new_root.mkdir(parents=True, exist_ok=True)
    username = read_active_username() or ""
    user_id = read_active_user() or ""
    if user_id:
        # New design: per-user slug = Supabase UUID (see set_user_workspace).
        target = new_root / user_id
    elif username:
        # Legacy callsite (no uid in [Session]) — keep email-slug behaviour.
        slug = _slug_username(username)
        target = new_root / (slug or "_guest")
    else:
        target = new_root / "_guest"
    target.mkdir(parents=True, exist_ok=True)
    _apply_workspace_config(
        workspace_root=new_root,
        user_config_dir=target,
        active_user=user_id,
        active_username=username,
    )
    reload_paths()


# ===========================================================================
# Merge helper (used when a returning account re-imports _guest data).
# ===========================================================================


def _merge_guest_into_slug(
    guest: Path,
    target: Path,
    *,
    prefer_existing: bool = True,
) -> int:
    """Non-destructively merge ``guest/`` into ``target/``.

    Walks guest one level at a time; directories present in both are
    recursed; file conflicts are resolved by ``prefer_existing`` (default
    keeps target's copy because target is canonical per-account state).
    Removes empty subdirs afterwards. Returns count of files moved.
    """
    moved = 0

    def _walk(src_dir: Path, dst_dir: Path) -> None:
        nonlocal moved
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            children = list(src_dir.iterdir())
        except OSError:
            return
        for entry in children:
            dst = dst_dir / entry.name
            try:
                if entry.is_dir():
                    if dst.exists() and dst.is_dir():
                        _walk(entry, dst)
                        try:
                            entry.rmdir()
                        except OSError:
                            pass
                    else:
                        shutil.move(str(entry), str(dst))
                        moved += 1
                else:
                    if dst.exists():
                        if prefer_existing:
                            try:
                                entry.unlink()
                            except OSError as exc:
                                log_warning(
                                    f"[workspace] merge: could not drop "
                                    f"{entry}: {exc}"
                                )
                        else:
                            try:
                                dst.unlink()
                                shutil.move(str(entry), str(dst))
                                moved += 1
                            except OSError as exc:
                                log_warning(
                                    f"[workspace] merge: overwrite failed for "
                                    f"{dst}: {exc}"
                                )
                    else:
                        shutil.move(str(entry), str(dst))
                        moved += 1
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[workspace] merge: could not move {entry} -> {dst}: {exc}"
                )

    _walk(guest, target)
    return moved


# ===========================================================================
# Hot workspace switch — the auth lifecycle entry point.
# ===========================================================================


def set_user_workspace(
    username: Optional[str],
    *,
    user_id: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Switch USER_CONFIG_DIR to ``<base>/{slug}/`` (or ``<base>/_guest/``).

    Behaviour:
      * ``username=""``/``None`` → ``<base>/_guest/`` (sign-out / never-signed-in).
      * Non-empty username → ``<base>/{slug}/``.

    First-login promotion:
      * if the live ``USER_CONFIG_DIR`` is ``_guest/`` and the slug dir
        does not yet exist → rename ``_guest/`` to ``{slug}/``;
      * if the slug dir already exists (returning user) and ``_guest/``
        has new data → merge it into the slug.

    Persists the new state to ``system.ini`` (Resources / Workspace /
    Session sections) and calls :func:`reload_paths` so the SDK picks
    up the change immediately. Caller may follow with
    :func:`reload_workspace_data` to drop in-memory caches.
    """
    if base_dir is None:
        base_dir = read_workspace_root()
    base = Path(base_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)

    if not username and not user_id:
        target = base / "_guest"
        active_uid = ""
        active_un = ""
    else:
        active_uid = user_id or ""
        active_un = username or ""
        # New design: per-user slug = Supabase UUID (filesystem-safe, stable,
        # matches cloud-side prefix). Fall back to email-slug only when the
        # caller did not supply a uid (legacy callers; should not happen in
        # normal sign-in).
        if active_uid:
            # Silent legacy-compat: if a prior install left an email-slug
            # directory for this user, rename it to the UUID dir before we
            # compute target. Idempotent — no-op on a fresh UUID workspace.
            _migrate_email_slug_to_uuid(base, active_uid, active_un)
            target = base / active_uid
        else:
            slug = _slug_username(active_un)
            target = base / (slug or "_guest")

    current = Path(Paths.USER_CONFIG_DIR)
    if (
        (username or user_id)
        and target != current
        and current.name == "_guest"
        and current.exists()
    ):
        try:
            if not target.exists():
                shutil.move(str(current), str(target))
                log_info(
                    f"[workspace] promoted _guest workspace -> {target} "
                    f"for user {username!r}"
                )
            else:
                moved = _merge_guest_into_slug(current, target, prefer_existing=True)
                if moved:
                    log_info(
                        f"[workspace] merged {moved} _guest item(s) into "
                        f"existing {target} for user {username!r}"
                    )
                try:
                    if not any(current.iterdir()):
                        current.rmdir()
                except OSError:
                    pass
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[workspace] could not promote/merge _guest -> {target}: {exc}"
            )

    target.mkdir(parents=True, exist_ok=True)
    _apply_workspace_config(
        workspace_root=base,
        user_config_dir=target,
        active_user=active_uid,
        active_username=active_un,
    )
    reload_paths()
    log_info(
        f"[workspace] hot-switch complete: USER_CONFIG_DIR = {target}; "
        f"active_user = {active_uid or '(guest)'}; "
        f"active_username = {active_un or '(guest)'}; "
        f"workspace_root = {base}"
    )
    return target


# ===========================================================================
# Boot migrators.
# ===========================================================================


def ensure_default_workspace_config(base_dir: Optional[Path] = None) -> bool:
    """Boot-time normalization. Always runs.

    CRITICAL SAFETY INVARIANTS (data-loss prevention):
      * NEVER overwrites a non-empty ``user_config_dir`` with ``_guest``.
        The current ``user_config_dir`` is the user's live workspace; if
        it's a per-account slug (e.g. ``alpharius-unitport.ai``), it
        stays as-is even when ``[Session]`` happens to be empty.
      * NEVER moves a slug directory anywhere. Slug dirs are sacred.
      * When ``[Session]`` is missing but a slug ``user_config_dir`` is
        set, attempts to recover ``active_user`` / ``active_username``
        from the on-disk ``session.json`` (the cached profile written by
        ``profile_cache``). This is what restores "auto sign-in" after
        an upgrade from a build that didn't persist the session.

    Steps:
      1. Rescue any legacy ``~/.unitport_paths.ini`` shim into
         ``system.ini`` (only filling gaps; never overwriting existing
         values), then delete the shim.
      2. If ``[Workspace].root`` and ``[Resources].user_config_dir``
         are BOTH unset → **return False immediately**. The install
         wizard's :class:`DataDirectoryPage` is the authoritative
         establisher; this function NEVER materialises a default
         directory on the user's behalf. Silent fallback would write
         files to a path the rest of the app never reads back from,
         producing the illusion of saved state.
      3. Otherwise (settings are configured): mkdir the configured
         paths if missing (the user explicitly chose them so creating
         is authorised), persist ``[Workspace].root`` if it was
         derivable from ``user_config_dir``, and validate.
      4. If ``[Session]`` is empty but ``user_config_dir`` is a slug,
         recover session from on-disk ``session.json``.

    Returns ``True`` if anything was rewritten (caller should
    :func:`reload_paths`).
    """
    # Step 1 — rescue legacy shim BEFORE deletion.
    rescued = _rescue_legacy_shim_into_system_ini()
    if rescued:
        merged: Dict[str, Dict[str, str]] = {}
        for compound, value in rescued.items():
            section, key = compound.split(".", 1)
            current = _read_system_ini_value(section, key)
            if not current:
                merged.setdefault(section, {})[key] = value
        if merged:
            _patch_system_ini(merged)
            log_info(
                f"[workspace] rescued shim state into system.ini: {merged}"
            )

    rewritten = _delete_bootstrap_shim()

    # Step 2 — refuse to materialise a default when both settings are blank.
    # The wizard's DataDirectoryPage is the sole authority; here we only
    # validate / mkdir paths the user has already authorised.
    recorded_root = _read_system_ini_value("Workspace", "root").strip()
    recorded_ucd = _read_system_ini_value("Resources", "user_config_dir").strip()
    if not recorded_root and not recorded_ucd:
        log_info(
            "[workspace] [Workspace].root and [Resources].user_config_dir "
            "are both unset; awaiting install wizard's DataDirectoryPage. "
            "Refusing to materialise a default — anything written via "
            "Storage.push before the wizard runs will raise loudly."
        )
        return rewritten   # only True if the legacy shim was deleted

    if base_dir is None:
        base_dir = read_workspace_root()
        if base_dir is None:
            # ``recorded_ucd`` was set but unparseable — log and skip
            # without inventing anything.
            log_warning(
                f"[workspace] user_config_dir={recorded_ucd!r} could not "
                f"be resolved to a Path; leaving system.ini alone. Re-run "
                f"the wizard to pick a valid location."
            )
            return rewritten
    base = Path(base_dir).expanduser()
    # User-authorised mkdir: this path comes from a value the user
    # picked (either now or in a past wizard run), so creating it is
    # legitimate.
    base.mkdir(parents=True, exist_ok=True)

    if recorded_root != str(base):
        _patch_system_ini({"Workspace": {"root": str(base)}})
        rewritten = True

    # Step 3 — validate user_config_dir WITHOUT clobbering an existing slug.
    if recorded_ucd:
        try:
            current_ucd = Path(recorded_ucd).expanduser()
        except OSError:
            current_ucd = None
    else:
        current_ucd = None

    if current_ucd is None:
        # Workspace.root is set but user_config_dir is unparseable.
        # Do NOT default-materialise _guest/ — that's the wizard's job.
        log_warning(
            "[workspace] [Workspace].root is set but "
            "[Resources].user_config_dir is unset/unparseable; awaiting "
            "wizard. NOT auto-creating _guest/."
        )
    elif current_ucd.parent.exists():
        # Both Workspace.root and user_config_dir are configured — the
        # user picked this path explicitly. Creating it is authorised.
        if not current_ucd.exists():
            try:
                current_ucd.mkdir(parents=True, exist_ok=True)
                log_info(
                    f"[workspace] re-created missing user_config_dir = {current_ucd}"
                )
            except OSError as exc:
                log_warning(
                    f"[workspace] could not create {current_ucd}: {exc}"
                )
    else:
        log_warning(
            f"[workspace] user_config_dir = {current_ucd} points at a "
            f"location whose parent does not exist; re-run the wizard "
            f"to pick a valid path. NOT auto-defaulting."
        )

    # Step 4 — recover [Session] from cached profile when missing but the
    # user_config_dir is a slug (i.e. someone was signed in before the
    # upgrade but the session bookkeeping didn't survive). For the
    # email-slug→UUID rebuild we also synthesize an active_user value
    # from the cached profile so Step 5 can rename the legacy dir.
    if not read_active_username():
        current_ucd = Path(_read_system_ini_value("Resources", "user_config_dir"))
        if current_ucd.name != "_guest":
            try:
                from application.service.auth.profile_cache import (
                    load_cached_profile,
                )
                profile = load_cached_profile()
                if profile and not profile.is_empty and (
                    profile.email or profile.user_id
                ):
                    recovered_username = profile.email or profile.user_id
                    _patch_system_ini({
                        "Session": {
                            "active_user": profile.user_id or "",
                            "active_username": recovered_username,
                        }
                    })
                    log_info(
                        f"[workspace] recovered session from cached profile: "
                        f"active_user={profile.user_id!r}, "
                        f"active_username={recovered_username!r}"
                    )
                    rewritten = True
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[workspace] could not recover session from profile cache: {exc}"
                )

    # Step 5 — silent email-slug → UUID-slug migration.
    #
    # If [Session].active_user is a Supabase UUID and the current
    # user_config_dir directory name is a *legacy email-derived slug*
    # (i.e. matches ``_slug_username(active_username)`` but not the
    # UUID), rename it in place so the new UUID-keyed layout is the
    # source of truth from this boot onward. The auth lifecycle's
    # ``set_user_workspace`` would normally do this on next sign-in,
    # but silent restore short-circuits before reaching it — so we do
    # it here, exactly once per upgrade.
    active_uid = read_active_user() or ""
    active_un = read_active_username() or ""
    if active_uid and active_un:
        current_ucd_str = _read_system_ini_value("Resources", "user_config_dir")
        if current_ucd_str:
            try:
                current_ucd = Path(current_ucd_str).expanduser()
            except OSError:
                current_ucd = None
            if (
                current_ucd is not None
                and current_ucd.name not in (active_uid, "_guest")
                and current_ucd.name == _slug_username(active_un)
            ):
                migrated = _migrate_email_slug_to_uuid(
                    base, active_uid, active_un,
                )
                if migrated is not None:
                    _patch_system_ini({
                        "Resources": {"user_config_dir": str(migrated)},
                    })
                    log_info(
                        f"[workspace] legacy-compat: user_config_dir "
                        f"re-anchored to UUID slug at {migrated}"
                    )
                    rewritten = True

    return rewritten


# Old name kept as an alias for callers that haven't been updated yet.
def ensure_default_workspace_shim(base_dir: Optional[Path] = None) -> bool:
    return ensure_default_workspace_config(base_dir)


def migrate_legacy_root_to_guest(base_dir: Optional[Path] = None) -> bool:
    """One-shot migrator for installs that predate per-account workspaces.

    Searches the WORKSPACE root for files/directories whose names match
    the well-known legacy layout (``user.ini``, ``projects/``, ``avatars/``,
    ``registers/``, ``engines/``, ``robot_assets/``, ``runtime/``,
    ``session.json``) and moves only those into ``_guest/``.

    **Slug directories** belonging to signed-in accounts (anything not in
    the legacy whitelist) are **NEVER** touched — they hold per-account
    data and moving them would destroy a user's workspace. The whitelist
    approach is a hard safety guarantee, not a heuristic.

    Idempotent. Returns ``True`` if anything was moved. Returns ``False``
    silently when the workspace root is not yet configured (fresh install
    pre-wizard).
    """
    if base_dir is None:
        base_dir = read_workspace_root()
    if base_dir is None:
        return False
    base = Path(base_dir).expanduser()

    if not base.exists():
        return False

    guest = base / "_guest"
    if guest.exists() and any(guest.iterdir()):
        # Migration already done in a previous run.
        return False

    try:
        candidates = []
        for child in base.iterdir():
            name = child.name
            if child.is_file() and name in _LEGACY_FILE_NAMES:
                candidates.append(child)
            elif child.is_dir() and name in _LEGACY_DIR_NAMES:
                candidates.append(child)
            # Anything else (per-account slug dirs, sentinels, machine-
            # level state) is left untouched — by design.
    except OSError as exc:
        log_warning(f"[workspace] could not scan {base}: {exc}")
        return False

    if not candidates:
        return False

    log_info(
        f"[workspace] legacy root layout detected at {base}; "
        f"moving {len(candidates)} whitelisted item(s) into _guest/"
    )
    guest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in candidates:
        dest = guest / child.name
        if dest.exists():
            log_warning(
                f"[workspace] skipping {child.name}: already present in _guest/"
            )
            continue
        try:
            shutil.move(str(child), str(dest))
            moved += 1
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[workspace] could not move {child} -> {dest}: {exc}")

    if moved == 0:
        return False

    # Only normalise system.ini's workspace_root; do NOT force
    # user_config_dir to _guest (an existing signed-in slug should remain
    # the active workspace if [Session] is set). The auth lifecycle / the
    # ensure_default_workspace_config call further down will fix
    # user_config_dir based on actual session state.
    if not _read_system_ini_value("Workspace", "root").strip():
        _patch_system_ini({"Workspace": {"root": str(base)}})
    log_info(
        f"[workspace] legacy migration complete; workspace_root = {base}"
    )
    return True


def migrate_user_state_to_machine_level(
    base_dir: Optional[Path] = None,
) -> bool:
    """Boot-time elevation of ``setup_state.json`` and ``sdk/install_state.json``.

    Older builds wrote these under USER_CONFIG_DIR (per-account); move
    them out to the WORKSPACE root so wizard / SDK-install state is
    shared across accounts. Idempotent. Returns ``False`` silently when
    the workspace root is not yet configured (fresh install pre-wizard).
    """
    if base_dir is None:
        base_dir = read_workspace_root()
    if base_dir is None:
        return False
    base = Path(base_dir).expanduser()
    if not base.exists():
        return False

    moved_any = False
    candidate_dirs: list[Path] = []
    guest = base / "_guest"
    if guest.exists():
        candidate_dirs.append(guest)
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if child.name in {"_guest", "_global", "users", "sdk"}:
                continue
            candidate_dirs.append(child)
    except OSError:
        pass

    for rel_user, rel_machine in _MACHINE_STATE_FILES:
        target = base / rel_machine
        if target.exists():
            continue
        for src_dir in candidate_dirs:
            src = src_dir / rel_user
            if not src.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(target))
                log_info(
                    f"[workspace] elevated {rel_user} from {src_dir} to "
                    f"{target} (machine-level)"
                )
                moved_any = True
                break
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[workspace] could not elevate {src} -> {target}: {exc}"
                )

    return moved_any


# ===========================================================================
# Workspace relocate (user-driven, via UserPanel "gear" button).
# ===========================================================================


def migrate_user_config_dir(new_dir: Path) -> Tuple[bool, str]:
    """Move all contents of the current ``USER_CONFIG_DIR`` to ``new_dir``.

    Used by the sidebar's manual relocate UI. After moving the live
    USER_CONFIG_DIR, rewrites system.ini so future boots pick it up.

    Refuses to proceed when:
      * ``new_dir`` resolves to the same path as the current workspace
      * ``new_dir`` is a sub-directory of the current workspace
      * ``new_dir`` already exists and is non-empty (avoids silent merge)
    """
    try:
        current = Path(Paths.USER_CONFIG_DIR).resolve()
    except OSError:
        current = Path(Paths.USER_CONFIG_DIR)
    try:
        target = Path(new_dir).expanduser().resolve()
    except OSError:
        target = Path(new_dir).expanduser()

    if target == current:
        return False, "Target equals the current workspace"

    try:
        if current in target.parents:
            return (
                False,
                "Cannot migrate into a sub-directory of the current workspace",
            )
    except (OSError, ValueError):
        pass

    if target.exists():
        try:
            non_empty = any(target.iterdir())
        except OSError as exc:
            return False, f"Cannot read target {target}: {exc}"
        if non_empty:
            return False, f"Target directory is non-empty: {target}"

    log_info(f"[workspace] migrating {current} -> {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if current.exists():
            if target.exists():
                target.rmdir()
            shutil.move(str(current), str(target))
        else:
            target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"Migration failed: {exc}"

    # Determine the workspace_root for the new location: parent if target
    # is _guest-shaped, else target itself.
    new_root = target.parent if target.name == "_guest" else target
    _apply_workspace_config(
        workspace_root=new_root,
        user_config_dir=target,
        active_user=read_active_user() or "",
        active_username=read_active_username() or "",
    )
    reload_paths()
    log_info(
        f"[workspace] migration complete; USER_CONFIG_DIR = {Paths.USER_CONFIG_DIR}"
    )
    return True, ""


def migrate_workspace_root(new_root: Path) -> Tuple[bool, str]:
    """Move the entire workspace tree to ``new_root`` and re-anchor paths.

    Used by the homepage Local Files card's Migrate button. The workspace
    root is the parent of every per-user dir (``_guest`` + each signed-in
    account's slug), so moving it relocates **all** users' projects,
    canvases, and cached state in one shot.

    Refuses to proceed when:
      * ``new_root`` resolves to the same path as the current root
      * ``new_root`` is a sub-directory of the current root (would loop)
      * the current root is already inside ``new_root`` (already there)
      * ``new_root`` already exists and is non-empty (avoid silent merge)

    On success, writes ``system.ini[Workspace].root = <new_root>`` and
    re-anchors ``USER_CONFIG_DIR`` to ``<new_root>/<active_slug>``. The
    caller must follow with :func:`reload_workspace_data` so cached
    state (DataManager, registries, ProjectStore) re-hydrates from the
    moved tree.

    Returns ``(ok, err_message)`` — ``err_message`` is empty on success.
    """
    try:
        current = read_workspace_root().resolve()
    except OSError:
        current = read_workspace_root()
    try:
        target = Path(new_root).expanduser().resolve()
    except OSError:
        target = Path(new_root).expanduser()

    if target == current:
        return False, "Target equals the current workspace root"

    try:
        if current in target.parents:
            return (
                False,
                "Cannot migrate into a sub-directory of the current workspace",
            )
        if target in current.parents:
            return (
                False,
                "Current workspace is already inside the target",
            )
    except (OSError, ValueError):
        pass

    if target.exists():
        try:
            non_empty = any(target.iterdir())
        except OSError as exc:
            return False, f"Cannot read target {target}: {exc}"
        if non_empty:
            return False, f"Target directory is non-empty: {target}"

    log_info(f"[workspace] migrating root {current} -> {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if current.exists():
            # shutil.move into an existing empty dir would nest under it
            # (target/<basename>) — rmdir first so the move performs a
            # rename instead.
            if target.exists():
                target.rmdir()
            shutil.move(str(current), str(target))
        else:
            target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"Migration failed: {exc}"

    # set_workspace_root rewrites system.ini and re-anchors USER_CONFIG_DIR
    # to <target>/<slug-or-_guest>; the moved tree already has that
    # structure inside ``target`` so the per-user data lines up.
    try:
        set_workspace_root(target)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"Path re-anchor failed: {exc}"
    log_info(f"[workspace] root migration complete; root = {target}")
    return True, ""


__all__ = [
    "default_user_config_dir",
    "read_workspace_root",
    "read_active_user",
    "read_active_username",
    "machine_config_dir",
    "read_machine_locale",
    "write_machine_locale",
    "apply_machine_locale_preference",
    "reload_paths",
    "reload_workspace_data",
    "set_user_workspace",
    "set_workspace_root",
    "migrate_user_config_dir",
    "migrate_workspace_root",
    "migrate_legacy_root_to_guest",
    "migrate_user_state_to_machine_level",
    "ensure_default_workspace_config",
    # Compat aliases — still imported by install_config_wizard.
    "write_shim",
    "clear_shim",
    "ensure_default_workspace_shim",
]
