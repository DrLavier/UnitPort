# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""EULA acceptance tracking for in-app installers.

Three responsibilities:

1. Read each licence's text from the static resources shipped under
   ``installers/data/*`` (NVOLA, Isaac Lab BSD-3, silent-prompts
   disclosure). The texts are vendored with the installer; we never
   download them at acceptance time so the user always sees the exact
   wording the installer will pass-through to `isaaclab.bat -i`.
2. Persist per-user acceptance records under
   ``Paths.USER_CONFIG_DIR / "eula_acceptance.json"`` so the wizard
   does not re-prompt on every reinstall. Records survive a clean
   ``git pull`` because they live outside the project tree (CLAUDE.md §1.4).
3. Verify the integrity of the vendored text files against the SHA256
   pinned in :mod:`._constants`. A mismatch is surfaced to the dialog
   so the user can review the diff before re-accepting.

Public surface:

* :class:`EulaRecord` — what we persist per accepted licence.
* :func:`list_required_eulas` — what the installer demands the user accept.
* :func:`read_eula_text` — load one licence's text + integrity status.
* :func:`load_acceptance` — read all stored records.
* :func:`record_acceptance` — append a new acceptance and flush to disk.
* :func:`required_eula_ids_satisfied` — terse check used by PostSetupTask.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from unitport_sdk import Paths, log_warning, push_data, read_data

from ._constants import EULA_TEXTS, EulaTextSpec

_DATA_DIR = Path(__file__).resolve().parent / "data"
_ACCEPTANCE_REL = "eula_acceptance.json"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EulaRecord:
    """One persisted acceptance.

    ``user_email`` is recorded so multi-user workspaces have an audit
    trail; empty string for guest sessions.
    """

    eula_id: str
    version: str
    accepted_at: str  # ISO-8601 UTC, e.g. 2026-05-18T13:42:01Z
    user_email: str = ""


@dataclass
class EulaText:
    """Loaded licence text + integrity status.

    ``hash_ok=False`` means the vendored file's SHA256 does not match
    the pinned constant. The dialog surfaces a warning banner in that
    case and points the user at ``upstream_url`` for a fresh copy.
    """

    spec: EulaTextSpec
    text: str
    actual_sha256: str
    hash_ok: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_required_eulas() -> tuple[EulaTextSpec, ...]:
    """Return the ordered list of licences the user must accept.

    Order matches :mod:`._constants.EULA_TEXTS`; the dialog renders one
    tab per entry in this order.
    """
    return EULA_TEXTS


def read_eula_text(spec: EulaTextSpec) -> EulaText:
    """Load one licence's text + verify its SHA256 against the pin.

    Hash mismatch is non-fatal — it returns a record with ``hash_ok=False``
    so the dialog can decide UX (we want the install to surface a
    warning, not silently install with text the user did not see).
    """
    path = _DATA_DIR / spec.file_name
    raw = path.read_bytes() if path.exists() else b""
    text = raw.decode("utf-8", errors="replace")
    actual = hashlib.sha256(raw).hexdigest()
    # Empty pinned hash means "do not check" (UnitPort-authored docs
    # like silent_prompts.md have no upstream to drift from). Otherwise
    # mismatch surfaces a banner in the dialog.
    hash_ok = (not spec.sha256) or (spec.sha256 == actual)
    return EulaText(spec=spec, text=text, actual_sha256=actual, hash_ok=hash_ok)


def _acceptance_path() -> Path:
    """Absolute path to the per-user acceptance store."""
    return Paths.USER_CONFIG_DIR / _ACCEPTANCE_REL


def load_acceptance() -> list[EulaRecord]:
    """Return every persisted acceptance record (empty list on first run)."""
    path = _acceptance_path()
    if not path.exists():
        return []
    try:
        payload = read_data(path)
    except Exception as exc:  # noqa: BLE001
        log_warning(f"[eula] could not read {path}: {exc}")
        return []
    if not isinstance(payload, dict):
        log_warning(f"[eula] {path} payload is not a dict; ignoring")
        return []
    records_raw = payload.get("records") or []
    out: list[EulaRecord] = []
    for entry in records_raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                EulaRecord(
                    eula_id=str(entry.get("eula_id", "")),
                    version=str(entry.get("version", "")),
                    accepted_at=str(entry.get("accepted_at", "")),
                    user_email=str(entry.get("user_email", "")),
                )
            )
        except Exception:
            continue
    return out


def record_acceptance(
    eula_id: str,
    version: str,
    *,
    user_email: str = "",
) -> bool:
    """Append an acceptance record. Idempotent on (eula_id, version).

    Returns True on a successful flush to disk; False (with log_warning)
    on I/O failure — the caller may still proceed with the install in
    memory but should NOT treat the user as having permanent acceptance.
    """
    now = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    new_record = EulaRecord(
        eula_id=str(eula_id),
        version=str(version),
        accepted_at=now,
        user_email=str(user_email or ""),
    )
    existing = load_acceptance()
    # Deduplicate by (eula_id, version): a re-accept overwrites the
    # earlier timestamp so the latest user action is what's audited.
    filtered = [
        r for r in existing
        if not (r.eula_id == new_record.eula_id and r.version == new_record.version)
    ]
    filtered.append(new_record)
    payload = {"records": [asdict(r) for r in filtered]}
    rel = _ACCEPTANCE_REL
    ok = push_data(rel, payload)
    if not ok:
        log_warning(
            f"[eula] failed to persist acceptance for {eula_id}@{version}; "
            f"the user may be re-prompted next launch"
        )
    return bool(ok)


def required_eula_ids_satisfied(
    *,
    require_specific_versions: bool = True,
) -> bool:
    """True iff every entry in :func:`list_required_eulas` has a record.

    When ``require_specific_versions`` is True (default), the record's
    ``version`` field must also match the currently-pinned version —
    so upgrading the installer to a new Isaac Sim release that ships
    a refreshed NVOLA forces a re-prompt. Set False for liberal checks
    (e.g. "the user has seen *some* version of these licences").
    """
    records = {(r.eula_id, r.version) for r in load_acceptance()}
    record_ids = {r.eula_id for r in load_acceptance()}
    for spec in list_required_eulas():
        if require_specific_versions:
            if (spec.eula_id, spec.version) not in records:
                return False
        else:
            if spec.eula_id not in record_ids:
                return False
    return True


def missing_eula_specs(
    *,
    require_specific_versions: bool = True,
) -> list[EulaTextSpec]:
    """Return the subset of required specs that have NOT been accepted.

    Used by the wizard to decide whether to open the dialog and by
    PostSetupTask's defensive guard against a hand-edited setup_state.json.
    """
    accepted_pairs = {(r.eula_id, r.version) for r in load_acceptance()}
    accepted_ids = {r.eula_id for r in load_acceptance()}
    missing: list[EulaTextSpec] = []
    for spec in list_required_eulas():
        if require_specific_versions:
            if (spec.eula_id, spec.version) not in accepted_pairs:
                missing.append(spec)
        else:
            if spec.eula_id not in accepted_ids:
                missing.append(spec)
    return missing


__all__ = [
    "EulaRecord",
    "EulaText",
    "list_required_eulas",
    "read_eula_text",
    "load_acceptance",
    "record_acceptance",
    "required_eula_ids_satisfied",
    "missing_eula_specs",
]
