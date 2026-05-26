# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reader for ``custom_mods/installs.txt`` -- the hot-plug install manifest.

Each non-comment line of ``installs.txt`` declares one optional third-party
git clone. The install wizard surfaces these as user-pickable checkboxes
(``BackendPage`` > "Reference Motion Libraries"); PostSetupTask clones only
the subset the user checked, into ``custom_mods/motions/<name>/``.

Format (whitespace-separated, see file header for full docs)::

    <relative_path>    <git_url>    [branch_or_commit]

- ``<relative_path>`` is resolved relative to ``custom_mods/``. The file
  ships with every entry under ``motions/`` so PostSetupTask has a single
  managed root.
- Lines starting with ``#`` and blank lines are ignored.

Public surface:

* :class:`CustomModEntry` -- one parsed line (``key``, ``relative_path``,
  ``url``, ``ref``, ``target_dir``).
* :func:`load_manifest_entries` -- parse the on-disk manifest into a list
  of entries. Missing file → empty list (the wizard simply shows no
  options); malformed lines log a warning and are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from unitport_sdk import Paths, log_warning


__all__ = [
    "CustomModEntry",
    "manifest_path",
    "load_manifest_entries",
    "entry_already_installed",
]


@dataclass(frozen=True)
class CustomModEntry:
    """One parsed line of ``custom_mods/installs.txt``.

    ``key`` is the basename of ``relative_path`` and acts as the stable
    identifier echoed into ``selections.backend.custom_mods``. ``target_dir``
    is the absolute on-disk clone destination resolved against the
    user's ``Paths.CUSTOM_MODS_DIR`` (overlay-resolved at module load).

    ``required_for`` is the parsed ``required_for=`` annotation: a tuple
    of ``"<backend>/<template>"`` strings pointing into
    ``custom_mods/canvas/<backend>/[<backend>] <template>.canvas.json``.
    An entry with a non-empty ``required_for`` is pre-checked in the
    install wizard and labelled so the user understands which shipped
    template would silently break without it.
    """

    key: str
    relative_path: str
    url: str
    ref: str
    required_for: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def target_dir(self) -> Path:
        return Path(Paths.CUSTOM_MODS_DIR) / self.relative_path

    @property
    def display_name(self) -> str:
        return self.key.replace("_", " ").replace("-", " ")

    @property
    def is_required(self) -> bool:
        return bool(self.required_for)


def manifest_path() -> Path:
    """Return the canonical on-disk path of ``installs.txt``."""
    return Path(Paths.CUSTOM_MODS_DIR) / "installs.txt"


def load_manifest_entries(path: Optional[Path] = None) -> List[CustomModEntry]:
    """Parse ``installs.txt`` into a list of :class:`CustomModEntry`.

    A missing manifest is treated as "no optional installs available" and
    returns an empty list -- the wizard simply renders no checkboxes.
    Malformed lines (fewer than two whitespace-separated fields) are
    skipped with a WARN; we do not raise, because a broken third-party
    entry must not block the rest of the wizard from advancing.
    """
    p = path or manifest_path()
    if not p.exists():
        return []

    entries: List[CustomModEntry] = []
    seen_keys: set[str] = set()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        log_warning(f"[custom_mods] cannot read {p}: {exc}")
        return []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            log_warning(
                f"[custom_mods] {p.name}:{lineno} ignored (need at "
                f"least <relative_path> <git_url>): {line!r}"
            )
            continue
        rel = parts[0].replace("\\", "/").strip("/")
        url = parts[1]
        ref = ""
        required_for: List[str] = []
        for token in parts[2:]:
            if "=" in token:
                ann_key, _, ann_val = token.partition("=")
                ann_key = ann_key.strip().lower()
                ann_val = ann_val.strip()
                if ann_key == "required_for":
                    required_for.extend(
                        s.strip() for s in ann_val.split(",") if s.strip()
                    )
                else:
                    log_warning(
                        f"[custom_mods] {p.name}:{lineno} unknown "
                        f"annotation {ann_key!r}; ignoring"
                    )
            elif not ref:
                ref = token
            else:
                log_warning(
                    f"[custom_mods] {p.name}:{lineno} extra positional "
                    f"token {token!r} after ref; ignoring"
                )
        key = Path(rel).name
        if not key:
            log_warning(
                f"[custom_mods] {p.name}:{lineno} ignored (empty key): {line!r}"
            )
            continue
        if key in seen_keys:
            log_warning(
                f"[custom_mods] {p.name}:{lineno} duplicate entry "
                f"{key!r}; keeping the first"
            )
            continue
        seen_keys.add(key)
        entries.append(
            CustomModEntry(
                key=key,
                relative_path=rel,
                url=url,
                ref=ref,
                required_for=tuple(required_for),
            )
        )
    return entries


def entry_already_installed(entry: CustomModEntry) -> bool:
    """Return True iff the entry's target dir already contains a clone.

    Probe is intentionally cheap: presence of ``.git`` (full clone) OR
    of ``HEAD`` (the bare-clone shape git uses when ``--depth`` was
    promoted to a treeless mirror). A non-empty directory without those
    markers is NOT treated as installed -- it may be a stale leftover
    from a previous failed clone, and the post-setup step will error
    loudly if it tries to clone over a non-empty target.
    """
    target = entry.target_dir
    return (target / ".git").exists() or (target / "HEAD").exists()
