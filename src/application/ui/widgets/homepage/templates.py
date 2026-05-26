# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Homepage templates — backend-scoped scan for the "Create" card.

Scan root: ``Paths.PROJECT_ROOT / "custom_mods" / "canvas" / <backend_id> /
*.canvas.json``. The backend subdirectory name is the registered engine
id (``"isaac_lab"`` / ``"sb3_mujoco"`` / ...), so the create card can
filter templates by the user's currently-picked backend without reading
each JSON to verify its ``backend`` field.

Synthetic entries:
    * Exactly one ``<Blank>`` entry per backend — represents an empty
      canvas with no nodes/edges. ``template_path`` is empty for this
      entry; the host treats that as "use the default seed".

The filename stem (``[IssacLab] Go2_PPO`` → "Go2_PPO" + subtitle
"IssacLab · Go2 · PPO") is preserved for the file-template title — the
``[engine_tag]`` prefix is decorative, the authoritative engine id is
the subdir name.

Convention: callers re-invoke :func:`list_templates_for_backend` every
time the backend selector flips or the user clicks "refresh" — there is
no cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from unitport_sdk import Paths, log_warning


_FNAME_RE = re.compile(
    r"^\[(?P<engine_tag>[^\]]+)\]\s+(?P<title>.+?)\.canvas\.json$"
)


@dataclass(frozen=True)
class Template:
    template_path: str   # absolute .canvas.json path; "" means blank canvas
    title: str           # display name (stem-derived, or "<Blank>")
    subtitle: str        # "<engine_tag> · <robot> · <algo>" / "Empty canvas"
    engine_id: str       # canvas backend id, e.g. "isaac_lab" / "sb3_mujoco"
    icon_name: str       # Assets.find_icon stem; "" => no icon
    is_blank: bool       # True when template_path == ""


def _templates_root() -> Path:
    return Paths.CUSTOM_MODS_DIR / "canvas"


def _backend_dir(backend_id: str) -> Path:
    return _templates_root() / backend_id


def _parse_filename(fname: str) -> tuple[str, str]:
    """``"[IssacLab] Go2_PPO.canvas.json"`` → ``("Go2_PPO", "IssacLab · Go2 · PPO")``.

    Falls back to stem + empty subtitle if the filename doesn't match the
    ``[engine] body`` pattern.
    """
    m = _FNAME_RE.match(fname)
    if not m:
        stem = fname[: -len(".canvas.json")] if fname.endswith(
            ".canvas.json"
        ) else fname
        return stem, ""
    engine_tag = m.group("engine_tag").strip()
    body = m.group("title").strip()
    if "_" in body:
        robot, _, algo = body.partition("_")
        subtitle = f"{engine_tag} · {robot} · {algo}"
    else:
        subtitle = engine_tag
    return body, subtitle


def _blank(backend_id: str) -> Template:
    """Synthetic blank template for the given backend — shown as 'Free Build'."""
    return Template(
        template_path="",
        title="Free Build",
        subtitle="Empty canvas",
        engine_id=backend_id,
        icon_name="icon_node",
        is_blank=True,
    )


def list_templates_for_backend(backend_id: str) -> List[Template]:
    """Return ``[<Blank>] + file templates`` for ``backend_id``.

    The blank entry is always first so the list is never empty. File
    templates are scanned from ``custom_mods/canvas/<backend_id>/`` —
    missing directory yields just the blank entry.
    """
    out: List[Template] = [_blank(backend_id)]
    if not backend_id:
        return out
    root = _backend_dir(backend_id)
    if not root.exists() or not root.is_dir():
        return out
    try:
        files = sorted(root.glob("*.canvas.json"))
    except OSError as exc:                                    # noqa: BLE001
        log_warning(f"[homepage.templates] glob failed {root}: {exc!r}")
        return out
    for p in files:
        title, subtitle = _parse_filename(p.name)
        out.append(Template(
            template_path=str(p),
            title=title,
            subtitle=subtitle,
            engine_id=backend_id,
            icon_name="note",
            is_blank=False,
        ))
    return out


def find_template(template_id: str) -> Optional[Template]:
    """Lookup by ``template_path`` (or ``"blank:<engine_id>"`` for synthetics).

    Walks every backend subdirectory because the caller may not know the
    backend any more (e.g. resolving a serialised selection).
    """
    if not template_id:
        return None
    root = _templates_root()
    if not root.exists():
        return None
    for backend_dir in root.iterdir():
        if not backend_dir.is_dir():
            continue
        backend_id = backend_dir.name
        for t in list_templates_for_backend(backend_id):
            key = t.template_path or f"blank:{t.engine_id}"
            if key == template_id:
                return t
    return None
