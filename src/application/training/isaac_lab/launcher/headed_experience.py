# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Generate a minimal viewport-only Isaac Sim experience for headed training.

Headed Isaac Lab training defaults to ``isaaclab.python.kit``, which loads the
full Kit GUI — menu bars, content browser, property/console/stage windows. On
the reported setup that GUI crashes during construction
(``omni.kit.menu.utils._build_menu`` → Windows access violation), and the full
GUI also maps far more buffers through the GPU's scarce PCIe BAR1 aperture.

This module derives a stripped experience from the **selected install's own**
``apps/isaaclab.python.kit`` (so extension names/versions always match that
install — never a static shipped file that drifts across Isaac Sim versions)
by removing the crashing menu/window dependencies, and writes it back into the
same ``apps/`` directory. It MUST live there: the stock kit's
``[settings.app.exts] folders`` use ``${app}/../source`` (``${app}`` = the .kit
file's own directory) to locate the Isaac Lab extensions — a file written
anywhere else would fail to resolve them.

Pure stdlib only (``re`` / ``os`` / ``pathlib``): this runs inside the Isaac
launcher subprocess, which does not initialise the UnitPort SDK.

Fail loud (CLAUDE.md §8): a missing/unreadable source kit, an unwritable
``apps/`` dir, or a post-transform sanity-check failure all raise — we never
silently fall back to the crashing default experience.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set

# Crash-critical (always dropped): the full-GUI menu + window + viewport-menubar
# extensions. ``omni.kit.menu.utils._build_menu`` is the observed access
# violation; the menubars depend on the menu system, the windows are pure GUI.
HEADED_DROP_DEPS: Set[str] = {
    "isaacsim.gui.menu",
    "omni.kit.menu.common",
    "omni.kit.menu.create",
    "omni.kit.menu.stage",
    "omni.kit.menu.utils",
    "omni.kit.window.console",
    "omni.kit.window.content_browser",
    "omni.kit.window.property",
    "omni.kit.window.script_editor",
    "omni.kit.window.stage",
    "omni.kit.window.status_bar",
    "omni.kit.window.toolbar",
    "omni.kit.viewport.menubar.camera",
    "omni.kit.viewport.menubar.display",
    "omni.kit.viewport.menubar.lighting",
    "omni.kit.viewport.menubar.render",
    "omni.kit.viewport.menubar.settings",
}

# Pure UI / editor extensions that are safe-ish to drop for an even leaner
# viewport, but NOT dropped by default — some (omni.graph.*) are runtime deps
# of replicator/sensors and removing them risks a Kit dependency-solver error.
# Exposed for a future "extra strip" UI toggle (pass via ``extra_drop``); the
# default transform only removes the proven crash source above.
OPTIONAL_DROP_DEPS: Set[str] = {
    "isaacsim.asset.browser",
    "omni.kit.material.library",
    "omni.kit.tool.asset_importer",
    "omni.kit.tool.collect",
    "omni.graph.action",
    "omni.graph.core",
    "omni.graph.nodes",
    "omni.graph.scriptnode",
    "omni.graph.ui_nodes",
    "omni.anim.curve.core",
    "omni.kit.telemetry",
}

# Must survive the transform — viewport rendering + physics + Isaac core/lab.
# Asserted present afterwards so a bad drop-list fails loud, not at Kit boot.
MUST_KEEP_DEPS: Sequence[str] = (
    "omni.hydra.rtx",
    "omni.hydra.engine.stats",
    "omni.kit.viewport.window",
    "omni.kit.viewport.legacy_gizmos",
    "omni.kit.viewport.scene_camera_model",
    "omni.rtx.settings.core",
    "omni.kit.mainwindow",
    "omni.kit.manipulator.camera",
    "isaacsim.simulation_app",
    "isaacsim.core.api",
    "omni.physx.bundle",
    "omni.physx.tensors",
    "omni.warp.core",
    "isaaclab",
)

OUTPUT_NAME = "unitport.exp.train_viewport.kit"
_SOURCE_NAME = "isaaclab.python.kit"
_MENU_SETTINGS_HEADER = '[settings.exts."omni.kit.menu.utils"]'

_DEP_KEY_RE = re.compile(r'\s*"([^"]+)"\s*=')


def _dep_key(line: str) -> str:
    """Return the quoted dependency key on a ``"<key>" = {...}`` line, else ""."""
    m = _DEP_KEY_RE.match(line)
    return m.group(1) if m else ""


def _is_section_header(stripped: str) -> bool:
    return stripped.startswith("[") and stripped.endswith("]")


def _transform(text: str, drop: Set[str]) -> str:
    """Drop ``drop`` deps from ``[dependencies]`` and the menu.utils settings
    block; retitle the package. Section-aware so settings lines that merely
    mention a dropped extension's name are left untouched."""
    out: List[str] = []
    in_deps = False
    skipping_menu_block = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if _is_section_header(stripped):
            skipping_menu_block = stripped == _MENU_SETTINGS_HEADER
            in_deps = stripped == "[dependencies]"
            if skipping_menu_block:
                continue  # drop the block header too
            out.append(line)
            continue
        if skipping_menu_block:
            continue  # drop block body until the next section header
        if in_deps and _dep_key(line) in drop:
            continue
        out.append(line)
    result = "".join(out)
    # Retitle for log clarity (exact [package] title only; the
    # [settings.app.window] title = "Isaac Sim" line has a different value).
    result = result.replace(
        'title = "Isaac Lab Python"', 'title = "UnitPort Train Viewport"', 1
    )
    return result


def ensure_headed_experience(
    isaac_root: Path,
    *,
    extra_drop: Iterable[str] = (),
    keep: Iterable[str] = (),
    extra_settings_lines: Iterable[str] = (),
) -> str:
    """Generate/refresh the minimal headed experience in ``<isaac_root>/apps/``.

    Reads that install's ``isaaclab.python.kit``, strips the crash-causing
    GUI menu/window dependencies (plus any ``extra_drop``, minus any ``keep``),
    optionally appends ``extra_settings_lines``, and writes
    ``unitport.exp.train_viewport.kit`` atomically next to the source. Returns
    the relative file name (AppLauncher resolves it under the install's apps/).

    Idempotent — safe to call every headed launch; a reinstall that clobbers
    ``apps/`` self-heals on the next run. Raises ``RuntimeError`` on any failure
    (§8) rather than letting the crashing default experience load.
    """
    apps_dir = Path(isaac_root) / "apps"
    source = apps_dir / _SOURCE_NAME
    if not source.is_file():
        raise RuntimeError(
            f"[headed_experience] Isaac Lab experience not found: {source}. "
            f"The install at {isaac_root} looks incomplete/corrupt — re-run the "
            f"Isaac Lab installer."
        )
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"[headed_experience] cannot read {source}: {exc}"
        ) from exc

    drop = (HEADED_DROP_DEPS | set(extra_drop)) - set(keep)
    result = _transform(text, drop)

    extra = [ln if ln.endswith("\n") else ln + "\n" for ln in extra_settings_lines]
    if extra:
        if not result.endswith("\n"):
            result += "\n"
        result += "\n# UnitPort extra settings\n" + "".join(extra)

    # Post-transform sanity (§8): every must-keep dependency still present and
    # the file actually shrank — guards against a future bad drop-list.
    present = {_dep_key(ln) for ln in result.splitlines() if _dep_key(ln)}
    missing = [k for k in MUST_KEEP_DEPS if k not in present]
    if missing:
        raise RuntimeError(
            f"[headed_experience] transform dropped required extension(s) "
            f"{missing} — drop-list is wrong; refusing to write a broken "
            f"experience for {source}."
        )
    if len(result) >= len(text):
        raise RuntimeError(
            f"[headed_experience] transform removed nothing from {source} "
            f"(expected to strip GUI extensions) — refusing to proceed."
        )

    out_path = apps_dir / OUTPUT_NAME
    tmp_path = apps_dir / (OUTPUT_NAME + ".tmp")
    try:
        tmp_path.write_text(result, encoding="utf-8")
        os.replace(tmp_path, out_path)
    except OSError as exc:
        raise RuntimeError(
            f"[headed_experience] cannot write {out_path}: {exc}. The Isaac Lab "
            f"apps/ directory must be writable (the experience must live here so "
            f"its ${{app}}/../source extension paths resolve)."
        ) from exc

    return OUTPUT_NAME
