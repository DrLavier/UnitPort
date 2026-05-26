# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.backends — 训练/仿真 后端清单 + 算法 + 场景 + 奖励预设.

⚠ 重命名说明 / Rename note:
    原 ``registers/engines/`` 目录与 ``application/engine/``（IR 运行时）名称冲突。
    本注册表的真实语义是 "可用 backends 清单"（Isaac/MuJoCo/SB3 等），
    因此重命名为 ``backends``。代码内运行时只剩单数 ``application/engine/``。

合并自原 engines/ 子目录（plan §11.6）/ Merged from former engines/ subdir:
- installed.json        → data/backends_installed.json   （registers/ 内唯一可写）
- algorithms.json       → data/backends_algorithms.json
- scenes.json           → data/backends_scenes.json
- rewards_presets.json  → data/backends_rewards.json
- refresh.py            → 内联 ``refresh_engine_availability()``
- training_items.py     → 占位（阶段 C 随 application/training/ 一并填）
- training_assets.py    → 占位（阶段 C 随 application/training/ 一并填）

用户级配置：``<USER_CONFIG_DIR>/engines/<engine>.json`` ← 各引擎 API 密钥/调参，
通过 SDK ``Storage.push_data`` 写入 USER_CONFIG_DIR。

API:
    load() -> int
    list_algorithms() / list_scenes()
    get_rewards_preset(name) / list_rewards_presets()
    get_installed(engine_id) / list_installed()
    list_available() / list_sim_backends()
    refresh_engine_availability()
    list_training_items() / list_training_assets()       # 阶段 C 填
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_debug, log_warning, read_data, save_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_ALGORITHMS = _DATA_DIR / "backends_algorithms.json"
_SCENES = _DATA_DIR / "backends_scenes.json"
_REWARDS = _DATA_DIR / "backends_rewards.json"
_INSTALLED = _DATA_DIR / "backends_installed.json"

_state: Dict[str, Any] = {
    "loaded": False,
    "algorithms": [],
    "scenes": [],
    "rewards_presets": {},
    "installed": {},
    "training_items": {},
    "training_assets": {},
}


def load() -> int:
    if _state["loaded"]:
        return len(_state["algorithms"])

    algos = read_data(_ALGORITHMS) or {}
    _state["algorithms"] = list(algos.get("algorithms", []) or [])

    scns = read_data(_SCENES) or {}
    _state["scenes"] = list(scns.get("scenes", []) or [])

    rwds = read_data(_REWARDS) or {}
    _state["rewards_presets"] = dict(rwds) if isinstance(rwds, dict) else {}

    inst = read_data(_INSTALLED) or {}
    _state["installed"] = dict(inst.get("engines", {}) or {})

    _state["loaded"] = True
    return len(_state["algorithms"])


def list_algorithms() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["algorithms"])


def list_scenes() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["scenes"])


def get_rewards_preset(name: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["rewards_presets"].get(name)


def list_rewards_presets() -> List[str]:
    if not _state["loaded"]:
        load()
    return [
        k for k in _state["rewards_presets"].keys()
        if not k.startswith("_") and k not in {"schema", "version"}
    ]


def get_installed(engine_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["installed"].get(engine_id)


def list_installed() -> Dict[str, Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return dict(_state["installed"])


# ---------------------------------------------------------------------------
# Typed BackendInfo accessor (Stage 2)
# Stage 1 ``application.training.backend.select_backend("auto")`` consults this
# to walk the preference chain. ``available`` reflects the last
# ``refresh_engine_availability()`` probe; ``enabled`` is a user pin from
# Settings UI (currently false by default — opt-in).
# ---------------------------------------------------------------------------


# Default human-facing labels for known engine IDs. Used as a fallback when a
# row in ``backends_installed.json`` has no ``display_name`` (e.g. freshly
# detected by ``refresh_engine_availability``). Existing user-set values are
# preserved on refresh.
_DEFAULT_DISPLAY_NAMES: Dict[str, str] = {
    "sb3": "SB3",
    "mujoco": "MuJoCo",
    "gymnasium": "Gymnasium",
    # ``sb3_mujoco`` is the composite training backend that bundles
    # SB3 + MuJoCo + Gymnasium — internally the id makes the dep chain
    # explicit, but to the user it IS the SB3 backend, so the
    # display label collapses to plain "SB3".
    "sb3_mujoco": "SB3",
    "isaac_lab": "IsaacLab",
}

# Theme-color slot per backend. Consumed by the Mission Control Files panel
# to color the [backend] badge of each canvas item. Slots resolve through
# ``Config.get_color(slot)`` against ``system.ini[Theme]`` — never put hex
# values here.
_DEFAULT_THEME_SLOTS: Dict[str, str] = {
    "isaac_lab": "theme_1",
    "sb3_mujoco": "theme_2",
    "sb3": "theme_2",
    "mujoco": "theme_2",
    "gymnasium": "theme_2",
}
_FALLBACK_THEME_SLOT = "theme_3"

# Canvas storage sub-directory per backend. Decoupled from
# ``_DEFAULT_DISPLAY_NAMES`` because the on-disk folder name is a stable
# storage contract (``projects/<p>/canvas/<canvas_subdir>/<file>``) while
# display_name is free to be a longer human label. Only the *training*
# backends that own canvases appear here — ``sb3``/``mujoco``/``gymnasium``
# rows in installed.json are component probes, not canvas owners, so they
# don't get their own canvas folder. The training backend that bundles them
# is ``sb3_mujoco``.
_CANVAS_SUBDIRS: Dict[str, str] = {
    "isaac_lab": "IsaacLab",
    "sb3_mujoco": "SB3",
}

# Canvas engine_id → per-node backend enum kind. The per-node enum used by
# rewards / terminations / domain_rand (and any future backend-keyed schema)
# is ``"sb3"`` / ``"isaac_lab"``; the canvas owner id is the full engine
# identifier (e.g. ``"sb3_mujoco"``). Used by NodeItem to derive the
# canvas-bound kind for ``conditional_on`` visibility, by spec_compiler
# when populating backend-keyed config slots, and by RegistryModuleEditor
# to pick the right registry. Unknown engine ids collapse to ``"sb3"`` so
# legacy / future canvases still resolve to a valid kind without branching.
_NODE_BACKEND_KIND: Dict[str, str] = {
    "isaac_lab": "isaac_lab",
    "sb3_mujoco": "sb3",
}

# In-tree training launcher script for each backend that needs one.
# These are project-internal framework files anchored against
# ``Paths.APP_ROOT`` (= ``<repo>/src/application``) — there is **no** user
# override and no user.ini key, because the launcher is part of the
# RELEASE source tree, not user configuration. If the project tree is
# missing the script, that's a framework-integrity bug, not a setup task
# the user should fix from a settings dialog.
#
# Path is stored as a tuple of path components so the joinpath() is
# OS-neutral. Future backends with launchers add a row here; backends
# without launchers (sim-only, dataset-only) simply have no row.
_TRAIN_LAUNCHERS: Dict[str, tuple] = {
    "isaac_lab": ("training", "isaac_lab", "launcher", "il_train_launcher.py"),
}


def train_launcher_path(engine_id: str) -> Optional[Path]:
    """Absolute path to the in-tree UnitPort training launcher for
    ``engine_id``, or ``None`` if the backend has no launcher concept.

    Pure function of where the source tree lives on disk (resolved via
    ``Paths.APP_ROOT``); no user customisation. Callers that need the
    file's existence to be a hard requirement should check
    ``returned_path.is_file()`` themselves and raise — this function
    deliberately doesn't validate disk presence so missing-launcher
    diagnostics can be emitted as typed ValidationIssue at the right
    layer (e.g. ``spec_validator._check_amp_wiring``).
    """
    rel = _TRAIN_LAUNCHERS.get(engine_id)
    if not rel:
        return None
    return Paths.APP_ROOT.joinpath(*rel)


# Sim-capable engines, in *preference order* (first is the default pick).
# Distinct from training-side classification: a "sim backend" is the physics
# engine that hosts a review/deploy episode. ``sb3``/``gymnasium`` are
# training-only component probes and never appear here; ``sb3_mujoco`` is
# a training canvas owner whose review path collapses onto plain ``mujoco``,
# so it's deliberately omitted to keep the sim-axis non-redundant.
# Consumed by :func:`list_sim_backends`.
_SIM_CAPABLE_ENGINES: tuple = ("mujoco", "isaac_lab")

# Sim engines whose availability can be probed cheaply in-process (just
# `importlib.util.find_spec`). Used by :func:`list_sim_backends` as a
# fallback when the engine isn't yet in ``backends_installed.json`` (e.g.
# the user hasn't run ``refresh_engine_availability`` after first install).
# Heavy probes (isaac_lab subprocess, 30 s timeout) deliberately stay out —
# trust the registry for those.
_SIM_LIGHT_PROBE_MODULES: Dict[str, tuple] = {
    "mujoco": ("mujoco",),
}


@dataclass(frozen=True)
class BackendInfo:
    """One row of ``backends_installed.json``."""

    id: str
    display_name: str
    available: bool
    enabled: bool
    version: str
    path: str


def _row_to_info(eid: str, row: Dict[str, Any]) -> BackendInfo:
    return BackendInfo(
        id=eid,
        display_name=str(row.get("display_name") or _DEFAULT_DISPLAY_NAMES.get(eid, eid)),
        available=bool(row.get("available", False)),
        enabled=bool(row.get("enabled", False)),
        version=str(row.get("version", "")),
        path=str(row.get("path", "")),
    )


def get_display_name(engine_id: str) -> str:
    """Return the human-facing label for ``engine_id``.

    Resolution order: row's ``display_name`` → ``_DEFAULT_DISPLAY_NAMES`` →
    the raw ``engine_id`` itself (so the UI never renders an empty string).
    """
    row = get_installed(engine_id) or {}
    name = str(row.get("display_name") or "").strip()
    if name:
        return name
    return _DEFAULT_DISPLAY_NAMES.get(engine_id, engine_id)


def get_theme_slot(engine_id: str) -> str:
    """Return the ``system.ini[Theme]`` slot name to use for ``engine_id``.

    Unknown ids fall back to :data:`_FALLBACK_THEME_SLOT` so callers can
    always paint *something* without branching on None.
    """
    return _DEFAULT_THEME_SLOTS.get(engine_id, _FALLBACK_THEME_SLOT)


def canvas_subdir(engine_id: str) -> str:
    """Return the canvas storage sub-directory name for ``engine_id``.

    Drives the storage layout ``<project>/canvas/<canvas_subdir>/*.canvas.json``.
    Distinct from :func:`get_display_name` (which is a free-form human
    label) because the folder name is a stable contract and must be 1:1
    with the engine id. Falls back to :func:`get_display_name` for engines
    not in :data:`_CANVAS_SUBDIRS` so unknown backends still resolve to
    *something* the disk can hold; future canvas-owning backends should
    add an explicit row to :data:`_CANVAS_SUBDIRS`.
    """
    fixed = _CANVAS_SUBDIRS.get(engine_id)
    if fixed:
        return fixed
    return get_display_name(engine_id)


def node_backend_kind(engine_id: Optional[str]) -> str:
    """Return the per-node backend enum kind for a canvas engine id.

    Canvas backends declare themselves via the engine id (e.g.
    ``"sb3_mujoco"``); node-side backend-keyed schemas use a coarser kind
    (``"sb3"`` / ``"isaac_lab"``). Any unknown / missing engine id collapses
    to ``"sb3"`` so legacy / future canvases still resolve to a valid kind.

    Used as the source of ``params["backend"]`` on rewards / terminations /
    domain_rand NodeItems — those nodes no longer expose a user-toggleable
    backend ParamSpec; the kind is derived from the parent CanvasPage's
    bound backend at spawn / load time.
    """
    if not isinstance(engine_id, str):
        return "sb3"
    return _NODE_BACKEND_KIND.get(engine_id.strip(), "sb3")


def list_canvas_owning_engines() -> List[str]:
    """Engine ids that own a canvas storage sub-directory.

    These are the engines a user can pick when creating a new canvas: the
    on-disk layout ``<project>/canvas/<canvas_subdir>/<file>`` is only
    well-defined for engines listed here. ``sb3``/``mujoco``/``gymnasium``
    rows in ``backends_installed.json`` are component probes (consumed by
    ``sb3_mujoco`` etc.) and are intentionally excluded.
    """
    return list(_CANVAS_SUBDIRS.keys())


def resolve_engine_id_from_subdir(subdir_name: str) -> Optional[str]:
    """Inverse of :func:`canvas_subdir`: ``"IsaacLab"`` → ``"isaac_lab"``.

    Resolution order: (1) :data:`_CANVAS_SUBDIRS` reverse map (the canonical
    1:1 contract), (2) live ``backends_installed`` ``display_name`` match,
    (3) :data:`_DEFAULT_DISPLAY_NAMES` reverse match. Returns ``None`` when
    no rule matches — caller decides whether to log / skip / raise.
    """
    if not subdir_name:
        return None
    for eid, sub in _CANVAS_SUBDIRS.items():
        if sub == subdir_name:
            return eid
    if not _state["loaded"]:
        load()
    for eid, row in _state["installed"].items():
        if not isinstance(row, dict):
            continue
        if str(row.get("display_name") or "").strip() == subdir_name:
            return eid
    for eid, dn in _DEFAULT_DISPLAY_NAMES.items():
        if dn == subdir_name:
            return eid
    return None


def set_display_name(engine_id: str, name: str) -> bool:
    """Persist ``display_name`` for ``engine_id`` to ``backends_installed.json``.

    Refuses to write when ``engine_id`` does not yet have an installed row —
    callers should run :func:`refresh_engine_availability` first so the row
    exists. Returns True on successful write.
    """
    if not _state["loaded"]:
        load()
    payload = read_data(_INSTALLED) or {}
    engines = dict(payload.get("engines", {}) or {})
    row = engines.get(engine_id)
    if not isinstance(row, dict):
        log_warning(
            f"[backends] set_display_name: unknown engine_id {engine_id!r}; "
            f"run refresh_engine_availability() first"
        )
        return False
    new_row = dict(row)
    new_row["display_name"] = str(name)
    engines[engine_id] = new_row
    payload["engines"] = engines
    save_data(_INSTALLED, payload)
    _state["installed"] = dict(engines)
    return True


def get_engine_info(engine_id: str) -> Optional[BackendInfo]:
    row = get_installed(engine_id)
    return _row_to_info(engine_id, row) if row else None


def is_available(engine_id: str) -> bool:
    """Cheap boolean query: is ``engine_id`` marked available in the installed table?

    Reads ``backends_installed.json::engines.<engine_id>.available`` — the
    per-installation truth maintained by :func:`refresh_engine_availability`.
    Returns ``False`` when the engine has no row, or its row lacks the
    ``available`` field, or the value is falsey.

    Used by :mod:`registers.review_backends` to keep the review picker's
    ``available`` flag in sync with the actual install state of the
    underlying engine (e.g. ``isaac_sim`` review backend → ``isaac_lab``
    engine row). Avoid duplicating ``get_installed(...).get('available')``
    everywhere — one helper, one source of truth.
    """
    row = get_installed(engine_id)
    if not isinstance(row, dict):
        return False
    return bool(row.get("available", False))


def list_available() -> List[BackendInfo]:
    """Return engines with ``available == True`` from the installed table.

    Order follows ``backends_installed.json`` insertion (dict order). Callers
    that need a preference order (e.g. ``select_backend("auto")``) should
    apply their own filter on top.
    """
    if not _state["loaded"]:
        load()
    return [
        _row_to_info(eid, row)
        for eid, row in _state["installed"].items()
        if isinstance(row, dict) and row.get("available")
    ]


def list_sim_backends() -> List[BackendInfo]:
    """Sim-capable engines that are available right now, in preference order.

    Source of truth for any UI / service that needs to pick a physics engine
    for a review or deploy episode (e.g. mission_panel's Policy Simulation
    card). Order follows :data:`_SIM_CAPABLE_ENGINES` — the first entry is
    the recommended default.

    Availability is computed by, in order:
      1. The ``backends_installed.json`` row for the engine.
      2. A cheap in-process probe via :data:`_SIM_LIGHT_PROBE_MODULES`
         (currently only ``mujoco`` — a hard ``requirements.txt`` dependency
         that may not yet have a row written if the user has not triggered
         :func:`refresh_engine_availability`).

    The probe result is **not** written back to ``backends_installed.json`` —
    that file is reserved for the user-triggered refresh path so this query
    stays read-only.
    """
    if not _state["loaded"]:
        load()
    out: List[BackendInfo] = []
    for eid in _SIM_CAPABLE_ENGINES:
        row = _state["installed"].get(eid)
        if isinstance(row, dict) and row.get("available"):
            out.append(_row_to_info(eid, row))
            continue
        det = _light_probe(eid)
        if not det["available"]:
            continue
        # Synthesize a row that looks like an installed entry but is in-memory
        # only; consumers cannot tell the difference, and we do not persist.
        synth = {
            "display_name": (row or {}).get("display_name")
                            or _DEFAULT_DISPLAY_NAMES.get(eid, eid),
            "available": True,
            "enabled": False,
            "version": det["version"],
            "path": det["path"],
        }
        out.append(_row_to_info(eid, synth))
    return out


def _light_probe(engine_id: str) -> Dict[str, Any]:
    """Cheap in-process availability check for a sim engine.

    Returns the same shape as :func:`_detect_module`. ``available=False`` for
    engines without a light probe entry (callers must rely on the installed
    table or a heavier explicit probe like :func:`_detect_isaac_lab`).
    """
    mods = _SIM_LIGHT_PROBE_MODULES.get(engine_id)
    if not mods:
        return {"available": False, "version": "", "path": ""}
    last_det: Dict[str, Any] = {"available": False, "version": "", "path": ""}
    for m in mods:
        det = _detect_module(m)
        if not det["available"]:
            return {"available": False, "version": "", "path": ""}
        last_det = det
    return last_det


def list_training_items() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["training_items"].values())


def list_training_assets() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["training_assets"].values())


# ---------------------------------------------------------------------------
# refresh_engine_availability — 由 Settings UI 主动触发，写回 backends_installed.json
# ---------------------------------------------------------------------------

def _detect_module(module_name: str) -> Dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "version": "", "path": ""}
    version = ""
    try:
        mod = importlib.import_module(module_name)
        version = str(getattr(mod, "__version__", ""))
    except Exception:  # noqa: BLE001
        pass
    return {
        "available": True,
        "version": version,
        "path": getattr(spec, "origin", "") or "",
    }


# ---------------------------------------------------------------------------
# Isaac Lab — separate venv, subprocess probe via registered root.
#
# Local-installation registration lives in this same ``backends_installed.json``
# (the per-installation runtime-writable SDK file). USER_CONFIG_DIR is reserved
# for cloud-server state only — never read here.
# ---------------------------------------------------------------------------

_ISAAC_PROBE_TIMEOUT_S = 30  # Isaac import is slow; first launch can take 20s+
_ISAAC_LAB_ENGINE_ID = "isaac_lab"


def _project_engine_root(engine_id: str) -> Path:
    """Default project-level install root: ``PROJECT_ROOT/Engines/<engine_id>/``.

    The ``Engines/`` convention is the documented install destination for
    heavy backends (Isaac Sim is ~30 GB) — mirrored in
    ``application.service.installers._paths.default_isaac_lab_install_root``.
    Duplicated rather than imported to keep ``registers/`` free of
    ``application/`` dependencies.
    """
    return Paths.PROJECT_ROOT / "Engines" / engine_id


def _has_isaac_markers(root: Path) -> bool:
    """True iff ``root`` looks like a valid Isaac Lab install skeleton.

    Markers match :meth:`EngineService.register_isaac_local` and the
    install_state probe in startup_tasks — single source of truth so
    auto-discovery and manual-register cannot disagree.
    """
    try:
        if not root.is_dir():
            return False
        has_launcher = (root / "isaaclab.sh").exists() or (root / "isaaclab.bat").exists()
        has_source = (root / "source").is_dir()
        return has_launcher and has_source
    except OSError:
        return False


def _resolve_isaac_local_root() -> Optional[Path]:
    """Resolve where Isaac Lab is installed on this machine.

    Resolution order:
      1. ``backends_installed.json::isaac_lab.local_root`` — explicit pin
         from the User panel's gear button, the in-app installer, or a
         prior auto-discovery.
      2. ``PROJECT_ROOT/Engines/isaac_lab/`` — the default install
         destination. If a user (or the install wizard) places Isaac Lab
         here, no manual registration step is needed.
      3. ``None`` — Isaac Lab is not on this machine.

    A registered root that no longer exists is logged loudly (rule §8 —
    no silent fallback) before we drop through to auto-discovery so the
    surface error is the broken pin, not a phantom auto-detect that
    masks it.
    """
    if not _state["loaded"]:
        load()
    row = _state["installed"].get(_ISAAC_LAB_ENGINE_ID) or {}
    declared = str(row.get("local_root") or "").strip()
    if declared:
        p = Path(declared).expanduser()
        if _has_isaac_markers(p):
            return p
        log_warning(
            f"[backends] registered isaac_lab root {p} no longer has markers "
            f"(isaaclab.sh|isaaclab.bat + source/); falling through to "
            f"auto-discovery under PROJECT_ROOT/Engines/"
        )
    auto = _project_engine_root(_ISAAC_LAB_ENGINE_ID)
    if _has_isaac_markers(auto):
        return auto
    return None


def _find_isaac_python(root: Path) -> Optional[str]:
    """Locate Isaac Lab's Python interpreter under ``root``.

    Mirrors DEMO ``isaac_lab_backend._find_isaac_python``:
      1. Isaac Sim kit (``_isaac_sim/python.bat|sh``) — official launcher path
      2. Local venv inside Isaac Lab root
      3. Active conda env (``CONDA_PREFIX``)
    """
    candidates = [
        root / "_isaac_sim" / "python.sh",
        root / "_isaac_sim" / "python.bat",
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "python.exe")
        candidates.append(Path(conda_prefix) / "bin" / "python")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _probe_isaac_root(root: Path) -> Dict[str, Any]:
    """Probe one *specific* Isaac Lab root via subprocess.

    Locates the Isaac Python via :func:`_find_isaac_python`, then runs
    ``python -c "import isaaclab; print(isaaclab.__version__)"``.

    Returns the same shape as :func:`_detect_module` — ``path`` carries the
    Isaac Lab root (not the .py file) since that is the actionable handle for
    callers (subprocess launchers, installer UI). Multi-version aware: callers
    that manage several registered installs probe each root through here.
    """
    python = _find_isaac_python(root)
    if python is None:
        log_warning(
            f"[backends] could not find Isaac Python under {root} "
            f"(checked _isaac_sim/, .venv/, CONDA_PREFIX)"
        )
        return {"available": False, "version": "", "path": str(root)}
    try:
        result = subprocess.run(
            [python, "-c", "import isaaclab; print(isaaclab.__version__)"],
            capture_output=True,
            text=True,
            timeout=_ISAAC_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        log_warning(
            f"[backends] isaac_lab probe timed out after "
            f"{_ISAAC_PROBE_TIMEOUT_S}s (python={python})"
        )
        return {"available": False, "version": "", "path": str(root)}
    except OSError as exc:
        log_warning(f"[backends] isaac_lab probe failed to spawn: {exc}")
        return {"available": False, "version": "", "path": str(root)}
    if result.returncode != 0:
        log_warning(
            f"[backends] isaac_lab probe non-zero exit "
            f"({result.returncode}): {result.stderr.strip()[:200]}"
        )
        return {"available": False, "version": "", "path": str(root)}
    version = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    return {"available": True, "version": version, "path": str(root)}


def _detect_isaac_lab() -> Dict[str, Any]:
    """Probe Isaac Lab via subprocess against the resolved installation.

    Resolves the install root via :func:`_resolve_isaac_local_root`
    (pinned row → ``PROJECT_ROOT/Engines/isaac_lab/`` auto-discovery), then
    delegates the version probe to :func:`_probe_isaac_root`. This is the
    single-root "is Isaac available at all" probe used by
    :func:`refresh_engine_availability`; the multi-version management surface
    probes each registered root individually via
    :func:`refresh_isaac_installations`.
    """
    root = _resolve_isaac_local_root()
    if root is None:
        return {"available": False, "version": "", "path": ""}
    return _probe_isaac_root(root)


def refresh_engine_availability() -> Dict[str, Dict[str, Any]]:
    """扫描本机引擎可用性并写入 backends_installed.json.

    sb3 走主进程 importlib 探针（与主进程同栈）；isaac_lab 走独立 venv
    子进程探针，根来源由 :func:`_resolve_isaac_local_root` 解析（先读本表
    ``local_root`` 字段，再回退到 ``PROJECT_ROOT/Engines/isaac_lab/`` 自动发现）。
    任何依赖 USER_CONFIG_DIR 的引擎注册都已迁出 —— 本地安装是"每台机器一份
    物理事实"，归 SDK 树管理。

    自动发现命中时把 ``local_root`` / ``registered=True`` / ``source="auto"``
    一并落盘，下次 Settings UI 打开就能直接看到，不再需要用户去 gear 按钮
    手动 locate。

    registers/ 内**唯一**允许运行时写入。

    Returns:
        {engine_id: {available, enabled, version, path, local_root,
                      registered, source, display_name}}
    """
    payload = read_data(_INSTALLED) or {}
    engines = dict(payload.get("engines", {}))

    # SB3+MuJoCo backend (Stage 1 placeholder, Stage 10 implementation) needs
    # all three deps; report ``sb3_mujoco`` as available only if the trio is
    # present. Individual ``sb3`` row is kept for backwards compat (Settings
    # UI / DEMO IL launcher both query it directly).
    sb3 = _detect_module("stable_baselines3")
    mujoco = _detect_module("mujoco")
    gymnasium = _detect_module("gymnasium")
    sb3_mujoco_available = sb3["available"] and mujoco["available"] and gymnasium["available"]

    detection = {
        "sb3": sb3,
        "mujoco": mujoco,
        "gymnasium": gymnasium,
        "sb3_mujoco": {
            "available": sb3_mujoco_available,
            "version": sb3["version"] if sb3_mujoco_available else "",
            "path": sb3["path"] if sb3_mujoco_available else "",
        },
        "isaac_lab": _detect_isaac_lab(),
    }

    for eid, det in detection.items():
        prev = engines.get(eid, {}) or {}
        row = {
            "display_name": prev.get("display_name") or _DEFAULT_DISPLAY_NAMES.get(eid, eid),
            "available": det["available"],
            "enabled": prev.get("enabled") if det["available"] else False,
            "version": det["version"],
            "path": det["path"],
            # Local-registration fields — preserved across refreshes.
            # ``local_root`` only exists for engines whose install root is
            # external to the .venv (currently isaac_lab); pip-installed
            # backends carry the .py path in ``path`` and have an empty
            # ``local_root``.
            "local_root": str(prev.get("local_root") or ""),
            "registered": bool(prev.get("registered", False)),
            "source": str(prev.get("source") or ""),
        }
        # Isaac Lab auto-discovery write-back: when the probe used the
        # ``Engines/isaac_lab/`` default and there's no prior pin, persist it
        # so the User panel / installer report "already registered" without
        # the user ever clicking the gear button.
        if eid == _ISAAC_LAB_ENGINE_ID and det["available"] and det.get("path"):
            if not row["local_root"]:
                row["local_root"] = str(det["path"])
                row["registered"] = True
                row["source"] = row["source"] or "auto"
        # Multi-version: keep the project-owned base in the installations list
        # (cheap — reuses the single probe just run, no extra subprocess) so the
        # User-panel management surface and the train-target dropdowns see it
        # without a separate refresh. External pins are probed lazily by
        # refresh_isaac_installations(); we only touch the base here.
        if eid == _ISAAC_LAB_ENGINE_ID:
            base = _project_engine_root(_ISAAC_LAB_ENGINE_ID)
            prev_list = prev.get("installations")
            items = (
                [dict(x) for x in prev_list if isinstance(x, dict)]
                if isinstance(prev_list, list) else []
            )
            if _has_isaac_markers(base):
                base_avail = det["available"] and _norm_root(
                    det.get("path", "")) == _norm_root(str(base))
                hit = next(
                    (x for x in items
                     if _norm_root(x.get("root", "")) == _norm_root(str(base))),
                    None,
                )
                if hit is None:
                    items.insert(0, {
                        "root": str(base),
                        "version": det["version"] if base_avail else "",
                        "source": "auto",
                        "available": base_avail,
                    })
                else:
                    hit["available"] = base_avail
                    if base_avail and det["version"]:
                        hit["version"] = det["version"]
            row["installations"] = items
        engines[eid] = row
        if not det["available"]:
            log_warning(f"[backends] {eid} 未安装")
        else:
            log_debug(f"[backends] {eid} 可用：{det['version']}")

    payload["engines"] = engines
    save_data(_INSTALLED, payload)
    # Refresh in-memory snapshot
    _state["installed"] = dict(engines)
    return engines


# ---------------------------------------------------------------------------
# Local-installation setters / getters — owned here because
# ``backends_installed.json`` is the per-installation source of truth.
# EngineService delegates to these; USER_CONFIG_DIR holds cloud state only.
# ---------------------------------------------------------------------------


def get_local_registration(engine_id: str) -> Dict[str, Any]:
    """Return the local-installation block for ``engine_id`` as a plain dict.

    Shape (always all keys present, falsy when unset):

        {"root": "<absolute path or ''>",
         "registered": bool,
         "enabled": bool,
         "source": "<manual|install|import_from_demo|boot_repair|auto|''>"}

    Reads from :data:`_INSTALLED` (``backends_installed.json``) — never
    from USER_CONFIG_DIR. ``root`` is the live ``local_root`` field; the
    sibling ``path`` field of the same row carries the same value once
    detection has run (kept distinct so future probes can write ``path``
    without clobbering an explicit user pin).
    """
    if not _state["loaded"]:
        load()
    row = _state["installed"].get(engine_id) or {}
    return {
        "root": str(row.get("local_root") or ""),
        "registered": bool(row.get("registered", False)),
        "enabled": bool(row.get("enabled", False)),
        "source": str(row.get("source") or ""),
    }


def set_local_registration(
    engine_id: str,
    *,
    root: Optional[str] = None,
    registered: Optional[bool] = None,
    enabled: Optional[bool] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Patch the local-installation block of ``engine_id`` in
    ``backends_installed.json`` and return the post-write row.

    Any field passed as ``None`` is left untouched; pass an empty string
    to clear ``root`` / ``source``. The row is created (with default
    ``display_name`` / empty probe fields) when ``engine_id`` has no row
    yet — useful for first-time registration before
    :func:`refresh_engine_availability` has ever run.

    Callers (EngineService.register_isaac_local, installer success path,
    UserPanel gear button) should always go through this rather than
    poking the JSON directly.
    """
    if not _state["loaded"]:
        load()
    payload = read_data(_INSTALLED) or {}
    engines = dict(payload.get("engines", {}) or {})
    row = dict(engines.get(engine_id) or {})
    row.setdefault("display_name", _DEFAULT_DISPLAY_NAMES.get(engine_id, engine_id))
    row.setdefault("available", False)
    row.setdefault("enabled", False)
    row.setdefault("version", "")
    row.setdefault("path", "")
    row.setdefault("local_root", "")
    row.setdefault("registered", False)
    row.setdefault("source", "")
    if root is not None:
        row["local_root"] = str(root)
    if registered is not None:
        row["registered"] = bool(registered)
    if enabled is not None:
        row["enabled"] = bool(enabled)
    if source is not None:
        row["source"] = str(source)
    engines[engine_id] = row
    payload["engines"] = engines
    save_data(_INSTALLED, payload)
    _state["installed"] = dict(engines)
    return dict(row)


# ---------------------------------------------------------------------------
# Multi-version Isaac Lab installation registry.
#
# A machine may carry several Isaac Lab installs: the project-owned "base"
# under ``PROJECT_ROOT/Engines/isaac_lab/`` (auto-installed / auto-discovered)
# plus any number of external roots the user redirected to. All of them are a
# per-installation physical fact, so they live in ``backends_installed.json``
# under ``engines.isaac_lab.installations`` (a list), NOT in USER_CONFIG_DIR.
#
# Each entry: {"root", "version", "source", "available"}. ``is_base`` / ``drive``
# / ``exists`` are derived at read time (not persisted) so a renamed project
# root or a moved drive can't leave a stale flag behind.
#
# The legacy single-pin fields (``local_root`` / ``registered`` / ``version``)
# are kept in sync with the primary (base, else first) entry so existing
# readers — ``_resolve_isaac_local_root``, ``get_local_registration``,
# ``status`` — keep working unchanged.
# ---------------------------------------------------------------------------


def _norm_root(root: str) -> str:
    """Normalise an install root for dedup/equality comparison.

    Case-folded, resolved, trailing-separator stripped. Used so
    ``D:\\IsaacLab`` and ``d:/IsaacLab/`` register as the same install.
    """
    s = str(root or "").strip()
    if not s:
        return ""
    try:
        s = str(Path(s).expanduser().resolve())
    except OSError:
        pass
    return s.rstrip("\\/").lower()


def _root_drive(root: str) -> str:
    """Return the drive anchor of ``root`` (``"A:"`` on Windows, ``"/"`` POSIX)."""
    s = str(root or "")
    if not s:
        return ""
    drive = os.path.splitdrive(s)[0]
    if drive:
        return drive
    # POSIX has no drive letter — surface the mount root instead.
    return "/" if s.startswith("/") else ""


def install_roots_equal(a: str, b: str) -> bool:
    """True iff two install roots resolve to the same location (case/sep-insensitive)."""
    na, nb = _norm_root(a), _norm_root(b)
    return bool(na) and na == nb


def is_base_isaac_root(root: str) -> bool:
    """True iff ``root`` is the project-owned base install under ``Engines/``."""
    base = _project_engine_root(_ISAAC_LAB_ENGINE_ID)
    return bool(root) and _norm_root(root) == _norm_root(str(base))


def _read_isaac_installations_raw() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    row = _state["installed"].get(_ISAAC_LAB_ENGINE_ID) or {}
    raw = row.get("installations")
    if isinstance(raw, list):
        return [dict(e) for e in raw if isinstance(e, dict)]
    # Legacy single-pin → synthesise a one-entry list so first read after an
    # upgrade still shows the previously-registered install (rule §8(c)).
    lr = str(row.get("local_root") or "").strip()
    if lr:
        return [{
            "root": lr,
            "version": str(row.get("version") or ""),
            "source": str(row.get("source") or "manual"),
            "available": bool(row.get("available", False)),
        }]
    return []


def _pick_primary_installation(
    items: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Base entry if present, else the first — drives the legacy single pin."""
    for e in items:
        if is_base_isaac_root(str(e.get("root") or "")):
            return e
    return items[0] if items else None


def _write_isaac_installations(items: List[Dict[str, Any]]) -> None:
    """Persist the installations list and keep legacy single-pin fields synced."""
    payload = read_data(_INSTALLED) or {}
    engines = dict(payload.get("engines", {}) or {})
    row = dict(engines.get(_ISAAC_LAB_ENGINE_ID) or {})
    row.setdefault(
        "display_name",
        _DEFAULT_DISPLAY_NAMES.get(_ISAAC_LAB_ENGINE_ID, _ISAAC_LAB_ENGINE_ID),
    )
    clean = [
        {
            "root": str(e.get("root") or ""),
            "version": str(e.get("version") or ""),
            "source": str(e.get("source") or "manual"),
            "available": bool(e.get("available", False)),
        }
        for e in items
        if str(e.get("root") or "").strip()
    ]
    row["installations"] = clean
    primary = _pick_primary_installation(clean)
    if primary is not None:
        row["local_root"] = primary["root"]
        row["registered"] = True
        if primary.get("version"):
            row["version"] = primary["version"]
        row["source"] = primary.get("source") or row.get("source") or "manual"
    else:
        row["local_root"] = ""
        row["registered"] = False
    engines[_ISAAC_LAB_ENGINE_ID] = row
    payload["engines"] = engines
    save_data(_INSTALLED, payload)
    _state["installed"] = dict(engines)


def list_isaac_installations(
    *, ensure_base: bool = True
) -> List[Dict[str, Any]]:
    """Return registered Isaac Lab installs, newest-API decorated.

    Each item: ``{"root", "version", "source", "available", "is_base",
    "drive", "exists"}``. ``ensure_base=True`` injects the auto-discovered
    project base (``Engines/isaac_lab/``) at the front when it has valid
    markers but is not yet persisted — so the management UI shows it without a
    prior refresh. Read-only: never persists (use
    :func:`refresh_isaac_installations` for that).
    """
    items = _read_isaac_installations_raw()
    if ensure_base:
        base = _project_engine_root(_ISAAC_LAB_ENGINE_ID)
        if _has_isaac_markers(base) and not any(
            _norm_root(e.get("root", "")) == _norm_root(str(base)) for e in items
        ):
            items.insert(
                0, {"root": str(base), "version": "", "source": "auto",
                    "available": False}
            )
    out: List[Dict[str, Any]] = []
    for e in items:
        root = str(e.get("root") or "")
        if not root:
            continue
        exists = _has_isaac_markers(Path(root))
        out.append({
            "root": root,
            "version": str(e.get("version") or ""),
            "source": str(e.get("source") or "manual"),
            "available": bool(e.get("available", False)),
            "is_base": is_base_isaac_root(root),
            "drive": _root_drive(root),
            "exists": exists,
        })
    return out


def is_isaac_root_registered(root: str) -> bool:
    """True iff ``root`` (normalised) is already in the installations list."""
    nr = _norm_root(root)
    if not nr:
        return False
    return any(
        _norm_root(e.get("root", "")) == nr
        for e in list_isaac_installations(ensure_base=True)
    )


def register_isaac_installation(
    root: str,
    *,
    version: str = "",
    source: str = "manual",
    available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Add (or update) an Isaac Lab install in the multi-version list.

    Dedups on the normalised root: re-registering an existing root patches its
    cached ``version`` / ``source`` / ``available`` instead of duplicating.
    Returns the persisted (raw) entry. Caller is responsible for validating
    markers (see :meth:`EngineService.register_isaac_local`).
    """
    items = _read_isaac_installations_raw()
    nr = _norm_root(root)
    for e in items:
        if _norm_root(e.get("root", "")) == nr:
            if version:
                e["version"] = version
            if source:
                e["source"] = source
            if available is not None:
                e["available"] = available
            _write_isaac_installations(items)
            return dict(e)
    try:
        resolved = str(Path(root).expanduser().resolve())
    except OSError:
        resolved = str(root)
    entry = {
        "root": resolved,
        "version": version,
        "source": source,
        "available": bool(available) if available is not None else False,
    }
    items.append(entry)
    _write_isaac_installations(items)
    return dict(entry)


def unregister_isaac_installation(root: str) -> bool:
    """Remove an install from the list. Returns True iff something was removed.

    Does NOT touch the install on disk — physical removal (base uninstall) is
    the caller's job. Removing the base entry is legal; it reappears via
    ``ensure_base`` on next read if the directory still has markers, so a true
    uninstall must delete the directory first.
    """
    items = _read_isaac_installations_raw()
    nr = _norm_root(root)
    kept = [e for e in items if _norm_root(e.get("root", "")) != nr]
    if len(kept) == len(items):
        return False
    _write_isaac_installations(kept)
    return True


def refresh_isaac_installations() -> List[Dict[str, Any]]:
    """Re-probe every registered Isaac Lab root and persist cached version/avail.

    One subprocess per install (each up to ``_ISAAC_PROBE_TIMEOUT_S``) — call
    from the management dialog / an explicit user refresh, NOT on the startup
    hot path (:func:`refresh_engine_availability` probes only the primary root).
    Drops entries whose directory no longer has Isaac markers and is not the
    base (a vanished external pin), logging each drop loudly (rule §8).
    """
    decorated = list_isaac_installations(ensure_base=True)
    persisted: List[Dict[str, Any]] = []
    for e in decorated:
        root = e["root"]
        if e["exists"]:
            det = _probe_isaac_root(Path(root))
            persisted.append({
                "root": root,
                "version": det["version"] or e.get("version", ""),
                "source": e.get("source") or "manual",
                "available": det["available"],
            })
        elif e["is_base"]:
            # Base directory missing markers — keep the slot but mark unavailable
            # so the UI can offer "download" rather than silently dropping it.
            persisted.append({
                "root": root, "version": "", "source": "auto", "available": False,
            })
        else:
            log_warning(
                f"[backends] dropping isaac_lab install {root!r}: directory no "
                f"longer has Isaac markers (isaaclab.sh|bat + source/)"
            )
    _write_isaac_installations(persisted)
    return list_isaac_installations(ensure_base=True)


__all__ = [
    "load",
    "list_algorithms",
    "list_scenes",
    "get_rewards_preset",
    "list_rewards_presets",
    "get_installed",
    "list_installed",
    "list_training_items",
    "list_training_assets",
    "refresh_engine_availability",
    "get_local_registration",
    "set_local_registration",
    "is_base_isaac_root",
    "install_roots_equal",
    "list_isaac_installations",
    "is_isaac_root_registered",
    "register_isaac_installation",
    "unregister_isaac_installation",
    "refresh_isaac_installations",
    "BackendInfo",
    "get_engine_info",
    "is_available",
    "list_available",
    "get_display_name",
    "set_display_name",
    "get_theme_slot",
    "canvas_subdir",
    "resolve_engine_id_from_subdir",
    "list_canvas_owning_engines",
    "train_launcher_path",
]
