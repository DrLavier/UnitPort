# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Motion Library — manages reference motion files.

Hot-pluggable assets are scanned from **two roots, always merged**:

1. **Project-shipped (read-only):** ``<PROJECT_ROOT>/custom_mods/motions/``.
   Built-in + community packs that ship with the install (e.g.
   ``AMP_for_hardware``, ``MetalHead``, ``rl_amp``, ``unitport_builtin``).
2. **User-installed (writable):** ``<USER_CONFIG_DIR>/scripts/training_motion/motions/``.
   User's own imports, sits under ``~/UnitPort/`` so it survives redeploy.

Directory layout (identical under both roots) — SB3 ``.npy`` clips and
community AMP packs co-exist under the same ``motions/`` root and are
disambiguated by their subtree shape:

    motions/
        # SB3 library side (category-rooted)
        quadruped/<model>/*.npy
        biped/<model>/*.npy
        wheeled/<model>/*.npy
        manipulator/<model>/*.npy
        generic/<model>/*.npy
        # Community AMP packs (package-rooted)
        <package>/datasets/<subdir>/*.{txt,json}

The legacy ``custom_mods/archives/`` directory was consolidated into
``custom_mods/motions/`` — every hot-plug reference-motion repo deploys
under ``motions/`` (catalog: ``installers/custom_mods_manifest.py``).

Files are browsed by category (derived from robot family) so that a Go2
user sees all quadruped motions, with their own model's files pinned first.
Community AMP archives surface as :class:`MotionPack` 📦 entries in the
picker.

Lookup contract: when the same path tail exists under both roots, the
**first hit wins** in scan order ``[custom_mods, user_config]`` — built-in
packs are canonical; users add new packs by depositing into either root.
Writes (``import_file``) always go to the user-config side.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from scripts.training_motion._robot_family import (
    ROBOT_FAMILY_BY_TYPE,
    normalize_robot_type,
)
from scripts.training_motion.labels import infer_task_tag


# ---------------------------------------------------------------------------
# Paths — dual root: project custom_mods/ + per-user USER_CONFIG_DIR
# (RELEASE CLAUDE.md §1.4 keeps user writes under USER_CONFIG_DIR; project
# ships built-in packs under custom_mods/ and must remain discoverable.)
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root, lazy to avoid SDK import at module load."""
    from unitport_sdk import Paths
    return Path(Paths.PROJECT_ROOT)


def _custom_mods_root() -> Path:
    """Return ``Paths.CUSTOM_MODS_DIR`` (overlay-resolved third-party assets root)."""
    from unitport_sdk import Paths
    return Path(Paths.CUSTOM_MODS_DIR)


def _user_config_root() -> Path:
    """Return the user-state root, lazy to avoid SDK import at module load."""
    from unitport_sdk import Paths
    return Path(Paths.USER_CONFIG_DIR)


def _project_motions_dir() -> Path:
    """Project-shipped motion library root (read-only at runtime)."""
    return _custom_mods_root() / "motions"


def _user_motions_dir() -> Path:
    """User-installed motion library root (writable; target of import_file)."""
    return _user_config_root() / "scripts" / "training_motion" / "motions"


def _custom_motions_dirs() -> List[Path]:
    """Both motion library roots in scan order: project first, user second.

    Returns every root unconditionally — callers handle non-existent dirs
    by skipping. We intentionally don't filter to existing dirs here so a
    later ``import_file`` that creates the user side mid-session doesn't
    require a registry-style refresh to surface.
    """
    return [_project_motions_dir(), _user_motions_dir()]


def _project_archive_root() -> Path:
    """Project-shipped AMP archive root (read-only at runtime).

    Consolidated with the motions root: the reference-motion catalog
    (``installers/custom_mods_manifest.py``) clones every hot-plug pack
    under ``custom_mods/motions/<package>/``, and the legacy
    ``custom_mods/archives/`` tree was removed. SB3 ``.npy`` library
    and community AMP packs co-exist here — the scanners disambiguate by
    subtree shape (``<category>/<model>/*.npy`` vs
    ``<package>/datasets/<subdir>/*.{txt,json}``), so reusing one root is
    safe.
    """
    return _project_motions_dir()


def _user_archive_root() -> Path:
    """User-installed AMP archive root (mirrors the project-side unification)."""
    return _user_motions_dir()


def _archive_roots() -> List[Path]:
    """Both archive roots in scan order: project first, user second."""
    return [_project_archive_root(), _user_archive_root()]


#: Extensions recognised as amp_legged_gym motion clips. The AMPLegGymLoader
#: reads both — .txt is the AMP_for_hardware default, .json is used by newer
#: forks like MetalHead's jump dataset.
_ARCHIVE_MOTION_EXTS: tuple = (".txt", ".json")


def _iter_archive_motion_subdirs(pkg_dir: Path) -> List[Path]:
    """Yield every ``<pkg>/datasets/<sub>`` directory that holds at least
    one ``.txt`` / ``.json`` clip.

    Auto-discovery rather than a whitelist — each new community pack
    (AMP_for_hardware, amp_go2, MetalHead, ...) drops its clips under an
    arbitrarily-named subdir (``mocap_motions``, ``go2_motion``,
    ``mocap_motions_go2``, ``mocap_motions_jump``, ...). A fixed list
    here silently hides any pack whose subdir name we did not anticipate.
    Tool-private directories (``__pycache__``, hidden dirs) are skipped.
    """
    out: List[Path] = []
    datasets_root = pkg_dir / "datasets"
    if not datasets_root.is_dir():
        return out
    for sub in sorted(datasets_root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name.startswith(".") or name.startswith("_") or name == "__pycache__":
            continue
        try:
            has_clip = any(
                f.is_file() and f.suffix.lower() in _ARCHIVE_MOTION_EXTS
                for f in sub.iterdir()
            )
        except OSError:
            continue
        if has_clip:
            out.append(sub)
    return out


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

#: Canonical directory names used as first-level subdirs under custom_motions/.
CATEGORIES = ("quadruped", "biped", "wheeled", "manipulator", "generic")

_FAMILY_TO_CATEGORY: Dict[str, str] = {
    "quadruped":          "quadruped",
    "biped":              "biped",
    "wheeled":            "wheeled",
    "manipulator":        "manipulator",
    "generic_locomotion": "generic",
}


def get_category(robot_type: str) -> str:
    """Return the motion library category for a robot type string."""
    family = ROBOT_FAMILY_BY_TYPE.get(
        normalize_robot_type(robot_type), "generic_locomotion"
    )
    return _FAMILY_TO_CATEGORY.get(family, "generic")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MotionEntry:
    name: str          # file stem, e.g. "walk"
    path: Path         # absolute path to the motion file
    robot_model: str   # model sub-directory name, e.g. "go2"
    category: str      # category directory name, e.g. "quadruped"
    task_tag: str = ""  # behavioral label, e.g. "walk", "trot", "stand"

    @property
    def rel_label(self) -> str:
        """Short path relative to motions/, used in UI detail lines."""
        return f"{self.category}/{self.robot_model}"


@dataclass
class MotionPack:
    """A bundle of motion clips that share a common origin.

    Packs are the natural granularity for AMP training: the discriminator
    learns the joint distribution of an entire pool of natural movements,
    not a single clip. The Reference Motion picker surfaces packs as
    📦 entries that the launcher expands into a comma-separated list of
    file paths just before subprocess launch.

    A ``pack_id`` has the form ``"pack:<package>:<subdir>"`` — e.g.
    ``"pack:AMP_for_hardware:mocap_motions"``. The picker writes this
    string into ``ReferenceMotionConfig.motion_source`` (which is the
    same field that already carries ``loco:`` and ``generate:`` source
    strings); the window-layer scanner detects the ``pack:`` prefix and
    calls :func:`expand_motion_pack` to materialise the file list.
    """

    package: str       # archive package name, e.g. "AMP_for_hardware"
    subdir: str        # mocap subdir name, e.g. "mocap_motions"
    file_count: int    # number of motion files in the pack
    files: List[Path] = field(default_factory=list)
    """Absolute paths to the motion files. Sorted, deterministic order."""

    @property
    def pack_id(self) -> str:
        return f"pack:{self.package}:{self.subdir}"

    @property
    def display_label(self) -> str:
        return f"{self.package} / {self.subdir} ({self.file_count} clips)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def list_entries(
    *,
    category: Optional[str] = None,
    robot_type: Optional[str] = None,
    include_archives: bool = True,
) -> List[MotionEntry]:
    """
    Return all discoverable reference motion entries.

    Scans two roots and merges:

    1. ``<PROJECT_ROOT>/custom_mods/motions/`` — project-shipped built-ins
       (both SB3 ``.npy`` clips under ``<category>/<model>/`` and community
       AMP packs under ``<package>/datasets/<subdir>/``).
    2. ``<USER_CONFIG_DIR>/scripts/training_motion/motions/`` — user-installed.

    Both are walked in that order. Entries with the same absolute path
    are deduplicated; otherwise duplicates with identical
    ``(category, robot_model, name)`` retain only the first occurrence
    (project canonical wins, mirroring resolution-side semantics).

    Archives carry community package mocap clips (AMP_for_hardware /
    MetalHead / amp_go2 / rl_amp). They show up under
    ``robot_model="<package_name>"`` and category ``"quadruped"`` (all
    known community AMP packages target quadrupeds today).

    Set ``include_archives=False`` to restrict the scan to the SB3
    library (useful when a call site only cares about .npy tracking
    clips, e.g. the legacy import workflow).

    If *robot_type* is given the category is inferred from it; that category
    is the only one scanned. If neither is given, all categories are scanned.
    """
    scan_cat: Optional[str] = category
    if scan_cat is None and robot_type:
        scan_cat = get_category(robot_type)

    entries: List[MotionEntry] = []
    seen_paths: set = set()
    seen_keys: set = set()

    # ── SB3 library side — both roots, project first ──────────────────
    cats = [scan_cat] if scan_cat else list(CATEGORIES)
    for motions_root in _custom_motions_dirs():
        if not motions_root.is_dir():
            continue
        for cat in cats:
            cat_dir = motions_root / cat
            if not cat_dir.is_dir():
                continue
            for model_dir in sorted(cat_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for npy in sorted(model_dir.glob("*.npy")):
                    p = npy.resolve()
                    key = (cat, model_dir.name, npy.stem)
                    if p in seen_paths or key in seen_keys:
                        continue
                    seen_paths.add(p)
                    seen_keys.add(key)
                    entries.append(MotionEntry(
                        name=npy.stem,
                        path=p,
                        robot_model=model_dir.name,
                        category=cat,
                        task_tag=infer_task_tag(npy.name),
                    ))

    # ── Archive side (AMP_for_hardware, MetalHead, amp_go2, …) ────────
    if include_archives and (scan_cat is None or scan_cat == "quadruped"):
        entries.extend(_list_archive_motion_entries())

    return entries


def _list_archive_motion_entries() -> List[MotionEntry]:
    """Walk both archive roots for AMP mocap clips.

    Returns one ``MotionEntry`` per file found across
    ``custom_mods/motions/`` and ``<USER_CONFIG>/scripts/training_motion/motions/``
    (archive and SB3-library content share one root post-consolidation).
    Packages are grouped via ``robot_model = <package_name>``; the
    category is fixed to ``quadruped`` because every community AMP
    package in scope today targets quadrupeds (A1, Go2, etc.).

    Missing roots or individual subdirs are silently skipped (they may
    legitimately not be installed). Within the merged listing, duplicate
    files (same absolute path or same ``<package>/<subdir>/<stem>``
    triple) are kept once, with project canonical winning on tie.
    """
    out: List[MotionEntry] = []
    seen_paths: set = set()
    seen_keys: set = set()
    for archive_root in _archive_roots():
        if not archive_root.is_dir():
            continue
        for pkg_dir in sorted(archive_root.iterdir()):
            if not pkg_dir.is_dir():
                continue
            if pkg_dir.name.startswith(".") or pkg_dir.name.startswith("_"):
                continue
            for base in _iter_archive_motion_subdirs(pkg_dir):
                for f in sorted(base.iterdir()):
                    if not f.is_file():
                        continue
                    if f.suffix.lower() not in _ARCHIVE_MOTION_EXTS:
                        continue
                    p = f.resolve()
                    key = (pkg_dir.name, base.name, f.stem)
                    if p in seen_paths or key in seen_keys:
                        continue
                    seen_paths.add(p)
                    seen_keys.add(key)
                    out.append(MotionEntry(
                        name=f.stem,
                        path=p,
                        # Package name serves as the "model" grouping so
                        # sorting pins archive clips together. Distinct from
                        # any user-defined SB3 library model (e.g. "go2")
                        # because no SB3 model directory ever uses a
                        # capitalised package name like "AMP_for_hardware".
                        robot_model=pkg_dir.name,
                        category="quadruped",
                        task_tag=infer_task_tag(f.name),
                    ))
    return out


# ---------------------------------------------------------------------------
# Motion packs (the natural granularity for AMP training)
# ---------------------------------------------------------------------------


def list_motion_packs() -> List[MotionPack]:
    """Enumerate motion packs across both archive roots.

    One ``MotionPack`` per ``<package>/datasets/<subdir>/`` directory
    that contains at least one .txt or .json clip. Used by the Reference
    Motion picker to surface 📦 entries that can be selected as a single
    UI action and expanded into the full file list at training launch.

    Packs from ``<PROJECT_ROOT>/custom_mods/motions/`` are listed first;
    same-named ``(package, subdir)`` pairs found later under USER_CONFIG
    are skipped so the project-shipped version wins on conflict (mirrors
    ``expand_motion_pack`` / ``resolve_pack_ref`` resolution order).
    """
    packs: List[MotionPack] = []
    seen: set = set()
    for archive_root in _archive_roots():
        if not archive_root.is_dir():
            continue
        for pkg_dir in sorted(archive_root.iterdir()):
            if not pkg_dir.is_dir():
                continue
            if pkg_dir.name.startswith(".") or pkg_dir.name.startswith("_"):
                continue
            for base in _iter_archive_motion_subdirs(pkg_dir):
                key = (pkg_dir.name, base.name)
                if key in seen:
                    continue
                files = sorted(
                    f.resolve() for f in base.iterdir()
                    if f.is_file() and f.suffix.lower() in _ARCHIVE_MOTION_EXTS
                )
                if not files:
                    continue
                seen.add(key)
                packs.append(MotionPack(
                    package=pkg_dir.name,
                    subdir=base.name,
                    file_count=len(files),
                    files=files,
                ))
    return packs


def expand_motion_pack(pack_id: str) -> List[Path]:
    """Resolve a ``"pack:<package>:<subdir>"`` identifier to its file list.

    Returns absolute paths in deterministic sorted order. Raises
    ``ValueError`` on a malformed identifier or unknown package/subdir.
    """
    if not pack_id.startswith("pack:"):
        raise ValueError(f"Not a pack identifier: {pack_id!r}")

    rest = pack_id[len("pack:"):]
    parts = rest.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Malformed pack identifier {pack_id!r}: "
            f"expected 'pack:<package>:<subdir>'"
        )
    package, subdir = parts

    # Probe both roots in scan order — project canonical first, then user.
    tried: List[Path] = []
    for archive_root in _archive_roots():
        base = archive_root / package / "datasets" / subdir
        tried.append(base)
        if not base.is_dir():
            continue
        files = sorted(
            f.resolve() for f in base.iterdir()
            if f.is_file() and f.suffix.lower() in _ARCHIVE_MOTION_EXTS
        )
        if files:
            return files
        # Directory exists but empty under this root — keep looking; the
        # other root may contain the actual clip files.
    raise ValueError(
        f"Pack {pack_id!r} not found on disk under any archive root; "
        f"searched: {', '.join(str(p) for p in tried)}"
    )


def resolve_pack_ref(ref: str) -> Path:
    """Resolve a per-clip pack reference ``"pack:<package>:<subdir>/<file>"`` to an absolute path.

    Distinct from :func:`expand_motion_pack`, which takes a pack-level
    identifier (``"pack:<package>:<subdir>"``) and returns the full file
    list. This helper takes a clip-level reference (same prefix, but
    the third segment contains a ``/<filename>`` suffix) and returns
    the absolute path to that one clip.
    """
    if not ref.startswith("pack:"):
        raise ValueError(f"Not a pack reference: {ref!r}")

    rest = ref[len("pack:"):]
    parts = rest.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Malformed pack reference {ref!r}: "
            f"expected 'pack:<package>:<subdir>/<file>'"
        )
    package, tail = parts

    if "/" not in tail:
        raise ValueError(
            f"Malformed pack reference {ref!r}: "
            f"expected 'pack:<package>:<subdir>/<file>'"
        )
    subdir, filename = tail.split("/", 1)
    if not subdir or not filename:
        raise ValueError(
            f"Malformed pack reference {ref!r}: "
            f"expected 'pack:<package>:<subdir>/<file>'"
        )
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(
            f"Clip reference {ref!r} must contain a flat filename "
            f"(no subdirectories)"
        )

    # Probe both roots in scan order — project canonical first, then user.
    tried: List[Path] = []
    for archive_root in _archive_roots():
        path = archive_root / package / "datasets" / subdir / filename
        tried.append(path)
        if path.is_file():
            if path.suffix.lower() not in _ARCHIVE_MOTION_EXTS:
                raise ValueError(
                    f"Clip reference {ref!r} has unsupported extension {path.suffix!r}"
                )
            return path.resolve()
    raise ValueError(
        f"Clip reference {ref!r} not found on disk under any archive root; "
        f"searched: {', '.join(str(p) for p in tried)}"
    )


def build_pack_ref(package: str, subdir: str, filename: str) -> str:
    """Construct a clip-level pack reference string.

    Inverse of :func:`resolve_pack_ref` parsing. Used by the UI when
    serializing a clip selection.
    """
    if not package or not subdir or not filename:
        raise ValueError("build_pack_ref: all of package/subdir/filename are required")
    if "/" in filename or "\\" in filename:
        raise ValueError(f"build_pack_ref: filename must be flat, got {filename!r}")
    return f"pack:{package}:{subdir}/{filename}"


def _resolve_single_bridge_source(source: str) -> Optional[str]:
    """Resolve a single-file ``loco:``/``generate:`` source to a ``.npy``.

    Delegates to ``loco_mujoco_bridge`` if available; returns ``None`` when
    the bridge module hasn't been migrated into RELEASE yet.
    """
    source = (source or "").strip()
    if not source:
        return None
    try:
        # NB: loco_mujoco_bridge is not part of RELEASE today. Once the
        # training-pipeline runtime lands, point this at its new home.
        from scripts.training_motion.loco_mujoco_bridge import (  # type: ignore[import]
            generate_simple_walk_reference,
            generate_standing_reference,
            is_available,
            load_trajectory_as_npy,
        )
        if source.startswith("loco:"):
            parts = source.split(":", 2)
            if len(parts) < 3:
                return None
            env_name, task = parts[1], parts[2]
            if not is_available():
                return None
            result = load_trajectory_as_npy(env_name, task=task)
            return str(result) if result else None
        if source.startswith("generate:"):
            gen_type = source.split(":", 1)[1].strip().lower()
            if gen_type == "standing":
                return str(generate_standing_reference("go2", fps=50.0))
            if gen_type == "walk":
                return str(generate_simple_walk_reference("go2", fps=50.0))
            return None
    except Exception:
        return None
    return None


def resolve_motion_source_files(source: str) -> List[str]:
    """Resolve a ``motion_source`` identifier to absolute file paths.

    Handles three prefix families:

    - ``pack:<package>:<subdir>`` — expands to the full clip list
      (delegates to :func:`expand_motion_pack`).
    - ``loco:<env>:<task>`` — single ``.npy`` via the loco_mujoco bridge
      (returns empty list when the bridge module is unavailable).
    - ``generate:<kind>`` — single synthetic ``.npy`` from the bridge
      (returns empty list when the bridge module is unavailable).

    Empty / unknown sources return an empty list. Single source of truth
    for motion_source expansion — the window launcher, UI preview panel,
    and ``TrainingMotionNode`` all delegate here.
    """
    source = (source or "").strip()
    if not source:
        return []
    if source.startswith("pack:"):
        return [str(p) for p in expand_motion_pack(source)]
    single = _resolve_single_bridge_source(source)
    return [single] if single else []


def apply_clip_enabled_filter(paths: List[str], enabled_json: str) -> List[str]:
    """Drop paths whose stem is explicitly disabled in ``enabled_json``.

    ``enabled_json`` is the JSON-encoded ``{stem: bool}`` map. Missing keys
    (or any parse error) default to *enabled* — opt-out semantics so existing
    canvases without the field behave unchanged.
    """
    if not paths:
        return list(paths)
    try:
        import json as _json
        enabled_map = _json.loads(enabled_json) if enabled_json else {}
    except Exception:
        return list(paths)
    if not isinstance(enabled_map, dict) or not enabled_map:
        return list(paths)

    result: List[str] = []
    for p in paths:
        stem = Path(p).stem
        val = enabled_map.get(stem, True)
        if isinstance(val, str):
            disabled = val.strip().lower() in ("false", "0", "no", "off")
        else:
            disabled = not bool(val)
        if not disabled:
            result.append(p)
    return result


def import_file(
    src: Path,
    *,
    robot_type: str,
    robot_model: Optional[str] = None,
) -> MotionEntry:
    """
    Copy *src* into the motion library under the correct category/model dir.

    *robot_model* defaults to the normalised robot_type when not given.
    If a file with the same stem already exists in that dir the name is made
    unique by appending ``_2``, ``_3`` … before the extension.

    Returns the newly registered ``MotionEntry``.
    """
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    cat = get_category(robot_type)
    model = str(robot_model or normalize_robot_type(robot_type) or "generic").lower()
    # Writes always land under USER_CONFIG_DIR — project's custom_mods/
    # tree is read-only at runtime (CLAUDE.md §1.4).
    dest_dir = _user_motions_dir() / cat / model
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    if dest.exists():
        stem, ext = src.stem, src.suffix
        i = 2
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{ext}"
            i += 1

    shutil.copy2(src, dest)
    return MotionEntry(
        name=dest.stem,
        path=dest.resolve(),
        robot_model=model,
        category=cat,
    )


__all__ = [
    "CATEGORIES",
    "MotionEntry",
    "MotionPack",
    "get_category",
    "list_entries",
    "list_motion_packs",
    "expand_motion_pack",
    "resolve_pack_ref",
    "build_pack_ref",
    "resolve_motion_source_files",
    "apply_clip_enabled_filter",
    "import_file",
]
