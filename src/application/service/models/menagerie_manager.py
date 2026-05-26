# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo Menagerie sparse-checkout manager (port of DEMO 1:1).

Single source of truth for both:

- Robot Asset sidebar's Menagerie browser dialog (:class:`MenagerieBrowserDialog`).
- :class:`InstallConfigWizard` Page 1 (:class:`MenagerieSelectPage`).
- :class:`PostSetupTask._step_menagerie` (delegates here).

Pure Python, no Qt -- UI layers wrap the callable surface in their
own QThread workers.  Functions that perform git/network work raise
``RuntimeError`` on failure; the caller decides whether to swallow or
surface.

Storage layout (matches DEMO bit-exact so a user crossing trees keeps
their existing checkout):

    custom_mods/models/menagerie/        -- sparse-checkout root (git work tree)
    runtime/_cache_/menagerie_icons/     -- preview PNG cache
        _index.json                      -- {dir_name: png_relpath_in_repo}
        <dir_name>.png                   -- per-package preview
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from unitport_sdk import Paths


MENAGERIE_REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_API_URL = (
    "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/"
)
MENAGERIE_TREES_API_URL = (
    "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/main"
    "?recursive=1"
)
MENAGERIE_RAW_BASE = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/"
)
ICON_CACHE_INDEX_FILE = "_index.json"

# Snapshot of mujoco_menagerie top-level packages (verified 2026-04-28
# in DEMO; bit-exact copy here so both trees use the same offline
# fallback when the GitHub API is unreachable).
MENAGERIE_PACKAGES_SNAPSHOT: Tuple[str, ...] = (
    "agilex_piper", "agility_cassie", "aloha",
    "anybotics_anymal_b", "anybotics_anymal_c",
    "apptronik_apollo", "arx_l5", "berkeley_humanoid",
    "bitcraze_crazyflie_2", "booster_t1", "boston_dynamics_spot",
    "dynamixel_2r", "flexiv_rizon4", "flybody",
    "fourier_n1", "franka_emika_panda", "franka_fr3", "franka_fr3_v2",
    "google_barkour_v0", "google_barkour_vb", "google_robot",
    "hello_robot_stretch", "hello_robot_stretch_3",
    "i2rt_yam", "iit_softfoot", "kinova_gen3", "kuka_iiwa_14",
    "leap_hand", "low_cost_robot_arm", "ms_human_700",
    "pal_talos", "pal_tiago", "pal_tiago_dual",
    "pndbotics_adam_lite", "realsense_d435i",
    "rethink_robotics_sawyer", "robot_soccer_kit",
    "robotiq_2f85", "robotiq_2f85_v4", "robotis_op3",
    "robotstudio_so101", "shadow_dexee", "shadow_hand",
    "skydio_x2", "stanford_tidybot",
    "tetheria_aero_hand_open", "toddlerbot_2xc", "toddlerbot_2xm",
    "trossen_vx300s", "trossen_wx250s", "trossen_wxai",
    "trs_so_arm100", "ufactory_lite6", "ufactory_xarm7",
    "umi_gripper",
    "unitree_a1", "unitree_g1", "unitree_go1", "unitree_go2",
    "unitree_h1", "unitree_z1",
    "universal_robots_ur10e", "universal_robots_ur5e",
    "wonik_allegro",
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def menagerie_root() -> Path:
    """Absolute path to ``custom_mods/models/menagerie`` (sparse-checkout root)."""
    return Paths.CUSTOM_MODS_DIR / "models" / "menagerie"


def icon_cache_root() -> Path:
    """Absolute path to ``runtime/_cache_/menagerie_icons``.

    Created on demand by :func:`ensure_cached_icon`. Lives under
    ``Paths.RUNTIME_DIR`` because it's a *runtime application cache*,
    not user-customised state -- safe to wipe and regenerate.
    """
    return Paths.RUNTIME_DIR / "_cache_" / "menagerie_icons"


# ---------------------------------------------------------------------------
# Local-disk inspection
# ---------------------------------------------------------------------------

def scan_installed_packages() -> Set[str]:
    """Return the set of package directory names currently materialised on disk."""
    root = menagerie_root()
    if not root.exists():
        return set()
    return {
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }


def has_git_clone() -> bool:
    """True iff the menagerie folder is a git working tree."""
    return (menagerie_root() / ".git").exists()


def registered_package_dirs() -> Set[str]:
    """Menagerie dirs that match a UnitPort-registered robot.

    Used by the wizard / browser dialog to highlight "this is one of
    yours". Resolves each registered SKU's ``assets.MJCF`` against the
    ``menagerie/<dir>/`` prefix convention so a card whose package is
    already wired into the canonical register surfaces with the blue
    "registered" title.

    Walks the merged registry (canonical + user overlay); variants whose
    parent declares the asset propagate automatically because the
    inheritance resolver fills in their merged ``assets.MJCF``.
    """
    # Imported lazily because this module is loaded during the wizard's
    # pre-Qt phase where the registry hasn't been initialised yet.
    try:
        from registers import robots as _robots
    except Exception:
        return set()
    out: Set[str] = set()
    for sku in _robots.list_skus():
        entry = _robots.get_robot(sku) or {}
        mjcf = (entry.get("assets") or {}).get("MJCF") or ""
        if not isinstance(mjcf, str) or not mjcf:
            continue
        # Canonical paths look like "menagerie/<dir>/<rest>.xml".
        if not mjcf.startswith("menagerie/"):
            continue
        tail = mjcf[len("menagerie/"):]
        dir_part = tail.split("/", 1)[0]
        if dir_part:
            out.add(dir_part)
    return out


def package_preview_image(dir_name: str) -> Optional[Path]:
    """Pick the most representative PNG inside a menagerie package folder.

    Heuristic:
      1. ``<dir>/<last_token>.png`` (e.g. ``unitree_go2`` -> ``go2.png``).
      2. Shortest top-level PNG by file name length.

    Returns ``None`` when the folder is missing or has no PNG at all.
    """
    folder = menagerie_root() / dir_name
    if not folder.exists() or not folder.is_dir():
        return None

    last_token = dir_name.rsplit("_", 1)[-1]
    candidate = folder / f"{last_token}.png"
    if candidate.exists():
        return candidate

    pngs = sorted(folder.glob("*.png"), key=lambda p: (len(p.name), p.name))
    return pngs[0] if pngs else None


# ---------------------------------------------------------------------------
# Icon cache
# ---------------------------------------------------------------------------

def cached_icon_path(name: str) -> Path:
    """Path to the cached preview PNG for *name* (does not have to exist)."""
    return icon_cache_root() / f"{name}.png"


def has_cached_icon(name: str) -> bool:
    return cached_icon_path(name).exists()


def load_icon_index() -> Dict[str, str]:
    """Return cached ``{dir: png_relpath_in_repo}`` map, or empty dict."""
    p = icon_cache_root() / ICON_CACHE_INDEX_FILE
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:  # noqa: BLE001
        pass
    return {}


def save_icon_index(index: Dict[str, str]) -> None:
    root = icon_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / ICON_CACHE_INDEX_FILE).write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def fetch_icon_index(timeout: float = 30.0) -> Dict[str, str]:
    """Hit the GitHub Trees API and return the per-package preview-PNG map.

    Raises ``RuntimeError`` on HTTP / network errors.
    """
    req = urllib.request.Request(
        MENAGERIE_TREES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "UnitPort-MenagerieManager",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub Trees API HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub Trees API unreachable: {exc.reason}") from exc

    by_dir: Dict[str, List[str]] = {}
    for entry in payload.get("tree") or []:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path") or "")
        if not path.endswith(".png"):
            continue
        parts = path.split("/", 1)
        if len(parts) != 2:
            continue
        dir_name, rest = parts
        if "/" in rest or dir_name.startswith("."):
            continue
        by_dir.setdefault(dir_name, []).append(path)

    index: Dict[str, str] = {}
    for dir_name, paths in by_dir.items():
        last_token = dir_name.rsplit("_", 1)[-1]
        preferred = f"{dir_name}/{last_token}.png"
        if preferred in paths:
            index[dir_name] = preferred
        else:
            paths.sort(key=lambda p: (len(p), p))
            index[dir_name] = paths[0]
    return index


def ensure_cached_icon(
    name: str,
    remote_relpath: Optional[str] = None,
    *,
    timeout: float = 30.0,
) -> Optional[Path]:
    """Make sure ``cache/<name>.png`` exists. Resolution order:

      1. Already cached -> return.
      2. Locally-installed package PNG -> copy into the cache.
      3. Download via ``raw.githubusercontent.com/<remote_relpath>``.

    Returns the cache path on success, ``None`` if every fallback failed.
    """
    target = cached_icon_path(name)
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    local = package_preview_image(name)
    if local is not None and local.exists():
        try:
            shutil.copy(str(local), str(target))
            return target
        except Exception:  # noqa: BLE001
            pass

    if not remote_relpath:
        return None

    raw_url = MENAGERIE_RAW_BASE + remote_relpath.lstrip("/")
    try:
        req = urllib.request.Request(
            raw_url,
            headers={"User-Agent": "UnitPort-MenagerieManager"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return None
        target.write_bytes(data)
        return target
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Remote fetch
# ---------------------------------------------------------------------------

def fetch_remote_packages(timeout: float = 15.0) -> List[str]:
    """Query the GitHub contents API and return the live top-level dir list.

    Raises ``RuntimeError`` (with a human-readable message) on HTTP /
    network errors. Returns a sorted list of package directory names.
    """
    req = urllib.request.Request(
        MENAGERIE_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "UnitPort-MenagerieManager",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub API HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API unreachable: {exc.reason}") from exc

    names: List[str] = []
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "dir":
            continue
        name = str(entry.get("name") or "")
        if not name or name.startswith("."):
            continue
        names.append(name)
    names.sort()
    return names


# ---------------------------------------------------------------------------
# Sparse-checkout operations
# ---------------------------------------------------------------------------

def _run_git(
    args: List[str],
    *,
    on_output: Optional[Callable[[str], None]] = None,
    stream: bool = False,
    timeout: Optional[float] = None,
) -> None:
    """Run a git command, raising on non-zero exit. Optionally streams output."""
    if stream:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            msg = line.strip()
            if msg and on_output is not None:
                on_output(msg)
        rc = process.wait(timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"{' '.join(args[:3])} exited with code {rc}")
        return

    completed = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False, timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        msg = stderr or f"{' '.join(args[:3])} failed (rc={completed.returncode})"
        raise RuntimeError(msg)


def bootstrap_with_packages(
    packages: Sequence[str],
    *,
    on_output: Optional[Callable[[str], None]] = None,
) -> None:
    """Initial sparse-clone of the menagerie repo with *packages* materialised.

    Steps: ``git init`` -> ``remote add origin`` -> ``sparse-checkout init
    --cone`` -> ``sparse-checkout set <packages>`` -> ``git pull --depth=1
    origin main``.

    Raises ``RuntimeError`` on failure. Caller is responsible for
    cleaning up a partial directory tree on error.
    """
    root = menagerie_root()
    if root.exists() and any(root.iterdir()) and not has_git_clone():
        raise RuntimeError(
            f"{root} exists without a .git directory; move it aside before "
            "bootstrapping menagerie."
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    if not has_git_clone():
        _run_git(["git", "init", str(root)], on_output=on_output)
        _run_git(
            ["git", "-C", str(root), "remote", "add", "origin", MENAGERIE_REPO_URL],
            on_output=on_output,
        )
        _run_git(
            ["git", "-C", str(root), "sparse-checkout", "init", "--cone"],
            on_output=on_output,
        )

    if packages:
        _run_git(
            ["git", "-C", str(root), "sparse-checkout", "set", *packages],
            on_output=on_output,
        )
    _run_git(
        ["git", "-C", str(root), "pull", "--depth=1", "--progress", "origin", "main"],
        on_output=on_output, stream=True,
    )


def add_packages(
    packages: Sequence[str],
    *,
    on_output: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Incrementally add *packages* to the existing sparse-checkout.

    Auto-bootstraps when ``.git`` is missing (uses ``--filter=blob:none
    --no-checkout`` + ``sparse-checkout init --cone`` + ``sparse-checkout
    add``).

    Raises ``RuntimeError`` on failure. Returns the list passed in
    (unchanged) so a caller doing ``installed |= set(add_packages(...))``
    works without re-deriving the input.
    """
    root = menagerie_root()
    pkgs = list(packages)

    if not has_git_clone():
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"{root} exists without a .git directory; move it aside before "
                "initializing menagerie."
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            [
                "git", "clone", "--filter=blob:none", "--no-checkout",
                "--depth", "1", MENAGERIE_REPO_URL, str(root),
            ],
            on_output=on_output, timeout=600,
        )
        _run_git(
            ["git", "-C", str(root), "sparse-checkout", "init", "--cone"],
            on_output=on_output,
        )

    if not pkgs:
        return []
    _run_git(
        ["git", "-C", str(root), "sparse-checkout", "add", *pkgs],
        on_output=on_output, timeout=600,
    )
    return pkgs


# ---------------------------------------------------------------------------
# OO facade
# ---------------------------------------------------------------------------

class MenagerieManager:
    """Thin OO facade over the module-level functions.

    The strict-naming requirement (``class MenagerieManager``) is so
    callers can reach the surface via a single import:

        from application.service.models import MenagerieManager
        mm = MenagerieManager()
        mm.bootstrap_with_packages(["unitree_go2"])

    Methods delegate 1:1 to module functions; the module-level API is
    the actual implementation. This keeps both styles working without
    code duplication.
    """

    # constants exposed for callers that want them off the class
    REPO_URL = MENAGERIE_REPO_URL
    PACKAGES_SNAPSHOT = MENAGERIE_PACKAGES_SNAPSHOT

    @staticmethod
    def root() -> Path:
        return menagerie_root()

    @staticmethod
    def icon_cache_root() -> Path:
        return icon_cache_root()

    @staticmethod
    def has_clone() -> bool:
        return has_git_clone()

    @staticmethod
    def scan_installed_packages() -> Set[str]:
        return scan_installed_packages()

    @staticmethod
    def registered_package_dirs() -> Set[str]:
        return registered_package_dirs()

    @staticmethod
    def package_preview_image(dir_name: str) -> Optional[Path]:
        return package_preview_image(dir_name)

    @staticmethod
    def cached_icon_path(name: str) -> Path:
        return cached_icon_path(name)

    @staticmethod
    def has_cached_icon(name: str) -> bool:
        return has_cached_icon(name)

    @staticmethod
    def load_icon_index() -> Dict[str, str]:
        return load_icon_index()

    @staticmethod
    def save_icon_index(index: Dict[str, str]) -> None:
        save_icon_index(index)

    @staticmethod
    def fetch_icon_index(timeout: float = 30.0) -> Dict[str, str]:
        return fetch_icon_index(timeout=timeout)

    @staticmethod
    def ensure_cached_icon(
        name: str,
        remote_relpath: Optional[str] = None,
        *,
        timeout: float = 30.0,
    ) -> Optional[Path]:
        return ensure_cached_icon(name, remote_relpath, timeout=timeout)

    @staticmethod
    def fetch_remote_packages(timeout: float = 15.0) -> List[str]:
        return fetch_remote_packages(timeout=timeout)

    @staticmethod
    def bootstrap_with_packages(
        packages: Sequence[str],
        *,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        bootstrap_with_packages(packages, on_output=on_output)

    @staticmethod
    def add_packages(
        packages: Sequence[str],
        *,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        return add_packages(packages, on_output=on_output)


__all__ = [
    "MENAGERIE_REPO_URL",
    "MENAGERIE_API_URL",
    "MENAGERIE_TREES_API_URL",
    "MENAGERIE_RAW_BASE",
    "MENAGERIE_PACKAGES_SNAPSHOT",
    "ICON_CACHE_INDEX_FILE",
    "MenagerieManager",
    "menagerie_root",
    "icon_cache_root",
    "scan_installed_packages",
    "has_git_clone",
    "registered_package_dirs",
    "package_preview_image",
    "cached_icon_path",
    "has_cached_icon",
    "load_icon_index",
    "save_icon_index",
    "fetch_icon_index",
    "ensure_cached_icon",
    "fetch_remote_packages",
    "bootstrap_with_packages",
    "add_packages",
]
