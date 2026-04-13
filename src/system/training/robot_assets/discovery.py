"""Dynamic scan + auto-register for robot assets.

Per AMP_design.yaml §todo.auto_discovery (originally phase_5 roadmap —
brought forward because the Canvas needs a pre-populated dropdown to be
usable without a file picker).

What it does
------------
Walks a small set of known asset locations, groups files by their parent
directory, builds a ``RobotAsset`` candidate per group, and routes each
through ``registry.register_asset()`` — the **same** entry point a
manual RobotAssetNode edit would use. Candidates that fail validation
are collected into the returned ``DiscoveryReport`` (not raised).

Scanned locations
-----------------
1. ``src/system/runtime/simulation/mujoco/menagerie/<robot>/`` — the
   vendored menagerie mirror. Each ``<robot>/`` directory becomes one
   asset with id = directory name (e.g. ``unitree_a1``).

2. ``custom_mods/archives/<package>/resources/robots/<robot>/`` — the
   community package layout. Each ``<robot>/`` directory becomes one
   asset with id = ``<package>_<robot>`` to prevent collisions across
   packages (so ``AMP_for_hardware_a1`` and ``MetalHead_a1`` can coexist).

3. ``custom_mods/archives/<package>/datasets/robots/`` — rare variant
   seen in some repos. Same collision-prefix rule.

The walker IS NOT recursive beyond the conventional depths above. This
is intentional: deep walks hit .git histories, __pycache__ dirs, and
stray example xml files that are not real MJCFs. Phase_5 may broaden
the heuristic with a proper candidate filter; phase_3 stays narrow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from src.system.training.robot_assets.registry import (
    RobotAsset,
    RobotAssetValidationError,
    register_asset,
)


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class RejectedCandidate:
    """An asset group that the validator refused."""
    asset_id: str
    root_dir: Path
    error: str


@dataclass
class DiscoveryReport:
    registered: List[str] = field(default_factory=list)
    rejected: List[RejectedCandidate] = field(default_factory=list)
    skipped_dirs: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"DiscoveryReport("
            f"registered={len(self.registered)}, "
            f"rejected={len(self.rejected)}, "
            f"skipped={len(self.skipped_dirs)}"
            f")"
        )


# ---------------------------------------------------------------------------
# Path roots
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # discovery.py → robot_assets → training → system → src → REPO_ROOT
    # parents[0]=robot_assets  [1]=training  [2]=system  [3]=src  [4]=REPO_ROOT
    return Path(__file__).resolve().parents[4]


def default_menagerie_root() -> Path:
    return _repo_root() / "src" / "system" / "runtime" / "simulation" / "mujoco" / "menagerie"


def default_archives_root() -> Path:
    return _repo_root() / "custom_mods" / "archives"


# ---------------------------------------------------------------------------
# Family inference (lightweight heuristic)
# ---------------------------------------------------------------------------


_BIPED_TOKENS = ("h1", "g1", "humanoid", "biped")
_QUADRUPED_TOKENS = ("a1", "go1", "go2", "b1", "b2",
                     "spot", "cyberdog", "anymal", "wildcat")
#: ``<base>w`` suffix on a quadruped marks it as a wheeled variant.
#: E.g. ``go2w`` is a wheeled go2. These must be checked BEFORE the
#: plain quadruped tokens because ``"go2" in "go2w"`` is True.
_WHEELED_VARIANTS = ("go1w", "go2w", "b1w", "b2w")
_WHEELED_TOKENS = ("wheel", "rover")


def _infer_family(name: str) -> str:
    low = name.lower()
    for tok in _BIPED_TOKENS:
        if tok in low:
            return "biped"
    # Wheeled detection — explicit wheel variants first (so "go2w"
    # doesn't get swept up by the "go2" quadruped match below), then
    # generic wheel / rover keywords.
    for tok in _WHEELED_VARIANTS:
        if tok in low:
            return "wheeled"
    for tok in _WHEELED_TOKENS:
        if tok in low:
            return "wheeled"
    for tok in _QUADRUPED_TOKENS:
        if tok in low:
            return "quadruped"
    # Everything else defaults to quadruped — the locomotion-heavy bias
    # of the community packages we target. User can override in node
    # params if needed.
    return "quadruped"


# ---------------------------------------------------------------------------
# Menagerie scanner
# ---------------------------------------------------------------------------


#: Menagerie asset_id → ``nucleus:`` marker pointing at the Isaac Lab
#: built-in USD library. The marker is a UnitPort-private prefix that
#: gets expanded at the boundary between main venv and Isaac venv:
#:
#:   * Compiler (main venv) emits ``f"{{ISAAC_NUCLEUS_DIR}}/<rel>"`` as
#:     Python code into the generated config file, plus the import.
#:   * Launcher (isaac venv) when handed ``--unitport_robot_usd nucleus:<rel>``
#:     imports ``ISAAC_NUCLEUS_DIR`` from ``isaaclab.utils.assets`` and
#:     substitutes the real Nucleus URL or local cache path.
#:
#: This indirection is needed because ``ISAAC_NUCLEUS_DIR`` is only
#: importable inside the isaac venv (it depends on Kit being booted),
#: so the main-venv side cannot pre-resolve it. The on-disk paths
#: under ``ISAAC_NUCLEUS_DIR`` mirror Isaac Lab's
#: ``isaaclab_assets/robots/unitree.py`` etc. — change them only if
#: Isaac Lab itself moves an asset.
_MENAGERIE_USD_URL = {
    "unitree_a1":   "nucleus:Robots/Unitree/A1/a1.usd",
    "unitree_go1":  "nucleus:Robots/Unitree/Go1/go1.usd",
    "unitree_go2":  "nucleus:Robots/Unitree/Go2/go2.usd",
    "unitree_h1":   "nucleus:Robots/Unitree/H1/h1.usd",
    "unitree_g1":   "nucleus:Robots/Unitree/G1/g1.usd",
    "boston_dynamics_spot": "nucleus:Robots/BostonDynamics/spot/spot.usd",
}


def _scan_menagerie(root: Path, report: DiscoveryReport) -> None:
    """Register each ``<root>/<robot_dir>/`` as an asset.

    MJCF resolution order within a robot dir:
      1. scene.xml     (preferred — includes ground plane)
      2. <dir_name>.xml
      3. any single *.xml in the top level (fallback)

    USD resolution:
      1. Any ``*.usd`` / ``*.usda`` file in the dir (rare in the
         menagerie; present only for users who manually dropped a
         USD next to the MJCF)
      2. The canonical ``isaac_lab_assets://...`` Nucleus URL from
         :data:`_MENAGERIE_USD_URL` — this is the normal case, since
         Isaac Lab's USD library is served through Nucleus, not as
         on-disk files.
    """
    if not root.is_dir():
        return

    for robot_dir in sorted(root.iterdir()):
        if not robot_dir.is_dir():
            continue
        if robot_dir.name.startswith(".") or robot_dir.name.startswith("_"):
            continue

        mjcf = _pick_mjcf(robot_dir)
        urdf = _pick_urdf(robot_dir)
        usd = _pick_usd(robot_dir)

        # Canonical Nucleus URL fallback when no on-disk .usd present.
        # Stored in the dedicated ``usd_url`` string field rather than
        # shoehorned into ``usd_path`` — ``pathlib.Path`` misparses
        # ``scheme://`` on Windows.
        usd_url = ""
        if usd is None:
            usd_url = _MENAGERIE_USD_URL.get(robot_dir.name, "")

        if mjcf is None and urdf is None and usd is None and not usd_url:
            report.skipped_dirs.append(robot_dir)
            continue

        asset_id = robot_dir.name
        _try_register(
            asset_id=asset_id,
            family=_infer_family(robot_dir.name),
            usd_path=usd,
            usd_url=usd_url,
            mjcf_path=mjcf,
            urdf_path=urdf,
            root_dir=robot_dir,
            report=report,
        )


# ---------------------------------------------------------------------------
# Archives scanner
# ---------------------------------------------------------------------------


def _scan_archives(root: Path, report: DiscoveryReport) -> None:
    """Register each ``<root>/<package>/resources/robots/<robot>/`` asset.

    The id is ``<package>_<robot>`` to avoid collisions between community
    packages that both ship an ``a1`` robot. Assets also scanned under
    ``datasets/robots/`` (rare legacy variant).
    """
    if not root.is_dir():
        return

    for package_dir in sorted(root.iterdir()):
        if not package_dir.is_dir():
            continue
        if package_dir.name.startswith(".") or package_dir.name.startswith("_"):
            continue

        # Conventional sub-paths inside a community package
        candidates: List[Path] = []
        for sub in ("resources/robots", "datasets/robots"):
            base = package_dir / sub
            if base.is_dir():
                candidates.extend(
                    d for d in base.iterdir() if d.is_dir()
                )

        for robot_dir in sorted(candidates, key=lambda p: p.name):
            if robot_dir.name.startswith(".") or robot_dir.name.startswith("_"):
                continue

            mjcf = _pick_mjcf_recursive(robot_dir, max_depth=3)
            urdf = _pick_urdf_recursive(robot_dir, max_depth=3)
            usd = _pick_usd_recursive(robot_dir, max_depth=3)

            if mjcf is None and urdf is None and usd is None:
                report.skipped_dirs.append(robot_dir)
                continue

            asset_id = f"{package_dir.name}_{robot_dir.name}"
            _try_register(
                asset_id=asset_id,
                family=_infer_family(robot_dir.name),
                usd_path=usd,
                usd_url="",  # archives don't get the canonical URL fallback
                mjcf_path=mjcf,
                urdf_path=urdf,
                root_dir=robot_dir,
                report=report,
            )


# ---------------------------------------------------------------------------
# File pickers
# ---------------------------------------------------------------------------


def _pick_mjcf(dir_: Path) -> Optional[Path]:
    """Shallow MJCF pick — only looks at the top level of ``dir_``."""
    if not dir_.is_dir():
        return None
    # scene.xml first (preferred — has ground)
    scene = dir_ / "scene.xml"
    if scene.is_file() and _is_mjcf_xml(scene):
        return scene
    # <dirname>.xml next
    named = dir_ / f"{dir_.name}.xml"
    if named.is_file() and _is_mjcf_xml(named):
        return named
    # Any single .xml as a fallback
    xmls = [p for p in dir_.glob("*.xml") if _is_mjcf_xml(p)]
    if len(xmls) == 1:
        return xmls[0]
    return None


def _pick_urdf(dir_: Path) -> Optional[Path]:
    urdfs = sorted(dir_.glob("*.urdf"))
    return urdfs[0] if urdfs else None


def _pick_usd(dir_: Path) -> Optional[Path]:
    usds = sorted(dir_.glob("*.usd")) + sorted(dir_.glob("*.usda"))
    return usds[0] if usds else None


def _pick_mjcf_recursive(dir_: Path, *, max_depth: int) -> Optional[Path]:
    for p in _walk_ext(dir_, "*.xml", max_depth):
        if _is_mjcf_xml(p):
            return p
    return None


def _pick_urdf_recursive(dir_: Path, *, max_depth: int) -> Optional[Path]:
    for p in _walk_ext(dir_, "*.urdf", max_depth):
        return p
    return None


def _pick_usd_recursive(dir_: Path, *, max_depth: int) -> Optional[Path]:
    for p in _walk_ext(dir_, "*.usd", max_depth):
        return p
    for p in _walk_ext(dir_, "*.usda", max_depth):
        return p
    return None


def _walk_ext(dir_: Path, pattern: str, max_depth: int) -> Iterable[Path]:
    """Depth-limited glob (avoids descending into .git / __pycache__ etc)."""
    if not dir_.is_dir():
        return
    denylist = {".git", "__pycache__", ".venv", "node_modules", ".idea"}
    base_depth = len(dir_.parts)
    for p in dir_.rglob(pattern):
        rel_depth = len(p.parts) - base_depth
        if rel_depth > max_depth:
            continue
        if any(seg in denylist for seg in p.parts):
            continue
        yield p


def _is_mjcf_xml(path: Path) -> bool:
    """Fast sniff test: does the file look like an MJCF vs a URDF or config?

    We peek at the first 2 KB and look for either ``<mujoco`` (MJCF) or
    ``<robot`` (URDF). Full mujoco.MjModel.from_xml_path validation is
    left to the validator during register_asset() — the sniff test keeps
    discovery cheap when walking thousands of files.
    """
    try:
        with path.open("rb") as f:
            head = f.read(2048)
    except OSError:
        return False
    if b"<mujoco" in head:
        return True
    if b"<robot" in head:
        return False
    return False


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def _try_register(
    *,
    asset_id: str,
    family: str,
    usd_path: Optional[Path],
    usd_url: str,
    mjcf_path: Optional[Path],
    urdf_path: Optional[Path],
    root_dir: Path,
    report: DiscoveryReport,
) -> None:
    asset = RobotAsset(
        asset_id=asset_id,
        family=family,
        usd_path=usd_path,
        usd_url=usd_url,
        mjcf_path=mjcf_path,
        urdf_path=urdf_path,
    )
    try:
        register_asset(asset)
        report.registered.append(asset_id)
    except RobotAssetValidationError as exc:
        report.rejected.append(
            RejectedCandidate(
                asset_id=asset_id, root_dir=root_dir, error=str(exc)
            )
        )
    except Exception as exc:  # defensive: never let one bad asset crash discovery
        report.rejected.append(
            RejectedCandidate(
                asset_id=asset_id,
                root_dir=root_dir,
                error=f"unexpected: {exc}",
            )
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def walk_and_register(
    menagerie_root: Optional[Path] = None,
    archives_root: Optional[Path] = None,
) -> DiscoveryReport:
    """Run both scanners and return a combined report.

    This is **idempotent**: registering the same asset_id twice is safe
    because ``register_asset`` overwrites the prior entry. Safe to call
    from UI widgets on every render.
    """
    report = DiscoveryReport()
    _scan_menagerie(
        menagerie_root if menagerie_root is not None else default_menagerie_root(),
        report,
    )
    _scan_archives(
        archives_root if archives_root is not None else default_archives_root(),
        report,
    )
    return report
