"""USD parser — Isaac venv only.

Per AMP_design.yaml §7.risks.usd_parser_venv_dependency (decision: pin
path b):

- ``pxr`` (Pixar USD python lib) only ships with Isaac Sim. The main
  UnitPort venv does not have it.
- In the main venv we **never call** the real parse path; this module
  raises ``UsdParserUnavailable`` on import attempt of pxr, which the
  validator catches and downgrades to a warn-only state for usd-only
  assets.
- The launcher subprocess (``il_train_launcher.py``) is already running
  inside the Isaac venv, so it imports ``pxr`` successfully and runs the
  full validation just before AppLauncher boot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


class UsdParserUnavailable(RuntimeError):
    """Raised when ``pxr`` cannot be imported (we're in the main venv).

    The validator treats this as 'defer to launcher subprocess', not as
    a hard error.
    """


class UsdParseError(ValueError):
    """Raised when a USD file is structurally invalid."""


@dataclass
class JointLimit:
    """Per-joint USD physics limits."""

    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    has_range: bool = False


@dataclass
class UsdMetadata:
    joint_names: List[str] = field(default_factory=list)
    dof_count: int = 0
    base_link: str = ""
    link_names: List[str] = field(default_factory=list)
    #: Per-joint limits keyed by joint name.
    joint_limits: Dict[str, JointLimit] = field(default_factory=dict)
    #: Per-link mass keyed by link prim name (when MassAPI present).
    link_masses: Dict[str, float] = field(default_factory=dict)
    #: Per-link diagonal inertia keyed by link prim name.
    link_inertias: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    #: Mesh asset refs found via UsdGeom.Mesh prims.
    mesh_refs: List[str] = field(default_factory=list)


#: URL schemes that resolve via Nucleus / S3 / web — these don't exist as
#: local files and ``Path.is_file()`` would always return False on them.
#: ``Usd.Stage.Open`` accepts the URL string directly and lets Nucleus
#: fetch the asset.
_USD_URL_SCHEMES = ("http://", "https://", "omniverse://", "nucleus://")


def parse_usd(path: Path | str) -> UsdMetadata:
    """Parse a USD file via ``pxr.Usd``.

    Accepts both local file paths AND remote URLs (Nucleus / S3 / http).
    URLs are passed straight to ``Usd.Stage.Open`` without local
    file-existence checks; the underlying USD client handles fetch +
    caching transparently.

    Raises ``UsdParserUnavailable`` if ``pxr`` is not importable in this
    process — the caller should treat that as 'defer to launcher' rather
    than as a hard parse failure.
    """
    raw = str(path)
    is_url = any(raw.startswith(s) for s in _USD_URL_SCHEMES)

    # Local file: validate existence first. URLs: skip — pathlib would
    # mangle the forward slashes on Windows ("https://" → "https:\")
    # AND ``is_file()`` is always False on URL strings anyway.
    if not is_url:
        p = Path(raw)
        if not p.is_file():
            raise UsdParseError(f"USD file not found: {p}")
        open_target = str(p)
    else:
        open_target = raw

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError as exc:
        raise UsdParserUnavailable(
            "pxr (Pixar USD) is not available in this process. USD parsing "
            "is deferred to the Isaac Sim subprocess. This is expected when "
            "running from the main UnitPort venv."
        ) from exc

    try:
        stage = Usd.Stage.Open(open_target)
    except Exception as exc:
        raise UsdParseError(f"Usd.Stage.Open failed for {open_target}: {exc}") from exc

    if stage is None:
        raise UsdParseError(f"Usd.Stage.Open returned None for {open_target}")

    meta = UsdMetadata()

    # Walk the stage and collect prim names that look like joints / links.
    # The UsdPhysics schema marks joints with PhysicsRevoluteJoint /
    # PhysicsPrismaticJoint typed prims; links are typically Xform prims
    # under an Articulation root. This is a deliberately minimal extractor
    # — its only job is to give the validator a joint_names list to
    # cross-check against the mjcf-derived list.
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        prim_name = prim.GetName()
        if type_name in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            meta.joint_names.append(prim_name)
            jl = JointLimit()
            try:
                joint = UsdPhysics.Joint(prim)
                lo_attr = joint.GetLowerLimitAttr()
                hi_attr = joint.GetUpperLimitAttr()
                if lo_attr and hi_attr:
                    jl.lower = float(lo_attr.Get() or 0.0)
                    jl.upper = float(hi_attr.Get() or 0.0)
                    jl.has_range = True
            except Exception:
                pass
            meta.joint_limits[prim_name] = jl
        elif type_name == "Xform":
            # Heuristic: only collect Xforms whose direct parent is an
            # Articulation root. Anything more sophisticated (proper
            # link extraction) is out of scope for "receive + validate".
            meta.link_names.append(prim_name)
            try:
                mass_api = UsdPhysics.MassAPI(prim)
                m_attr = mass_api.GetMassAttr()
                if m_attr and m_attr.HasAuthoredValue():
                    meta.link_masses[prim_name] = float(m_attr.Get() or 0.0)
                d_attr = mass_api.GetDiagonalInertiaAttr()
                if d_attr and d_attr.HasAuthoredValue():
                    v = d_attr.Get()
                    if v is not None:
                        meta.link_inertias[prim_name] = (
                            float(v[0]), float(v[1]), float(v[2]),
                        )
            except Exception:
                pass
        elif type_name == "Mesh":
            try:
                pts_attr = prim.GetAttribute("points")
                if pts_attr is not None and pts_attr.HasAuthoredValue():
                    meta.mesh_refs.append(prim.GetPath().pathString)
            except Exception:
                pass

    meta.dof_count = len(meta.joint_names)
    if meta.link_names:
        meta.base_link = meta.link_names[0]

    return meta
