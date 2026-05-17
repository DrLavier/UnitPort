"""URDF → MJCF asset conversion (Stage 6+ stub).

The Stage 6 env consumes :attr:`RobotSpec.mjcf_path` directly. URDFs are
accepted on the registers side (``robots_canonical.json`` ``assets.URDF``),
but the env never converts them at runtime — the conversion path is
upstream (asset import UI → MJCF artefact saved alongside URDF).

This module is reserved for that conversion utility. It is intentionally
empty in Stage 6 because:
    * MuJoCo's official ``mjcf_compile`` CLI is the right tool when the
      URDF + meshes are well-formed.
    * In-process URDF parsing pulls in heavy deps (urdfpy / xmltodict /
      mesh loaders) we don't yet need.
    * The Stage 6 acceptance test runs against MJCF assets; the URDF
      branch is not on the canvas → bundle critical path.

When a real conversion is needed (a brand adapter that ships URDF
without a paired MJCF), implement :func:`convert_urdf_to_mjcf` here so
the rest of the codebase has a single call site to dispatch through.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


class UrdfConversionError(RuntimeError):
    """Raised when the URDF → MJCF pipeline cannot complete."""


def convert_urdf_to_mjcf(
    urdf_path: Union[Path, str],
    out_dir: Union[Path, str],
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a URDF + meshes to an MJCF tree under *out_dir*.

    Stage 6 stub — raises :exc:`UrdfConversionError`. Wire to MuJoCo's
    ``mjcf_compile`` (or urdfpy) when a brand adapter actually needs
    runtime conversion. Until then, asset packs must ship an ``.xml``
    MJCF alongside their URDF.
    """
    raise UrdfConversionError(
        "URDF → MJCF conversion not yet wired (Stage 6 stub). "
        "Either ship an MJCF alongside the URDF (recommended; runs through "
        "RobotSpec.mjcf_path directly) or implement this function against "
        "MuJoCo's mjcf_compile CLI. urdf_path={!r} out_dir={!r}".format(
            str(urdf_path), str(out_dir),
        )
    )


def has_paired_mjcf(urdf_path: Union[Path, str]) -> Optional[Path]:
    """Return the sibling ``.xml`` if it exists, else ``None``.

    Convention: an asset pack with ``robot.urdf`` typically ships
    ``robot.xml`` (MJCF) in the same directory. The asset registry
    populates ``RobotSpec.mjcf_path`` from this when present.
    """
    p = Path(urdf_path)
    candidate = p.with_suffix(".xml")
    return candidate if candidate.is_file() else None


__all__ = [
    "UrdfConversionError",
    "convert_urdf_to_mjcf",
    "has_paired_mjcf",
]
