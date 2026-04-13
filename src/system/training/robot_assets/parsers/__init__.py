"""Robot asset parsers — read-only metadata extraction.

These parsers **do not modify** the source files. They extract joint names,
dof count, link tree, default joint positions, foot links, base link and
mass for the registry.

Per AMP_design.yaml §7.risks.usd_parser_venv_dependency (path b):

- ``mjcf_parser`` is the **primary** metadata source in the main venv
  (mujoco python wheel installs cleanly outside Isaac Sim).
- ``urdf_parser`` is the **secondary** source (pure xml, zero deps).
- ``usd_parser`` requires ``pxr`` which only ships with Isaac Sim. In the
  main venv it raises ``UsdParserUnavailable``; the launcher subprocess
  (already inside the Isaac venv) does the real parse.
"""
from __future__ import annotations

from src.system.training.robot_assets.parsers.urdf_parser import (
    parse_urdf,
    UrdfParseError,
)
from src.system.training.robot_assets.parsers.mjcf_parser import (
    parse_mjcf,
    MjcfParseError,
)
from src.system.training.robot_assets.parsers.usd_parser import (
    parse_usd,
    UsdParserUnavailable,
    UsdParseError,
)

__all__ = [
    "parse_urdf",
    "UrdfParseError",
    "parse_mjcf",
    "MjcfParseError",
    "parse_usd",
    "UsdParserUnavailable",
    "UsdParseError",
]
