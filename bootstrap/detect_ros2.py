#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform native ROS2 detection helper.

Prints a single JSON object to stdout so batch/bash callers can parse with a
one-liner. Never raises — absence of ROS2 is reported as `{"found": false}`.

Detection order:
  1. ROS_DISTRO environment variable combined with AMENT_PREFIX_PATH / a
     discoverable `ros2` executable.
  2. `ros2` on PATH.
  3. Conventional install roots per platform (/opt/ros/<distro>,
     C:\\opt\\ros\\<distro>, C:\\dev\\ros2_<distro>).

Output schema:
  {
    "found": true,
    "distro": "humble",
    "ros_root": "/opt/ros/humble",
    "executable": "/opt/ros/humble/bin/ros2",
    "source": "env" | "path" | "conventional"
  }
  or {"found": false}
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


_WINDOWS_ROOTS = [
    Path(r"C:\opt\ros"),
    Path(r"C:\dev"),
    Path(r"C:\ros2"),
]
_POSIX_ROOTS = [
    Path("/opt/ros"),
    Path.home() / "ros2",
]
_KNOWN_DISTROS = (
    "humble", "iron", "jazzy", "rolling",
    "galactic", "foxy", "eloquent",
)


def _ros_executable_name() -> str:
    return "ros2.exe" if os.name == "nt" else "ros2"


def _infer_root_from_exec(exec_path: Path) -> Optional[Path]:
    # Typical layout: <root>/bin/ros2(.exe)
    if exec_path.parent.name.lower() == "bin":
        return exec_path.parent.parent
    return exec_path.parent


def _infer_distro_from_root(root: Path) -> Optional[str]:
    name = root.name.lower()
    if name in _KNOWN_DISTROS:
        return name
    for parent in root.parents:
        if parent.name.lower() in _KNOWN_DISTROS:
            return parent.name.lower()
    return None


def _try_env() -> Optional[dict]:
    distro = os.environ.get("ROS_DISTRO", "").strip().lower()
    if not distro:
        return None
    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    ros_root: Optional[Path] = None
    for entry in ament.split(os.pathsep):
        p = Path(entry.strip())
        if p.name.lower() == distro and p.exists():
            ros_root = p
            break
    exec_path = shutil.which(_ros_executable_name())
    if exec_path:
        exec_p = Path(exec_path)
        if ros_root is None:
            ros_root = _infer_root_from_exec(exec_p)
        return {
            "found": True,
            "distro": distro,
            "ros_root": str(ros_root) if ros_root else "",
            "executable": str(exec_p),
            "source": "env",
        }
    if ros_root is not None:
        bin_dir = ros_root / "bin"
        exec_p = bin_dir / _ros_executable_name()
        return {
            "found": True,
            "distro": distro,
            "ros_root": str(ros_root),
            "executable": str(exec_p) if exec_p.exists() else "",
            "source": "env",
        }
    return None


def _try_path() -> Optional[dict]:
    exec_path = shutil.which(_ros_executable_name())
    if not exec_path:
        return None
    exec_p = Path(exec_path)
    ros_root = _infer_root_from_exec(exec_p)
    distro = _infer_distro_from_root(ros_root) if ros_root else None
    if not distro:
        # Last-chance: maybe ROS_DISTRO is exported anyway
        distro = os.environ.get("ROS_DISTRO", "").strip().lower() or "unknown"
    return {
        "found": True,
        "distro": distro,
        "ros_root": str(ros_root) if ros_root else "",
        "executable": str(exec_p),
        "source": "path",
    }


def _try_conventional_roots() -> Optional[dict]:
    roots = _WINDOWS_ROOTS if os.name == "nt" else _POSIX_ROOTS
    for base in roots:
        if not base.exists() or not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            name = child.name.lower()
            distro: Optional[str] = None
            if name in _KNOWN_DISTROS:
                distro = name
                ros_root = child
            else:
                # e.g. C:\dev\ros2_humble
                for d in _KNOWN_DISTROS:
                    if d in name:
                        distro = d
                        ros_root = child
                        break
                else:
                    continue
            bin_dir = ros_root / "bin"
            exec_p = bin_dir / _ros_executable_name()
            return {
                "found": True,
                "distro": distro,
                "ros_root": str(ros_root),
                "executable": str(exec_p) if exec_p.exists() else "",
                "source": "conventional",
            }
    return None


def detect() -> dict:
    for probe in (_try_env, _try_path, _try_conventional_roots):
        try:
            result = probe()
        except Exception:
            result = None
        if result:
            return result
    return {"found": False}


def main() -> int:
    result = detect()
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
