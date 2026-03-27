from __future__ import annotations

from typing import Iterable, List


def canonicalize_joint_name(name: str) -> str:
    """Normalize equivalent joint/actuator names to a shared runtime form."""
    text = str(name or "").strip()
    if text.endswith("_joint"):
        return text[:-6]
    return text


def canonicalize_joint_names(names: Iterable[str]) -> List[str]:
    """Return canonicalized joint names preserving input order."""
    return [canonicalize_joint_name(name) for name in names]
