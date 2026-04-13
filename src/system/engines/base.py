"""Shared constants and helpers for engine installers.

Every engine installer (Isaac, Newton, ...) shares the same parent
directory convention so the user's machine stays organized:

    ~/UnitPort/engines/
        isaac/          — Isaac Lab + Isaac Sim 5.1
        newton/         — (future)

The parent ``~/UnitPort/engines/`` is the default; users can override
per-engine via the setup wizard or settings.
"""

from __future__ import annotations

import platform
from pathlib import Path

# ── UnitPort user data root ──────────────────────────────────────────────
# All user-private data (engine installs, SSH configs, registry) lives here,
# outside the project tree.  Never committed to git.
UNITPORT_USER_ROOT: Path = Path.home() / "UnitPort"

# ── Default root directory for all engine installations ──────────────────
ENGINES_DEFAULT_ROOT: Path = UNITPORT_USER_ROOT / "engines"


def engine_default_dir(engine_id: str) -> Path:
    """Return the default installation directory for *engine_id*.

    >>> engine_default_dir("isaac")
    PosixPath('/home/user/UnitPort/engines/isaac')
    """
    return ENGINES_DEFAULT_ROOT / engine_id


# ── Structured output protocol (shared by all engine install scripts) ────
# Install scripts emit these markers on stdout so the QThread wrapper can
# parse progress without engine-specific knowledge.

OUTPUT_STEP   = "STEP"        # [STEP N/TOTAL] description
OUTPUT_OK     = "OK"          # [OK] message
OUTPUT_WARN   = "WARN"        # [WARN] message
OUTPUT_ERROR  = "ERROR"       # [ERROR] message — fatal
OUTPUT_DONE   = "DONE"        # [DONE] — installation succeeded
OUTPUT_PROGRESS = "PROGRESS"  # [PROGRESS] 0-100
