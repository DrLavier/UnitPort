"""Windows non-ASCII path compatibility shim for MuJoCo XML loads.

MuJoCo's ``MjModel.from_xml_path`` calls into tinyxml2's C++ parser,
which opens files via the C runtime ``fopen``. On Windows ``fopen``
takes ``const char*`` and interprets the bytes through the system
ANSI codepage (``CP_ACP``). When the Python binding hands a UTF-8
encoded path to ``fopen``, any byte outside the ANSI range is
mis-translated and the open fails with::

    ValueError: ParseXML: Error opening file 'A:\\测试\\…\\scene.xml'

Python itself opens the same path via ``_wfopen`` (Unicode-aware),
so ``pathlib.Path.exists()`` and ``open(path)`` work; only MuJoCo's
C++ side breaks.

Two-tier fix:

1. **``GetShortPathNameW`` fast path** — when the volume has 8.3
   short-name generation enabled (the default on system drives),
   Windows returns a pure-ASCII alias like ``A:\\6C18~1\\UnitPort\\…``
   that ``fopen`` opens correctly. Zero file copies, ~microseconds.

2. **Copy-to-temp-ASCII fallback** — when 8.3 is disabled on the
   volume (``fsutil 8dot3name set <vol> 1`` was never run, or it was
   set to ``2`` per-directory and our directory was not flagged),
   ``GetShortPathNameW`` returns a path that is still non-ASCII.
   In that case we copy the scene file's parent directory to a temp
   directory under ``%TEMP%`` (which always lives under a profile
   path Windows guarantees is openable by the ANSI codepage in
   practice — even Chinese Windows uses ``C:\\Users\\<latin-name>\\AppData``
   for the standard user account; the entire ``AppData`` subtree is
   ASCII). MuJoCo then opens the copy.

   The copy includes the *whole parent directory* because MJCF files
   reference meshes and includes via paths relative to the scene
   file's directory. Menagerie robot dirs are 1-50 MB; one copy per
   robot dir, cached for the process lifetime. ``atexit`` cleanup
   removes the temp tree on normal shutdown.

Both branches are gated on the path being non-ASCII on Windows; the
helper is a no-op on POSIX and on ASCII-only Windows paths, so it
costs nothing for the common case.

This is the only legitimate place in the codebase to hide a path
translation behind a helper (CLAUDE.md §1.8 fallback rule (b) —
cross-platform branches, documented intent). Anywhere a future
caller hands a path to MuJoCo, route it through here.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Union

__all__ = ["safe_mjcf_path"]


# Module-level cache so repeated loads of the same MJCF (or sibling files
# under the same robot directory) reuse a single temp tree. Keyed on the
# resolved parent directory; value is the matching temp directory.
_alias_lock = threading.Lock()
_alias_cache: Dict[Path, Path] = {}


def _try_short_path(s: str) -> Optional[str]:
    """Best-effort short-path lookup; returns None if it would still
    contain non-ASCII characters (volume has 8.3 disabled), or on any
    Win32 error."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        GetShortPathNameW.restype = wintypes.DWORD
    except (OSError, AttributeError):
        return None

    buf = ctypes.create_unicode_buffer(512)
    n = GetShortPathNameW(s, buf, len(buf))
    if n == 0:
        return None
    if n > len(buf):
        buf = ctypes.create_unicode_buffer(n + 1)
        n = GetShortPathNameW(s, buf, len(buf))
        if n == 0:
            return None
    candidate = buf.value
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError:
        # Volume has 8.3 disabled for this subtree; the "short" path is
        # the same long-name UTF-16 string we passed in. Caller falls
        # through to the copy path.
        return None
    return candidate


def _alias_via_copy(p: Path) -> str:
    """Copy the file's parent dir to an ASCII temp location; return the
    path of ``p`` under the copy.

    The whole directory is copied (not just the requested file) because
    MJCF scenes use relative paths to reference included XMLs and mesh
    files. Caching is keyed on the *real* parent directory so two scenes
    in the same robot dir share one copy.
    """
    parent = p.parent.resolve()
    with _alias_lock:
        cached = _alias_cache.get(parent)
        if cached is not None:
            return str(cached / p.name)

        # mkdtemp returns an ASCII path under %TEMP% — the standard
        # Windows temp dir is under C:\Users\<latin-name>\AppData on
        # every locale (account names default to Latin; Chinese
        # account names are rare and even those typically have an
        # ASCII profile dir set by the installer).
        tmp_root = Path(tempfile.mkdtemp(prefix="unitport_mjcf_alias_"))
        try:
            # copytree refuses to create over an existing directory;
            # mkdtemp returned a fresh path, so the dst is the temp
            # root and we walk parent's contents in.
            for child in parent.iterdir():
                dst = tmp_root / child.name
                if child.is_dir():
                    shutil.copytree(child, dst, symlinks=False)
                else:
                    shutil.copy2(child, dst)
        except OSError:
            # Cleanup partial copy; let MuJoCo report its own error on
            # the original path so the user sees something actionable.
            shutil.rmtree(tmp_root, ignore_errors=True)
            return str(p)

        _alias_cache[parent] = tmp_root
        # Schedule cleanup at process exit. ignore_errors keeps a slow
        # AV scan from blocking shutdown.
        atexit.register(shutil.rmtree, str(tmp_root), True)
        return str(tmp_root / p.name)


def safe_mjcf_path(path: Union[str, Path]) -> str:
    """Return a path string MuJoCo's C++ XML parser can open on Windows.

    No-op on POSIX and on ASCII-only Windows paths. On Windows paths
    with non-ASCII components, first attempts ``GetShortPathNameW``;
    if 8.3 is disabled on the volume, falls back to copying the file's
    parent directory under ``%TEMP%`` (cached for the process lifetime)
    and returning the path within the copy.

    Returns a *string* path (not a Path) because the most common
    callsite is ``mujoco.MjModel.from_xml_path(<str>)``.
    """
    s = str(path)
    if os.name != "nt":
        return s
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        pass

    short = _try_short_path(s)
    if short is not None:
        return short

    return _alias_via_copy(Path(path))
