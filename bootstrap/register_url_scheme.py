"""Register the ``unitport://`` URL scheme with the OS.

Called once at install time (via install.bat / install.sh). The scheme
handler lets OAuth redirects like ``unitport://auth-callback?code=...``
launch (or hand off to) the primary UnitPort process.

**Windows**: writes per-user registry keys under
``HKCU\\Software\\Classes\\unitport``. No admin needed.

**Linux**: writes ``~/.local/share/applications/unitport.desktop`` and
calls ``xdg-mime default`` to register the handler. Falls back silently
if ``xdg-mime`` is unavailable.

**macOS**: not implemented (current project is Windows-first). macOS
requires ``CFBundleURLTypes`` entries inside an ``.app`` bundle.

The script is idempotent: re-running overwrites the existing entries
with the current python + main.py paths, which is usually desirable
after updates.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_SCHEME = "unitport"
_APP_NAME = "UnitPort"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MAIN_PY = _PROJECT_ROOT / "main.py"


def _venv_python(prefer_windowed: bool = False) -> Path:
    """Prefer the project-local .venv311 interpreter when present.

    If ``prefer_windowed`` is true (Windows), returns pythonw.exe so the
    URL-scheme launch doesn't flash a black console window.
    """
    if platform.system() == "Windows":
        scripts = _PROJECT_ROOT / ".venv311" / "Scripts"
        if prefer_windowed and (scripts / "pythonw.exe").exists():
            return scripts / "pythonw.exe"
        venv = scripts / "python.exe"
    else:
        venv = _PROJECT_ROOT / ".venv311" / "bin" / "python"
    return venv if venv.exists() else Path(sys.executable)


def _ensure_windows_launcher(source_python: Path) -> Path:
    """Create (or refresh) ``.venv311/Scripts/UnitPort.exe`` as a copy of
    the venv interpreter. The sole purpose is so that Chrome's "Open in
    UnitPort?" prompt, the Windows taskbar tooltip, and the ``tasklist``
    entry all read ``UnitPort.exe`` instead of ``python.exe`` / ``pythonw.exe``.

    Copying (not hard-linking) preserves venv semantics — Python uses the
    interpreter's directory to locate its ``pyvenv.cfg``, so the copy must
    live in the same ``Scripts/`` directory as the original.
    """
    target = source_python.with_name(f"{_APP_NAME}.exe")
    try:
        if target.exists():
            # Refresh if the source changed (Python upgrade, new venv, …).
            if target.stat().st_size != source_python.stat().st_size:
                shutil.copy2(source_python, target)
        else:
            shutil.copy2(source_python, target)
    except (OSError, shutil.SameFileError) as exc:
        print(
            f"[url-scheme] Failed to create {target.name}: {exc} — "
            f"falling back to {source_python.name}",
            file=sys.stderr,
        )
        return source_python
    return target


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _register_windows(python_exe: Path) -> None:
    import winreg  # type: ignore[import-not-found]

    # Create a copy of the interpreter named ``UnitPort.exe`` so every OS
    # surface (browser "Open external" prompt, taskbar, Task Manager) shows
    # the app name instead of "python" / "pythonw".
    launcher = _ensure_windows_launcher(python_exe)

    cmd = f'"{launcher}" "{_MAIN_PY}" "%1"'
    base_path = rf"Software\Classes\{_SCHEME}"

    # Pre-create subkeys.
    for key_path in (
        base_path,
        rf"{base_path}\DefaultIcon",
        rf"{base_path}\shell",
        rf"{base_path}\shell\open",
        rf"{base_path}\shell\open\command",
    ):
        key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.CloseKey(key)

    # Write values.
    def _set(path: str, name: str, value: str) -> None:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        winreg.CloseKey(key)

    _set(base_path, "", f"URL:{_APP_NAME} Protocol")
    _set(base_path, "URL Protocol", "")
    _set(base_path, "FriendlyTypeName", _APP_NAME)
    _set(rf"{base_path}\DefaultIcon", "", f'"{launcher}",0')
    _set(rf"{base_path}\shell\open\command", "", cmd)

    # ``Applications\<exe>`` lets us attach a FriendlyAppName that Chrome /
    # Edge show in the "Open in another app" prompt. Without this the
    # prompt falls back to the exe's internal FileDescription ("Python").
    apps_key = rf"Software\Classes\Applications\{launcher.name}"
    for sub in (apps_key, rf"{apps_key}\shell\open\command"):
        key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, sub, 0, winreg.KEY_SET_VALUE)
        winreg.CloseKey(key)
    _set(apps_key, "FriendlyAppName", _APP_NAME)
    _set(apps_key, "ApplicationName", _APP_NAME)
    _set(rf"{apps_key}\shell\open\command", "", cmd)

    print(f"[url-scheme] Registered '{_SCHEME}://' → {cmd}")


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------


def _register_linux(python_exe: Path) -> None:
    apps_dir = Path.home() / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = apps_dir / "unitport.desktop"
    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=UnitPort\n"
        "Comment=UnitPort Robot Visual Programming Platform\n"
        f"Exec={python_exe} {_MAIN_PY} %u\n"
        "Terminal=false\n"
        f"MimeType=x-scheme-handler/{_SCHEME};\n"
        "NoDisplay=true\n"
    )
    desktop_path.write_text(content, encoding="utf-8")
    try:
        os.chmod(desktop_path, 0o755)
    except OSError:
        pass
    print(f"[url-scheme] Wrote {desktop_path}")

    # Register as the default handler for the scheme.
    try:
        subprocess.run(
            ["xdg-mime", "default", "unitport.desktop", f"x-scheme-handler/{_SCHEME}"],
            check=False,
        )
        subprocess.run(
            ["update-desktop-database", str(apps_dir)],
            check=False,
        )
    except FileNotFoundError:
        # xdg-utils not installed — file is in place but won't be activated.
        print("[url-scheme] xdg-mime not found; install xdg-utils for auto-handler.")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    # On Windows prefer pythonw.exe so the URL-scheme launch doesn't flash
    # a console window. On Linux the shebang / venv bin/python is fine.
    python_exe = _venv_python(prefer_windowed=True)
    system = platform.system()
    if system == "Windows":
        try:
            _register_windows(python_exe)
        except Exception as exc:  # noqa: BLE001
            print(f"[url-scheme] Windows registration failed: {exc}", file=sys.stderr)
            return 1
    elif system == "Linux":
        try:
            _register_linux(python_exe)
        except Exception as exc:  # noqa: BLE001
            print(f"[url-scheme] Linux registration failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            f"[url-scheme] Platform {system!r} not supported — "
            "register the unitport:// scheme manually (see bootstrap/register_url_scheme.py).",
            file=sys.stderr,
        )
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
