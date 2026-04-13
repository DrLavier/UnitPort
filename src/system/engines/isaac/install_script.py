#!/usr/bin/env python3
"""Cross-platform Isaac Lab + Isaac Sim installer.

Called by UnitPort's setup wizard (via IsaacLabInstallThread) or directly
from the command line.

Outputs structured progress markers on stdout so the UI can parse them.
See ``src/system/engines/base.py`` for the marker protocol.

Usage:
    python -m src.system.engines.isaac.install_script [--install-dir DIR] [--python PYTHON_CMD]

    # Or directly:
    python src/system/engines/isaac/install_script.py --install-dir ~/UnitPort/engines/isaac
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Pinned versions ─────────────────────────────────────────────────────────
# Must stay in sync with deploy/install.sh (the original Linux-only script).
ISAACLAB_REPO = "https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT = "f4aa17f87e2e5db5484f0b5974918573e8918ce2"
ISAACSIM_VERSION = "5.1.0.0"
TORCH_VERSION = "2.7.0"
TORCHVISION_VERSION = "0.22.0"
CUDA_TAG = "cu128"

TOTAL_STEPS = 10

IS_WINDOWS = platform.system() == "Windows"

# Default install directory — matches engines/base.py convention
_DEFAULT_DIR = str(Path.home() / "UnitPort" / "engines" / "isaac")


# ── Structured output helpers ────────────────────────────────────────────────

def emit(tag: str, msg: str = "") -> None:
    line = f"[{tag}] {msg}" if msg else f"[{tag}]"
    print(line, flush=True)


def step(n: int, desc: str) -> None:
    emit(f"STEP {n}/{TOTAL_STEPS}", desc)


def run(cmd: list[str], *, cwd: str | None = None,
        env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, streaming output line-by-line."""
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=cwd, env=merged_env, text=True, errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip()
        if stripped:
            print(f"  {stripped}", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        emit("ERROR", f"Command failed (exit {proc.returncode}): {' '.join(cmd[:4])}")
        sys.exit(1)
    return subprocess.CompletedProcess(cmd, proc.returncode)


# ── Python 3.11 discovery ───────────────────────────────────────────────────

def find_python311(hint: str = "") -> str:
    """Locate a Python 3.11 interpreter.  *hint* is tried first."""
    def is_311(exe: str) -> bool:
        try:
            out = subprocess.check_output(
                [exe, "-c",
                 "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            ).strip()
            return out == "3.11"
        except Exception:
            return False

    if hint and is_311(hint):
        return hint

    candidates: list[str] = []
    if IS_WINDOWS:
        candidates += ["py -3.11", "python"]
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates.append(
                str(Path(local) / "Programs/Python/Python311/python.exe"))
        candidates += [
            r"C:\Python311\python.exe",
            r"C:\Program Files\Python311\python.exe",
        ]
    else:
        candidates += ["python3.11", "python3", "python"]
        candidates += [
            "/usr/bin/python3.11",
            "/usr/local/bin/python3.11",
        ]

    for c in candidates:
        parts = c.split()  # handle "py -3.11"
        exe = parts[0]
        if shutil.which(exe) or Path(parts[-1]).exists():
            if len(parts) == 1 and is_311(c):
                return c
            if len(parts) > 1:
                try:
                    out = subprocess.check_output(
                        parts + ["-c",
                                 "import sys; print(f'{sys.version_info.major}."
                                 "f{sys.version_info.minor}')"],
                        text=True, stderr=subprocess.DEVNULL, timeout=10,
                    ).strip()
                    if out == "3.11":
                        return c
                except Exception:
                    pass

    emit("ERROR", "Python 3.11 not found. Install it first.")
    sys.exit(1)


# ── Main installer ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install Isaac Lab + Isaac Sim into a dedicated directory")
    parser.add_argument("--install-dir", default=_DEFAULT_DIR,
                        help=f"Installation directory (default: {_DEFAULT_DIR})")
    parser.add_argument("--python", default="",
                        help="Path to Python 3.11 interpreter (auto-detected)")
    args = parser.parse_args()

    install_dir = Path(args.install_dir).resolve()
    emit("OK", f"Install directory: {install_dir}")

    # ── Step 1: Preflight ────────────────────────────────────────────────
    step(1, "Preflight checks")

    python_cmd = find_python311(args.python)
    emit("OK", f"Python 3.11: {python_cmd}")

    if not shutil.which("git"):
        emit("ERROR", "git not found. Please install git first.")
        sys.exit(1)
    emit("OK", "git available")

    has_gpu = shutil.which("nvidia-smi") is not None
    if has_gpu:
        try:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            ).strip().split("\n")[0]
            emit("OK", f"GPU: {gpu_name}")
        except Exception:
            emit("WARN", "nvidia-smi found but GPU query failed")
    else:
        emit("WARN", "No NVIDIA GPU detected -- CUDA may not work")

    # ── Step 2: Directory structure ──────────────────────────────────────
    step(2, "Create directory structure")
    install_dir.mkdir(parents=True, exist_ok=True)
    emit("OK", f"Directory ready: {install_dir}")

    # ── Step 3: Clone IsaacLab ───────────────────────────────────────────
    step(3, "Clone IsaacLab (pinned commit)")
    isaac_lab_dir = install_dir / "IsaacLab"

    if (isaac_lab_dir / ".git").is_dir():
        emit("OK", "IsaacLab already cloned, checking out pinned commit...")
        run(["git", "fetch", "origin"], cwd=str(isaac_lab_dir))
        run(["git", "checkout", ISAACLAB_COMMIT], cwd=str(isaac_lab_dir))
    else:
        run(["git", "clone", ISAACLAB_REPO, str(isaac_lab_dir)])
        run(["git", "checkout", ISAACLAB_COMMIT], cwd=str(isaac_lab_dir))
    emit("OK", f"IsaacLab at commit: {ISAACLAB_COMMIT[:12]}")

    # ── Step 4: Create venv ──────────────────────────────────────────────
    step(4, "Create Python 3.11 virtual environment")

    if IS_WINDOWS:
        venv_python = str(install_dir / "venv" / "Scripts" / "python.exe")
    else:
        venv_python = str(install_dir / "venv" / "bin" / "python")

    if Path(venv_python).exists():
        emit("OK", "venv already exists, skipping creation")
    else:
        py_parts = python_cmd.split()
        run([*py_parts, "-m", "venv", str(install_dir / "venv")])
        emit("OK", f"venv created at {install_dir / 'venv'}")

    run([venv_python, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"])
    emit("OK", "pip upgraded")

    # ── Step 5: Install Isaac Sim ────────────────────────────────────────
    step(5, f"Install Isaac Sim {ISAACSIM_VERSION} (largest download, ~12 GB)")
    emit("PROGRESS", "0")

    run([venv_python, "-m", "pip", "install",
         f"isaacsim[all,extscache]=={ISAACSIM_VERSION}",
         "--extra-index-url", "https://pypi.nvidia.com"])
    emit("OK", "Isaac Sim installed")

    # ── Step 6: Install IsaacLab packages ────────────────────────────────
    step(6, "Install IsaacLab packages (editable, --no-deps)")

    pip_args = [venv_python, "-m", "pip", "install", "--no-deps"]
    for pkg_name in ("isaaclab", "isaaclab_assets", "isaaclab_rl",
                      "isaaclab_tasks", "isaaclab_mimic", "isaaclab_contrib"):
        pip_args.extend(["-e", str(isaac_lab_dir / "source" / pkg_name)])
    run(pip_args)
    emit("OK", "6 IsaacLab packages installed in editable mode")

    # ── Step 7: Install PyTorch ──────────────────────────────────────────
    step(7, f"Install PyTorch {TORCH_VERSION}+{CUDA_TAG} (--no-deps)")

    run([venv_python, "-m", "pip", "install", "--no-deps",
         f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
         "--index-url", f"https://download.pytorch.org/whl/{CUDA_TAG}"])
    emit("OK", "PyTorch installed")

    try:
        cuda_check = subprocess.check_output(
            [venv_python, "-c",
             "import torch; print(f'torch {torch.__version__}, "
             "CUDA: {torch.cuda.is_available()}')"],
            text=True, stderr=subprocess.DEVNULL, timeout=30,
        ).strip()
        emit("OK", cuda_check)
    except Exception:
        emit("WARN", "torch CUDA verification skipped")

    # ── Step 8: Install pinned dependencies ──────────────────────────────
    step(8, "Install IsaacLab dependencies (with isaacsim pin locks)")

    run([venv_python, "-m", "pip", "install",
         "numpy==1.26.0", "Pillow==11.3.0", "filelock==3.13.1",
         "fsspec==2024.6.1", "MarkupSafe==2.1.3", "networkx==3.3",
         "sympy==1.13.3", "typing_extensions==4.12.2", "starlette==0.45.3",
         "onnx>=1.18.0", "prettytable==3.3.0", "hidapi==0.14.0.post2",
         "gymnasium==1.2.1", "pyglet<2", "transformers==4.57.6",
         "einops", "warp-lang", "pytest", "pytest-mock", "junitparser",
         "flatdict==4.0.0", "flaky", "protobuf>=4.25.8,!=5.26.0",
         "hydra-core", "h5py==3.13.0", "tensorboard", "moviepy",
         "ipywidgets==8.1.5", "tomli"])
    emit("OK", "IsaacLab dependencies installed")

    run([venv_python, "-m", "pip", "install", "--no-deps",
         "rsl-rl-lib==5.0.1", "onnxscript>=0.5", "tensordict",
         "GitPython", "gitdb", "smmap", "onnx_ir",
         "importlib_metadata", "orjson", "pyvers"])
    emit("OK", "RSL-RL extras installed")

    # ── Step 9: Create compatibility stubs ───────────────────────────────
    step(9, "Create compatibility stubs")

    # sitecustomize.py (h5py preload)
    site_pkg_cmd = [venv_python, "-c",
                    "import site; print(site.getsitepackages()[0])"]
    site_pkg = subprocess.check_output(site_pkg_cmd, text=True).strip()
    sitecustomize = Path(site_pkg) / "sitecustomize.py"
    sitecustomize.write_text(
        "# Preloads h5py before SimulationApp hijacks the loader.\n"
        "try:\n"
        "    import h5py  # noqa: F401\n"
        "except ImportError:\n"
        "    pass\n",
        encoding="utf-8",
    )
    emit("OK", f"sitecustomize.py written to {site_pkg}")

    # _isaac_sim/python.sh or python.bat stub for _find_isaac_python()
    stub_dir = isaac_lab_dir / "_isaac_sim"
    stub_dir.mkdir(parents=True, exist_ok=True)

    if IS_WINDOWS:
        stub_path = stub_dir / "python.bat"
        venv_scripts = install_dir / "venv" / "Scripts"
        stub_path.write_text(
            "@echo off\r\n"
            f'set "PATH={venv_scripts};%PATH%"\r\n'
            f'"{venv_scripts / "python.exe"}" %*\r\n',
            encoding="utf-8",
        )
        emit("OK", f"python.bat stub created")
    else:
        stub_path = stub_dir / "python.sh"
        venv_bin = install_dir / "venv" / "bin"
        stub_path.write_text(
            "#!/usr/bin/env bash\n"
            f'export PATH="{venv_bin}:$PATH"\n'
            f'exec "{venv_bin / "python"}" "$@"\n',
            encoding="utf-8",
        )
        stub_path.chmod(0o755)
        emit("OK", f"python.sh stub created")

    # env activation script
    if IS_WINDOWS:
        env_script = install_dir / "env.bat"
        venv_activate = install_dir / "venv" / "Scripts" / "activate.bat"
        env_script.write_text(
            "@echo off\r\n"
            "set OMNI_KIT_ACCEPT_EULA=YES\r\n"
            f'set ISAAC_LAB_PATH={isaac_lab_dir}\r\n'
            f'call "{venv_activate}"\r\n',
            encoding="utf-8",
        )
        emit("OK", "env.bat created")
    else:
        env_script = install_dir / "env.sh"
        env_script.write_text(
            "#!/usr/bin/env bash\n"
            "export OMNI_KIT_ACCEPT_EULA=YES\n"
            f'export ISAAC_LAB_PATH="{isaac_lab_dir}"\n'
            f'source "{install_dir / "venv" / "bin" / "activate"}"\n',
            encoding="utf-8",
        )
        env_script.chmod(0o755)
        emit("OK", "env.sh created")

    # ── Step 10: Verification ────────────────────────────────────────────
    step(10, "Verification")

    verify_script = (
        "import sys\n"
        "print(f'Python: {sys.version}')\n"
        "import torch\n"
        "print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\n"
        "import h5py\n"
        "print(f'h5py: {h5py.__version__}')\n"
        "import gymnasium\n"
        "print(f'gymnasium: {gymnasium.__version__}')\n"
        "try:\n"
        "    import isaaclab\n"
        "    print(f'isaaclab: {isaaclab.__version__}')\n"
        "except Exception as e:\n"
        "    print(f'isaaclab: import warning ({e})')\n"
        "print('All imports OK')\n"
    )
    result = run([venv_python, "-c", verify_script], check=False)
    if result.returncode != 0:
        emit("WARN", "Some imports failed -- check GPU driver and display settings")
    else:
        emit("OK", "All verification imports passed")

    # ── Done ─────────────────────────────────────────────────────────────
    emit("ISAAC_LAB_ROOT", str(isaac_lab_dir))
    emit("DONE")


if __name__ == "__main__":
    main()
