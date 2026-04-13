@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: UnitPort install.bat
:: Strategy: project-local .venv311 (Python 3.11)
::
:: Creates a project-local virtual environment at .venv311\
:: and installs all dependencies into it.
:: Never installs packages into global Python.
::
:: This project always installs into and runs from .venv311.
:: runtime\python\python.exe is ignored for dependency installation.
:: ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "VENV_DIR=%PROJECT_ROOT%\.venv311"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "WHEELS_DIR=%PROJECT_ROOT%\runtime\wheels"
set "REQUIREMENTS=%PROJECT_ROOT%\requirements.txt"
set "ENV_DIR=%PROJECT_ROOT%\runtime\env"
set "INSTALL_STATE=%ENV_DIR%\install_state.json"

echo [install] UnitPort environment setup
echo [install] Project root: %PROJECT_ROOT%

:: -------------------------------------------------------
:: Step 1: resolve Python 3.11 for env creation
:: -------------------------------------------------------
:: Strategy (in order, most reliable first):
::   1) py -3.11 launcher (official Windows multi-version tool)
::   2) Common install locations (per-user, all-users, C:\Python311)
::   3) PATH commands: python, python3.11, python3

set "INSTALL_MODE=venv311"
set "BASE_PYTHON="

:: 1) py launcher
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do (
            set "BASE_PYTHON=%%p"
            echo [install] Mode: project-local .venv311 via py -3.11 launcher
            goto :base_python_found
        )
    )
)

:: 2) Common install locations
for %%P in (
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
    "C:\Python311\python.exe"
    "C:\Python\Python311\python.exe"
) do (
    if exist %%P (
        set "BASE_PYTHON=%%~P"
        echo [install] Mode: project-local .venv311 via %%~P
        goto :base_python_found
    )
)

:: 3) PATH commands
for %%C in (python python3.11 python3) do (
    where %%C >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('where %%C 2^>nul') do (
            "%%p" -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
            if not errorlevel 1 (
                set "BASE_PYTHON=%%p"
                echo [install] Mode: project-local .venv311 via %%C in PATH
                goto :base_python_found
            )
        )
    )
)

:: 4) Conda environments (base + envs) - scan common install roots
set "CONDA_PY="
set "CONDA_NAME="
for %%R in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "%ProgramData%\miniconda3"
    "%ProgramData%\anaconda3"
    "%ProgramData%\miniforge3"
) do (
    if not defined CONDA_PY (
        if exist "%%~R\python.exe" (
            "%%~R\python.exe" -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
            if not errorlevel 1 (
                set "CONDA_PY=%%~R\python.exe"
                set "CONDA_NAME=%%~nxR-base"
            )
        )
    )
    if not defined CONDA_PY (
        if exist "%%~R\envs" (
            for /d %%E in ("%%~R\envs\*") do (
                if not defined CONDA_PY (
                    if exist "%%~E\python.exe" (
                        "%%~E\python.exe" -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
                        if not errorlevel 1 (
                            set "CONDA_PY=%%~E\python.exe"
                            set "CONDA_NAME=%%~nxE"
                        )
                    )
                )
            )
        )
    )
)

if defined CONDA_PY (
    echo.
    echo [install] Found conda environment "!CONDA_NAME!" with Python 3.11:
    echo [install]   !CONDA_PY!
    set /p "USE_CONDA=Use this conda environment to create .venv311? [y/N]: "
    if /i "!USE_CONDA!"=="y" (
        set "BASE_PYTHON=!CONDA_PY!"
        echo [install] Mode: project-local .venv311 via conda env "!CONDA_NAME!"
        goto :base_python_found
    )
    echo [install] Skipping conda environment.
)

:: 5) Offer to auto-download Python 3.11.9 (official installer, per-user, no admin)
echo.
echo [install] Python 3.11 not found on this system.
set /p "DL_CONFIRM=Download and install Python 3.11.9 now? (~25 MB, per-user, no admin required) [y/N]: "
if /i not "!DL_CONFIRM!"=="y" goto :install_failed

set "DL_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "DL_DEST=%TEMP%\unitport-python-3.11.9-amd64.exe"
echo [install] Downloading %DL_URL% ...

where curl >nul 2>&1
if not errorlevel 1 (
    curl -fL --retry 3 -o "%DL_DEST%" "%DL_URL%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%DL_URL%' -OutFile '%DL_DEST%' -UseBasicParsing } catch { Write-Host $_.Exception.Message; exit 1 }"
)
if errorlevel 1 (
    echo [ERROR] Download failed. Check network connectivity.
    goto :install_failed
)
if not exist "%DL_DEST%" (
    echo [ERROR] Downloaded file missing: %DL_DEST%
    goto :install_failed
)

echo [install] Running Python 3.11.9 silent installer (per-user) ...
"%DL_DEST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 AssociateFiles=0 Shortcuts=0
set "DL_RC=!errorlevel!"
del "%DL_DEST%" >nul 2>&1
if not "!DL_RC!"=="0" (
    echo [ERROR] Python installer exited with code !DL_RC!
    goto :install_failed
)

set "NEW_PY=%LocalAppData%\Programs\Python\Python311\python.exe"
if not exist "!NEW_PY!" (
    echo [ERROR] Install finished but python.exe not found at !NEW_PY!
    goto :install_failed
)
set "BASE_PYTHON=!NEW_PY!"
echo [install] Python 3.11.9 installed successfully.
echo [install] Mode: project-local .venv311 via freshly installed Python 3.11.9
goto :base_python_found

:install_failed
echo.
echo [ERROR] Python 3.11 not available.
echo [ERROR] UnitPort requires Python 3.11 specifically (not 3.10, 3.12, or 3.13).
echo [ERROR]
echo [ERROR] Install manually from:
echo [ERROR]   - https://www.python.org/downloads/release/python-3119/
echo [ERROR]   - Microsoft Store: search "Python 3.11"
echo [ERROR]
echo [ERROR] During installation, check "Add python.exe to PATH" and "py launcher".
echo [ERROR] Then reopen the terminal and run install.bat again.
exit /b 1

:base_python_found
set "PY_VER=unknown"
for /f "tokens=2" %%v in ('"%BASE_PYTHON%" --version 2^>^&1') do set "PY_VER=%%v"
echo [install] Base Python: %PY_VER%

:: -------------------------------------------------------
:: Step 2: create or verify .venv311
:: -------------------------------------------------------

if exist "%VENV_PYTHON%" (
    for /f "tokens=*" %%v in ('"%VENV_PYTHON%" --version 2^>^&1') do echo [install] Existing .venv311: %%v
) else (
    echo [install] Creating .venv311 ...
    "%BASE_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv311
        exit /b 1
    )
    echo [install] .venv311 created.
)

set "TARGET_PYTHON=%VENV_PYTHON%"

:packages

:: -------------------------------------------------------
:: Step 3: upgrade pip
:: -------------------------------------------------------

echo [install] Upgrading pip ...
"%TARGET_PYTHON%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 (
    echo [WARNING] pip upgrade failed. Continuing with existing version.
)

:: -------------------------------------------------------
:: Step 4a: install PyTorch with CUDA (if NVIDIA GPU present)
:: -------------------------------------------------------
:: PyPI only hosts CPU-only torch builds.  The CUDA builds live on
:: PyTorch's own index.  We detect NVIDIA hardware first and install
:: the CUDA wheel so that the plain "torch>=2.0.0" line in
:: requirements.txt is already satisfied when Step 4b runs.

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [install] NVIDIA GPU detected -- installing PyTorch with CUDA ...
    "%TARGET_PYTHON%" -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124 --quiet
    if errorlevel 1 (
        echo [WARNING] CUDA PyTorch install failed; will fall back to CPU version.
    ) else (
        echo [install] PyTorch CUDA installed.
    )
) else (
    echo [install] No NVIDIA GPU detected -- PyTorch CPU will be installed.
)

:: -------------------------------------------------------
:: Step 4b: install packages
:: -------------------------------------------------------

if not exist "%REQUIREMENTS%" (
    echo [ERROR] requirements.txt not found: %REQUIREMENTS%
    exit /b 1
)

:: Count available wheels
set "WHEEL_COUNT=0"
if exist "%WHEELS_DIR%" (
    for %%f in ("%WHEELS_DIR%\*.whl") do set /a WHEEL_COUNT+=1
)

if %WHEEL_COUNT% GTR 0 (
    echo [install] Installing from local wheelhouse (%WHEEL_COUNT% wheels^) ...
    "%TARGET_PYTHON%" -m pip install --no-index --find-links="%WHEELS_DIR%" -r "%REQUIREMENTS%"
    if errorlevel 1 (
        echo [install] Offline install incomplete, retrying with network ...
        "%TARGET_PYTHON%" -m pip install -r "%REQUIREMENTS%"
        if errorlevel 1 ( echo [ERROR] Package install failed & exit /b 1 )
    )
) else (
    echo [install] Installing from network (no local wheels found^) ...
    "%TARGET_PYTHON%" -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 ( echo [ERROR] Package install failed & exit /b 1 )
)

:: -------------------------------------------------------
:: Step 5: clone loco-mujoco if missing
:: -------------------------------------------------------

if not exist "%PROJECT_ROOT%\custom_mods\motions\loco-mujoco\.git" (
    echo [install] Cloning loco-mujoco reference motion library ...
    git clone --depth=1 --progress "https://github.com/robfiras/loco-mujoco.git" "%PROJECT_ROOT%\custom_mods\motions\loco-mujoco"
    if errorlevel 1 (
        echo [WARNING] loco-mujoco clone failed (optional, reference motions unavailable^).
    ) else (
        echo [install] loco-mujoco cloned.
    )
) else (
    echo [install] loco-mujoco already present.
)

:: -------------------------------------------------------
:: Step 6: verify critical imports
:: -------------------------------------------------------

echo [install] Verifying imports ...
"%TARGET_PYTHON%" -c "import PySide6; print('[install] PySide6 OK:', PySide6.__version__)" 2>nul
if errorlevel 1 echo [WARNING] PySide6 import failed.

"%TARGET_PYTHON%" -c "import torch; cuda='CUDA '+torch.version.cuda if torch.cuda.is_available() else 'CPU only'; print('[install] PyTorch OK:', torch.__version__, cuda)" 2>nul
if errorlevel 1 echo [WARNING] PyTorch import failed.

"%TARGET_PYTHON%" -c "import mujoco; print('[install] MuJoCo  OK:', mujoco.__version__)" 2>nul
if errorlevel 1 echo [WARNING] MuJoCo import failed (optional).

"%TARGET_PYTHON%" -c "import cyclonedds; print('[install] CycloneDDS OK:', cyclonedds.__file__)" 2>nul
if errorlevel 1 (
    echo [WARNING] CycloneDDS import failed. Attempting reinstall ...
    "%TARGET_PYTHON%" -m pip install --only-binary :all: "cyclonedds>=0.10.2" --quiet
    "%TARGET_PYTHON%" -c "import cyclonedds; print('[install] CycloneDDS OK (retry):', cyclonedds.__file__)" 2>nul
    if errorlevel 1 echo [WARNING] CycloneDDS still unavailable. Real-robot communication may not work.
)

:: -------------------------------------------------------
:: Step 7: write install_state.json (valid JSON, batch echo)
:: -------------------------------------------------------

if not exist "%ENV_DIR%" mkdir "%ENV_DIR%"

:: Pre-compute boolean (avoids > operator in scripts)
set "WHEELS_BOOL=false"
if %WHEEL_COUNT% GTR 0 set "WHEELS_BOOL=true"

:: Pre-compute Python bool literal (avoids > operator inside -c string)
set "WHEELS_BOOL=False"
if %WHEEL_COUNT% GTR 0 set "WHEELS_BOOL=True"

:: Single-line Python -c: outer double-quotes, inner single-quotes only -- no redirect issue
"%TARGET_PYTHON%" -c "import json,datetime; from pathlib import Path; s={'installed':True,'install_timestamp':datetime.datetime.utcnow().isoformat()+'Z','install_mode':'venv311','python_version':'%PY_VER%','runtime_python_verified':False,'cyclonedds_verified':False,'wheels_installed':%WHEELS_BOOL%,'notes':'Written by install.bat'}; Path(r'%INSTALL_STATE%').write_text(json.dumps(s,indent=2),encoding='utf-8')"
echo [install] install_state.json written.

:: -------------------------------------------------------
:: Summary
:: -------------------------------------------------------

echo.
echo [install] Installation complete.
echo [install] Mode   : venv311
echo [install] Python : %VENV_PYTHON%
echo [install] Launch : start.bat

endlocal
