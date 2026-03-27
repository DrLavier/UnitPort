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

:: Use system Python 3.11 only to create/repair .venv311.
set "SYS_PY311=%LocalAppData%\Programs\Python\Python311\python.exe"
if exist "!SYS_PY311!" (
    set "BASE_PYTHON=!SYS_PY311!"
    set "INSTALL_MODE=venv311"
    echo [install] Mode: project-local .venv311 via system Python 3.11
    goto :base_python_found
)

:: Try PATH
where python >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%p in ('where python 2^>nul') do (
        "%%p" -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
        if not errorlevel 1 (
            set "BASE_PYTHON=%%p"
            set "INSTALL_MODE=venv311"
            echo [install] Mode: project-local .venv311 via Python 3.11 in PATH
            goto :base_python_found
        )
    )
)

echo [ERROR] Python 3.11 not found.
echo [ERROR] Install Python 3.11 for the current user.
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
:: Step 3.5: install torch (GPU-aware)
::
:: torch is installed BEFORE requirements.txt so that pip can select
:: the correct CUDA or CPU build.  requirements.txt intentionally omits
:: torch and only documents the minimum version constraint.
:: -------------------------------------------------------

echo [install] Detecting GPU ...
set "TORCH_VARIANT=cpu"
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [install] NVIDIA GPU detected.
    set "TORCH_VARIANT=cu124"
) else (
    echo [install] No NVIDIA GPU found, using CPU torch.
)

:: Check whether torch is already installed in the venv
"%TARGET_PYTHON%" -c "import torch" >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%v in ('"%TARGET_PYTHON%" -c "import torch; print(torch.__version__)" 2^>nul') do (
        echo [install] torch already installed: %%v  ^(skipping torch download^)
        goto :torch_done
    )
)

if "!TORCH_VARIANT!"=="cu124" (
    echo [install] Installing torch with CUDA 12.4 support ...
    "%TARGET_PYTHON%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --quiet
    if errorlevel 1 (
        echo [WARNING] CUDA torch install failed, falling back to CPU torch ...
        "%TARGET_PYTHON%" -m pip install torch --quiet
        if errorlevel 1 ( echo [ERROR] torch install failed & exit /b 1 )
    )
) else (
    echo [install] Installing CPU torch ...
    "%TARGET_PYTHON%" -m pip install torch --quiet
    if errorlevel 1 ( echo [ERROR] torch install failed & exit /b 1 )
)

:torch_done

:: -------------------------------------------------------
:: Step 4: install packages
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
:: Step 5: verify critical imports
:: -------------------------------------------------------

echo [install] Verifying imports ...
"%TARGET_PYTHON%" -c "import PySide6; print('[install] PySide6           OK:', PySide6.__version__)" 2>nul
if errorlevel 1 echo [WARNING] PySide6 import failed.

"%TARGET_PYTHON%" -c "import mujoco; print('[install] MuJoCo            OK:', mujoco.__version__)" 2>nul
if errorlevel 1 echo [WARNING] MuJoCo import failed.

"%TARGET_PYTHON%" -c "import torch; v=torch.__version__; cuda=torch.cuda.is_available(); print('[install] torch             OK:', v, '| CUDA:', cuda)" 2>nul
if errorlevel 1 echo [WARNING] torch import failed.

"%TARGET_PYTHON%" -c "import stable_baselines3; print('[install] stable-baselines3 OK:', stable_baselines3.__version__)" 2>nul
if errorlevel 1 echo [WARNING] stable_baselines3 import failed.

:: -------------------------------------------------------
:: Step 6: write install_state.json (valid JSON, batch echo)
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
