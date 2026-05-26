REM SPDX-FileCopyrightText: 2026 SU CHANG
REM SPDX-License-Identifier: Apache-2.0

@echo off
setlocal EnableDelayedExpansion

:: -------------------------------------------------------
:: ASCII logo (always) + LICENSE panel (first install only)
:: First-install gate: %~dp0runtime\env\install_state.json absence.
:: reset.bat clears that file -> factory reset re-prompts the license.
::
:: UNITPORT_ASCII_PRINTED=1 means our caller (start.bat) already painted
:: the logo, so we skip it to avoid the double-print. Direct runs of
:: install.bat still see the logo because the var is not set.
:: -------------------------------------------------------
if not defined UNITPORT_ASCII_PRINTED call :print_ascii

if exist "%~dp0runtime\env\install_state.json" goto :license_done

call :print_license
echo.

:ask_license
set "LICENSE_ANSWER="
set /p "LICENSE_ANSWER=Have you read and agreed to the terms above? [Y/N]: "
if /i "!LICENSE_ANSWER!"=="Y" goto :license_done
if /i "!LICENSE_ANSWER!"=="N" (
    echo.
    echo [install] Cancelled. Run install.bat again to review the terms.
    endlocal
    exit /b 0
)
echo   Please enter Y or N.
echo.
goto :ask_license

:license_done
echo.

:: ============================================================
:: UnitPort install.bat
:: Strategy: project-local .venv311 (Python 3.11).
::
:: Creates a project-local virtual environment at .venv311\
:: and installs all dependencies into it.
:: Never installs packages into global Python.
::
:: This project always installs into and runs from .venv311.
:: ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "VENV_DIR=%PROJECT_ROOT%\.venv311"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "WHEELS_DIR=%PROJECT_ROOT%\runtime\wheels"
set "REQUIREMENTS=%PROJECT_ROOT%\requirements.txt"
set "ENV_DIR=%PROJECT_ROOT%\runtime\env"
set "INSTALL_STATE=%ENV_DIR%\install_state.json"

:: -------------------------------------------------------
:: Redirect pip temp + cache onto the project drive.
:: pip downloads GB-scale wheels (torch/CUDA) through %TEMP%; on systems
:: where the OS drive is small this fills C: and aborts install with
:: "No space left on device". Keep everything on the same drive as the
:: project so the available space is whatever this drive has.
:: Scoped to this setlocal block -- does not leak to the parent shell.
:: -------------------------------------------------------
set "PIP_TMP_DIR=%PROJECT_ROOT%\.pip-tmp"
set "PIP_CACHE=%PROJECT_ROOT%\.pip-cache"
if not exist "%PIP_TMP_DIR%" mkdir "%PIP_TMP_DIR%"
if not exist "%PIP_CACHE%" mkdir "%PIP_CACHE%"
set "TMP=%PIP_TMP_DIR%"
set "TEMP=%PIP_TMP_DIR%"
set "PIP_CACHE_DIR=%PIP_CACHE%"

echo [install] UnitPort RELEASE environment setup
echo [install] Project root: %PROJECT_ROOT%
echo [install] pip tmp     : %PIP_TMP_DIR%
echo [install] pip cache   : %PIP_CACHE%

:: -------------------------------------------------------
:: Step 1: resolve Python 3.11 for env creation
:: -------------------------------------------------------
:: Strategy (in order, most reliable first):
::   1) py -3.11 launcher (official Windows multi-version tool)
::   2) Common install locations (per-user, all-users, C:\Python311)
::   3) PATH commands: python, python3.11, python3
::   4) Conda environments
::   5) Optionally download Python 3.11.9 official installer

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

:: -------------------------------------------------------
:: Verify SHA-256 of the downloaded installer.
:: Plan finding P0-2: HTTPS alone is not enough -- a corporate root CA or
:: a poisoned resolver can serve a tampered Python installer that then
:: runs with the user's privileges. Constant below MUST be set to the
:: SHA-256 published on https://www.python.org/downloads/release/python-3119/
:: When bumping Python version, rotate this constant (see CLAUDE.md SOP).
:: -------------------------------------------------------
:: MAINTAINER: replace the value below with the official SHA-256 of
::   python-3.11.9-amd64.exe taken from python.org BEFORE shipping a release.
::   While empty, this script refuses to run the installer (fail-closed).
set "DL_SHA256=5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde"

if "!DL_SHA256!"=="" (
    echo [ERROR] DL_SHA256 is not set in install.bat -- refusing to run an unverified installer.
    echo [ERROR] Maintainer: fill in the SHA-256 from python.org and recommit.
    del "%DL_DEST%" >nul 2>&1
    goto :install_failed
)

set "DL_GOT="
for /f "delims=" %%H in ('powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "(Get-FileHash -Algorithm SHA256 -LiteralPath '%DL_DEST%').Hash.ToLower()" 2^>nul') do set "DL_GOT=%%H"

if "!DL_GOT!"=="" (
    echo [ERROR] Could not compute SHA-256 of the downloaded installer.
    echo [ERROR] PowerShell Get-FileHash failed; refusing to run the installer.
    del "%DL_DEST%" >nul 2>&1
    goto :install_failed
)

if /i not "!DL_GOT!"=="!DL_SHA256!" (
    echo [ERROR] SHA-256 mismatch -- the downloaded Python installer is not what we expected.
    echo [ERROR]   expected: !DL_SHA256!
    echo [ERROR]   got     : !DL_GOT!
    echo [ERROR] Refusing to execute. If you just bumped Python version, update DL_SHA256.
    del "%DL_DEST%" >nul 2>&1
    goto :install_failed
)
echo [install] SHA-256 verified.

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
:: Step 4: bootstrap install -- minimum to launch MainWindow
:: -------------------------------------------------------
:: All heavy provisioning (torch CUDA, requirements.txt, loco-mujoco,
:: URL scheme, ROS2) is deferred to the in-app ProvisioningTask which
:: streams pip stdout into the LoadingScreen. Bootstrap only needs the
:: minimum to draw a Qt window and route logs:
::   PyQt6      -- Qt platform
::   loguru     -- log sink consumed by unitport_sdk
::   cyclonedds -- hard requirement of idl_messages/builtin_interfaces.py
::                 (class base IdlStruct); MainWindow's widget chain pulls
::                 it in eagerly via adapters -> ros2 native bridge. Without
::                 it the LoadingScreen itself cannot paint.
:: start.bat's Gate 3 also probes ``unitport_sdk``; that is local source
:: under src/, no install needed.
::
:: --only-binary :all: on cyclonedds is non-negotiable: 0.10.x has no
:: Windows source-build path that survives without the CycloneDDS C
:: library + a C++17 toolchain. The flag makes pip refuse to build, so
:: pip selects the newest wheel that matches the current platform/Python
:: -- or fails with "No matching distribution" (which we surface here as
:: a hard error so the user can install a compatible Python).

echo [install] Installing bootstrap packages (PyQt6, loguru) ...
"%TARGET_PYTHON%" -m pip install --disable-pip-version-check PyQt6 loguru
if errorlevel 1 (
    echo [ERROR] Bootstrap install failed.
    exit /b 1
)

echo [install] Installing cyclonedds (--only-binary :all:; MainWindow import gate) ...
"%TARGET_PYTHON%" -m pip install --disable-pip-version-check --only-binary :all: --upgrade "cyclonedds>=0.10.2"
if errorlevel 1 (
    echo [ERROR] cyclonedds install failed.
    echo [ERROR] pip could not find a binary wheel for cyclonedds on this Python.
    echo [ERROR] MainWindow cannot import without it -- the app will not start.
    echo [ERROR] Verify you are on Python 3.11 ^(``%TARGET_PYTHON%`` --version^),
    echo [ERROR] or check https://pypi.org/project/cyclonedds/#files for a matching wheel.
    exit /b 1
)

:: -------------------------------------------------------
:: Step 5: write minimal install_state.json
:: -------------------------------------------------------
:: ProvisioningTask flips ``provisioning_pending`` to false and adds
:: torch_cuda / loco_mujoco / url_scheme facts after it runs.

if not exist "%ENV_DIR%" mkdir "%ENV_DIR%"

"%TARGET_PYTHON%" -c "import json,datetime; from pathlib import Path; s={'bootstrap':True,'provisioning_pending':True,'install_timestamp':datetime.datetime.utcnow().isoformat()+'Z','install_mode':'venv311','python_version':'%PY_VER%','notes':'Written by RELEASE/install.bat (bootstrap stage)'}; Path(r'%INSTALL_STATE%').write_text(json.dumps(s,indent=2),encoding='utf-8')"
echo [install] install_state.json written.

:: -------------------------------------------------------
:: Summary
:: -------------------------------------------------------

echo.
echo [install] Bootstrap complete. Heavy dependencies will install on first launch via LoadingScreen.
echo [install] Mode   : venv311
echo [install] Python : %VENV_PYTHON%
echo [install] Launch : start.bat

endlocal
exit /b 0

:: ============================================================
:: Subroutines: ASCII logo + LICENSE panel (120 chars wide).
:: Kept inline so install.bat works without external resources.
:: ============================================================

:print_ascii
echo(========================================================================================================================
echo(------------------------------------------------------------------------------------------------------------------------
echo(         ####
echo(     ####   ####                                          ####
echo( #####   ############      #####     #####                #####  #####   ############                            ####
echo(##    ##########    ##     #####     #####                       #####   #############                          #####
echo(##  #########       ##     #####     #####  ###########   ##### ######## #####    #####   #########   ####### #########
echo(##  ######     #### ##     #####     #####  ############  ##### ######## #####    ##### ############  ####### #########
echo(##  ######   ###### ##     #####     #####  ####    ####  #####  #####   #############  ####    ##### #####     #####
echo(##  ######   ###### ##     #####     #####  ####    ####  #####  #####   ###########    ####    ##### #####     #####
echo(##    ####   ###    ##     ######   ######  ####    ####  #####  #####   #####          #####   ##### #####     #####
echo(##                ###       #############   ####    ####  #####  ####### #####           ###########  #####     #######
echo(  #####      ######           ########      ####    ####  #####   ###### #####             #######    #####       #####
echo(     #####   ###
echo(         ####
echo(------------------------------------------------------------------------------------------------------------------------
echo(========================================================================================================================
goto :eof

:print_license
echo.
echo ========================================================================================================================
echo  ^| UnitPort Studio  -  License and Pre-install Notice                                                                   ^|
echo  ^|                                                                                                                      ^|
echo  ^| [LICENSE - Key Terms]                                                                                                ^|
echo  ^|   1. Data Security  : User data is processed and stored locally by default.                                          ^|
echo  ^|                       Cloud storage is hosted on Supabase, optional, and does not block normal usage.                ^|
echo  ^|   2. No Repackaging : Redistributing or commercially reselling this software (or derivatives) is prohibited.         ^|
echo  ^|   3. Copyright      : Protected under the EU Copyright Directive 2019/790. Violations will be prosecuted.            ^|
echo  ^|                                                                                                                      ^|
echo  ^| [Installation Notice]                                                                                                ^|
echo  ^|   First-time install may take 15 - 45 minutes, depending on selected components and network conditions.              ^|
echo  ^|   Components include Isaac Lab, loco-mujoco, and vendor SDKs.                                                        ^|
echo  ^|   Please keep the network stable and avoid power loss during installation.                                           ^|
echo ========================================================================================================================
goto :eof
