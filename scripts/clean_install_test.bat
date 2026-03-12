@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: UnitPort - scripts/clean_install_test.bat
:: LOCAL CLEAN INSTALL SIMULATION
::
:: Simulates a clean-machine install experience WITHOUT needing
:: a real isolated VM. Strips the current shell's Python and
:: CycloneDDS environment, then runs install.bat + verify.
::
:: What it does:
::   1. Unsets PYTHONHOME, PYTHONPATH, CYCLONEDDS_HOME
::   2. Removes known global Python/CycloneDDS paths from PATH
::   3. Runs install.bat (must succeed using packaged runtime only)
::   4. Runs verify_runtime.py (must pass all critical checks)
::
:: What it does NOT do:
::   - Actually uninstall Python from the machine
::   - Isolate registry / COM / system DLL state
::   For true clean-machine validation, use an actual VM (Phase 3 final).
::
:: Usage:
::   scripts\clean_install_test.bat
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

set "RUNTIME_PYTHON=%PROJECT_ROOT%\runtime\python\python.exe"

echo ============================================================
echo UnitPort Clean Install Test (local simulation)
echo Project root: %PROJECT_ROOT%
echo ============================================================
echo.

:: -- Preflight: packaged runtime must exist ---------------------------------

if not exist "%RUNTIME_PYTHON%" (
    echo [ERROR] Packaged Python not found at runtime\python\python.exe
    echo [ERROR] Run scripts\build_runtime.bat before running this test.
    exit /b 1
)

:: -- Strip global Python/CycloneDDS from this shell's environment -----------

echo [test] Stripping global Python and CycloneDDS from shell environment...

:: Remove env vars that would allow fallback to global dependencies
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONSTARTUP="
set "CYCLONEDDS_HOME="

:: Remove known global Python install paths from PATH
:: (Covers common per-user and system-wide locations)
call :strip_from_path "%LocalAppData%\Programs\Python\Python311\Scripts"
call :strip_from_path "%LocalAppData%\Programs\Python\Python311"
call :strip_from_path "%LocalAppData%\Programs\Python\Python310\Scripts"
call :strip_from_path "%LocalAppData%\Programs\Python\Python310"
call :strip_from_path "%LocalAppData%\Programs\Python\Python313\Scripts"
call :strip_from_path "%LocalAppData%\Programs\Python\Python313"
call :strip_from_path "C:\Python311\Scripts"
call :strip_from_path "C:\Python311"
call :strip_from_path "C:\Python310"
call :strip_from_path "C:\Python313"
call :strip_from_path "C:\CycloneDDS\bin"
call :strip_from_path "C:\CycloneDDS"

echo [test] Environment stripped.
echo [test] PYTHONHOME  = (unset)
echo [test] PYTHONPATH  = (unset)
echo [test] CYCLONEDDS_HOME = (unset)

:: Verify global python.exe is no longer resolvable
where python >nul 2>&1
if %errorlevel% EQU 0 (
    for /f "tokens=*" %%p in ('where python 2^>nul') do (
        echo [test] NOTE: python.exe still visible in PATH: %%p
        echo [test] This may come from a system install or conda.
        echo [test] Test continues  install.bat must still use packaged runtime only.
    )
) else (
    echo [test] Global python.exe not visible in PATH  clean simulation confirmed.
)

echo.

:: -- STEP 1: Run install.bat ------------------------------------------------

echo [test] STEP 1: Running install.bat...
echo --------------------------------------------------------

call "%PROJECT_ROOT%\install.bat"
set "INSTALL_RC=%errorlevel%"

echo --------------------------------------------------------
if %INSTALL_RC% NEQ 0 (
    echo [FAIL] install.bat exited with code %INSTALL_RC%
    exit /b 1
)
echo [PASS] install.bat completed successfully.
echo.

:: -- STEP 2: Run verify_runtime.py -----------------------------------------

echo [test] STEP 2: Running verify_runtime.py...
echo --------------------------------------------------------

"%RUNTIME_PYTHON%" "%PROJECT_ROOT%\scripts\verify_runtime.py"
set "VERIFY_RC=%errorlevel%"

echo --------------------------------------------------------
if %VERIFY_RC% NEQ 0 (
    echo [FAIL] verify_runtime.py reported critical failures.
    echo [FAIL] Review output above and fix before distribution.
    exit /b 1
)
echo [PASS] verify_runtime.py passed.
echo.

:: -- STEP 3: Smoke test  import project core ------------------------------

echo [test] STEP 3: Smoke test  project imports...
set "PYTHONPATH=%PROJECT_ROOT%"
"%RUNTIME_PYTHON%" -c ^"
import sys
sys.path.insert(0, r'%PROJECT_ROOT%')
from bin.core.config_manager import ConfigManager
from models.sdk_manager import verify_registered_sdks, configure_cyclonedds_env
print('[PASS] Core project imports OK')
^"
if errorlevel 1 (
    echo [FAIL] Core project import smoke test failed.
    exit /b 1
)

:: -- Summary ----------------------------------------------------------------

echo.
echo ============================================================
echo Clean install test PASSED.
echo The packaged runtime operates independently of global Python.
echo Ready for:
echo   - Folder distribution packaging: scripts\package_distribution.bat --zip
echo   - Real VM validation (Phase 3 final)
echo ============================================================
exit /b 0


:: -- Helper: strip a directory from PATH -----------------------------------
:strip_from_path
set "STRIP_TARGET=%~1"
set "NEW_PATH="
for %%d in ("%PATH:;=" "%") do (
    set "SEGMENT=%%~d"
    if /i not "!SEGMENT!"=="%STRIP_TARGET%" (
        if "!NEW_PATH!"=="" (
            set "NEW_PATH=!SEGMENT!"
        ) else (
            set "NEW_PATH=!NEW_PATH!;!SEGMENT!"
        )
    )
)
set "PATH=!NEW_PATH!"
exit /b 0
