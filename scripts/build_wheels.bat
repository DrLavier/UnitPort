@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: UnitPort - scripts/build_wheels.bat
:: MAINTAINER / CI USE ONLY
::
:: Pre-downloads all Python package wheels for offline install.
:: Output: runtime/wheels/*.whl
::
:: Run AFTER scripts/build_runtime.bat so packaged Python is ready.
:: If packaged Python is absent, falls back to any Python 3.11 on PATH.
::
:: Usage:
::   cd <project-root>
::   scripts\build_wheels.bat
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

set "RUNTIME_DIR=%PROJECT_ROOT%\runtime"
set "PYTHON_DIR=%RUNTIME_DIR%\python"
set "WHEELS_DIR=%RUNTIME_DIR%\wheels"
set "ENV_DIR=%RUNTIME_DIR%\env"
set "REQUIREMENTS=%PROJECT_ROOT%\requirements.txt"

:: -------------------------------------------------------
:: Resolve Python
:: -------------------------------------------------------

if exist "%PYTHON_DIR%\python.exe" (
    set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
    echo [wheels] Using packaged Python: %PYTHON_DIR%\python.exe
) else (
    echo [wheels] Packaged Python not found. Falling back to system Python.
    echo [wheels] Wheels should be downloaded with a Python version matching the target runtime.
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No Python found. Run scripts\build_runtime.bat first.
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

for /f "tokens=2" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do echo [wheels] Python: %%v

:: -------------------------------------------------------
:: Verify requirements.txt
:: -------------------------------------------------------

if not exist "%REQUIREMENTS%" (
    echo [ERROR] requirements.txt not found at %REQUIREMENTS%
    exit /b 1
)

:: -------------------------------------------------------
:: Create wheels dir
:: -------------------------------------------------------

if not exist "%WHEELS_DIR%" mkdir "%WHEELS_DIR%"

:: -------------------------------------------------------
:: Download wheels
:: -------------------------------------------------------

echo [wheels] Downloading wheels to runtime\wheels\ ...
echo [wheels] Requirements: %REQUIREMENTS%

"%PYTHON_EXE%" -m pip download ^
    --dest "%WHEELS_DIR%" ^
    --platform win_amd64 ^
    --python-version 3.11 ^
    --only-binary=:all: ^
    -r "%REQUIREMENTS%"

if errorlevel 1 (
    echo.
    echo [wheels] Binary-only download failed for some packages.
    echo [wheels] Retrying without --only-binary to allow source packages...
    "%PYTHON_EXE%" -m pip download --dest "%WHEELS_DIR%" -r "%REQUIREMENTS%"
    if errorlevel 1 (
        echo [ERROR] Wheel download failed.
        exit /b 1
    )
)

:: -------------------------------------------------------
:: Count wheels
:: -------------------------------------------------------

set "WHEEL_COUNT=0"
for %%f in ("%WHEELS_DIR%\*.whl") do set /a WHEEL_COUNT+=1
set "TAR_COUNT=0"
for %%f in ("%WHEELS_DIR%\*.tar.gz") do set /a TAR_COUNT+=1

echo.
echo [wheels] Downloaded: %WHEEL_COUNT% wheel(s), %TAR_COUNT% sdist(s)

:: -------------------------------------------------------
:: Update manifest via PowerShell
:: -------------------------------------------------------

set "MANIFEST=%ENV_DIR%\manifest.json"

if exist "%MANIFEST%" (
    powershell -NoProfile -Command ^
      "try {" ^
      "  $m = Get-Content '%MANIFEST%' -Raw | ConvertFrom-Json;" ^
      "  $wh = Get-ChildItem '%WHEELS_DIR%' -Filter '*.whl' | Select-Object -ExpandProperty Name | Sort-Object;" ^
      "  $m | Add-Member -Force -NotePropertyName wheel_packages -NotePropertyValue $wh;" ^
      "  $m | Add-Member -Force -NotePropertyName wheels_count -NotePropertyValue $wh.Count;" ^
      "  ($m | ConvertTo-Json -Depth 5) | Set-Content -Encoding UTF8 '%MANIFEST%';" ^
      "  Write-Host '[wheels] manifest.json updated with' $wh.Count 'packages.'" ^
      "} catch { Write-Host '[wheels] manifest update skipped:' $_.Exception.Message }"
) else (
    echo [wheels] manifest.json not found  run build_runtime.bat first.
)

:: -------------------------------------------------------
:: Summary
:: -------------------------------------------------------

echo.
echo ============================================================
echo Wheels build complete.
echo   runtime\wheels\   %WHEEL_COUNT% wheel(s) ready for offline install
echo.
echo Verify with:
echo   runtime\python\python.exe scripts\verify_runtime.py
echo ============================================================

endlocal
