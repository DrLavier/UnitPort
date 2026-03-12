@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: UnitPort - scripts/package_distribution.bat
:: MAINTAINER / CI USE ONLY
::
:: Produces a self-contained folder distribution:
::   dist_package\UnitPort-<version>\     (folder, ready to zip/share)
::   dist_package\UnitPort-<version>.zip  (optional, if --zip passed)
::
:: Prerequisites:
::   scripts\build_runtime.bat must have completed successfully.
::   scripts\build_wheels.bat  must have completed successfully.
::
:: Usage:
::   scripts\package_distribution.bat
::   scripts\package_distribution.bat --zip
::   scripts\package_distribution.bat --version 1.0.0-beta
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

:: -- Parse arguments --------------------------------------------------------

set "DO_ZIP=0"
set "VERSION=dev"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--zip" (
    set "DO_ZIP=1"
    shift & goto :parse_args
)
if /i "%~1"=="--version" (
    set "VERSION=%~2"
    shift & shift & goto :parse_args
)
shift & goto :parse_args
:args_done

set "PACKAGE_NAME=UnitPort-%VERSION%"
set "OUTPUT_ROOT=%PROJECT_ROOT%\dist_package"
set "PACKAGE_DIR=%OUTPUT_ROOT%\%PACKAGE_NAME%"
set "RUNTIME_DIR=%PROJECT_ROOT%\runtime"
set "PYTHON_EXE=%RUNTIME_DIR%\python\python.exe"

echo ============================================================
echo UnitPort Distribution Packager
echo Version      : %VERSION%
echo Package name : %PACKAGE_NAME%
echo Output dir   : %OUTPUT_ROOT%
echo ============================================================
echo.

:: -- Preflight: verify runtime is assembled ---------------------------------

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Packaged Python not found at runtime\python\python.exe
    echo [ERROR] Run scripts\build_runtime.bat first.
    exit /b 1
)

set "WHEEL_COUNT=0"
for %%f in ("%RUNTIME_DIR%\wheels\*.whl") do set /a WHEEL_COUNT+=1
if %WHEEL_COUNT% EQU 0 (
    echo [WARNING] No wheels found in runtime\wheels\
    echo [WARNING] Run scripts\build_wheels.bat first.
    echo [WARNING] Continuing  install.bat will fall back to network install.
    echo.
)

:: -- Run verification before packaging -------------------------------------

echo [package] Running pre-package verification...
"%PYTHON_EXE%" scripts\verify_runtime.py --json > "%OUTPUT_ROOT%\_verify_tmp.json" 2>nul
if errorlevel 1 (
    echo [WARNING] Verification reported failures. See details above.
    echo [WARNING] Packaging will continue  check the output carefully.
) else (
    echo [package] Verification passed.
)
if exist "%OUTPUT_ROOT%\_verify_tmp.json" del "%OUTPUT_ROOT%\_verify_tmp.json"

:: -- Create/clean output directory -----------------------------------------

if exist "%PACKAGE_DIR%" (
    echo [package] Removing previous package at %PACKAGE_DIR% ...
    rmdir /s /q "%PACKAGE_DIR%"
)
mkdir "%PACKAGE_DIR%"
echo [package] Output: %PACKAGE_DIR%

:: -- Copy project source ----------------------------------------------------

echo [package] Copying project source...

:: Use robocopy for filtering; exit codes 0,1 = success
robocopy "%PROJECT_ROOT%" "%PACKAGE_DIR%" /E /NP /NFL /NDL ^
    /XD ".git" ".venv311" "__pycache__" ".pytest_cache" ".cache" ^
         "dist_package" "runtime\_downloads" ^
         "__cache__" "scripts" ^
    /XF "*.pyc" "*.pyo" "*.pyd" "log.txt" "*.log" "*.tmp" "*.swp" "*.swo" ^
        ".gitignore" "AGENTS.md" "desktop.ini" ".DS_Store" ^
        "run_app.bat" "run_tests.bat" "setup_venv.bat"

set "RC=%errorlevel%"
if %RC% GTR 7 (
    echo [ERROR] robocopy failed with exit code %RC%
    exit /b 1
)

:: -- Copy runtime payload ---------------------------------------------------

echo [package] Copying runtime payload...

:: runtime/python/
echo [package]   runtime\python\ ...
robocopy "%RUNTIME_DIR%\python" "%PACKAGE_DIR%\runtime\python" /E /NP /NFL /NDL
set "RC=%errorlevel%"
if %RC% GTR 7 ( echo [ERROR] Failed to copy runtime\python\ & exit /b 1 )

:: runtime/cyclonedds/ (optional)
set "CDDS_READY=false"
for %%f in ("%RUNTIME_DIR%\cyclonedds\bin\*.dll") do set "CDDS_READY=true"

if "%CDDS_READY%"=="true" (
    echo [package]   runtime\cyclonedds\ ...
    robocopy "%RUNTIME_DIR%\cyclonedds" "%PACKAGE_DIR%\runtime\cyclonedds" /E /NP /NFL /NDL
    set "RC=!errorlevel!"
    if !RC! GTR 7 ( echo [ERROR] Failed to copy runtime\cyclonedds\ & exit /b 1 )
) else (
    echo [package]   runtime\cyclonedds\ not populated  Unitree SDK features will be unavailable.
)

:: runtime/wheels/
if %WHEEL_COUNT% GTR 0 (
    echo [package]   runtime\wheels\ (%WHEEL_COUNT% wheels^) ...
    robocopy "%RUNTIME_DIR%\wheels" "%PACKAGE_DIR%\runtime\wheels" /E /NP /NFL /NDL
    set "RC=!errorlevel!"
    if !RC! GTR 7 ( echo [ERROR] Failed to copy runtime\wheels\ & exit /b 1 )
)

:: runtime/env/ (manifest only  not machine-local state files)
if not exist "%PACKAGE_DIR%\runtime\env" mkdir "%PACKAGE_DIR%\runtime\env"
if exist "%RUNTIME_DIR%\env\manifest.json" (
    copy /y "%RUNTIME_DIR%\env\manifest.json" "%PACKAGE_DIR%\runtime\env\manifest.json" >nul
)

:: Write fresh placeholder state files for the recipient machine
> "%PACKAGE_DIR%\runtime\env\install_state.json" (
    echo {"installed": false, "install_timestamp": null, "runtime_python_verified": false, "cyclonedds_verified": false, "wheels_installed": false, "install_mode": null, "notes": "Run install.bat to populate."}
)
> "%PACKAGE_DIR%\runtime\env\bootstrap_state.json" (
    echo {"sdk_verified": false, "last_startup_check": null, "degraded_mode": false, "missing_optional_sdks": [], "notes": "Written by startup verification layer."}
)

:: -- Write package metadata -------------------------------------------------

set "PACK_TIMESTAMP="
for /f "tokens=*" %%t in ('"%PYTHON_EXE%" -c "import datetime; print(datetime.datetime.utcnow().isoformat())"') do set "PACK_TIMESTAMP=%%t"

> "%PACKAGE_DIR%\runtime\env\package_info.json" (
    echo {
    echo   "package_name": "%PACKAGE_NAME%",
    echo   "version": "%VERSION%",
    echo   "packaged_at": "%PACK_TIMESTAMP%",
    echo   "wheel_count": %WHEEL_COUNT%
    echo }
)

:: -- Count output ----------------------------------------------------------

set "FILE_COUNT=0"
for /r "%PACKAGE_DIR%" %%f in (*) do set /a FILE_COUNT+=1

echo.
echo [package] Package contents: %FILE_COUNT% files

:: -- Optional zip ----------------------------------------------------------

if "%DO_ZIP%"=="1" (
    set "ZIP_PATH=%OUTPUT_ROOT%\%PACKAGE_NAME%.zip"
    echo [package] Creating zip: !ZIP_PATH! ...
    if exist "!ZIP_PATH!" del "!ZIP_PATH!"
    powershell -NoProfile -Command ^
        "Compress-Archive -Path '%PACKAGE_DIR%' -DestinationPath '!ZIP_PATH!' -CompressionLevel Optimal"
    if errorlevel 1 (
        echo [ERROR] Zip creation failed.
        exit /b 1
    )
    echo [package] Zip ready: !ZIP_PATH!
)

:: -- Summary ----------------------------------------------------------------

echo.
echo ============================================================
echo Distribution package ready.
echo   Folder : %PACKAGE_DIR%
if "%DO_ZIP%"=="1" (
    echo   Zip    : %OUTPUT_ROOT%\%PACKAGE_NAME%.zip
)
echo.
echo Test the package on a clean machine:
echo   cd %PACKAGE_DIR%
echo   install.bat
echo   start.bat
echo   runtime\python\python.exe scripts\verify_runtime.py
echo ============================================================

endlocal
