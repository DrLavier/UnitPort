@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: UnitPort - scripts/build_runtime.bat
:: MAINTAINER / CI USE ONLY
::
:: Assembles the project-local runtime payload:
::   runtime/python/       Python 3.11 embeddable runtime + pip
::   runtime/cyclonedds/   CycloneDDS prebuilt installation
::
:: Prerequisites (maintainer machine only):
::   - PowerShell 5+ (available on Windows 10/11)
::   - Internet access for initial download
::   - 7-Zip (7z.exe in PATH) OR Windows built-in Expand-Archive
::
:: Usage:
::   cd <project-root>
::   scripts\build_runtime.bat
::
:: After completion, run:
::   scripts\build_wheels.bat   (to populate runtime/wheels/)
::   scripts\verify_runtime.py  (to confirm assembly)
:: ============================================================

:: -- Version pins ----------------------------------------------------------
:: Update these variables when bumping the runtime baseline.

set "PYTHON_VERSION=3.11.9"
set "PYTHON_ZIP=python-3.11.9-embed-amd64.zip"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"

set "CDDS_VERSION=0.10.2"
set "CDDS_ZIP=CycloneDDS-0.10.2-win64.zip"
:: GitHub release page: https://github.com/eclipse-cyclonedds/cyclonedds/releases/tag/0.10.2
:: Verify exact filename before running if this URL returns 404.
set "CDDS_URL=https://github.com/eclipse-cyclonedds/cyclonedds/releases/download/0.10.2/CycloneDDS-0.10.2-win64.zip"

set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

:: -- Paths -----------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

set "RUNTIME_DIR=%PROJECT_ROOT%\runtime"
set "PYTHON_DIR=%RUNTIME_DIR%\python"
set "CDDS_DIR=%RUNTIME_DIR%\cyclonedds"
set "DOWNLOADS_DIR=%RUNTIME_DIR%\_downloads"
set "ENV_DIR=%RUNTIME_DIR%\env"

echo ============================================================
echo UnitPort Runtime Build
echo Project root : %PROJECT_ROOT%
echo Python       : %PYTHON_VERSION%
echo CycloneDDS   : %CDDS_VERSION%
echo ============================================================
echo.

:: -- Create directories -----------------------------------------------------

if not exist "%DOWNLOADS_DIR%" mkdir "%DOWNLOADS_DIR%"
if not exist "%ENV_DIR%"       mkdir "%ENV_DIR%"

:: -- STEP 1: Python embeddable runtime -------------------------------------

echo [build] STEP 1: Python embeddable runtime

if exist "%PYTHON_DIR%\python.exe" (
    echo [build] runtime\python\python.exe already present  skipping download.
    goto :python_done
)

set "PYTHON_ARCHIVE=%DOWNLOADS_DIR%\%PYTHON_ZIP%"

if not exist "%PYTHON_ARCHIVE%" (
    echo [build] Downloading Python %PYTHON_VERSION% embeddable...
    powershell -NoProfile -Command ^
        "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ARCHIVE%' -UseBasicParsing"
    if errorlevel 1 (
        echo [ERROR] Python download failed.
        echo [ERROR] URL: %PYTHON_URL%
        echo [ERROR] Download manually and place at: %PYTHON_ARCHIVE%
        exit /b 1
    )
    echo [build] Download complete.
) else (
    echo [build] Cached archive found: %PYTHON_ARCHIVE%
)

echo [build] Extracting Python embeddable to runtime\python\ ...
if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"

powershell -NoProfile -Command ^
    "Expand-Archive -Path '%PYTHON_ARCHIVE%' -DestinationPath '%PYTHON_DIR%' -Force"
if errorlevel 1 (
    echo [ERROR] Python extraction failed.
    exit /b 1
)

:: -- Enable site-packages in embeddable layout ------------------------------
:: The embeddable zip ships with "import site" commented out in python3XX._pth.
:: Uncommenting it enables site-packages so pip-installed packages are visible.

echo [build] Enabling site-packages in embeddable runtime...
for %%f in ("%PYTHON_DIR%\python3*._pth") do (
    powershell -NoProfile -Command ^
        "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
    echo [build] Patched: %%~nxf
)

:: -- Bootstrap pip into embeddable runtime ---------------------------------

echo [build] Bootstrapping pip...
set "GETPIP_SCRIPT=%DOWNLOADS_DIR%\get-pip.py"

if not exist "%GETPIP_SCRIPT%" (
    powershell -NoProfile -Command ^
        "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%GETPIP_SCRIPT%' -UseBasicParsing"
    if errorlevel 1 (
        echo [ERROR] get-pip.py download failed.
        exit /b 1
    )
)

"%PYTHON_DIR%\python.exe" "%GETPIP_SCRIPT%" --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] pip bootstrap failed.
    exit /b 1
)
echo [build] pip installed successfully.

:python_done
for /f "tokens=*" %%v in ('"%PYTHON_DIR%\python.exe" --version 2^>^&1') do echo [build] Python ready: %%v

:: -- STEP 2: CycloneDDS prebuilt --------------------------------------------

echo.
echo [build] STEP 2: CycloneDDS prebuilt

set "CDDS_READY=false"
for %%f in ("%CDDS_DIR%\bin\*.dll") do set "CDDS_READY=true"

if "%CDDS_READY%"=="true" (
    echo [build] runtime\cyclonedds\ already populated with DLLs  skipping download.
    goto :cdds_done
)

set "CDDS_ARCHIVE=%DOWNLOADS_DIR%\%CDDS_ZIP%"

if not exist "%CDDS_ARCHIVE%" (
    echo [build] Downloading CycloneDDS %CDDS_VERSION% prebuilt...
    powershell -NoProfile -Command ^
        "Invoke-WebRequest -Uri '%CDDS_URL%' -OutFile '%CDDS_ARCHIVE%' -UseBasicParsing"
    if errorlevel 1 (
        echo.
        echo [WARNING] CycloneDDS download failed.
        echo [WARNING] URL: %CDDS_URL%
        echo [WARNING] If the release filename has changed, update CDDS_URL in this script.
        echo [WARNING] You can also download manually and place at: %CDDS_ARCHIVE%
        echo [WARNING] Skipping CycloneDDS  Unitree SDK integration will be unavailable.
        echo [WARNING] Re-run this script after placing the archive to complete the setup.
        goto :cdds_skip
    )
    echo [build] Download complete.
) else (
    echo [build] Cached archive found: %CDDS_ARCHIVE%
)

echo [build] Extracting CycloneDDS to runtime\cyclonedds\ ...
set "CDDS_EXTRACT_TMP=%DOWNLOADS_DIR%\_cdds_tmp"
if exist "%CDDS_EXTRACT_TMP%" rmdir /s /q "%CDDS_EXTRACT_TMP%"
mkdir "%CDDS_EXTRACT_TMP%"

powershell -NoProfile -Command ^
    "Expand-Archive -Path '%CDDS_ARCHIVE%' -DestinationPath '%CDDS_EXTRACT_TMP%' -Force"
if errorlevel 1 (
    echo [ERROR] CycloneDDS extraction failed.
    exit /b 1
)

:: CycloneDDS zip may have a top-level folder  find the actual installation root
set "CDDS_INNER="
for /d %%d in ("%CDDS_EXTRACT_TMP%\*") do (
    if exist "%%d\bin" (
        set "CDDS_INNER=%%d"
        goto :cdds_inner_found
    )
)
:: No nested folder  extract root is the installation root
set "CDDS_INNER=%CDDS_EXTRACT_TMP%"

:cdds_inner_found
if not exist "%CDDS_DIR%" mkdir "%CDDS_DIR%"
echo [build] Staging CycloneDDS from: %CDDS_INNER%

powershell -NoProfile -Command ^
    "Copy-Item -Path '%CDDS_INNER%\*' -Destination '%CDDS_DIR%' -Recurse -Force"
if errorlevel 1 (
    echo [ERROR] CycloneDDS staging failed.
    exit /b 1
)

rmdir /s /q "%CDDS_EXTRACT_TMP%"
echo [build] CycloneDDS staged to runtime\cyclonedds\

:cdds_done
echo [build] CycloneDDS ready: %CDDS_DIR%
goto :manifest_step

:cdds_skip
echo [build] CycloneDDS skipped.

:: -- STEP 3: Write manifest -------------------------------------------------

:manifest_step
echo.
echo [build] STEP 3: Writing runtime manifest...

:: Check for actual DLL files, not just directory skeleton
set "CDDS_PRESENT=false"
for %%f in ("%CDDS_DIR%\bin\*.dll") do set "CDDS_PRESENT=true"

:: Write manifest via PowerShell (handles timestamp and JSON reliably)
powershell -NoProfile -Command ^
  "$ts = (Get-Date).ToUniversalTime().ToString('o');" ^
  "$cdds = '%CDDS_PRESENT%' -eq 'true';" ^
  "$m = [ordered]@{" ^
  "  manifest_version = '1';" ^
  "  python_version = '%PYTHON_VERSION%';" ^
  "  python_source = 'embeddable-amd64';" ^
  "  cyclonedds_version = '%CDDS_VERSION%';" ^
  "  cyclonedds_present = $cdds;" ^
  "  build_timestamp = $ts;" ^
  "  wheel_packages = @();" ^
  "  compatibility_notes = 'Python %PYTHON_VERSION% embeddable + CycloneDDS %CDDS_VERSION% (Windows x64)';" ^
  "  unitport_version = 'phase2'" ^
  "};" ^
  "$json = ($m | ConvertTo-Json);" ^
  "[System.IO.File]::WriteAllText('%ENV_DIR%\manifest.json', $json, [System.Text.UTF8Encoding]::new(`$false))"
echo [build] Manifest written to runtime\env\manifest.json

:: -- Summary ----------------------------------------------------------------

echo.
echo ============================================================
echo Runtime build complete.
echo   runtime\python\     Python %PYTHON_VERSION% embeddable
if "%CDDS_PRESENT%"=="true" (
    echo   runtime\cyclonedds\ CycloneDDS %CDDS_VERSION%
) else (
    echo   runtime\cyclonedds\ NOT built ^(optional, needed for Unitree SDK^)
)
echo.
echo Next steps:
echo   scripts\build_wheels.bat        ^(populate offline wheelhouse^)
echo   runtime\python\python.exe scripts\verify_runtime.py
echo ============================================================

endlocal
