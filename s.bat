@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: UnitPort start.bat - single entry point
:: Checks .venv311; if missing or broken, runs install.bat first.
:: Then launches the application.
:: ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "VENV_PYTHON=%PROJECT_ROOT%\.venv311\Scripts\python.exe"
set "CDDS_DIR=%PROJECT_ROOT%\runtime\cyclonedds"

:: -------------------------------------------------------
:: Auto-install if .venv311 is missing or broken
:: -------------------------------------------------------

set "NEED_INSTALL=0"

if not exist "%VENV_PYTHON%" (
    echo [start] .venv311 not found, running install.bat ...
    set "NEED_INSTALL=1"
    goto :do_install_check
)

"%VENV_PYTHON%" -c "import sys; sys.exit(0)" >nul 2>&1
if errorlevel 1 (
    echo [start] .venv311 Python broken, running install.bat ...
    set "NEED_INSTALL=1"
)

:do_install_check
if "!NEED_INSTALL!"=="1" (
    call "%PROJECT_ROOT%\install.bat"
    if errorlevel 1 (
        echo [start] install.bat failed, cannot continue.
        exit /b 1
    )
    if not exist "%VENV_PYTHON%" (
        echo [start] install.bat finished but .venv311 still missing, cannot continue.
        exit /b 1
    )
)

:: Quick sanity: can we import PySide6?
"%VENV_PYTHON%" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo [start] PySide6 missing, re-running install.bat ...
    call "%PROJECT_ROOT%\install.bat"
    if errorlevel 1 (
        echo [start] install.bat failed, cannot continue.
        exit /b 1
    )
)

set "PYTHON_EXE=%VENV_PYTHON%"
set "LAUNCH_MODE=venv311"

:: -------------------------------------------------------
:: Inject project-local CycloneDDS (best-effort)
:: -------------------------------------------------------

if exist "%CDDS_DIR%\bin" (
    set "CYCLONEDDS_HOME=%CDDS_DIR%"
    set "PATH=%CDDS_DIR%\bin;%PATH%"
)

:: -------------------------------------------------------
:: Launch
:: -------------------------------------------------------

cd /d "%PROJECT_ROOT%"

echo [start] UnitPort - mode: %LAUNCH_MODE%
echo [start] Python: %PYTHON_EXE%

"%PYTHON_EXE%" main.py %*

endlocal
