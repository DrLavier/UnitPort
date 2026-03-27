@echo off
setlocal

:: ============================================================
:: UnitPort start.bat
:: Launches UnitPort using ONLY the project-local .venv311 environment.
:: Never falls back to runtime\python or any global Python.
:: Always injects project-local CYCLONEDDS_HOME if available.
:: ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "VENV_PYTHON=%PROJECT_ROOT%\.venv311\Scripts\python.exe"
set "CDDS_DIR=%PROJECT_ROOT%\runtime\cyclonedds"
set "INSTALL_STATE=%PROJECT_ROOT%\runtime\env\install_state.json"

:: -------------------------------------------------------
:: Select Python executable
:: -------------------------------------------------------

if exist "%VENV_PYTHON%" (
    set "PYTHON_EXE=%VENV_PYTHON%"
    set "LAUNCH_MODE=venv311"
) else (
    echo [ERROR] .venv311 Python not found.
    echo [ERROR] Checked: .venv311\Scripts\python.exe
    echo [ERROR] Run install.bat first.
    exit /b 1
)

:: -------------------------------------------------------
:: Inject project-local CycloneDDS (best-effort, non-blocking)
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
