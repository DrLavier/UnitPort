@echo off
setlocal

:: ============================================================
:: UnitPort start.bat
:: Launches UnitPort using the project-local Python environment.
::
:: Priority order:
::   1. runtime\python\python.exe (packaged runtime, if present)
::   2. .venv311\Scripts\python.exe (project-local venv, default)
::
:: Always injects project-local CYCLONEDDS_HOME if available.
:: Never relies on whichever Python the shell resolves first.
:: ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "RUNTIME_PYTHON=%PROJECT_ROOT%\runtime\python\python.exe"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv311\Scripts\python.exe"
set "CDDS_DIR=%PROJECT_ROOT%\runtime\cyclonedds"
set "INSTALL_STATE=%PROJECT_ROOT%\runtime\env\install_state.json"

:: -------------------------------------------------------
:: Select Python executable
:: -------------------------------------------------------

if exist "%RUNTIME_PYTHON%" (
    set "PYTHON_EXE=%RUNTIME_PYTHON%"
    set "LAUNCH_MODE=packaged"
) else if exist "%VENV_PYTHON%" (
    set "PYTHON_EXE=%VENV_PYTHON%"
    set "LAUNCH_MODE=venv311"
) else (
    echo [ERROR] No project-local Python found.
    echo [ERROR] Checked: runtime\python\python.exe
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
