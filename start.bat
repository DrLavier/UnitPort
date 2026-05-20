@echo off
setlocal EnableDelayedExpansion

call :print_ascii
:: Flag install.bat so it doesn't repaint the ASCII when start.bat falls through to it.
set "UNITPORT_ASCII_PRINTED=1"

:: ============================================================
:: UnitPort RELEASE start.bat - single entry point
::
:: 1) Verify .venv311 exists and Python boots
:: 2) Verify the BOOTSTRAP imports (PyQt6 + loguru + unitport_sdk) so
::    main.py can paint the LoadingScreen. Everything else (httpx,
::    paramiko, mujoco, ...) is verified inside the loading screen by
::    DepCheckTask, which streams pip install progress to the log widget.
:: 3) Inject project-local CycloneDDS (best-effort)
:: 4) Launch main.py
:: 5) On non-zero exit, run install.bat once and retry the launch
::    (defensive net for crashes that occur before DepCheckTask ran)
::
:: At any failed gate, install.bat is invoked transparently.
:: ============================================================

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "VENV_PYTHON=%PROJECT_ROOT%\.venv311\Scripts\python.exe"
set "CDDS_DIR=%PROJECT_ROOT%\runtime\cyclonedds"

:: -------------------------------------------------------
:: Gate 1 + 2: .venv311 present and Python boots
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

:: -------------------------------------------------------
:: Gate 3: BOOTSTRAP import check (PyQt6 + loguru + unitport_sdk only)
:: -------------------------------------------------------
:: These three are required to paint the LoadingScreen. If any is missing
:: we cannot show the loading screen at all, so we fall back to the
:: console-mode install.bat. Every other requirements.txt package is
:: verified by DepCheckTask AFTER the loading screen is up, so the user
:: sees pip-install progress streamed live.

"%VENV_PYTHON%" -c "import sys; sys.path.insert(0, r'%PROJECT_ROOT%\src'); import PyQt6, loguru, unitport_sdk" >nul 2>&1
if errorlevel 1 (
    echo [start] bootstrap imports missing ^(PyQt6 / loguru / unitport_sdk^), running install.bat ...
    call "%PROJECT_ROOT%\install.bat"
    if errorlevel 1 (
        echo [start] install.bat failed, cannot continue.
        exit /b 1
    )
    "%VENV_PYTHON%" -c "import sys; sys.path.insert(0, r'%PROJECT_ROOT%\src'); import PyQt6, loguru, unitport_sdk" >nul 2>&1
    if errorlevel 1 (
        echo [start] bootstrap imports still missing after install.bat; aborting.
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

cd /d "%PROJECT_ROOT%"

echo [start] UnitPort RELEASE - mode: %LAUNCH_MODE%
echo [start] Python: %PYTHON_EXE%

:: -------------------------------------------------------
:: Launch
:: -------------------------------------------------------
:: NOTE: do NOT auto-rerun install.bat on non-zero exit. A GB-scale pip
:: reinstall must never be triggered by a runtime crash or a dirty Qt
:: shutdown (e.g. "QThread: Destroyed while thread is still running"
:: produces a non-zero exit code but the environment is fine). Surface
:: the exit code so the user sees the real error.

"%PYTHON_EXE%" main.py %*
set "RC=!errorlevel!"
if not "!RC!"=="0" (
    echo.
    echo [start] main.py exited with code !RC!.
    echo [start] If you suspect a broken environment, run install.bat manually.
)
endlocal
exit /b !RC!

:: ============================================================
:: Subroutine: ASCII logo (120 chars wide), inlined for self-contained launch.
:: ============================================================

:print_ascii
echo(========================================================================================================================
echo(------------------------------------------------------------------------------------------------------------------------
echo(         ####
echo(     ####   ####                                          ####
echo( #####    ###########      #####     #####                #####  #####   ############                            ####
echo(##    ##########    ##     #####     #####                       #####   #############                          #####
echo(##  #########       ##     #####     #####  ###########   ##### ######## #####    #####   #########   ####### #########
echo(##  ######     #### ##     #####     #####  ############  ##### ######## #####    ##### ############  ####### #########
echo(##  ######   ###### ##     #####     #####  ####    ####  #####  #####   #############  ####    ##### #####     #####
echo(##  ######   ###### ##     #####     #####  ####    ####  #####  #####   ###########    ####    ##### #####     #####
echo(##    ####   ###    ##     ######   ######  ####    ####  #####  #####   #####          #####   ##### #####     #####
echo(###               ###       #############   ####    ####  #####  ####### #####           ###########  #####     #######
echo(  #####      ######           ########      ####    ####  #####   ###### #####             #######    #####       #####
echo(     #####   ###
echo(         ####
echo(------------------------------------------------------------------------------------------------------------------------
echo(========================================================================================================================
goto :eof
