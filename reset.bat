@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo  ============================================================
echo        UnitPort RELEASE - Factory Reset
echo  ============================================================
echo.
echo  This will restore the project to its initial (pre-install)
echo  state.  The next "start.bat" will trigger a full install.
echo.
echo  WILL BE DELETED:
echo.
echo    .venv311\                         Python virtual environment
echo    runtime\env\                      Install state
echo    custom_mods\motions\loco-mujoco\  Cloned reference motion library
echo    log\                              Log files
echo    __pycache__ / *.pyc               Python cache (everywhere)
echo    .pip-tmp\ / .pip-cache\           pip scratch + cache
echo    .pytest_cache\ / .cache\          Test/build caches
echo.
echo  NOT TOUCHED:
echo.
echo    src\config\system.ini             SDK factory defaults (read-only)
echo    src\                              Source tree
echo    custom_mods\nodes\                User-authored nodes
echo    projects\                         User project data
echo    %%USERPROFILE%%\UnitPort\         Per-user state (user.ini, tokens, etc.)
echo.

set /p "CONFIRM=  Proceed with factory reset? [y/N] "
if /i not "%CONFIRM%"=="y" (
    echo  Aborted.
    goto :eof
)

echo.
echo  [1/5] Removing virtual environment ...
call :del_dir ".venv311"

echo  [2/5] Removing runtime install state ...
call :del_dir "runtime\env"

echo  [3/5] Removing cloned reference libraries ...
call :del_dir "custom_mods\motions\loco-mujoco"

echo  [4/5] Clearing cache and logs ...
call :del_dir "log"
call :del_dir ".pip-tmp"
call :del_dir ".pip-cache"
call :del_dir ".pytest_cache"
call :del_dir ".cache"
for /d /r "%ROOT%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" >nul 2>&1
)
echo   [DEL]  __pycache__ (all)
del /s /q "%ROOT%\*.pyc" >nul 2>&1

echo  [5/5] Hint: per-user state (user.ini, auth tokens, telemetry caches) lives
echo        under %%USERPROFILE%%\UnitPort\ and is NOT touched by reset.
echo        Delete it manually if you want a truly clean first-run experience.

echo.
echo  ============================================================
echo   Factory reset complete.
echo.
echo   Run "install.bat" then "start.bat" to begin fresh setup,
echo   or just run "start.bat" -- it auto-installs on a missing venv.
echo  ============================================================
echo.
goto :eof


:del_file
set "TARGET=%ROOT%\%~1"
if exist "%TARGET%" (
    del /f /q "%TARGET%" >nul 2>&1
    echo   [DEL]  %~1
) else (
    echo   [OK]   %~1  (not present)
)
goto :eof

:del_dir
set "TARGET=%ROOT%\%~1"
if exist "%TARGET%\" (
    rmdir /s /q "%TARGET%" >nul 2>&1
    echo   [DEL]  %~1\
) else (
    echo   [OK]   %~1\  (not present)
)
goto :eof
