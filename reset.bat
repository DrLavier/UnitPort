@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo  ============================================================
echo        UnitPort - Factory Reset
echo  ============================================================
echo.
echo  This will restore the project to its initial (pre-install)
echo  state.  The next "start.bat" will trigger a full install +
echo  setup wizard.
echo.
echo  WILL BE DELETED:
echo.
echo    .venv311\                         Python virtual environment
echo    runtime\env\                      Install state
echo    src\system\runtime\sdk\*          Downloaded SDKs
echo    src\system\runtime\simulation\    MuJoCo menagerie
echo    custom_mods\                      All custom content
echo    training_workspaces\              Training workspace data
echo    logs\                             Log files
echo    __pycache__ / *.pyc               Python cache (everywhere)
echo    src\config\setup_state.json       Setup wizard state
echo    src\config\isaaclab.json          Isaac Lab registration
echo    src\config\rewards_presets.json   Reward preset overrides
echo    src\system\models\.sdk_install_state.json
echo.
echo  WILL BE RESTORED (git checkout):
echo.
echo    src\config\system.ini             Default system config
echo    src\config\user.ini               Default user config
echo    custom_mods\                      Git-tracked skeleton
echo.
echo  NOT TOUCHED:
echo.
echo    projects\                         User project data
echo.

set /p "CONFIRM=  Proceed with factory reset? [y/N] "
if /i not "%CONFIRM%"=="y" (
    echo  Aborted.
    goto :eof
)

echo.
echo  [1/7] Removing virtual environment ...
call :del_dir ".venv311"

echo  [2/7] Removing runtime state ...
call :del_dir "runtime\env"
call :del_file "src\system\models\.sdk_install_state.json"

echo  [3/7] Removing downloaded SDKs ...
for /d %%d in ("%ROOT%\src\system\runtime\sdk\*") do (
    rmdir /s /q "%%d" >nul 2>&1
    echo   [DEL]  src\system\runtime\sdk\%%~nxd\
)
if exist "%ROOT%\src\system\runtime\simulation\mujoco\menagerie\.git" (
    pushd "%ROOT%"
    git submodule deinit -f src/system/runtime/simulation/mujoco/menagerie >nul 2>&1
    popd
    echo   [DEL]  menagerie submodule (deinit)
)
call :del_dir "src\system\runtime\simulation\mujoco\menagerie"

echo  [4/7] Removing custom_mods and training data ...
call :del_dir "custom_mods"
call :del_dir "training_workspaces"

echo  [5/7] Clearing config registry ...
call :del_file "src\config\setup_state.json"
call :del_file "src\config\isaaclab.json"
call :del_file "src\config\rewards_presets.json"

echo  [6/7] Clearing cache and logs ...
call :del_dir "logs"
for /d /r "%ROOT%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" >nul 2>&1
)
echo   [DEL]  __pycache__ (all)
del /s /q "%ROOT%\*.pyc" >nul 2>&1
call :del_dir ".pytest_cache"
call :del_dir ".cache"

echo  [7/7] Restoring git-tracked defaults ...
pushd "%ROOT%"
git checkout -- src/config/system.ini >nul 2>&1
if not errorlevel 1 ( echo   [RST]  src\config\system.ini ) else ( echo   [SKIP] src\config\system.ini )
git checkout -- src/config/user.ini >nul 2>&1
if not errorlevel 1 ( echo   [RST]  src\config\user.ini ) else ( echo   [SKIP] src\config\user.ini )
git checkout -- src/config/ui.ini >nul 2>&1
if not errorlevel 1 ( echo   [RST]  src\config\ui.ini ) else ( echo   [SKIP] src\config\ui.ini )
git checkout -- src/config/node_registry.json >nul 2>&1
if not errorlevel 1 ( echo   [RST]  src\config\node_registry.json ) else ( echo   [SKIP] src\config\node_registry.json )
git checkout -- custom_mods/ >nul 2>&1
if not errorlevel 1 ( echo   [RST]  custom_mods\ ) else ( echo   [SKIP] custom_mods\ )
popd

echo.
echo  ============================================================
echo   Factory reset complete.
echo.
echo   Run "install.bat" then "start.bat" to begin fresh setup.
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
