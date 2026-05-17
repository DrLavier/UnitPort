@echo off
REM install_ros2.bat -- Windows ROS2 bridge setup.
REM
REM Verifies Docker Desktop is running, pulls the Humble base image,
REM builds the UnitPort ros2_bridge image, and records state.
REM
REM Optional first argument: --check-native-first
REM   When set, runs bootstrap\detect_ros2.py first; if a native ROS2 Humble
REM   install is found the script records ros2_mode=native in install_state.json
REM   and exits 0 without touching Docker.

setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "STATE_DIR=%PROJECT_ROOT%\runtime\env"
set "STATE_FILE=%STATE_DIR%\install_state.json"
if "%UNITPORT_ROS2_IMAGE%"=="" set "UNITPORT_ROS2_IMAGE=unitport/ros2_bridge:humble"

set "PY_EXE=%PROJECT_ROOT%\.venv311\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

if /i "%~1"=="--check-native-first" (
    echo [install_ros2] Probing for native ROS2 Humble...
    if not exist "%STATE_DIR%" mkdir "%STATE_DIR%"
    for /f "usebackq delims=" %%D in (`"%PY_EXE%" "%SCRIPT_DIR%detect_ros2.py" 2^>nul`) do set "ROS2_JSON=%%D"
    if defined ROS2_JSON (
        "%PY_EXE%" -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if (d.get('found') and d.get('distro')=='humble') else 1)" "!ROS2_JSON!" >nul 2>&1
        if not errorlevel 1 (
            echo [install_ros2] Native ROS2 Humble detected; skipping Docker bridge build.
            "%PY_EXE%" -c "import json, time, pathlib, sys; d=json.loads(sys.argv[1]); sf=pathlib.Path(sys.argv[2]); s={}; s.update(json.loads(sf.read_text(encoding='utf-8'))) if sf.exists() else None; s['ros2_mode']='native'; s['ros2_distro']=d.get('distro',''); s['ros2_ros_root']=d.get('ros_root',''); s['ros2_verified_at']=int(time.time()); sf.write_text(json.dumps(s, indent=2), encoding='utf-8'); print(f'[install_ros2] State written to {sf}')" "!ROS2_JSON!" "%STATE_FILE%"
            endlocal
            exit /b 0
        )
    )
    echo [install_ros2] No native ROS2 Humble found; proceeding with Docker bridge build.
)

echo [install_ros2] Checking Docker Desktop...
where docker >nul 2>&1
if errorlevel 1 (
    echo [install_ros2][ERROR] docker CLI not found on PATH.
    echo   Install Docker Desktop: https://www.docker.com/products/docker-desktop/
    exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
    echo [install_ros2][ERROR] docker info failed -- is Docker Desktop running?
    echo   Also ensure "Use the WSL 2 based engine" is enabled in Settings.
    exit /b 1
)

echo [install_ros2] WSL2 diagnostics...
wsl --status 2>nul

echo [install_ros2] Pulling ros:humble-ros-base...
docker pull ros:humble-ros-base || exit /b 1

echo [install_ros2] Building %UNITPORT_ROS2_IMAGE%...
docker build -t %UNITPORT_ROS2_IMAGE% "%PROJECT_ROOT%\runtime\docker\ros2_bridge" || exit /b 1

echo [install_ros2] Smoke-testing image...
docker run --rm %UNITPORT_ROS2_IMAGE% bash -lc "ros2 --help >/dev/null && echo ok" || exit /b 1

if not exist "%STATE_DIR%" mkdir "%STATE_DIR%"
set "PY_EXE=%PROJECT_ROOT%\.venv311\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"
"%PY_EXE%" -c "import json, os, time, pathlib; sf=pathlib.Path(r'%STATE_FILE%'); s={}; s.update(json.loads(sf.read_text(encoding='utf-8'))) if sf.exists() else None; s['ros2_bridge_verified']=True; s['ros2_bridge_image']=r'%UNITPORT_ROS2_IMAGE%'; s['ros2_bridge_verified_at']=int(time.time()); s['ros2_mode']='both' if s.get('ros2_mode')=='native' else 'docker'; sf.write_text(json.dumps(s, indent=2), encoding='utf-8'); print(f'[install_ros2] State written to {sf}')"

echo [install_ros2] Done. Bridge image: %UNITPORT_ROS2_IMAGE%
endlocal
