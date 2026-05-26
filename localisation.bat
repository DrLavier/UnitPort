REM SPDX-FileCopyrightText: 2026 SU CHANG
REM SPDX-License-Identifier: Apache-2.0

@echo off
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv311\Scripts\python.exe"
if not exist "%PY%" (
    echo .venv311 not found. Run install.bat first.
    exit /b 1
)
"%PY%" "%ROOT%bootstrap\export_localisation.py" %*
exit /b %ERRORLEVEL%
