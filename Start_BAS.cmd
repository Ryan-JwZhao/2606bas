@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PYW=.venv\Scripts\pythonw.exe"
set "YOLO_CONFIG_DIR=%CD%\local_settings\ultralytics"
set "MPLCONFIGDIR=%CD%\local_settings\matplotlib"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%" >nul 2>nul
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>nul

if not exist "%VENV_PY%" (
    echo .venv not found. Run Setup_Environment.cmd once, then start again.
    pause
    exit /b 1
)

if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" -m bas ui
) else (
    start "" "%VENV_PY%" -m bas ui
)

exit /b 0
