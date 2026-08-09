@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "HF_HOME=huggingface"
set "PYTHONUTF8=1"
set "MIKAZUKI_PORT=28000"
set "MIKAZUKI_SCHEMA_HOT_RELOAD=1"

:: Source/venv launcher. Portable packages use run_gui_portable.bat instead.

if exist "venv\Scripts\python.exe" goto :launch
if exist "python\python.exe" goto :launch

findstr /C:"2.7.0+cu128" "%~dp0install-cn.ps1" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] 安装脚本过旧或与当前仓库不符。
    echo   请在本目录执行 git pull，或下载最新 Release 整合包后双击 run_gui.bat。
    echo.
    pause
    exit /b 1
)

echo [First run] Installing dependencies for source environment, please wait...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-cn.ps1"
if errorlevel 1 (
    echo Install failed. Check network and retry.
    pause
    exit /b 1
)

:launch
:: Prefer the project interpreter by absolute path. Bare `python` after
:: activate.bat can still hit system Python312 when PATH order is wrong
:: (common on Windows), which loads CPU torch / mismatched site-packages.
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "python\python.exe" (
    set "PYTHON_EXE=%~dp0python\python.exe"
    set "PATH=%~dp0python;%PATH%"
) else (
    echo [ERROR] No project Python found. Run install-cn.ps1 first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" scripts\prefetch_default_tagger.py --if-missing
"%PYTHON_EXE%" gui.py %*
set "EXIT_CODE=%errorlevel%"
if %EXIT_CODE% neq 0 pause
exit /b %EXIT_CODE%
