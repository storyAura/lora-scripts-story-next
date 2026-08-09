# Source/venv launcher. Portable packages use run_gui_portable.bat instead.
# Prefer run_gui.bat on Windows (avoids execution policy issues).

$Env:HF_HOME = "huggingface"
$Env:PYTHONUTF8 = "1"
$Env:MIKAZUKI_PORT = "28000"
$Env:MIKAZUKI_SCHEMA_HOT_RELOAD = "1"

# Prefer project interpreter by absolute path. Activating then calling bare
# `python` can still resolve to system Python312 when PATH order is wrong.
if (Test-Path -LiteralPath "venv\Scripts\python.exe") {
    Write-Host -ForegroundColor green "Using project venv Python..."
    $pythonExe = (Resolve-Path -LiteralPath "venv\Scripts\python.exe").Path
}
elseif (Test-Path -LiteralPath "python\python.exe") {
    Write-Host -ForegroundColor green "Using python from python folder..."
    $pythonExe = (Resolve-Path -LiteralPath "python\python.exe").Path
    $env:PATH = "$(Split-Path -Parent $pythonExe);$env:PATH"
}
else {
    Write-Host -ForegroundColor Red "[ERROR] No project Python found. Run install-cn.ps1 first."
    exit 1
}

& $pythonExe gui.py @args
exit $LASTEXITCODE
