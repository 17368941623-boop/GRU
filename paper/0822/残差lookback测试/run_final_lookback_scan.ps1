$ErrorActionPreference = "Stop"

$TaskPython = "D:\conda\envs\pytorch\python.exe"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "run_final_lookback_scan.py"

if (-not (Test-Path -LiteralPath $TaskPython)) {
    throw "Configured PyTorch Python was not found: $TaskPython"
}

& $TaskPython $Runner @args
if ($LASTEXITCODE -ne 0) {
    throw "Final lookback scan failed with exit code $LASTEXITCODE"
}
