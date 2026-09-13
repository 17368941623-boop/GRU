$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsDir = Join-Path $ScriptDir "modelout_lb65_paper"
$PidFile = Join-Path $ResultsDir "pipeline.pid"
$ProgressFile = Join-Path $ResultsDir "progress.json"
$StdoutLog = Join-Path $ResultsDir "pipeline_stdout.log"
$StderrLog = Join-Path $ResultsDir "pipeline_stderr.log"

if (Test-Path -LiteralPath $PidFile) {
    $PipelinePid = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
    $Process = Get-Process -Id $PipelinePid -ErrorAction SilentlyContinue
    if ($null -eq $Process) {
        Write-Output "PROCESS=not running (PID=$PipelinePid)"
    }
    else {
        Write-Output "PROCESS=running (PID=$PipelinePid)"
    }
}
else {
    Write-Output "PROCESS=not started"
}

if (Test-Path -LiteralPath $ProgressFile) {
    Write-Output ""
    Write-Output "PROGRESS"
    Get-Content -LiteralPath $ProgressFile -Raw
}

if (Test-Path -LiteralPath $StdoutLog) {
    Write-Output ""
    Write-Output "LATEST PIPELINE MESSAGES"
    Get-Content -LiteralPath $StdoutLog -Tail 12
}

if ((Test-Path -LiteralPath $StderrLog) -and (Get-Item -LiteralPath $StderrLog).Length -gt 0) {
    Write-Output ""
    Write-Output "LATEST PIPELINE ERRORS"
    Get-Content -LiteralPath $StderrLog -Tail 20
}
