$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "D:\conda\envs\pytorch\python.exe"
$Runner = Join-Path $ScriptDir "run_lb65_paper_study.py"
$DataDir = Join-Path (Split-Path -Parent $ScriptDir) "processed_data"
$ResultsDir = Join-Path $ScriptDir "modelout_lb65_paper"
$PidFile = Join-Path $ResultsDir "pipeline.pid"
$StdoutLog = Join-Path $ResultsDir "pipeline_stdout.log"
$StderrLog = Join-Path $ResultsDir "pipeline_stderr.log"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "PyTorch Python was not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Study runner was not found: $Runner"
}
if (-not (Test-Path -LiteralPath $DataDir)) {
    throw "Processed data directory was not found: $DataDir"
}

New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

if (Test-Path -LiteralPath $PidFile) {
    $ExistingPid = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
    $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($null -ne $ExistingProcess) {
        Write-Output "The lookback=65 study is already running (PID=$ExistingPid)."
        exit 0
    }
}

$Arguments = @(
    $Runner,
    "--data-dir", $DataDir,
    "--results-dir", $ResultsDir
)

# Some managed desktop sessions expose both Path and PATH.  Windows
# Start-Process uses a case-insensitive dictionary and rejects that duplicate,
# so normalize the process environment before launching the detached worker.
$EffectivePath = $env:Path
[System.Environment]::SetEnvironmentVariable("PATH", $null, [System.EnvironmentVariableTarget]::Process)
[System.Environment]::SetEnvironmentVariable("Path", $null, [System.EnvironmentVariableTarget]::Process)
[System.Environment]::SetEnvironmentVariable("Path", $EffectivePath, [System.EnvironmentVariableTarget]::Process)

$Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $Arguments `
    -WorkingDirectory $ScriptDir `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
@{
    pid = $Process.Id
    started_local_time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    python = $PythonExe
    runner = $Runner
    results_dir = $ResultsDir
    expected_training_runs = 70
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ResultsDir "process_info.json") -Encoding utf8

Write-Output "Lookback=65 paper study started in the background."
Write-Output "PID=$($Process.Id)"
Write-Output "RESULTS=$ResultsDir"
Write-Output "PROGRESS=$(Join-Path $ResultsDir 'progress.json')"
