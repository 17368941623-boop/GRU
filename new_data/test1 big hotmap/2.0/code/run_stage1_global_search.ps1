#Requires -Version 5.1
<#
.SYNOPSIS
  Windows PowerShell launcher for stage-1 global GRU lookback/hidden search.
  Mirrors run_stage1_global_search.sh for Windows Server with CUDA.
#>

param(
    [int]$Workers = 10,
    [int]$Epochs = 100,
    [int]$BatchSize = 256,
    [int]$Patience = 12,
    [int]$LoaderWorkers = 0,
    [string]$Device = "auto",
    [string]$Overwrite = "0",
    [string]$DataDir = "",
    [string]$ResultsDir = "",
    [string]$PythonBin = "python"
)

$ErrorActionPreference = "Continue"

function Fail([string]$Message, [int]$Code) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit $Code
}

$CodeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $CodeDir

if ($DataDir -eq "") {
    $ServerDataDir = Join-Path (Split-Path -Parent $ProjectDir) "processed_data"
    $LocalDataDir = Join-Path (Split-Path -Parent (Split-Path -Parent $ProjectDir)) "processed_data"
    if (Test-Path (Join-Path $ServerDataDir "data_build_config.json")) {
        $DataDir = $ServerDataDir
    } elseif (Test-Path (Join-Path $LocalDataDir "data_build_config.json")) {
        $DataDir = $LocalDataDir
    } else {
        Fail "processed_data was not found beside the project folder or in the archive root" 2
    }
}

if ($ResultsDir -eq "") {
    $ResultsDir = Join-Path $ProjectDir "output"
}

if ($Workers -lt 1 -or $Workers -gt 20) { Fail "Workers must be an integer from 1 to 20" 2 }
if ($Epochs -lt 1) { Fail "Epochs must be positive" 2 }
if ($BatchSize -lt 1) { Fail "BatchSize must be positive" 2 }
if ($Patience -lt 1) { Fail "Patience must be positive" 2 }
if ($Overwrite -ne "0" -and $Overwrite -ne "1") { Fail "Overwrite must be 0 or 1" 2 }

if (-not (Test-Path -LiteralPath $PythonBin)) {
    $resolved = Get-Command $PythonBin -ErrorAction SilentlyContinue
    if ($null -eq $resolved) { Fail "Python executable not found: $PythonBin" 2 }
    $PythonBin = $resolved.Source
}

$LauncherLogs = Join-Path $ResultsDir "launcher_logs"
New-Item -ItemType Directory -Path $LauncherLogs -Force | Out-Null

$LockDir = Join-Path $ResultsDir ".stage1_global_search.lock"
try {
    New-Item -ItemType Directory -Path $LockDir -ErrorAction Stop | Out-Null
} catch {
    Fail "The launcher may already be running: $LockDir`nIf no process is active, remove only this stale lock directory and retry." 3
}

$processes = @()

function Cleanup {
    foreach ($proc in $processes) {
        if ($null -ne $proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Remove-Item $LockDir -Force -ErrorAction SilentlyContinue
}

try {

$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"

Write-Host "PHASE 0/3: environment and no-leakage preflight"
& $PythonBin -c "import joblib,numpy,pandas,scipy,torch; print('DEPENDENCIES_OK=true'); print('TORCH_VERSION='+torch.__version__); print('CUDA_AVAILABLE='+str(torch.cuda.is_available())); print('CUDA_DEVICE='+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))" 2>&1 | Tee-Object (Join-Path $LauncherLogs "environment.log")
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }

& $PythonBin (Join-Path $CodeDir "preflight.py") 2>&1 | Tee-Object (Join-Path $LauncherLogs "preflight.log")
if ($LASTEXITCODE -ne 0) { throw "Preflight failed" }

& $PythonBin (Join-Path $CodeDir "smoke_test.py") 2>&1 | Tee-Object (Join-Path $LauncherLogs "smoke_test.log")
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }

Write-Host "PHASE 1/3: 120 runs = 6 lookbacks x 4 hidden sizes x 5 paired seeds"
Write-Host "WORKERS=$Workers EPOCHS=$Epochs BATCH_SIZE=$BatchSize DEVICE=$Device"

for ($shard = 0; $shard -lt $Workers; $shard++) {
    $workerArgs = @(
        (Join-Path $CodeDir "run_shard.py"),
        "--shard-index", "$shard",
        "--num-shards", "$Workers",
        "--data-dir", "$DataDir",
        "--results-dir", "$ResultsDir",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--patience", "$Patience",
        "--num-workers", "$LoaderWorkers",
        "--device", "$Device"
    )
    if ($Overwrite -eq "1") { $workerArgs += "--overwrite" }
    $stdoutLog = Join-Path $LauncherLogs "worker_$shard.log"
    $stderrLog = Join-Path $LauncherLogs "worker_$shard.err.log"
    $proc = Start-Process -FilePath $PythonBin -ArgumentList $workerArgs `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog `
        -NoNewWindow -PassThru
    $processes += $proc
    Write-Host "WORKER_${shard}_PID=$($proc.Id)"
}

$failed = $false
foreach ($proc in $processes) {
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        Write-Host "WARNING: worker PID $($proc.Id) exited with code $($proc.ExitCode)" -ForegroundColor Yellow
        $failed = $true
    }
}
if ($failed) { throw "At least one worker failed; inspect output/launcher_logs." }

Write-Host "PHASE 2/3: validation-only summary and one-SE coarse region"
& $PythonBin (Join-Path $CodeDir "summarize_results.py") --results-dir $ResultsDir 2>&1 | Tee-Object (Join-Path $LauncherLogs "summary.log")
if ($LASTEXITCODE -ne 0) { throw "Summary failed" }

Write-Host "PHASE 3/3: complete"
Write-Host "SEED_RUNS=$ResultsDir\validation_summary\validation_seed_runs.csv"
Write-Host "CELL_SUMMARY=$ResultsDir\validation_summary\validation_cell_summary.csv"
Write-Host "MEAN_MATRIX=$ResultsDir\validation_summary\validation_rmse_mean_matrix.csv"
Write-Host "FINE_REGION=$ResultsDir\validation_summary\coarse_search_recommendation.json"
Write-Host "TEST_DATA_OPENED=false"

} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
} finally {
    Cleanup
}
