param(
    [string]$PythonPath = "D:\conda\envs\pytorch\python.exe",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Code = Join-Path $Root "model_code"
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Status = Join-Path $Logs "training_status.json"
$Seeds = @(42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
$Completed = 0

try {
    Set-Content -LiteralPath $Status -Encoding UTF8 -Value '{"state":"preparing","completed":0,"total":10}'
    $TrainCache = Join-Path $Root "data_cache\train_raw54.pkl"
    $ValidationCache = Join-Path $Root "data_cache\validation_raw54.pkl"
    if (-not ((Test-Path -LiteralPath $TrainCache) -and (Test-Path -LiteralPath $ValidationCache))) {
        & $PythonPath (Join-Path $Code "prepare_raw54_data.py") 2>&1 | Tee-Object -FilePath (Join-Path $Logs "prepare.log")
        if ($LASTEXITCODE -ne 0) { throw "Data preparation failed" }
    }
    & $PythonPath (Join-Path $Code "preflight.py") 2>&1 | Tee-Object -FilePath (Join-Path $Logs "preflight.log")
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed" }

    foreach ($Seed in $Seeds) {
        $StatusPayload = @{state="training"; current_seed=$Seed; completed=$Completed; total=10; updated=(Get-Date).ToString("s")} | ConvertTo-Json -Compress
        Set-Content -LiteralPath $Status -Encoding UTF8 -Value $StatusPayload
        $SeedLog = Join-Path $Logs ("seed_{0}.log" -f $Seed)
        & $PythonPath (Join-Path $Code "train_one.py") --seed $Seed --device $Device 2>&1 | Tee-Object -FilePath $SeedLog
        if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $Seed" }
        $Completed += 1
    }

    & $PythonPath (Join-Path $Code "summarize.py") 2>&1 | Tee-Object -FilePath (Join-Path $Logs "summarize.log")
    if ($LASTEXITCODE -ne 0) { throw "Summary failed" }
    $StatusPayload = @{state="complete"; completed=10; total=10; updated=(Get-Date).ToString("s")} | ConvertTo-Json -Compress
    Set-Content -LiteralPath $Status -Encoding UTF8 -Value $StatusPayload
} catch {
    $Failure = @{state="failed"; completed=$Completed; total=10; error=$_.Exception.Message; updated=(Get-Date).ToString("s")} | ConvertTo-Json -Compress
    Set-Content -LiteralPath $Status -Encoding UTF8 -Value $Failure
    throw
}
