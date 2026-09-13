param(
    [string]$PythonPath = "python",
    [int]$PredictSteps = 15,
    [int]$Epochs = 100,
    [int]$BatchSize = 256,
    [string]$Device = "auto",
    [int]$WorkerCount = 0,
    [int]$LoaderWorkers = 0
)
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $RootDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Runner = Join-Path $RootDir "model_code\run_shard.py"
$GpuCount = [int](& $PythonPath -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
if ($WorkerCount -le 0) {
    if (($Device -eq "auto" -or $Device -eq "cuda") -and $GpuCount -gt 0) {
        $WorkerCount = [Math]::Min($GpuCount, 5)
    }
    else {
        $WorkerCount = 1
    }
}
$Processes = @()
for ($Worker = 0; $Worker -lt $WorkerCount; $Worker++) {
    $WorkerDevice = $Device
    if ($GpuCount -gt 0 -and ($Device -eq "auto" -or $Device -eq "cuda")) {
        $WorkerDevice = "cuda:$($Worker % $GpuCount)"
    }
    $Arguments = @(
        $Runner, "--worker-index", "$Worker", "--worker-count", "$WorkerCount",
        "--predict-steps", "$PredictSteps", "--epochs", "$Epochs",
        "--batch-size", "$BatchSize", "--num-workers", "$LoaderWorkers",
        "--device", $WorkerDevice
    )
    $Processes += Start-Process -FilePath $PythonPath -ArgumentList $Arguments `
        -WorkingDirectory $RootDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir "worker_${Worker}.out.log") `
        -RedirectStandardError (Join-Path $LogDir "worker_${Worker}.err.log")
}
$Processes | Wait-Process
$Failed = $Processes | Where-Object { $_.ExitCode -ne 0 }
if ($Failed) { throw "One or more training workers failed. Check logs/." }
& $PythonPath (Join-Path $RootDir "model_code\summarize.py") --stage validation --predict-steps $PredictSteps
exit $LASTEXITCODE
