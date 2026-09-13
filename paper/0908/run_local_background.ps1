param(
    [string]$PythonPath = "D:\conda\envs\pytorch\python.exe",
    [int]$PredictSteps = 15,
    [int]$Bootstrap = 20,
    [int]$Epochs = 100,
    [int]$BatchSize = 256
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $RootDir "logs"
$StatusPath = Join-Path $LogDir "local_pipeline_status.json"
$OutputLog = Join-Path $LogDir "local_pipeline.out.log"
$ErrorLog = Join-Path $LogDir "local_pipeline.err.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Status {
    param([string]$Stage, [string]$State, [string]$Message)
    $Payload = [ordered]@{
        process_id = $PID
        stage = $Stage
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        python = $PythonPath
        predict_steps = $PredictSteps
        bootstrap = $Bootstrap
        epochs = $Epochs
        batch_size = $BatchSize
        worker_count = 1
        output_directory = (Join-Path $RootDir "outputs")
    }
    $Payload | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-PythonStage {
    param([string]$Stage, [string[]]$Arguments)
    Write-Status -Stage $Stage -State "RUNNING" -Message "Stage started"
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] START $Stage" | Add-Content -LiteralPath $OutputLog
    & $PythonPath @Arguments 1>> $OutputLog 2>> $ErrorLog
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] COMPLETE $Stage" | Add-Content -LiteralPath $OutputLog
}

try {
    Invoke-PythonStage -Stage "preflight" -Arguments @(
        (Join-Path $RootDir "model_code\preflight.py")
    )
    Invoke-PythonStage -Stage "causal_discovery" -Arguments @(
        (Join-Path $RootDir "model_code\discover_causal_graph.py"),
        "--bootstrap", "$Bootstrap"
    )
    Invoke-PythonStage -Stage "development_training" -Arguments @(
        (Join-Path $RootDir "model_code\run_shard.py"),
        "--worker-index", "0", "--worker-count", "1",
        "--predict-steps", "$PredictSteps", "--epochs", "$Epochs",
        "--batch-size", "$BatchSize", "--device", "cuda"
    )
    Invoke-PythonStage -Stage "validation_summary" -Arguments @(
        (Join-Path $RootDir "model_code\summarize.py"),
        "--stage", "validation", "--predict-steps", "$PredictSteps"
    )
    Write-Status -Stage "complete" -State "COMPLETE" -Message "All development runs and validation summary completed; frozen test has not been opened."
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $ErrorLog
    Write-Status -Stage "failed" -State "FAILED" -Message $_.Exception.Message
    exit 1
}
