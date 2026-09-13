$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $Root "logs\training.pid"
if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Output "No PID file found."
    exit 0
}
$TrainingPid = [int](Get-Content -LiteralPath $PidFile -Raw)
$Process = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
if ($Process) {
    Stop-Process -Id $TrainingPid
    Write-Output "Stopped training PID $TrainingPid"
} else {
    Write-Output "Training PID $TrainingPid is not running"
}
