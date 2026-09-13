param(
    [string]$PythonPath = "D:\conda\envs\pytorch\python.exe",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$PidFile = Join-Path $Logs "training.pid"
if (Test-Path -LiteralPath $PidFile) {
    $ExistingPid = [int](Get-Content -LiteralPath $PidFile -Raw)
    if (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue) {
        throw "Training is already running with PID $ExistingPid"
    }
}
$OutLog = Join-Path $Logs "pipeline.stdout.log"
$ErrLog = Join-Path $Logs "pipeline.stderr.log"
$Runner = Join-Path $Root "run_raw54_10seeds.ps1"
$Arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner,
    "-PythonPath", $PythonPath, "-Device", $Device
)
$Process = Start-Process -FilePath "powershell.exe" -ArgumentList $Arguments -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Set-Content -LiteralPath $PidFile -Encoding ASCII -Value $Process.Id
Write-Output ("Started Raw54-GRU training PID={0}" -f $Process.Id)
