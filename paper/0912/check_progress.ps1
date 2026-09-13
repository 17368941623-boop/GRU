$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Status = Join-Path $Root "logs\training_status.json"
if (Test-Path -LiteralPath $Status) {
    Get-Content -LiteralPath $Status -Raw -Encoding UTF8
} else {
    Write-Output '{"state":"not_started"}'
}
$Metrics = Get-ChildItem -LiteralPath (Join-Path $Root "outputs\development\raw54_gru") -Filter metrics.json -File -Recurse -ErrorAction SilentlyContinue
Write-Output ("completed_metric_files={0}" -f @($Metrics).Count)
if ($Metrics) {
    $Rows = foreach ($File in $Metrics) {
        $Payload = Get-Content -LiteralPath $File.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        [pscustomobject]@{
            seed = $Payload.seed
            best_epoch = $Payload.best_epoch
            trajectory_rmse_k = [math]::Round($Payload.validation.trajectory_rmse_k, 6)
            final_rmse_k = [math]::Round($Payload.validation.final_horizon_rmse_k, 6)
            seconds = [math]::Round($Payload.training_seconds, 1)
        }
    }
    $Rows | Sort-Object seed | Format-Table -AutoSize
}
