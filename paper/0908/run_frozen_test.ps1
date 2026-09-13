param(
    [string]$PythonPath = "python",
    [int]$PredictSteps = 15,
    [string]$Device = "auto"
)
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $PythonPath (Join-Path $RootDir "model_code\evaluate_test.py") --predict-steps $PredictSteps --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonPath (Join-Path $RootDir "model_code\summarize.py") --stage test --predict-steps $PredictSteps
exit $LASTEXITCODE
