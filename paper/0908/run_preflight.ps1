param(
    [string]$PythonPath = "python"
)
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $PythonPath (Join-Path $RootDir "model_code\preflight.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonPath (Join-Path $RootDir "model_code\smoke_test.py")
exit $LASTEXITCODE
