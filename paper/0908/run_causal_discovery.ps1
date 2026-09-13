param(
    [string]$PythonPath = "python",
    [int]$Bootstrap = 20
)
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $PythonPath (Join-Path $RootDir "model_code\discover_causal_graph.py") --bootstrap $Bootstrap
exit $LASTEXITCODE
