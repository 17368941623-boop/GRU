$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$envName = if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { 'pytorch' }

conda run --no-capture-output -n $envName python "$scriptDir\validate_graph_constraints.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
conda run --no-capture-output -n $envName python "$scriptDir\test_graph_constraints.py" -v
exit $LASTEXITCODE
