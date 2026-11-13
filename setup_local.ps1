param(
    [string]$BasePython = 'D:\environment\envs\vnvc\python.exe',
    [switch]$CleanInstall
)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $BasePython)) {
    throw 'Pass -BasePython with the path to a Python 3.12 interpreter.'
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $venvArguments = @('-m', 'venv')
    if (-not $CleanInstall) { $venvArguments += '--system-site-packages' }
    $venvArguments += (Join-Path $projectPath '.venv')
    & $BasePython @venvArguments
    if ($LASTEXITCODE -ne 0) { throw 'Could not create venv.' }
}
& $pythonPath -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA PyTorch.' }
}
& $pythonPath -m pip install -r (Join-Path $projectPath 'requirements-dev.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install project dependencies.' }
& $pythonPath -m pytest (Join-Path $projectPath 'tests') -q
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
