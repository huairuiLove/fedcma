param([string]$OutputDirectory = '')
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_local.ps1 first.' }
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $PSScriptRoot ('experiments\local_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$settings = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'configs\local_ihm.json') -Raw | ConvertFrom-Json
$runArguments = @('-u', (Join-Path $PSScriptRoot 'run_experiments.py'), '--out_dir', $OutputDirectory)
foreach ($property in $settings.PSObject.Properties) {
    $runArguments += ('--' + $property.Name)
    foreach ($value in @($property.Value)) {
        $runArguments += [string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $value)
    }
}
& $pythonPath @runArguments
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; see the run logs.' }
