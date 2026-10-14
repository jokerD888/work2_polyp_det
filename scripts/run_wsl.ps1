# Example: ./scripts/run_wsl.ps1 train --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenchArgs
)
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$TaskWindowsRoot = $TaskRoot.Replace('\', '/')
$TaskConvertedRoot = & wsl.exe -d Ubuntu-22.04 -- wslpath -a $TaskWindowsRoot
if ($LASTEXITCODE -ne 0) { throw 'Failed to resolve project path in WSL' }
$TaskWslRoot = $TaskConvertedRoot.Trim()
& wsl.exe -d Ubuntu-22.04 -- bash "$TaskWslRoot/scripts/wsl_entry.sh" $TaskWslRoot @BenchArgs
exit $LASTEXITCODE
