param([Parameter(ValueFromRemainingArguments = $true)][string[]]$MethodArgs)
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$TaskWslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a $TaskRoot.Replace('\', '/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Failed to resolve WSL workspace' }
& wsl.exe -d Ubuntu-22.04 -- bash "$TaskWslRoot/methods/mssl/wsl_entry.sh" $TaskWslRoot -m methods.mssl @MethodArgs
exit $LASTEXITCODE
