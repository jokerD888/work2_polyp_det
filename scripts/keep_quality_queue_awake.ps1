param([Parameter(Mandatory = $true)][string]$QueueRoot)
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class QualityQueuePower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$queueState = Join-Path $QueueRoot 'state.json'
$powerRecord = Join-Path $QueueRoot 'windows_power_guard.json'
function Write-PowerRecord([string]$Status) {
    @{ status = $Status; pid = $PID; updated_at = [DateTimeOffset]::UtcNow.ToString('o');
       policy = 'Prevent automatic system sleep while the finite queue runs; allow display sleep' } |
        ConvertTo-Json | Set-Content -LiteralPath $powerRecord -Encoding utf8
}
try {
    if ([QualityQueuePower]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'SetThreadExecutionState failed'
    }
    Write-PowerRecord 'active'
    while ($true) {
        try {
            $queueStatus = (Get-Content -LiteralPath $queueState -Raw | ConvertFrom-Json).status
            if ($queueStatus -in @('completed', 'failed', 'paused')) { break }
        } catch {
            # State is replaced by the running queue; retry a transient read.
        }
        Start-Sleep -Seconds 30
    }
} finally {
    [void][QualityQueuePower]::SetThreadExecutionState([uint32]2147483648)
    Write-PowerRecord 'released'
}
