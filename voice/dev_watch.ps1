$ErrorActionPreference = "Stop"

function Get-WatchedFiles {
    param(
        [string]$Root
    )

    $files = @()
    $envFile = Join-Path $Root ".env"
    if (Test-Path $envFile) {
        $files += Get-Item $envFile
    }

    $appDir = Join-Path $Root "app"
    if (Test-Path $appDir) {
        $files += Get-ChildItem -Path $appDir -Recurse -File -Filter *.py
    }

    $configDir = Join-Path $Root "config"
    if (Test-Path $configDir) {
        $files += Get-ChildItem -Path $configDir -Recurse -File -Filter *.json
    }

    return $files
}

function Get-StateSnapshot {
    param(
        [System.IO.FileInfo[]]$Files
    )

    $state = @{}
    foreach ($f in $Files) {
        $state[$f.FullName] = $f.LastWriteTimeUtc.Ticks
    }
    return $state
}

function Has-Changes {
    param(
        [hashtable]$OldState,
        [hashtable]$NewState
    )

    if ($OldState.Count -ne $NewState.Count) {
        return $true
    }

    foreach ($k in $NewState.Keys) {
        if (-not $OldState.ContainsKey($k)) {
            return $true
        }
        if ($OldState[$k] -ne $NewState[$k]) {
            return $true
        }
    }
    return $false
}

function Start-VoiceProcess {
    param(
        [string]$Root
    )

    Write-Host "[voice-watch] starting backend..." -ForegroundColor Cyan
    return Start-Process -FilePath "python" `
        -ArgumentList "-m uvicorn app.voice_wake_sherpa:app --host 0.0.0.0 --port 8010" `
        -WorkingDirectory $Root `
        -PassThru
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$files = Get-WatchedFiles -Root $root
$state = Get-StateSnapshot -Files $files
$proc = Start-VoiceProcess -Root $root

try {
    while ($true) {
        Start-Sleep -Milliseconds 900

        if ($proc.HasExited) {
            Write-Host "[voice-watch] backend exited (code $($proc.ExitCode)); restarting..." -ForegroundColor Yellow
            $proc = Start-VoiceProcess -Root $root
            $files = Get-WatchedFiles -Root $root
            $state = Get-StateSnapshot -Files $files
            continue
        }

        $filesNow = Get-WatchedFiles -Root $root
        $stateNow = Get-StateSnapshot -Files $filesNow
        if (Has-Changes -OldState $state -NewState $stateNow) {
            Write-Host "[voice-watch] change detected; restarting backend..." -ForegroundColor Yellow
            try {
                Stop-Process -Id $proc.Id -Force
            } catch {
                # Ignore process-stop races.
            }
            Start-Sleep -Milliseconds 250
            $proc = Start-VoiceProcess -Root $root
            $files = $filesNow
            $state = $stateNow
        }
    }
}
finally {
    if ($proc -and -not $proc.HasExited) {
        try {
            Stop-Process -Id $proc.Id -Force
        } catch {
            # Best effort shutdown.
        }
    }
}
