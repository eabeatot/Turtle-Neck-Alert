# Zero-downtime monitor: runs as an independent process during the CD job.
# Sends the fixed test input to the real analysis API, waits 1 second after each response completes,
# and records one JSON line per request (UTF-8, no BOM).
param(
  [Parameter(Mandatory=$true)][string]$Output,
  [Parameter(Mandatory=$true)][string]$PhaseFile,
  [Parameter(Mandatory=$true)][string]$StopFile,
  [Parameter(Mandatory=$true)][string]$BodyFile,
  [string]$Url = 'http://127.0.0.1:8000/api/predict',
  [int]$IntervalMs = 1000
)
$ErrorActionPreference = 'Continue'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$body = [System.IO.File]::ReadAllText($BodyFile, $utf8)
$seq = 0
while (-not (Test-Path -LiteralPath $StopFile)) {
  $seq++
  $phase = if (Test-Path -LiteralPath $PhaseFile) { ([System.IO.File]::ReadAllText($PhaseFile)).Trim() } else { 'before' }
  $start = Get-Date
  $status = 0; $label = ''; $confidence = $null; $app = ''; $model = ''; $errorText = ''
  try {
    $res = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 5
    $status = [int]$res.StatusCode
    $json = $res.Content | ConvertFrom-Json
    $label = [string]$json.label; $confidence = $json.confidence
    $app = [string]$json.app_version; $model = [string]$json.model_version
  } catch {
    $errorText = $_.Exception.Message
    try { $status = [int]$_.Exception.Response.StatusCode } catch { $status = 0 }
  }
  $end = Get-Date
  $valid = ($status -eq 200) -and ($label -eq 'good' -or $label -eq 'bad') -and ($app -ne '') -and ($model -ne '')
  $record = [ordered]@{
    seq = $seq; phase = $phase
    start = $start.ToString('o'); end = $end.ToString('o')
    latency_ms = [math]::Round(($end - $start).TotalMilliseconds, 1)
    status = $status; ok = $valid; label = $label; confidence = $confidence
    app_version = $app; model_version = $model; error = $errorText
  }
  [System.IO.File]::AppendAllText($Output, (($record | ConvertTo-Json -Compress) + "`n"), $utf8)
  Start-Sleep -Milliseconds $IntervalMs
}
