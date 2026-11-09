param(
  [Parameter(Mandatory=$true)][string]$Output,
  [Parameter(Mandatory=$true)][string]$PhaseFile,
  [Parameter(Mandatory=$true)][string]$StopFile
)
$points = @(0..32 | ForEach-Object { @{ index=$_; x=0.5; y=(0.3 + ($_ * 0.01)); z=0 } })
$body = @{ points=$points } | ConvertTo-Json -Compress -Depth 5
while (-not (Test-Path -LiteralPath $StopFile)) {
  $phase = if (Test-Path -LiteralPath $PhaseFile) { (Get-Content -Raw $PhaseFile).Trim() } else { 'before' }
  $started = Get-Date
  $ok = $false; $app = ''; $model = ''; $errorText = ''
  try {
    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/predict' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 3
    $ok = $true; $app = [string]$response.app_version; $model = [string]$response.model_version
  } catch { $errorText = $_.Exception.Message }
  [pscustomobject]@{ timestamp=$started.ToUniversalTime().ToString('o'); phase=$phase; ok=$ok; app_version=$app; model_version=$model; error=$errorText } |
    ConvertTo-Json -Compress | Add-Content -LiteralPath $Output -Encoding utf8
  Start-Sleep -Milliseconds 200
}
