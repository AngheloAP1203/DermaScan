param(
    [int]$Port = 5001,
    [string]$HostAddress = "127.0.0.1",
    [string]$BackendStoreUri = "sqlite:///mlflow.db",
    [string]$ArtifactsDestination = "./mlartifacts"
)

$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
} else {
    $Python = "python"
}

New-Item -ItemType Directory -Force -Path $ArtifactsDestination | Out-Null

Write-Host "[DermaScan] MLflow Tracking Server"
Write-Host "  Python:        $Python"
Write-Host "  URL:           http://$HostAddress`:$Port"
Write-Host "  Backend store: $BackendStoreUri"
Write-Host "  Artifacts:     $ArtifactsDestination"
Write-Host ""
Write-Host "En otra terminal usa:"
Write-Host "  `$env:MLFLOW_TRACKING_URI=`"http://$HostAddress`:$Port`""
Write-Host ""

& $Python -m mlflow server `
    --backend-store-uri $BackendStoreUri `
    --serve-artifacts `
    --artifacts-destination $ArtifactsDestination `
    --host $HostAddress `
    --port $Port
