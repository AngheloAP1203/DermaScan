# ============================================================================
# DermaScan — Ejecutar Pipeline Batch con PySpark (Windows PowerShell)
# ============================================================================
#
# Uso:
#   .\run_pipeline.ps1 <ruta_imagenes> [<ruta_salida>]
#
# Ejemplos:
#   .\run_pipeline.ps1 .\datos\HAM10000
#   .\run_pipeline.ps1 .\datos\HAM10000 .\resultados
#
# ============================================================================

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$RutaImagenes,

    [Parameter(Mandatory=$false, Position=1)]
    [string]$RutaSalida
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Pipeline  = Join-Path $ScriptDir "pipeline_batch.py"

# Verificar Java
try {
    $null = java -version 2>&1
} catch {
    Write-Host "ERROR: Java no esta instalado. Spark requiere Java 11+." -ForegroundColor Red
    Write-Host "  Descargar desde: https://adoptium.net/"
    exit 1
}

# Verificar MLflow (pipeline_batch.py resuelve el modelo "champion" desde el
# Model Registry, no desde un .keras local)
if (-not $env:MLFLOW_TRACKING_URI) {
    Write-Host "ERROR: MLFLOW_TRACKING_URI no esta definida." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DermaScan - Iniciando Pipeline Batch" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Construir argumentos
$args_pipeline = @($Pipeline, $RutaImagenes)
if ($RutaSalida) {
    $args_pipeline += $RutaSalida
}

# Ejecutar con spark-submit si existe, sino python directo
$sparkSubmit = Get-Command spark-submit -ErrorAction SilentlyContinue
if ($sparkSubmit) {
    & spark-submit --master "local[*]" --driver-memory 4g --conf spark.driver.maxResultSize=2g @args_pipeline
} else {
    Write-Host "[INFO] spark-submit no encontrado, usando python directo (pyspark embebido)" -ForegroundColor Yellow
    & python @args_pipeline
}
