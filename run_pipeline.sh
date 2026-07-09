#!/usr/bin/env bash
# ============================================================================
# DermaScan — Ejecutar Pipeline Batch con PySpark
# ============================================================================
#
# Uso:
#   ./run_pipeline.sh <ruta_imagenes> [<ruta_salida>]
#
# Ejemplos:
#   ./run_pipeline.sh ./datos/HAM10000_ISIC
#   ./run_pipeline.sh ./datos/HAM10000_ISIC ./resultados
#   ./run_pipeline.sh /ruta/absoluta/imagenes /ruta/absoluta/salida
#
# Requisitos:
#   - Python 3.11+ con pyspark, tensorflow, keras, opencv-python-headless
#   - Java 11+ (requerido por Spark)
#   - MLFLOW_TRACKING_URI opcional. Si existe, usa Model Registry; si no,
#     usa modelo_dermascan.keras local.
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="$SCRIPT_DIR/pipeline_batch.py"

# Verificar Java
if ! command -v java &> /dev/null; then
    echo "ERROR: Java no está instalado. Spark requiere Java 11+."
    echo "  Instalar con: apt-get install -y default-jre (Linux)"
    echo "                o descargar desde https://adoptium.net/"
    exit 1
fi

# Informar fuente del modelo
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    echo "[INFO] MLFLOW_TRACKING_URI no está definida; usando modelo_dermascan.keras local."
else
    echo "[INFO] Usando MLflow Model Registry: $MLFLOW_TRACKING_URI"
fi

# Verificar argumentos
if [ $# -lt 1 ]; then
    echo "Uso: $0 <ruta_imagenes> [<ruta_salida>]"
    exit 1
fi

echo ""
echo "============================================================"
echo "  DermaScan — Iniciando Pipeline Batch"
echo "============================================================"
echo ""

# Ejecutar con spark-submit si está disponible, sino con python directo
# (pyspark incluye spark-submit embebido cuando se instala via pip)
if command -v spark-submit &> /dev/null; then
    spark-submit \
        --master "local[*]" \
        --driver-memory 4g \
        --conf spark.driver.maxResultSize=2g \
        "$PIPELINE" "$@"
else
    echo "[INFO] spark-submit no encontrado, usando python directo (pyspark embebido)"
    python "$PIPELINE" "$@"
fi
