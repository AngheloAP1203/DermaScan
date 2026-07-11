"""Contenido para la pagina /arquitectura.

La pagina resume la arquitectura vigente de feature/MLflow y conserva, como
evidencia historica, capturas del despliegue experimental que existia en main.
"""

ARQUITECTURA_API = {
    'titulo': 'Servicio de inferencia Flask + ONNX Runtime',
    'descripcion': (
        'La API separa el diagnostico rapido de los calculos pesados de '
        'explicabilidad. /analizar usa ONNX Runtime y /explicabilidad usa '
        'TensorFlow/Keras cuando se requieren Grad-CAM, ABCDE e incertidumbre.'
    ),
    'evidencia': [
        'POST /analizar recibe una imagen base64, valida calidad y responde clase, confianza, probabilidad, modo y riesgo.',
        'El fast path carga modelo_dermascan.onnx mediante onnxruntime para evitar depender de TensorFlow en la ruta principal.',
        'POST /explicabilidad agrupa Grad-CAM, analisis ABCDE, visualizaciones e incertidumbre MC Dropout fuera del camino rapido.',
        'GET /salud y GET /metricas exponen estado operativo y metricas finales del modelo.',
    ],
}

ARQUITECTURA_MLOPS = {
    'titulo': 'MLOps con MLflow y modo local reproducible',
    'descripcion': (
        'En produccion, el modelo se resuelve desde MLflow Model Registry. En '
        'desarrollo local, si no existe MLFLOW_TRACKING_URI, se usa el archivo '
        'modelo_dermascan.keras incluido en el repositorio.'
    ),
    'evidencia': [
        'MLflow usa el modelo dermascan-clasificador-piel y el alias champion como version aprobada.',
        'Los umbrales umbral_optimo y umbral_screening se leen como tags de la version del modelo en produccion.',
        'El modo local usa umbral balanceado 0.70 y screening 0.45 para que la app funcione sin servidor MLflow.',
        'convertir_onnx.py puede convertir desde MLflow o desde el modelo Keras local, segun la configuracion disponible.',
    ],
}

ARQUITECTURA_BATCH = {
    'titulo': 'Pipeline batch con PySpark',
    'descripcion': (
        'El pipeline procesa lotes de imagenes dermatoscopicas con Spark, carga '
        'el modelo una vez por particion y guarda resultados analiticos para '
        'revision posterior.'
    ),
    'evidencia': [
        'spark.read.format("binaryFile") lee imagenes reales desde disco local o filesystems compatibles con Hadoop.',
        'mapPartitions evita cargar el modelo por cada imagen y reduce la sobrecarga de serializacion.',
        'La salida se persiste en Parquet, CSV distribuido y resumen JSON.',
        'El mismo src.config.py alimenta a la API Flask y al pipeline batch para mantener modelo, tamano de imagen y umbral sincronizados.',
    ],
}

EVIDENCIA_HISTORICA = {
    'titulo': 'Evidencia historica de despliegue experimental',
    'descripcion': (
        'La rama main incluia capturas de un entorno experimental con Spark '
        'standalone y MinIO. Se conservan como evidencia de pruebas previas, '
        'pero no son requisito para ejecutar la rama feature/MLflow actual.'
    ),
    'evidencia': [
        'Spark standalone fue usado como evidencia de procesamiento distribuido fuera de la API de inferencia.',
        'MinIO fue usado como data lake compatible con S3 en la validacion experimental anterior.',
        'La rama feature/MLflow actual documenta como camino soportado el pipeline PySpark batch y salidas Parquet/CSV/JSON.',
    ],
}

NOTA_ARQUITECTURA = (
    'La inferencia sincrona de una imagen no debe depender de Spark, Kafka ni '
    'MinIO. Esos componentes pertenecen a la arquitectura de datos offline o '
    'experimental. El camino de usuario en tiempo real se mantiene en Flask + '
    'ONNX Runtime, y los procesos de gobierno del modelo se resuelven con '
    'MLflow cuando esta configurado.'
)
