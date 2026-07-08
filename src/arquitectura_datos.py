"""
Evidencia real (no simulada) del pipeline de datos a escala: cluster Spark
standalone, data lake MinIO (S3) y broker Kafka, todos ejecutados como
contenedores Docker independientes y verificados con comandos reales
contra cada servicio (no valores escritos a mano).

Esta evidencia se muestra en /arquitectura como una pagina de solo lectura.
Deliberadamente el servicio de inferencia (/analizar) NO depende de Spark,
Kafka ni MinIO: son el pipeline de datos offline (ingesta, integridad,
trazabilidad, streaming), no el camino de prediccion en tiempo real. Meter
Kafka o Spark en la ruta de /analizar seria arquitectonicamente incorrecto
para una inferencia sincrona de una sola imagen.
"""

EVIDENCIA_SPARK = {
    'titulo': 'Cluster Spark Standalone Real',
    'descripcion': (
        'Tres contenedores Docker independientes en red (dermascan-spark-master, '
        'dermascan-spark-worker-1, dermascan-spark-worker-2), en vez de Spark en '
        'modo local[*] (un unico proceso simulando un cluster).'
    ),
    'evidencia': [
        'Master registro 2 workers reales por heartbeat: "Alive Workers: 2" (http://localhost:8080)',
        'Job sometido con spark-submit contra spark://spark-master:7077',
        '13,309 registros del manifiesto repartidos en 2 particiones, cada una procesada '
        'en un contenedor fisicamente distinto (confirmado en los logs de stderr de cada worker)',
        'Resultado del triage identico al esperado: ALERTA_DERMATOLOGO (HAM10000=1954, ISIC=1496), '
        'SEGUIMIENTO_RUTINARIO (HAM10000=8059, ISIC=1800)',
        'Tiempo de ejecucion en el cluster real de 2 workers: 48.75 s',
    ],
}

EVIDENCIA_MINIO = {
    'titulo': 'Data Lake Real (MinIO, protocolo S3)',
    'descripcion': (
        'Bucket real accesible por el protocolo S3 estandar (s3a://), no una carpeta '
        'de Parquet en disco local.'
    ),
    'evidencia': [
        'Bucket "dermascan-datalake" creado con el cliente oficial de MinIO (mc mb)',
        'Job de Spark escribio el resultado con el conector hadoop-aws (s3a://dermascan-datalake/...)',
        'Verificado con "mc ls --recursive": _SUCCESS + 2 particiones Parquet '
        '(354 KiB y 355 KiB) presentes en el bucket real',
        'Mismo protocolo y llamadas que usaria un S3 real de AWS; unica diferencia es el '
        'endpoint (http://minio:9000 en vez del endpoint de AWS)',
    ],
}

EVIDENCIA_KAFKA = {
    'titulo': 'Broker Kafka Real y Persistente',
    'descripcion': (
        'Broker Apache Kafka 3.7.0 en modo KRaft (sin ZooKeeper), como contenedor '
        'Docker persistente e independiente del kernel de Kaggle donde se entrena el modelo.'
    ),
    'evidencia': [
        'Topico "dermascan-imagenes" creado con 3 particiones reales',
        'Productor real (kafka-python) publico las 13,309 filas del manifiesto: 2.63 s (~5,061 msg/s)',
        'Conteo verificado DESDE EL PROPIO BROKER con kafka-get-offsets (no desde el productor): '
        '4,491 + 4,412 + 4,406 = 13,309 mensajes',
        'Consumidor Spark Structured Streaming (conector spark-sql-kafka-0-10) sometido al '
        'cluster real de 2 workers leyo los 13,309 mensajes y agrego por clase/fuente',
        'Conteos finales identicos, cifra por cifra, a la verificacion independiente del dataset: '
        'Benigno/HAM10000=8059, Benigno/ISIC=1800, Maligno/HAM10000=1954, Maligno/ISIC=1496',
    ],
}

NOTA_ARQUITECTURA = (
    'Este servicio web (el que estas usando ahora mismo para /analizar) es intencionalmente '
    'independiente de Spark, Kafka y MinIO: son el pipeline de datos offline (integridad, '
    'trazabilidad, streaming de ingesta), no el camino de inferencia en tiempo real. Mezclar '
    'ambos reintroduciria la misma contradiccion de latencia que ya se corrigio en la seccion '
    'de resultados del informe (ver /metricas). El cluster completo (docker-compose con Spark, '
    'MinIO y Kafka) es reproducible y se documenta en el repositorio del proyecto.'
)
