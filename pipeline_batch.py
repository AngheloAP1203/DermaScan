"""Pipeline de procesamiento por lotes (Batch) con PySpark.

Lee un directorio de imágenes dermatoscópicas reales (ej. HAM10000) desde
CUALQUIER filesystem soportado por Hadoop (local, HDFS, S3/S3A, ABFS, GCS...),
ejecuta inferencia con el modelo "champion" del Model Registry de MLflow en
paralelo, y persiste los resultados en formato Parquet + CSV distribuidos.

Uso local (dev/laptop):
    spark-submit pipeline_batch.py ./datos/HAM10000 ./resultados

Uso en un cluster real (YARN, K8s, Databricks...) — el paralelismo, la
memoria y el master los decide quien lanza el submit, NUNCA el código:
    spark-submit \
        --master yarn \
        --num-executors 20 --executor-cores 4 --executor-memory 8g \
        --conf spark.python.worker.timeout=300 \
        --conf spark.executorEnv.MLFLOW_TRACKING_URI=https://mlflow.miempresa.com \
        pipeline_batch.py s3a://mi-bucket/HAM10000 s3a://mi-bucket/resultados

El modelo se carga UNA sola vez por partición (no por imagen), eliminando la
sobrecarga de serialización que colapsa los pipelines ingenuos. El threshold
y la URI del modelo se resuelven contra el mismo MLflow Model Registry que
usa la API Flask (src/config.py) — ver Fase 1.
"""
import os
import sys
import time
import json
from datetime import datetime, timezone

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    BooleanType, IntegerType,
)

# ── Configuración ──────────────────────────────────────────────────────────
# UMBRAL, MODEL_URI e IMG_SIZE se resuelven UNA sola vez aquí (en el driver)
# contra el mismo Model Registry de MLflow que usa la API Flask (src/config.py)
# — es la única fuente de verdad, no se hardcodean valores propios que puedan
# desincronizarse del servicio en tiempo real.
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
from src.config import UMBRAL, IMG_SIZE, MODEL_URI, MARGEN_DUDA  # noqa: E402

EXTENSIONES = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']
GLOB_IMAGENES = '*.{' + ','.join(EXTENSIONES) + '}'

# Esquema de salida (Spark)
SCHEMA = StructType([
    StructField('imagen_id',       StringType(),    False),
    StructField('ruta',            StringType(),    False),
    StructField('prob_maligno',    FloatType(),     True),
    StructField('clase',           StringType(),    True),
    StructField('confianza_pct',   StringType(),    True),
    StructField('dudoso',          BooleanType(),   True),
    StructField('riesgo_banda',    StringType(),    True),
    StructField('riesgo_nivel',    IntegerType(),   True),
    StructField('calidad_nitidez', FloatType(),     True),
    StructField('calidad_piel',    BooleanType(),   True),
    StructField('latencia_ms',     FloatType(),     True),
    StructField('error',           StringType(),    True),
    StructField('procesado_en',    StringType(),    True),
])


# ── Helpers de rutas: local o URI (s3a://, hdfs://, abfss://...) ───────────
# os.path.abspath/os.path.join/os.makedirs asumen filesystem local — sobre
# una URI de object storage las destrozan. Estos helpers son la unica parte
# del pipeline que sabe distinguir "estoy en mi laptop" de "esto es un URI
# de cluster", para que el resto del codigo no tenga que saberlo.

def _es_uri(ruta):
    return '://' in ruta


def _normalizar_entrada(ruta):
    return ruta if _es_uri(ruta) else os.path.abspath(ruta)


def _unir(base, nombre):
    return f"{base.rstrip('/')}/{nombre}" if _es_uri(base) else os.path.join(base, nombre)


# ── Funciones de análisis (independientes de los módulos src/) ─────────────
# Se reimplementan aquí de forma autocontenida para que los workers de Spark
# no dependan de imports de Flask ni de src.modelo (que cargaría el modelo
# completo también en el proceso del driver). UMBRAL/IMG_SIZE/MODEL_URI sí se
# importan de src.config arriba, pero solo se resuelven una vez en el driver
# y viajan a los workers como valores primitivos (str/float) capturados por
# closure — no como un import del paquete src en cada partición.

def _evaluar_riesgo(prob):
    """Banda de riesgo clínico."""
    if prob < min(0.30, UMBRAL):
        return 'Bajo', 1
    if prob < UMBRAL:
        return 'Moderado', 2
    if prob < 0.85:
        return 'Alto', 3
    return 'Muy alto', 4


def _calidad_basica(img_rgb):
    """Chequeo rápido de nitidez y detección de piel (OpenCV puro)."""
    import cv2
    import numpy as np

    img  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())

    # Detección simple de piel por YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127))
    piel_pct = float(mask.mean() * 100.0)

    return nitidez, piel_pct > 5.0


def _procesar_particion(filas):
    """Función de partición: carga el modelo UNA vez, procesa todas las
    imágenes del bloque (ya traídas en memoria como bytes por Spark vía
    binaryFile — sin I/O local a disco) y yield Rows con resultados.

    Al ser una función generadora, nada de su cuerpo se ejecuta hasta que
    Spark empieza a iterarla — por eso primero se comprueba si la partición
    tiene datos ANTES de cargar TensorFlow/el modelo. Si no hiciéramos este
    peek, una partición vacía (repartition(N) con N mayor al numero de
    imagenes reales, algo normal en el ultimo batch de un dataset grande)
    igual pagaría el costo completo de bajar y cargar el modelo (~1-2 GB)
    sin procesar una sola imagen — multiplicado por cada partición vacía.
    """
    filas = iter(filas)
    try:
        primera_fila = next(filas)
    except StopIteration:
        return  # partición vacía: no cargar nada, no hacer nada

    import itertools
    import cv2
    import numpy as np

    # Imports pesados — solo se ejecutan una vez por worker/partición, y solo
    # si la partición tiene al menos una imagen que procesar.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import mlflow.keras

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    # Carga del modelo UNA sola vez para toda la partición, resuelta contra el
    # mismo Model Registry que sirve la API (MODEL_URI, capturado por closure
    # desde el driver). Requiere que cada nodo del cluster tenga acceso de red
    # al tracking server y al artifact store de MLflow (variable de entorno
    # MLFLOW_TRACKING_URI propagada vía spark.executorEnv en el submit).
    modelo = mlflow.keras.load_model(MODEL_URI)

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
    def inferir(x):
        return modelo(x, training=False)

    for fila in itertools.chain([primera_fila], filas):
        ruta   = fila.path
        nombre = os.path.basename(ruta)
        ahora  = datetime.now(timezone.utc).isoformat()

        try:
            # El contenido ya viaja en la fila (leido de forma distribuida
            # por spark.read.format("binaryFile")) — no hay open()/read()
            # local aqui, funciona igual sobre HDFS, S3 o disco local.
            arr = np.frombuffer(fila.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('no se pudo decodificar la imagen')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocesar → tensor
            resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')
            tensor  = tf.expand_dims(tf.convert_to_tensor(resized), 0)

            # Inferencia
            t0   = time.time()
            prob = float(inferir(tensor)[0][0])
            ms   = round((time.time() - t0) * 1000, 1)

            # Clasificación
            clase  = 'Maligno' if prob >= UMBRAL else 'Benigno'
            conf   = prob if clase == 'Maligno' else 1 - prob
            dudoso = abs(prob - UMBRAL) <= MARGEN_DUDA

            banda, nivel = _evaluar_riesgo(prob)

            # Calidad
            nitidez, es_piel = _calidad_basica(img_rgb)

            yield Row(
                imagen_id=nombre, ruta=ruta,
                prob_maligno=round(prob, 4),
                clase=clase,
                confianza_pct=f'{conf*100:.1f}%',
                dudoso=dudoso,
                riesgo_banda=banda, riesgo_nivel=nivel,
                calidad_nitidez=round(nitidez, 1),
                calidad_piel=es_piel,
                latencia_ms=ms,
                error=None,
                procesado_en=ahora,
            )
        except Exception as e:
            yield Row(
                imagen_id=nombre, ruta=ruta,
                prob_maligno=None, clase=None, confianza_pct=None,
                dudoso=None, riesgo_banda=None, riesgo_nivel=None,
                calidad_nitidez=None, calidad_piel=None,
                latencia_ms=None,
                error=str(e),
                procesado_en=ahora,
            )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Uso: spark-submit pipeline_batch.py <ruta_imagenes> [<ruta_salida>] [<n_particiones>]")
        print()
        print("  <ruta_imagenes>  Directorio o URI con imagenes dermatoscopicas")
        print("                   (local, s3a://, hdfs://, abfss://...)")
        print("  <ruta_salida>    Directorio o URI para resultados (por defecto: ./resultados)")
        print("  <n_particiones>  Opcional. Por defecto: paralelismo del cluster detectado")
        print()
        print("  El master, la memoria y el resto de recursos NO se configuran aqui:")
        print("  se deciden con flags de spark-submit (--master, --num-executors,")
        print("  --executor-memory, etc.) — ver docstring del modulo.")
        sys.exit(1)

    dir_imagenes = _normalizar_entrada(sys.argv[1])
    dir_salida   = _normalizar_entrada(sys.argv[2]) if len(sys.argv) > 2 else _unir(RAIZ, 'resultados')

    # Solo tiene sentido validar existencia local si es una ruta local real
    # (una URI de S3/HDFS se valida al leerla, no antes — no tenemos por que
    # saber hablar con ese filesystem desde el driver fuera de Spark).
    if not _es_uri(dir_imagenes) and not os.path.isdir(dir_imagenes):
        print(f"ERROR: El directorio '{dir_imagenes}' no existe.")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  DermaScan — Pipeline Batch con PySpark")
    print(f"  Modelo:    {MODEL_URI} ({IMG_SIZE}x{IMG_SIZE}, umbral={UMBRAL})")
    print(f"  Entrada:   {dir_imagenes}")
    print(f"  Salida:    {dir_salida}")
    print(f"{'='*60}")

    # Crear SparkSession — SIN .master() ni configs de recursos hardcodeadas.
    # El master (local[*], yarn, k8s://...), la memoria del driver/executors,
    # el numero de shuffle partitions, etc. se deciden en el `spark-submit`
    # (o en spark-defaults.conf del cluster), no en el codigo: cualquier
    # `.config(...)` puesto aqui tendria PRECEDENCIA sobre los flags de
    # spark-submit y le quitaria el control al administrador del cluster.
    spark = SparkSession.builder.appName('DermaScan-Batch-EfficientNetB4').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Paralelismo por defecto = el que Spark detecta del cluster real
    # (executors x cores asignados), no los cores de la maquina del driver.
    paralelismo_cluster = spark.sparkContext.defaultParallelism
    n_particiones = int(sys.argv[3]) if len(sys.argv) > 3 else paralelismo_cluster
    print(f"  Paralelismo detectado del cluster: {paralelismo_cluster}")
    print(f"  Particiones a usar:                {n_particiones}")
    print()

    # ── Lectura distribuida nativa ──────────────────────────────────────
    # spark.read.format("binaryFile") lista y lee los archivos usando la
    # capa Hadoop FileSystem: funciona igual sobre disco local, HDFS, S3A,
    # ABFS, GCS... sin que este codigo sepa (ni le importe) cual es. El
    # contenido de cada imagen viaja en la columna `content` (bytes) —
    # nada de os.walk() en el driver ni open().read() en el worker.
    df_rutas = (spark.read.format('binaryFile')
                .option('pathGlobFilter', GLOB_IMAGENES)
                .option('recursiveFileLookup', 'true')
                .load(dir_imagenes)
                .select('path', 'content')
                .repartition(n_particiones))

    if df_rutas.limit(1).count() == 0:
        print(f"ERROR: No se encontraron imagenes en '{dir_imagenes}'.")
        spark.stop()
        sys.exit(1)

    # Procesar con mapPartitions (modelo cargado UNA vez por partición)
    t_inicio = time.time()

    rdd_resultado = df_rutas.rdd.mapPartitions(_procesar_particion)
    df_resultado  = spark.createDataFrame(rdd_resultado, schema=SCHEMA)

    # Forzar ejecución y cachear
    df_resultado.cache()
    total = df_resultado.count()

    t_total = time.time() - t_inicio

    # ── Estadísticas ──
    errores  = df_resultado.filter(df_resultado.error.isNotNull()).count()
    exitosos = total - errores

    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"  Total procesadas:  {total:,}")
    print(f"  Exitosas:          {exitosos:,}")
    print(f"  Con error:         {errores:,}")
    print(f"  Tiempo total:      {t_total:.1f} s")
    if exitosos > 0:
        print(f"  Velocidad:         {exitosos/t_total:.1f} imgs/s")

    stats = None
    if exitosos > 0:
        from pyspark.sql import functions as F

        stats = df_resultado.filter(df_resultado.error.isNull()).agg(
            F.avg('prob_maligno').alias('prob_media'),
            F.avg('latencia_ms').alias('latencia_media'),
            F.sum(F.when(df_resultado.clase == 'Maligno', 1).otherwise(0)).alias('n_malignos'),
            F.sum(F.when(df_resultado.clase == 'Benigno', 1).otherwise(0)).alias('n_benignos'),
            F.sum(F.when(df_resultado.dudoso == True, 1).otherwise(0)).alias('n_dudosos'),
        ).collect()[0]

        print(f"\n  Distribución:")
        print(f"    Benignos:  {int(stats.n_benignos):,}")
        print(f"    Malignos:  {int(stats.n_malignos):,}")
        print(f"    Dudosos:   {int(stats.n_dudosos):,}")
        print(f"  Prob. media:   {stats.prob_media:.4f}")
        print(f"  Latencia media: {stats.latencia_media:.1f} ms")

    # ── Guardar resultados: escrituras distribuidas, nunca via el driver ──
    # os.makedirs solo aplica si la salida es realmente un path local; una
    # URI de S3/HDFS la crea el propio writer de Spark al escribir.
    if not _es_uri(dir_salida):
        os.makedirs(dir_salida, exist_ok=True)

    ruta_parquet = _unir(dir_salida, 'predicciones.parquet')
    ruta_csv     = _unir(dir_salida, 'predicciones.csv')
    ruta_resumen = _unir(dir_salida, 'resumen.json')

    # Parquet: escritura distribuida, cada executor escribe su propia
    # porcion — nunca pasa por el driver.
    df_resultado.write.mode('overwrite').parquet(ruta_parquet)
    print(f"\n  Parquet: {ruta_parquet}")

    # CSV: igual, escritura distribuida nativa de Spark (produce un
    # directorio con varios part-*.csv, no un unico archivo). Para
    # "datos hospitalarios masivos" eso es correcto — forzar un solo
    # archivo con coalesce(1)/toPandas() reintroduce el mismo cuello de
    # botella de driver que se elimino aqui. Si un consumidor rio abajo
    # necesita un solo CSV, que lo combine el fuera (o que lea Parquet).
    df_resultado.write.mode('overwrite').option('header', True).csv(ruta_csv)
    print(f"  CSV:     {ruta_csv}")

    # Resumen: para rutas locales se guarda como un unico JSON legible
    # (comodo en dev); para URIs de cluster se escribe con el writer
    # distribuido de Spark, igual que parquet/csv, para no depender de
    # open() local contra un filesystem que el driver no sabe hablar.
    resumen = {
        'modelo':           MODEL_URI,
        'img_size':         IMG_SIZE,
        'umbral':           UMBRAL,
        'entrada':          dir_imagenes,
        'total_imagenes':   total,
        'exitosas':         exitosos,
        'errores':          errores,
        'tiempo_total_s':   round(t_total, 1),
        'velocidad_imgs_s': round(exitosos / t_total, 1) if t_total > 0 else 0,
        'n_particiones':    n_particiones,
        'paralelismo_cluster': paralelismo_cluster,
        'procesado_en':     datetime.now(timezone.utc).isoformat(),
    }
    if stats is not None:
        resumen.update({
            'prob_media':     round(float(stats.prob_media), 4),
            'latencia_media': round(float(stats.latencia_media), 1),
            'n_benignos':     int(stats.n_benignos),
            'n_malignos':     int(stats.n_malignos),
            'n_dudosos':      int(stats.n_dudosos),
        })

    if _es_uri(dir_salida):
        spark.createDataFrame([Row(**{k: str(v) for k, v in resumen.items()})]) \
             .write.mode('overwrite').json(ruta_resumen)
    else:
        with open(ruta_resumen, 'w', encoding='utf-8') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"  Resumen: {ruta_resumen}")

    print(f"\n{'='*60}")
    print(f"  Pipeline completado exitosamente.")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == '__main__':
    main()
