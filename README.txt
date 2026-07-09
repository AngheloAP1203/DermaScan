# DermaScan - Despliegue Completo (Seccion 7 MLOps)

Proyecto: DermaScan | UPAO 2026-10 | Percepcion Computacional
Accuracy : 92.39%  |  AUC ROC: 0.9706  |  Latencia fast path: 16 ms
Modelo   : EfficientNetB4 | Dataset: HAM10000 + ISIC
Datos    : 13,309 imagenes fusionadas y deduplicadas | Test: 1,997 imagenes

========================================================
## CONTENIDO
========================================================
  modelo_dermascan.keras     Pesos entrenados de EfficientNetB4 (380 px)
  modelo_dermascan.onnx      Grafo ONNX para el fast path de /analizar
  app.py                     API REST Flask + interfaz web
  registrar_experimento.py   Registra metricas y pesos en MLflow
  convertir_onnx.py          Convierte el modelo Keras/MLflow a ONNX
  pipeline_batch.py          Pipeline batch con PySpark
  requirements.txt           Dependencias de runtime
  Dockerfile                 Imagen de despliegue con Gunicorn

========================================================
## PASO 1 - Instalar dependencias
========================================================
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt

  Para generar el ONNX tambien se requiere:

  pip install tf2onnx

========================================================
## PASO 2 - Fuente del modelo
========================================================

  ### Opcion A: desarrollo local sin MLflow

  Si no existe MLFLOW_TRACKING_URI, DermaScan usa:

    modelo_dermascan.keras

  Umbrales locales por defecto:

    balanceado = 0.70
    screening  = 0.45

  Tambien pueden sobrescribirse con:

    DERMASCAN_UMBRAL_OPTIMO
    DERMASCAN_UMBRAL_SCREENING
    DERMASCAN_MODEL_LOCAL_PATH

  ### Opcion B: produccion con MLflow Model Registry

  Configurar:

    $env:MLFLOW_TRACKING_URI="https://<tu-servidor-mlflow>"

  Registrar/promover el modelo:

    python registrar_experimento.py `
        --modelo modelo_dermascan.keras `
        --predicciones predicciones_val.csv `
        --dataset "HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)" `
        --promover

  La version promovida queda accesible mediante el alias:

    champion

  En produccion, la API Flask y el pipeline Spark leen el modelo y los
  umbrales desde MLflow Model Registry mediante los tags:

    umbral_optimo
    umbral_screening

========================================================
## PASO 3 - Generar modelo ONNX
========================================================

  python convertir_onnx.py

  Sin MLFLOW_TRACKING_URI, convierte modelo_dermascan.keras local.
  Con MLFLOW_TRACKING_URI, convierte el modelo champion desde MLflow.

========================================================
## PASO 4 - Correr el servidor web
========================================================

  ### Opcion A: Python directo

    python app.py
    Abre: http://localhost:5000

  ### Opcion B: Gunicorn (Linux/produccion)

    gunicorn --bind 0.0.0.0:7860 --workers 4 --threads 4 --worker-class gthread app:app
    Abre: http://localhost:7860

  ### Opcion C: Docker

    docker build -t dermascan:2.0 .
    docker run -p 7860:7860 dermascan:2.0
    Abre: http://localhost:7860

  Para produccion con MLflow:

    docker run -p 7860:7860 -e MLFLOW_TRACKING_URI="https://<tu-servidor-mlflow>" dermascan:2.0

========================================================
## PIPELINE BATCH CON PYSPARK
========================================================

  Windows:

    .\run_pipeline.ps1 .\datos\HAM10000_ISIC .\resultados

  Linux/macOS:

    ./run_pipeline.sh ./datos/HAM10000_ISIC ./resultados

  El pipeline usa la misma configuracion que la API:

    - Con MLFLOW_TRACKING_URI: Model Registry, alias champion.
    - Sin MLFLOW_TRACKING_URI: modelo_dermascan.keras local.

  Salidas:

    resultados/predicciones.parquet
    resultados/predicciones.csv
    resultados/resumen.json

========================================================
## ENDPOINTS DE LA API
========================================================
  GET  /                Interfaz web principal
  GET  /salud           Health-check del modelo/configuracion
  GET  /metricas        Metricas finales del modelo
  GET  /roc             Curva ROC en base64
  POST /analizar        Fast path ONNX: diagnostico principal
  POST /predict         Alias de /analizar
  POST /explicabilidad  Grad-CAM, ABCDE, visuales e incertidumbre

========================================================
## METRICAS FINALES
========================================================
  Dataset final:         HAM10000 + ISIC
  Total imagenes:        13,309
  Test set:              1,997
  Modelo final:          EfficientNetB4
  Accuracy:              92.39%
  AUC ROC:               0.9706
  F1 macro:              0.9009
  Precision macro:       0.9009
  Recall macro:          0.9009
  Baseline HOG + SVM:    74.26%
  Mejora vs baseline:    +18.13 puntos porcentuales
  Umbral balanceado:     0.70
  Umbral screening:      0.45

========================================================
## NOTA ACADEMICA
========================================================
  Esta herramienta es academica y no sustituye el diagnostico de un
  dermatologo certificado. El modelo fue entrenado sobre HAM10000 + ISIC y
  esta limitado a lesiones similares a esos datasets.
