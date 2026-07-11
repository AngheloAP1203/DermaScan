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
  docker-compose.yml         Levanta app + MLflow + registro champion
  run_mlflow.ps1             Levanta MLflow local en Windows
  run_mlflow.sh              Levanta MLflow local en Linux/macOS
  .env.example               Variables de entorno de referencia

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
## PASO 2 - Levantar todo con Docker Compose
========================================================

  Este es el modo recomendado para despliegue local completo:

    docker compose up --build

  Servicios:

    dermascan-mlflow     MLflow Tracking Server + Model Registry
    dermascan-registrar  Registra/promueve el modelo como alias champion
    dermascan-app        API Flask/Gunicorn consumiendo MLflow

  URLs:

    App DermaScan:  http://localhost:7860
    MLflow UI:      http://localhost:5001

  El registro es idempotente: si el alias champion ya existe, no crea una
  version nueva del modelo. Los datos de MLflow persisten en el volumen Docker
  mlflow-data.

  Para detener:

    docker compose down

  Para borrar tambien el Registry/artefactos locales de MLflow:

    docker compose down -v

========================================================
## PASO 3 - Fuente del modelo
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

  Primero levantar o apuntar a un servidor MLflow.

  Para desarrollo local en Windows:

    .\run_mlflow.ps1

  Si la API Flask correra dentro de Docker y MLflow en tu PC, exponer MLflow
  en todas las interfaces para que el contenedor pueda conectarse:

    .\run_mlflow.ps1 -HostAddress 0.0.0.0

  Para Linux/macOS:

    ./run_mlflow.sh

  Luego, en otra terminal, configurar:

    $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
    $env:MLFLOW_MODEL_NAME="dermascan-clasificador-piel"
    $env:MLFLOW_MODEL_ALIAS="champion"

  Registrar/promover el modelo usando metricas ya documentadas:

    python registrar_experimento.py `
        --modelo modelo_dermascan.keras `
        --dataset "HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)" `
        --umbral-optimo 0.70 `
        --umbral-screening 0.45 `
        --accuracy 0.9239 `
        --auc-roc 0.9706 `
        --fuente "Informe/metricas finales del entrenamiento EfficientNetB4; test set independiente de 1997 imagenes." `
        --promover

  Alternativamente, si se cuenta con predicciones reales y_true,y_score:

    python registrar_experimento.py `
        --modelo modelo_dermascan.keras `
        --predicciones predicciones_val.csv `
        --dataset "HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)" `
        --promover

  La version promovida queda accesible mediante el alias:

    champion

  Verificar desde Python:

    python -c "import src.config as c; print(c.USAR_MLFLOW, c.MODEL_URI, c.UMBRAL, c.MODOS)"

  En produccion, la API Flask y el pipeline Spark leen el modelo y los
  umbrales desde MLflow Model Registry mediante los tags:

    umbral_optimo
    umbral_screening

========================================================
## PASO 4 - Generar modelo ONNX
========================================================

  python convertir_onnx.py

  Sin MLFLOW_TRACKING_URI, convierte modelo_dermascan.keras local.
  Con MLFLOW_TRACKING_URI, convierte el modelo champion desde MLflow.

========================================================
## PASO 5 - Correr el servidor web sin Compose
========================================================

  ### Opcion A: Python directo

    python app.py
    Abre: http://localhost:5000

  ### Opcion B: Gunicorn (Linux/produccion)

    gunicorn --bind 0.0.0.0:7860 --workers 4 --threads 4 --worker-class gthread app:app
    Abre: http://localhost:7860

  ### Opcion C: Docker simple, sin MLflow

    docker build -t dermascan:2.0 .
    docker run -p 7860:7860 dermascan:2.0
    Abre: http://localhost:7860

  Para produccion con MLflow externo:

    docker run -p 7860:7860 -e MLFLOW_TRACKING_URI="https://<tu-servidor-mlflow>" dermascan:2.0

  Si el MLflow local corre en tu PC y la app corre dentro de Docker en Windows:

    docker run -p 7860:7860 `
      -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" `
      -e MLFLOW_MODEL_NAME="dermascan-clasificador-piel" `
      -e MLFLOW_MODEL_ALIAS="champion" `
      dermascan:2.0

  Nota: run_mlflow.ps1/run_mlflow.sh usan MLflow Artifact Proxy
  (--serve-artifacts) y guardan artefactos en ./mlartifacts. Esto permite que
  clientes remotos o contenedores descarguen el modelo desde el servidor
  MLflow por HTTP, sin depender de rutas locales del host.

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
  GET  /arquitectura    Vista tecnica de API, MLOps y pipeline batch
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
