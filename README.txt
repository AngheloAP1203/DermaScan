# DermaScan – Despliegue Completo (Sección 7 MLOps)

Proyecto: DermaScan | UPAO 2026-10 | Percepción Computacional
Accuracy : 89.69%  |  AUC ROC: 0.9479  |  Latencia producción: 21 ms
Modelo   : EfficientNetB4 | Dataset: HAM10000

========================================================
## CONTENIDO
========================================================
  modelo_dermascan.keras     Pesos entrenados de EfficientNetB4 (380 px)
  app.py                     API REST Flask + Interfaz web animada
  registrar_experimento.py   Registra métricas y pesos en MLflow
  requirements.txt           Dependencias exactas
  Dockerfile                 Para despliegue en cualquier servidor Docker

========================================================
## PASO 1 – Instalar dependencias
========================================================
  pip install -r requirements.txt

========================================================
## PASO 2 – Registrar el modelo en MLflow (Model Registry)
========================================================
  export MLFLOW_TRACKING_URI=https://<tu-servidor-mlflow>   # obligatorio

  python registrar_experimento.py \
      --modelo modelo_dermascan.keras \
      --predicciones predicciones_val.csv \
      --dataset "HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)" \
      --promover

  `predicciones_val.csv` (y_true,y_score) se exporta del set de
  validación/prueba del notebook de entrenamiento — el umbral óptimo y el de
  screening se CALCULAN desde ahí (Youden's J / sensibilidad objetivo), no
  se escriben a mano. Sin `--promover` el modelo queda registrado pero no
  sirve producción (ver alias "champion").

  La API Flask y el pipeline de Spark leen el modelo y el umbral desde este
  Registry en cada arranque (src/config.py) — no hay más constantes locales.

========================================================
## PASO 3 – Correr el servidor web (Sección 7.1)
========================================================

  ### Opción A: Python directo
    python app.py
    Abre: http://localhost:5000

  ### Opción B: Gunicorn (producción)
    gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app
    Abre: http://localhost:5000

  ### Opción C: Docker (recomendado para despliegue real)
    docker build -t dermascan:2.0 .
    docker run -p 5000:5000 dermascan:2.0
    Abre: http://localhost:5000

========================================================
## ENDPOINTS DE LA API
========================================================
  GET  /           Interfaz web principal
  POST /analizar   Recibe JSON {"imagen": "<base64>"}, devuelve diagnóstico
  POST /predict    Alias de /analizar (nomenclatura estándar)
  GET  /salud      Health-check: estado del modelo

========================================================
## NOTA ACADÉMICA
========================================================
  Esta herramienta es académica y no sustituye el diagnóstico de un dermatólogo.
  Modelo entrenado sobre HAM10000. Limitado a lesiones similares al dataset.
