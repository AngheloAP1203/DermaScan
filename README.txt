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
## PASO 2 – Registrar el experimento en MLflow (Sección 7.2 + 7.3)
========================================================
  python registrar_experimento.py

  Genera la carpeta mlruns/ con:
    - Experimento "DermaScan-EfficientNetB4" con hiperparámetros y métricas
    - Artifact: modelo_dermascan.keras
    - Model Registry "dermascan-clasificador-piel" v1 en estado "Production"

  Para ver la UI (y capturar screenshots del informe):
    mlflow ui --port 5001
    Abre: http://localhost:5001

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
