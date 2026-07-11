FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY registrar_experimento.py ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
# modelo_dermascan.onnx: generado localmente con `python convertir_onnx.py`
# ANTES del build. Horneado en la imagen a proposito: el Fast Path (/analizar)
# lo usa directo, sin red.
COPY modelo_dermascan.onnx ./
COPY modelo_dermascan.keras ./
EXPOSE 7860

# En produccion, src/config.py resuelve el modelo contra el Model Registry de
# MLflow en arranque (alias "champion"). En desarrollo local, si no se define
# MLFLOW_TRACKING_URI, usa modelo_dermascan.keras horneado en la imagen:
#   MLFLOW_TRACKING_URI   servidor MLflow remoto (produccion)
#   MLFLOW_MODEL_NAME     opcional, default "dermascan-clasificador-piel"
#   MLFLOW_MODEL_ALIAS    opcional, default "champion"

# gthread + multiples workers/threads: un solo worker serializaba TODAS las
# peticiones (el bug que la auditoria original senalo). Cada worker carga su
# propia copia completa del modelo (TF/Keras + ONNX Runtime, ver src/modelo.py
# y src/modelo_onnx.py), y src/modelo.py ademas configura TF para usar TODOS
# los nucleos de CPU por worker — con varios workers eso compite por los
# mismos nucleos. Los defaults de abajo (2 workers x 2 threads = 4 peticiones
# concurrentes) estan pensados para correr en una laptop de desarrollo sin
# saturar RAM/CPU; en un servidor real con mas nucleos, subir
# GUNICORN_WORKERS/GUNICORN_THREADS por variable de entorno (regla practica:
# workers = 2-4 x nucleos del host).
# --timeout es logico, no el default de gunicorn: cubre el peor caso real
# (endpoints pesados como /explicabilidad o /incertidumbre), no el Fast Path
# (/analizar), que debe resolver muy por debajo de eso. Un worker que tarda
# mas que esto se considera colgado y gunicorn lo mata y reemplaza.
ENV GUNICORN_WORKERS=2
ENV GUNICORN_THREADS=2
CMD gunicorn --bind 0.0.0.0:7860 \
     --workers ${GUNICORN_WORKERS} --threads ${GUNICORN_THREADS} --worker-class gthread \
     --timeout 30 --graceful-timeout 15 \
     --max-requests 500 --max-requests-jitter 50 \
     app:app
