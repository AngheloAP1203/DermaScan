FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY src/ ./src/
COPY templates/ ./templates/
# modelo_dermascan.onnx: generado localmente con `python convertir_onnx.py`
# ANTES del build. Horneado en la imagen a proposito (prioridad = velocidad
# de despliegue ya, no gobernanza) — el Fast Path (/analizar) lo usa directo,
# sin red. src/modelo.py (TF, para /explicabilidad) SIGUE resolviendo desde
# el Registry en runtime, así que MLFLOW_TRACKING_URI sigue siendo obligatoria.
COPY modelo_dermascan.onnx ./
EXPOSE 7860

# src/modelo.py (TF) sigue resolviendo el modelo contra el Model Registry de
# MLflow en arranque (alias "champion") — solo para /explicabilidad. Inyectar
# en runtime, no en build (docker run -e / secretos del orquestador):
#   MLFLOW_TRACKING_URI   servidor MLflow remoto (obligatoria)
#   MLFLOW_MODEL_NAME     opcional, default "dermascan-clasificador-piel"
#   MLFLOW_MODEL_ALIAS    opcional, default "champion"

# gthread + multiples workers/threads: un solo worker serializaba TODAS las
# peticiones (el bug que la auditoria original senalo). Con 4 workers x 4
# threads hay 16 peticiones concurrentes en vuelo — dimensionar segun CPU
# real del host (regla practica: workers = 2-4 x nucleos).
# --timeout es logico, no el default de gunicorn: cubre el peor caso real
# (endpoints pesados como /explicabilidad o /incertidumbre), no el Fast Path
# (/analizar), que debe resolver muy por debajo de eso. Un worker que tarda
# mas que esto se considera colgado y gunicorn lo mata y reemplaza — con 1
# solo worker eso significa caida total del servicio; con 4, solo se pierde
# 1/4 de la capacidad mientras se recicla.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", \
     "--workers", "4", "--threads", "4", "--worker-class", "gthread", \
     "--timeout", "30", "--graceful-timeout", "15", \
     "--max-requests", "500", "--max-requests-jitter", "50", \
     "app:app"]
