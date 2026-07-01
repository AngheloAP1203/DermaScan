FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modelo_dermascan.keras .
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/

ENV PORT=7860
EXPOSE 7860
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 300 app:app
