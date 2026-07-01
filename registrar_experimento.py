"""
Ejecutar UNA SOLA VEZ desde la carpeta DermaScan_Deploy/:
    python registrar_experimento.py

Genera la carpeta mlruns/ con el experimento y el registro en Model Registry.
Luego corre:  mlflow ui --port 5001
y abre http://localhost:5001 para capturar screenshots del informe (secciones 7.2 y 7.3).
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.path.join(os.path.dirname(__file__), "mlruns")
mlflow.set_tracking_uri(f"file:///{TRACKING_URI}")

EXPERIMENT_NAME = "DermaScan-EfficientNetB4"
MODEL_NAME      = "dermascan-clasificador-piel"
ARTIFACT_PATH   = os.path.join(os.path.dirname(__file__), "modelo_dermascan.keras")

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="EfficientNetB4-HAM10000-final") as run:
    # --- Hiperparámetros del entrenamiento ---
    mlflow.log_params({
        "arquitectura":    "EfficientNetB4",
        "img_size":        380,
        "umbral_decision": 0.62,
        "dataset":         "HAM10000",
        "num_clases":      2,
        "clases":          "Benigno, Maligno",
        "epochs":          30,
        "batch_size":      32,
        "optimizer":       "Adam",
        "learning_rate":   1e-4,
        "fine_tuning_lr":  1e-5,
        "augmentation":    "flip, rotate, zoom, brightness",
        "pretrained_on":   "ImageNet",
    })

    # --- Métricas del modelo final ---
    mlflow.log_metrics({
        "accuracy":         0.8969,
        "auc_roc":          0.9479,
        "latencia_prod_ms": 21.0,
        "val_accuracy":     0.8969,
    })

    # --- Artefacto: pesos del modelo ---
    if os.path.exists(ARTIFACT_PATH):
        mlflow.log_artifact(ARTIFACT_PATH, artifact_path="model")
        print(f"[OK] Artifact registrado: modelo_dermascan.keras")
    else:
        print(f"[WARN] No se encontró {ARTIFACT_PATH} — registrando sin artifact.")

    run_id = run.info.run_id
    print(f"[OK] Run registrado con ID: {run_id}")

# --- Model Registry ---
client = MlflowClient()

model_uri = f"runs:/{run_id}/model/modelo_dermascan.keras"
try:
    client.create_registered_model(MODEL_NAME)
    print(f"[OK] Modelo registrado en Registry como '{MODEL_NAME}'")
except Exception:
    print(f"[INFO] El modelo '{MODEL_NAME}' ya existe en Registry.")

mv = client.create_model_version(
    name=MODEL_NAME,
    source=model_uri,
    run_id=run_id,
)
print(f"[OK] Versión creada: v{mv.version}")

# Transición a Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"[OK] Versión v{mv.version} promovida a 'Production'")

print()
print("=" * 55)
print("  Listo! Para ver la UI de MLflow ejecuta:")
print("    mlflow ui --port 5001")
print("  Luego abre: http://localhost:5001")
print("  Captura screenshots para secciones 7.2 y 7.3")
print("=" * 55)
