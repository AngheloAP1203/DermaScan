"""Convierte el modelo 'champion' de MLflow a ONNX.
Uso: python convertir_onnx.py
Requiere: pip install tf2onnx
"""
import os
import glob
import tensorflow as tf
import tf2onnx
import mlflow.artifacts
import keras

from src.config import MODEL_URI, IMG_SIZE

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_dermascan.onnx")

# mlflow.keras.load_model() falla en Windows nativo (bug conocido: interpreta
# "C:" como scheme de URI). Workaround: descargar el artifact directo y
# cargarlo con Keras nativo (mismo resultado, sin pasar por ese bug).
_local_dir = mlflow.artifacts.download_artifacts(MODEL_URI)
_keras_file = glob.glob(_local_dir + "/**/model.keras", recursive=True)[0]
modelo = keras.models.load_model(_keras_file, compile=False)
spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
tf2onnx.convert.from_keras(modelo, input_signature=spec, opset=13, output_path=OUT_PATH)
print(f"[OK] ONNX guardado en {OUT_PATH}")
