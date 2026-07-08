"""Sesión ONNX Runtime para el Fast Path de /analizar — no depende de
TensorFlow en tiempo de inferencia. El grafo de TF (src.modelo) sigue
existiendo aparte para /explicabilidad (Grad-CAM/saliency/integrated
necesitan gradientes reales, que ONNX Runtime no provee)."""
import os
import numpy as np
import onnxruntime as ort

ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelo_dermascan.onnx")

_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def inferir_onnx(tensor):
    """tensor: [1, H, W, 3] float32 (mismo formato que preprocesar()). Devuelve prob_maligno (float)."""
    x = np.asarray(tensor, dtype=np.float32)
    salida = _session.run(None, {_input_name: x})
    return float(salida[0][0][0])
