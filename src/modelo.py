"""Carga del modelo, funcion de inferencia y capa usada por Grad-CAM."""
import tensorflow as tf
import keras
import mlflow.keras

from .config import MODEL_URI, IMG_SIZE, USAR_MLFLOW

# Usar todos los núcleos de CPU disponibles (0 = TensorFlow decide según el sistema)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

# Carga unica del modelo al importar el modulo.
if USAR_MLFLOW:
    modelo = mlflow.keras.load_model(MODEL_URI)
else:
    modelo = keras.models.load_model(MODEL_URI, compile=False)


@tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
def inferir(x):
    """Inferencia optimizada (grafo compilado) sobre un batch de imágenes."""
    return modelo(x, training=False)


def _buscar_ultima_conv(m):
    """Busca recursivamente la última capa con salida 4D (mapa de activación)."""
    ultima = None
    for capa in m.layers:
        if hasattr(capa, 'layers') and capa.layers:          # submodelo anidado
            sub = _buscar_ultima_conv(capa)
            if sub is not None:
                ultima = sub
        else:
            try:
                if len(capa.output.shape) == 4:
                    ultima = capa
            except Exception:
                pass
    return ultima


# Detección de la capa convolucional para Grad-CAM (se hace una sola vez)
CAPA_CONV = None
try:
    CAPA_CONV = _buscar_ultima_conv(modelo)
    print(f"[GradCAM] Capa convolucional detectada: {CAPA_CONV.name if CAPA_CONV else 'NINGUNA'}")
except Exception as e:
    print(f"[GradCAM] No se pudo detectar capa conv: {e}")
