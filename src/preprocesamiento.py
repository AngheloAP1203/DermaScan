"""Decodificación de imágenes base64 y preparación del tensor de entrada."""
import base64
import numpy as np
import tensorflow as tf

from .config import IMG_SIZE


def decodificar(b64):
    """Convierte una imagen base64 en un arreglo RGB uint8."""
    import cv2
    datos = base64.b64decode(b64)
    arr   = np.frombuffer(datos, np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocesar(img_rgb):
    """Redimensiona la imagen RGB y la convierte en tensor batch [1, H, W, 3]."""
    t = tf.image.resize(tf.cast(img_rgb, tf.float32), [IMG_SIZE, IMG_SIZE])
    return tf.expand_dims(t, 0)
