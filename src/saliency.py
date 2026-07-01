"""SmoothGrad : mapa de relevancia a nivel de píxel. Calcula el gradiente de la
puntuación de malignidad respecto a la imagen de entrada, repetido sobre varias
copias con ruido, y promedia el valor absoluto. Más fino que Grad-CAM (resolución
por píxel). Tercera técnica de explicabilidad (XAI)."""
import base64
import numpy as np
import tensorflow as tf

from .config import IMG_SIZE
from .modelo import modelo


def _smoothgrad(tensor, n=5, sigma=0.12, chunk=5):
    """Mapa de saliencia [0,1] HxW promediando gradientes con ruido. Procesa las
    copias con ruido en bloques pequeños (rápido y sin agotar memoria)."""
    base = tf.convert_to_tensor(tensor, dtype=tf.float32)[0]       # (H,W,3) en [0,255]
    escala = 255.0 * sigma
    acum = tf.zeros((IMG_SIZE, IMG_SIZE), dtype=tf.float32)
    hecho = 0
    while hecho < n:
        c = min(chunk, n - hecho)
        ruido = tf.random.normal((c, IMG_SIZE, IMG_SIZE, 3), mean=0.0, stddev=escala)
        xs = base[None] + ruido                                    # (c,H,W,3)
        with tf.GradientTape() as tape:
            tape.watch(xs)
            preds = modelo(xs, training=False)
            total = tf.reduce_sum(preds[:, 0])
        grads = tape.gradient(total, xs)                           # (c,H,W,3)
        acum += tf.reduce_sum(tf.reduce_max(tf.abs(grads), axis=-1), axis=0)
        hecho += c
    sal = (acum / n).numpy()
    p99 = np.percentile(sal, 99)
    if p99 > 0:
        sal = np.clip(sal / p99, 0, 1)
    return sal


def saliency(img_rgb):
    """Devuelve overlay y heatmap de saliencia en base64."""
    import cv2
    try:
        base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')
        tensor = tf.expand_dims(tf.convert_to_tensor(base), 0)
        sal = _smoothgrad(tensor)

        sal = cv2.GaussianBlur(sal, (5, 5), 0)
        sal_u8 = np.uint8(255 * sal)
        sal_color = cv2.applyColorMap(sal_u8, cv2.COLORMAP_VIRIDIS)
        sal_color = cv2.cvtColor(sal_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base.astype('uint8'), 0.5, sal_color, 0.5, 0)

        def _b64(im):
            ok, buf = cv2.imencode('.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf).decode('utf-8') if ok else None

        return {'overlay_b64': _b64(overlay), 'heatmap_b64': _b64(sal_color)}
    except Exception as e:
        print(f"[Saliency] error: {e}", flush=True)
        return {'error': str(e)}
