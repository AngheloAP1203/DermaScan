"""Integrated Gradients: técnica XAI que atribuye la decisión integrando los
gradientes a lo largo de un camino desde una imagen base (negro) hasta la imagen
real. Vectorizado: todas las interpolaciones en un solo lote (rápido)."""
import base64
import numpy as np
import tensorflow as tf

from .config import IMG_SIZE
from .modelo import modelo


def _b64(img_rgb):
    import cv2
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8') if ok else None


def integrated_gradients(img_rgb, steps=5, chunk=5):
    """Devuelve overlay y heatmap de atribución en base64. Procesa las
    interpolaciones en bloques pequeños (rápido y sin agotar memoria)."""
    import cv2
    try:
        base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')
        x = tf.convert_to_tensor(base)                       # (H,W,3) en [0,255]
        baseline = tf.zeros_like(x)

        alphas = tf.linspace(0.0, 1.0, steps)
        grad_acum = tf.zeros_like(x)
        for i in range(0, steps, chunk):
            a = tf.reshape(alphas[i:i+chunk], (-1, 1, 1, 1))
            interp = baseline + a * (x - baseline)           # (c,H,W,3)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                preds = modelo(interp, training=False)
                total = tf.reduce_sum(preds[:, 0])
            g = tape.gradient(total, interp)                 # (c,H,W,3)
            grad_acum += tf.reduce_sum(g, axis=0)
        avg_grad = grad_acum / float(steps)
        ig = (x - baseline) * avg_grad

        attr = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()
        p99 = np.percentile(attr, 99)
        if p99 > 0:
            attr = np.clip(attr / p99, 0, 1)
        attr = cv2.GaussianBlur(attr, (5, 5), 0)

        u8 = np.uint8(255 * attr)
        col = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base.astype('uint8'), 0.5, col, 0.5, 0)

        return {'overlay_b64': _b64(overlay), 'heatmap_b64': _b64(col)}
    except Exception as e:
        print(f"[IntegratedGrad] error: {e}", flush=True)
        return {'error': str(e)}
