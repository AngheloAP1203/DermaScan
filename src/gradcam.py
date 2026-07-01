"""Grad-CAM : mapa de calor que explica qué zonas de la imagen influyeron más en
la decisión del modelo (explicabilidad / XAI).

Maneja el caso (común en transfer learning) de un EfficientNetB4 anidado como
submodelo: construir un grad-model apuntando directamente a una capa interna
provoca "Graph disconnected" en Keras 3, así que se opera sobre el submodelo base
y se reejecuta la "cola" de capas (pooling + densas) para obtener la predicción."""
import base64
import numpy as np
import tensorflow as tf
import keras

from .config import IMG_SIZE
from .modelo import modelo, _buscar_ultima_conv


def _localizar_conv_y_base():
    """Devuelve (submodelo_base o None, capa_conv). Si el conv está directo en el
    modelo, base es None. Si está dentro de un submodelo anidado, lo devuelve."""
    for capa in modelo.layers:
        if isinstance(capa, keras.Model):               # submodelo anidado (EfficientNet)
            conv = _buscar_ultima_conv(capa)
            if conv is not None:
                return capa, conv
    return None, _buscar_ultima_conv(modelo)


def heatmap_array(tensor):
    """API pública: mapa Grad-CAM normalizado [0,1] como ndarray (para comparativas)."""
    return _heatmap(tensor)


def _heatmap(tensor):
    """Grad-CAM clásico robusto. Devuelve ndarray HxW normalizado [0,1]."""
    base, conv = _localizar_conv_y_base()
    if conv is None:
        raise ValueError("no se encontró capa convolucional")

    if base is None:
        # Modelo plano: la capa conv es accesible desde la entrada del modelo
        grad_model = keras.models.Model(modelo.inputs, [conv.output, modelo.output])
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(tensor, training=False)
            score = pred[:, 0]
        grads = tape.gradient(score, conv_out)
    else:
        # Modelo anidado: grad-model sobre el submodelo base + reejecutar la cola
        sub = keras.models.Model(base.inputs, [conv.output, base.output])
        idx_base = modelo.layers.index(base)
        cola = modelo.layers[idx_base + 1:]
        with tf.GradientTape() as tape:
            conv_out, x = sub(tensor, training=False)
            for capa in cola:
                x = capa(x, training=False)
            score = x[:, 0]
        grads = tape.gradient(score, conv_out)

    if grads is None:
        raise ValueError("gradientes nulos")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))      # peso por canal
    conv_a = conv_out[0]
    heat = tf.reduce_sum(conv_a * pooled, axis=-1)
    heat = tf.nn.relu(heat)
    maxv = tf.reduce_max(heat)
    if maxv > 0:
        heat = heat / maxv
    return heat.numpy()


def generar_gradcam(tensor, img_rgb_uint8):
    """Devuelve dict con overlay y heatmap puro en base64, o {'error': ...} si falla.

    {'overlay_b64': PNG superpuesto, 'heatmap_b64': PNG del mapa de calor coloreado}
    """
    import cv2
    try:
        heat = _heatmap(tensor)

        heat = cv2.resize(heat, (IMG_SIZE, IMG_SIZE))
        heat_u8    = np.uint8(255 * heat)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        base    = cv2.resize(img_rgb_uint8, (IMG_SIZE, IMG_SIZE))
        overlay = cv2.addWeighted(base, 0.6, heat_color, 0.4, 0)

        def _b64(img_rgb):
            ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf).decode('utf-8') if ok else None

        return {'overlay_b64': _b64(overlay), 'heatmap_b64': _b64(heat_color)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GradCAM] error: {e}", flush=True)
        return {'error': str(e)}
