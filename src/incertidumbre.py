"""Incertidumbre por Monte Carlo Dropout: deja el dropout ACTIVO en inferencia y
hace varias pasadas. La dispersión de las predicciones estima cuánto "duda" el
modelo (incertidumbre epistémica). El mapa espacial se obtiene de la varianza de
las activaciones convolucionales bajo dropout. Vectorizado: una sola pasada con
un lote de n copias (rápido, sin gradientes)."""
import base64
import numpy as np
import tensorflow as tf
import keras

from .config import IMG_SIZE
from .modelo import modelo, CAPA_CONV, _buscar_ultima_conv


def _b64(img_rgb):
    import cv2
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8') if ok else None


def _localizar_base_conv():
    for capa in modelo.layers:
        if isinstance(capa, keras.Model):
            cc = _buscar_ultima_conv(capa)
            if cc is not None:
                return capa, cc
    return None, CAPA_CONV


def mc_dropout(img_rgb, n=6, chunk=6):
    """Stats de incertidumbre + mapa espacial. Procesa en bloques pequeños."""
    import cv2
    try:
        base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')

        base_sub, conv = _localizar_base_conv()
        if conv is None:
            raise ValueError("sin capa convolucional")

        if base_sub is None:
            gm = keras.models.Model(modelo.inputs, [conv.output, modelo.output])
        else:
            sub = keras.models.Model(base_sub.inputs, [conv.output, base_sub.output])
            idx = modelo.layers.index(base_sub); cola = modelo.layers[idx+1:]

        probs_all, conv_all = [], []
        hecho = 0
        while hecho < n:
            c = min(chunk, n - hecho)
            xs = tf.convert_to_tensor(np.repeat(base[None], c, axis=0))   # (c,H,W,3)
            if base_sub is None:
                conv_out, preds = gm(xs, training=True)
            else:
                conv_out, x = sub(xs, training=True)
                for capa in cola:
                    x = capa(x, training=True)
                preds = x
            probs_all.append(preds[:, 0].numpy())
            conv_all.append(conv_out.numpy())
            hecho += c

        probs = np.concatenate(probs_all)
        media = float(probs.mean()); std = float(probs.std())

        conv_np = np.concatenate(conv_all, axis=0)                 # (n,h,w,C)
        mapa = conv_np.std(axis=0).mean(axis=-1)                   # (h,w)
        mx = mapa.max()
        if mx > 0:
            mapa = mapa / mx

        mapa = cv2.resize(mapa, (IMG_SIZE, IMG_SIZE))
        u8 = np.uint8(255 * np.clip(mapa, 0, 1))
        col = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base.astype('uint8'), 0.55, col, 0.45, 0)

        if std < 0.05:    nivel = 'Baja incertidumbre'
        elif std < 0.12:  nivel = 'Incertidumbre moderada'
        else:             nivel = 'Alta incertidumbre'

        return {
            'n_pasadas': n,
            'prob_media': round(media, 4),
            'incertidumbre': round(std, 4),
            'nivel': nivel,
            'probs': [round(float(p), 4) for p in probs],
            'overlay_b64': _b64(overlay),
            'heatmap_b64': _b64(col),
        }
    except Exception as e:
        print(f"[MC-Dropout] error: {e}", flush=True)
        return {'error': str(e)}
