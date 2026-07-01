"""Análisis avanzados que ejercitan el modelo entrenado de forma intensiva:

- TTA (Test-Time Augmentation): promedia la predicción sobre varias vistas
  aumentadas → predicción más robusta + estimación de incertidumbre.
- Mapa de oclusión: tapa regiones de la imagen y mide cuánto cae la predicción
  de malignidad → segunda técnica de explicabilidad complementaria al Grad-CAM.

Ambos son costosos en CPU, por eso se invocan bajo demanda (endpoints propios)."""
import base64
import numpy as np
import tensorflow as tf

from .config import IMG_SIZE, UMBRAL
from .modelo import inferir


def tta(img_rgb):
    """Test-Time Augmentation. Devuelve probabilidad media, incertidumbre (std) y
    la lista de probabilidades por vista."""
    import cv2
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')

    vistas = [
        base,
        cv2.flip(base, 1),                 # espejo horizontal
        cv2.flip(base, 0),                 # espejo vertical
        cv2.flip(base, -1),                # ambos
        np.rot90(base, 1).copy(),          # 90°
        np.rot90(base, 2).copy(),          # 180°
        np.rot90(base, 3).copy(),          # 270°
        np.clip(base * 1.1, 0, 255),       # +brillo
    ]
    batch = tf.stack([tf.convert_to_tensor(v, dtype=tf.float32) for v in vistas])
    preds = inferir(batch).numpy().flatten()

    media = float(preds.mean())
    std   = float(preds.std())
    clase = 'Maligno' if media >= UMBRAL else 'Benigno'
    conf  = media if clase == 'Maligno' else 1 - media

    return {
        'n_vistas': len(vistas),
        'prob_media': round(media, 4),
        'incertidumbre': round(std, 4),
        'clase': clase,
        'confianza_pct': f'{conf*100:.1f}%',
        'probs': [round(float(p), 4) for p in preds],
        'prob_min': round(float(preds.min()), 4),
        'prob_max': round(float(preds.max()), 4),
    }


def _b64(img_rgb):
    import cv2
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8') if ok else None


def _oclusion_array(img_rgb, grid=5):
    """Calcula el mapa de oclusión. Devuelve (heat_380 [0,1], prob_base, caida_max)."""
    import cv2
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')

    p0 = float(inferir(tf.expand_dims(tf.convert_to_tensor(base), 0))[0][0])

    paso = IMG_SIZE // grid
    gris = 128.0
    ocluidas, coords = [], []
    for i in range(grid):
        for j in range(grid):
            y0, x0 = i * paso, j * paso
            img_o = base.copy()
            img_o[y0:y0+paso, x0:x0+paso] = gris
            ocluidas.append(img_o)
            coords.append((i, j))

    batch = tf.stack([tf.convert_to_tensor(o, dtype=tf.float32) for o in ocluidas])
    preds = inferir(batch).numpy().flatten()

    mapa = np.zeros((grid, grid), np.float32)
    for (i, j), p in zip(coords, preds):
        mapa[i, j] = max(p0 - p, 0.0)            # solo caídas (zonas que sostienen la decisión)

    maxv = float(mapa.max())
    if maxv > 0:
        mapa = mapa / maxv

    heat = cv2.resize(mapa, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    return heat, p0, maxv


def oclusion(img_rgb, grid=5):
    """Mapa de sensibilidad por oclusión. Devuelve overlay y heatmap base64 +
    la caída máxima de probabilidad."""
    import cv2
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('uint8')
    heat, p0, maxv = _oclusion_array(img_rgb, grid)

    heat_u8    = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)

    return {
        'overlay_b64': _b64(overlay),
        'heatmap_b64': _b64(heat_color),
        'prob_base': round(p0, 4),
        'caida_max': round(maxv, 4),
        'grid': grid,
    }


def robustez(img_rgb):
    """Mide cómo cambia la probabilidad de malignidad al rotar y variar el brillo.
    Una inferencia por lotes para rotaciones y otra para brillos."""
    import cv2
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')

    p0 = float(inferir(tf.expand_dims(tf.convert_to_tensor(base), 0))[0][0])

    angs = [0, 90, 180, 270]
    rot_imgs = [np.rot90(base, k).copy() for k in range(4)]
    batch_r = tf.stack([tf.convert_to_tensor(v, dtype=tf.float32) for v in rot_imgs])
    probs_r = inferir(batch_r).numpy().flatten()

    factores = [0.7, 0.85, 1.0, 1.15, 1.3]
    bri_imgs = [np.clip(base * f, 0, 255) for f in factores]
    batch_b = tf.stack([tf.convert_to_tensor(v, dtype=tf.float32) for v in bri_imgs])
    probs_b = inferir(batch_b).numpy().flatten()

    todas = np.concatenate([probs_r, probs_b])
    estabilidad = float(1.0 - todas.std())

    return {
        'prob_base': round(p0, 4),
        'rotaciones': [{'ang': a, 'prob': round(float(p), 4)} for a, p in zip(angs, probs_r)],
        'brillos':    [{'factor': f, 'prob': round(float(p), 4)} for f, p in zip(factores, probs_b)],
        'estabilidad': round(estabilidad, 4),
    }


def diagnostico_enfocado(img_rgb):
    """Recorta la lesión segmentada y la vuelve a clasificar (predicción enfocada).
    Compara con la predicción global."""
    import cv2
    from .abcde import segmentar
    from .config import UMBRAL

    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')
    prob_global = float(inferir(tf.expand_dims(tf.convert_to_tensor(base), 0))[0][0])

    seg = segmentar(img_rgb)
    if seg is None:
        return {'error': 'No se pudo segmentar una lesión para enfocar.'}
    img, mask_c, c, area = seg
    x, y, w, h = cv2.boundingRect(c)
    m = int(0.12 * max(w, h))                                # margen
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, IMG_SIZE), min(y + h + m, IMG_SIZE)
    crop = img[y0:y1, x0:x1]
    crop_r = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype('float32')
    prob_enf = float(inferir(tf.expand_dims(tf.convert_to_tensor(crop_r), 0))[0][0])

    def clase(p): return 'Maligno' if p >= UMBRAL else 'Benigno'
    coincide = clase(prob_global) == clase(prob_enf)
    interp = ('La predicción enfocada coincide con la global, lo que refuerza el diagnóstico.'
              if coincide else
              'La predicción enfocada difiere de la global: el contexto de la piel influye en la decisión.')

    return {
        'prob_global': round(prob_global, 4), 'clase_global': clase(prob_global),
        'prob_enfocado': round(prob_enf, 4), 'clase_enfocado': clase(prob_enf),
        'coincide': coincide, 'interpretacion': interp,
        'recorte_b64': _b64(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype('uint8')),
    }


def robustez_ruido(img_rgb):
    """Añade ruido gaussiano creciente y mide cómo cae la probabilidad. Una sola
    inferencia por lotes."""
    import cv2
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')

    niveles = [0, 5, 10, 20, 40]
    imgs = []
    for s in niveles:
        if s == 0:
            imgs.append(base.copy())
        else:
            ruido = np.random.normal(0, s, base.shape).astype('float32')
            imgs.append(np.clip(base + ruido, 0, 255))
    batch = tf.stack([tf.convert_to_tensor(v, dtype=tf.float32) for v in imgs])
    probs = inferir(batch).numpy().flatten()

    caida = float(abs(probs[0] - probs[-1]))
    return {
        'niveles': [{'sigma': s, 'prob': round(float(p), 4)} for s, p in zip(niveles, probs)],
        'prob_base': round(float(probs[0]), 4),
        'caida_total': round(caida, 4),
        'estable': caida < 0.15,
    }


def deteccion_ventana(img_rgb, grid=5):
    """Mapa de detección por ventana deslizante: recorta regiones solapadas, las
    clasifica y arma un mapa de malignidad. Localiza la zona más sospechosa.
    Una sola inferencia por lotes (grid x grid ventanas)."""
    import cv2
    full = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('uint8')

    win  = IMG_SIZE // 2                      # ventana = mitad de la imagen
    if grid > 1:
        paso = (IMG_SIZE - win) // (grid - 1)
    else:
        paso = 0

    ventanas, centros = [], []
    for i in range(grid):
        for j in range(grid):
            y0, x0 = i * paso, j * paso
            crop = full[y0:y0+win, x0:x0+win]
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype('float32')
            ventanas.append(crop)
            centros.append((y0 + win // 2, x0 + win // 2))

    batch = tf.stack([tf.convert_to_tensor(v, dtype=tf.float32) for v in ventanas])
    probs = inferir(batch).numpy().flatten()

    # acumular en un mapa por la posición del centro de cada ventana
    mapa = np.zeros((grid, grid), np.float32)
    k = 0
    for i in range(grid):
        for j in range(grid):
            mapa[i, j] = probs[k]; k += 1

    idx = int(np.argmax(probs))
    cy, cx = centros[idx]
    prob_max = float(probs.max())

    heat = cv2.resize(mapa, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    heat_u8    = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(full, 0.55, heat_color, 0.45, 0)

    # marcar la región más sospechosa
    r = win // 2
    cv2.rectangle(overlay, (max(cx-r, 0), max(cy-r, 0)),
                  (min(cx+r, IMG_SIZE), min(cy+r, IMG_SIZE)), (255, 255, 255), 2)

    return {
        'overlay_b64': _b64(overlay),
        'heatmap_b64': _b64(heat_color),
        'prob_max': round(prob_max, 4),
        'grid': grid,
        'n_ventanas': len(ventanas),
    }


def analisis_canales(img_rgb):
    """Pasa la imagen por el modelo aislando canales de color para ver cuál pesa
    más en la decisión. Una sola inferencia por lotes."""
    import cv2
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')

    solo_r = base.copy(); solo_r[:, :, 1] = 0; solo_r[:, :, 2] = 0
    solo_g = base.copy(); solo_g[:, :, 0] = 0; solo_g[:, :, 2] = 0
    solo_b = base.copy(); solo_b[:, :, 0] = 0; solo_b[:, :, 1] = 0
    gris1  = cv2.cvtColor(base.astype('uint8'), cv2.COLOR_RGB2GRAY)
    gris3  = cv2.cvtColor(gris1, cv2.COLOR_GRAY2RGB).astype('float32')

    variantes = [('Original', base), ('Solo R', solo_r), ('Solo G', solo_g),
                 ('Solo B', solo_b), ('Grises', gris3)]
    batch = tf.stack([tf.convert_to_tensor(v, dtype=tf.float32) for _, v in variantes])
    probs = inferir(batch).numpy().flatten()

    return {
        'canales': [{'nombre': n, 'prob': round(float(p), 4)} for (n, _), p in zip(variantes, probs)],
        'prob_base': round(float(probs[0]), 4),
    }
