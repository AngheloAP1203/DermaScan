"""Análisis morfológico ABCDE de la lesión mediante procesamiento de imagen
(OpenCV): Asimetría, Borde, Color y Diámetro."""
import numpy as np

from .config import IMG_SIZE


def _etiqueta(score):
    if score < 0.34:  return 'Bajo'
    if score < 0.67:  return 'Medio'
    return 'Alto'


def _crit(nombre, score, valor):
    return {'nombre': nombre, 'score': round(float(score), 2),
            'etiqueta': _etiqueta(score), 'valor': valor}


def segmentar(img_rgb):
    """Segmenta la lesión por umbral de Otsu y morfología.

    Devuelve (img_redimensionada, mask_contorno, contorno, area) o None si no
    se detecta una lesión clara. Reutilizado por ABCDE y por las visualizaciones.
    """
    import cv2
    img  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu invertido: la lesión suele ser más oscura que la piel
    _, mask = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c    = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < (IMG_SIZE * IMG_SIZE * 0.005):                  # lesión no detectada
        return None

    mask_c = np.zeros_like(mask)
    cv2.drawContours(mask_c, [c], -1, 255, -1)
    return img, mask_c, c, area


def analisis_abcde(img_rgb):
    """Calcula los criterios ABCDE. Devuelve dict o None si no hay lesión clara."""
    import cv2
    try:
        seg = segmentar(img_rgb)
        if seg is None:
            return None
        img, mask_c, c, area = seg

        # --- A: Asimetría ---------------------------------------------------
        x, y, w, h = cv2.boundingRect(c)
        roi     = mask_c[y:y+h, x:x+w]
        dif_h   = np.count_nonzero(cv2.bitwise_xor(roi, cv2.flip(roi, 1)))
        dif_v   = np.count_nonzero(cv2.bitwise_xor(roi, cv2.flip(roi, 0)))
        base_px = max(np.count_nonzero(roi), 1)
        asim    = min((dif_h + dif_v) / (2 * base_px), 1.0)

        # --- B: Borde (irregularidad por circularidad) ----------------------
        perim = cv2.arcLength(c, True)
        circ  = (perim ** 2) / (4 * np.pi * area) if area > 0 else 1.0
        borde = min(max((circ - 1.0) / 2.0, 0.0), 1.0)

        # --- C: Color (variedad cromática dentro de la lesión) --------------
        pix = img[mask_c == 255]
        if len(pix) > 0:
            std   = np.std(pix.astype(np.float32), axis=0).mean()
            color = min(std / 60.0, 1.0)
        else:
            std, color = 0.0, 0.0

        # --- D: Diámetro equivalente (relativo, en px) ----------------------
        diam_px    = int(np.sqrt(4 * area / np.pi))
        diam_score = min(diam_px / IMG_SIZE, 1.0)

        return {
            'asimetria': _crit('Asimetria', asim,  f'{asim*100:.0f}% disimetria'),
            'borde':     _crit('Borde',     borde, f'circularidad {circ:.2f}'),
            'color':     _crit('Color',     color, f'variacion {std:.0f}' if len(pix) else 'n/d'),
            'diametro':  _crit('Diametro',  diam_score, f'~{diam_px} px'),
        }
    except Exception as e:
        print(f"[ABCDE] error: {e}")
        return None
