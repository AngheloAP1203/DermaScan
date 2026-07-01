"""Chequeo de calidad de la imagen previo al diagnóstico: nitidez, iluminación y
si la imagen realmente parece piel humana fotografiada (y no una ilustración,
dibujo o foto de otra cosa). Todo con OpenCV puro, sin invocar el modelo — es una
validación de sanidad, no una predicción."""
import numpy as np

from .config import IMG_SIZE

UMBRAL_NITIDEZ = 80.0
BRILLO_MIN, BRILLO_MAX = 40, 220
UMBRAL_SCORE_PIEL = 0.50   # score combinado 0..1


def _mascara_piel(img_rgb):
    """Máscara YCrCb del tono de piel humano clásico."""
    import cv2
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype('uint8') * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _blob_mayor_ratio(mask):
    """Proporción de la imagen que ocupa el contorno de piel más grande. Una foto
    real de piel tiene una región contigua grande; ilustraciones con fragmentos
    de tono similar (ojos, detalles) dan blobs pequeños y dispersos."""
    import cv2
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0.0
    area_mayor = max(cv2.contourArea(c) for c in contornos)
    return float(area_mayor) / (mask.shape[0] * mask.shape[1])


def _diversidad_color(img_rgb, mask):
    """Ratio de colores únicos (cuantizados) dentro de la zona de piel candidata,
    normalizado por el número de píxeles. Fotos reales tienen degradados continuos
    (ratio alto); ilustraciones con sombreado plano ("cel shading") usan pocos
    colores en áreas grandes (ratio bajo)."""
    pix = img_rgb[mask > 0]
    n = len(pix)
    if n < 50:
        return 0.0
    cuantizado = (pix // 16).astype(np.int32)          # 16 bins por canal
    claves = cuantizado[:, 0] * 256 + cuantizado[:, 1] * 16 + cuantizado[:, 2]
    n_unicos = len(np.unique(claves))
    return min(n_unicos / (n * 0.15), 1.0)              # normalizado, tope en 1.0


def _porcentaje_piel(mask):
    return float((mask > 0).mean() * 100.0)


def _saturacion_extrema_pct(img_rgb):
    """% de píxeles de TODA la imagen con saturación muy alta. Las ilustraciones
    (anime, dibujos) usan colores planos muy saturados (rojos, azules, amarillos
    puros); las fotos reales de piel son mucho más apagadas y naturales."""
    import cv2
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    return float((sat > 180).mean() * 100.0)


def _densidad_bordes_pct(gris):
    """% de píxeles que son borde fuerte (Canny). El line-art / contorno de tinta
    de dibujos e ilustraciones produce muchos más bordes marcados que una foto de
    piel real, que tiene transiciones suaves."""
    import cv2
    bordes = cv2.Canny(gris, 80, 160)
    return float((bordes > 0).mean() * 100.0)


def evaluar_calidad(img_rgb):
    """Devuelve {nitidez, iluminacion, piel_pct, es_piel, es_piel_confianza, advertencias:[...]}."""
    import cv2
    try:
        img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        nitidez_val = float(cv2.Laplacian(gris, cv2.CV_64F).var())
        nitidez_ok = nitidez_val >= UMBRAL_NITIDEZ

        brillo_val = float(gris.mean())
        iluminacion_ok = BRILLO_MIN <= brillo_val <= BRILLO_MAX
        if brillo_val < BRILLO_MIN:
            iluminacion_txt = 'Imagen muy oscura'
        elif brillo_val > BRILLO_MAX:
            iluminacion_txt = 'Imagen sobreexpuesta'
        else:
            iluminacion_txt = 'Iluminación adecuada'

        mask = _mascara_piel(img)
        piel_pct = _porcentaje_piel(mask)
        blob_ratio = _blob_mayor_ratio(mask)
        diversidad = _diversidad_color(img, mask)
        sat_extrema = _saturacion_extrema_pct(img)
        bordes_pct = _densidad_bordes_pct(gris)

        # Señales a favor de "foto real": blob de piel contiguo, degradados naturales,
        # % de piel. Señales en contra (ilustración/dibujo): colores muy saturados y
        # planos, exceso de bordes marcados (line-art). Un cuerpo grande dibujado
        # (ej. anime) puede superar las primeras 3 señales, pero casi siempre delata
        # colores saturados y/o contornos de tinta -> penalización fuerte.
        score_favor = 0.35 * min(blob_ratio / 0.15, 1.0) + 0.40 * diversidad + 0.25 * min(piel_pct / 15.0, 1.0)
        score_sat    = max(0.0, 1.0 - sat_extrema / 25.0)      # >=25% saturación extrema -> 0
        score_bordes = max(0.0, 1.0 - bordes_pct / 12.0)       # >=12% de bordes fuertes -> 0
        score = 0.55 * score_favor + 0.25 * score_sat + 0.20 * score_bordes
        es_piel = score >= UMBRAL_SCORE_PIEL

        advertencias = []
        if not nitidez_ok:
            advertencias.append('La imagen parece borrosa — el análisis puede ser menos preciso.')
        if not iluminacion_ok:
            advertencias.append(f'{iluminacion_txt} — verifique la iluminación de la foto.')
        if not es_piel:
            advertencias.append('Esta imagen no parece una foto real de piel humana — el resultado puede no ser confiable.')

        return {
            'nitidez': {'valor': round(nitidez_val, 1), 'ok': nitidez_ok},
            'iluminacion': {'valor': round(brillo_val, 1), 'ok': iluminacion_ok, 'texto': iluminacion_txt},
            'piel_pct': round(piel_pct, 1),
            'es_piel': es_piel,
            'es_piel_confianza': round(score, 3),
            'advertencias': advertencias,
        }
    except Exception as e:
        print(f"[Calidad] error: {e}", flush=True)
        return {'error': str(e)}
