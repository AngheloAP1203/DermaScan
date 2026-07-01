"""Visualizaciones de visión por computadora: máscara de segmentación de la
lesión y histograma de distribución de color dentro de ella."""
import base64
import io
import numpy as np

import matplotlib
matplotlib.use('Agg')                 # backend sin GUI (funciona dentro de Docker)
import matplotlib.pyplot as plt

from .abcde import segmentar


def _png_b64_desde_cv(img_bgr):
    import cv2
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        return None
    return base64.b64encode(buf).decode('utf-8')


def generar_visuales(img_rgb):
    """Devuelve {'segmentacion_b64', 'histograma_b64'} o None si no hay lesión."""
    import cv2
    try:
        seg = segmentar(img_rgb)
        if seg is None:
            return None
        img, mask_c, c, area = seg                            # img en RGB, uint8

        # --- 1. Imagen con el contorno de la lesión resaltado ---------------
        overlay = img.copy()
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)    # contorno verde
        # sombreado semitransparente de la zona segmentada
        relleno = img.copy()
        relleno[mask_c == 255] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.8, relleno, 0.2, 0)
        seg_b64 = _png_b64_desde_cv(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # --- 2. Histograma de color (solo píxeles de la lesión) -------------
        fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=100)
        fig.patch.set_facecolor('#12122a')
        ax.set_facecolor('#12122a')
        colores = [('R', '#ff6b6b'), ('G', '#51cf66'), ('B', '#4dabf7')]
        for i, (nombre, col) in enumerate(colores):
            hist = cv2.calcHist([img], [i], mask_c, [64], [0, 256]).flatten()
            ax.plot(np.linspace(0, 255, 64), hist, color=col, linewidth=1.6, label=nombre)
        ax.set_title('Distribucion de color en la lesion', color='#c4b5fd', fontsize=9)
        ax.tick_params(colors='#8888bb', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333366')
        ax.legend(facecolor='#1a1a3e', edgecolor='#333366', labelcolor='#e8e8ff', fontsize=7)
        ax.grid(alpha=0.12)
        fig.tight_layout()

        bio = io.BytesIO()
        fig.savefig(bio, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        bio.seek(0)
        hist_b64 = base64.b64encode(bio.read()).decode('utf-8')

        # --- 3. Detección de bordes (Canny) sobre toda la imagen -----------
        bordes_b64 = None
        try:
            gris_l = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gris_l = cv2.GaussianBlur(gris_l, (3, 3), 0)
            bordes = cv2.Canny(gris_l, 40, 120)                # bordes de toda la lesión/textura
            bordes_rgb = np.zeros((bordes.shape[0], bordes.shape[1], 3), np.uint8)
            bordes_rgb[bordes > 0] = (77, 171, 247)            # bordes en cian
            # superponer el contorno de la lesión segmentada (verde) como referencia
            cv2.drawContours(bordes_rgb, [c], -1, (0, 255, 0), 2)
            bordes_b64 = _png_b64_desde_cv(cv2.cvtColor(bordes_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"[Visuales] Canny error: {e}")

        # --- 4. Paleta de colores dominantes (k-means) ---------------------
        paleta = None
        try:
            pix = img[mask_c == 255].astype(np.float32)
            if len(pix) >= 5:
                k = 5
                criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                _, etiquetas, centros = cv2.kmeans(pix, k, None, criterios, 3, cv2.KMEANS_PP_CENTERS)
                etiquetas = etiquetas.flatten()
                total = len(etiquetas)
                paleta = []
                for i in range(k):
                    pct = float(np.count_nonzero(etiquetas == i)) / total
                    r, g, b = centros[i].astype(int)
                    paleta.append({'hex': f'#{r:02x}{g:02x}{b:02x}', 'pct': round(pct * 100, 1)})
                paleta.sort(key=lambda x: x['pct'], reverse=True)
        except Exception as e:
            print(f"[Visuales] kmeans error: {e}")

        return {'segmentacion_b64': seg_b64, 'histograma_b64': hist_b64,
                'bordes_b64': bordes_b64, 'paleta': paleta}
    except Exception as e:
        print(f"[Visuales] error: {e}")
        return None
