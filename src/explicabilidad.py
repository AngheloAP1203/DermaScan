"""Comparativa de explicabilidad: enfrenta el mapa Grad-CAM (basado en gradientes)
con el mapa de oclusión (basado en perturbaciones). Si ambas técnicas, que son
independientes, resaltan la misma zona, mayor confianza en la explicación."""
import base64
import numpy as np

from .config import IMG_SIZE
from .gradcam import heatmap_array
from .avanzado import _oclusion_array


def _colorear(heat, img_base, cmap):
    import cv2
    heat_u8    = np.uint8(255 * np.clip(heat, 0, 1))
    heat_color = cv2.applyColorMap(heat_u8, cmap)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_base, 0.55, heat_color, 0.45, 0)
    ok, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8') if ok else None


def comparar(tensor, img_rgb):
    """Devuelve los 2 overlays + % de acuerdo entre Grad-CAM y oclusión."""
    import cv2
    try:
        base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('uint8')

        gc = heatmap_array(tensor)                       # 380x380 [0,1]
        gc = cv2.resize(gc, (IMG_SIZE, IMG_SIZE))
        oc, _, _ = _oclusion_array(img_rgb)              # 380x380 [0,1]

        # Correlación de Pearson sobre una rejilla reducida (más estable)
        g_s = cv2.resize(gc, (16, 16)).flatten()
        o_s = cv2.resize(oc, (16, 16)).flatten()
        if g_s.std() > 1e-6 and o_s.std() > 1e-6:
            corr = float(np.corrcoef(g_s, o_s)[0, 1])
        else:
            corr = 0.0
        acuerdo = max(corr, 0.0) * 100.0

        if acuerdo >= 60:
            interp = 'Alto acuerdo: ambas tecnicas senalan la misma region, la explicacion es solida.'
        elif acuerdo >= 30:
            interp = 'Acuerdo parcial: las tecnicas coinciden en parte de la region relevante.'
        else:
            interp = 'Bajo acuerdo: las tecnicas resaltan zonas distintas, interpretar con cautela.'

        return {
            'gradcam_b64': _colorear(gc, base, cv2.COLORMAP_JET),
            'oclusion_b64': _colorear(oc, base, cv2.COLORMAP_INFERNO),
            'acuerdo_pct': round(acuerdo, 1),
            'correlacion': round(corr, 3),
            'interpretacion': interp,
        }
    except Exception as e:
        print(f"[Explicabilidad] error: {e}", flush=True)
        return {'error': str(e)}
