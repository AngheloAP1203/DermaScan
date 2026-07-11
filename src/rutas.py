"""Endpoints HTTP de la aplicación (Flask Blueprint)."""
import time
from flask import Blueprint, request, jsonify, render_template, Response

from .config import (UMBRAL, NOMBRE_MODELO, ACCURACY, MODOS, MARGEN_DUDA, VERSION_MODELO, DATASET,
                     UMBRAL_ABSTENCION, N_PASADAS_ABSTENCION)
from .modelo_onnx import inferir_onnx
from .preprocesamiento import decodificar, preprocesar
from .gradcam import generar_gradcam
from .abcde import analisis_abcde
from .visuales import generar_visuales
from .riesgo import evaluar_riesgo
from .metricas import METRICAS_MODELO
from .avanzado import (tta, oclusion, robustez, analisis_canales, deteccion_ventana,
                       diagnostico_enfocado, robustez_ruido)
from .explicabilidad import comparar
from .saliency import saliency
from .incertidumbre import mc_dropout, incertidumbre_rapida
from .integrated import integrated_gradients
from .calidad import evaluar_calidad
from .roc import curva_roc_b64
from .arquitectura_datos import (
    ARQUITECTURA_API, ARQUITECTURA_MLOPS, ARQUITECTURA_BATCH,
    EVIDENCIA_HISTORICA, NOTA_ARQUITECTURA,
)

bp = Blueprint('dermascan', __name__)


_FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" rx="8" fill="#22d3ee"/>
<path d="M16 7v18M7 16h18" stroke="#06283d" stroke-width="4" stroke-linecap="round"/>
</svg>"""


@bp.route('/favicon.ico')
def favicon():
    return Response(_FAVICON, mimetype='image/svg+xml')


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/salud')
def salud():
    return jsonify({'estado': 'activo', 'modelo': NOMBRE_MODELO, 'version': VERSION_MODELO,
                    'dataset': DATASET, 'umbral': UMBRAL, 'accuracy': ACCURACY})


@bp.route('/metricas')
def metricas():
    return jsonify(METRICAS_MODELO)


@bp.route('/arquitectura')
def arquitectura():
    return render_template(
        'arquitectura.html',
        api=ARQUITECTURA_API,
        mlops=ARQUITECTURA_MLOPS,
        batch=ARQUITECTURA_BATCH,
        historica=EVIDENCIA_HISTORICA,
        nota=NOTA_ARQUITECTURA,
    )


@bp.route('/roc')
def roc():
    return jsonify({'roc_b64': curva_roc_b64()})


def _decodificar_o_400(datos):
    """Decodifica el campo 'imagen' o devuelve None + la respuesta de error
    ya lista para retornar. Compartido por todos los endpoints que reciben
    una imagen: sin esto, un payload sin 'imagen' o una imagen corrupta
    tiraba una KeyError/excepcion de OpenCV sin manejar (500 crudo) en vez
    de un 400 legible."""
    if not datos or 'imagen' not in datos:
        return None, (jsonify({'error': 'Solicitud invalida: falta el campo "imagen".'}), 400)
    try:
        img_rgb = decodificar(datos['imagen'])
        if img_rgb is None:
            raise ValueError('imagen no decodificable')
    except Exception:
        return None, (jsonify({'error': 'No se pudo decodificar la imagen. Envie una imagen valida en base64.'}), 400)
    return img_rgb, None


@bp.route('/tta', methods=['POST'])
def tta_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(tta(img_rgb))


@bp.route('/oclusion', methods=['POST'])
def oclusion_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(oclusion(img_rgb))


@bp.route('/robustez', methods=['POST'])
def robustez_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(robustez(img_rgb))


@bp.route('/canales', methods=['POST'])
def canales_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(analisis_canales(img_rgb))


@bp.route('/comparar', methods=['POST'])
def comparar_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    tensor = preprocesar(img_rgb)
    return jsonify(comparar(tensor, img_rgb))


@bp.route('/saliency', methods=['POST'])
def saliency_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(saliency(img_rgb))


@bp.route('/deteccion', methods=['POST'])
def deteccion_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(deteccion_ventana(img_rgb))


@bp.route('/incertidumbre', methods=['POST'])
def incertidumbre_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(mc_dropout(img_rgb))


@bp.route('/integrated', methods=['POST'])
def integrated_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(integrated_gradients(img_rgb))


@bp.route('/enfocado', methods=['POST'])
def enfocado_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(diagnostico_enfocado(img_rgb))


@bp.route('/ruido', methods=['POST'])
def ruido_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(robustez_ruido(img_rgb))


@bp.route('/calidad', methods=['POST'])
def calidad_endpoint():
    img_rgb, error = _decodificar_o_400(request.get_json(silent=True))
    if error:
        return error
    return jsonify(evaluar_calidad(img_rgb))


@bp.route('/analizar', methods=['POST'])
def analizar():
    """Fast Path: forward pass via ONNX Runtime (sin TensorFlow) + riesgo.
    SIN barrera de incertidumbre (movida a /explicabilidad) — el veredicto
    ya no chequea si la imagen esta fuera de dominio antes de responder."""
    datos = request.get_json(silent=True)
    img_rgb, error = _decodificar_o_400(datos)
    if error:
        return error

    modo   = datos.get('modo', 'balanceado')
    umbral = MODOS.get(modo, UMBRAL)

    calidad = evaluar_calidad(img_rgb)
    if (not datos.get('forzar')) and calidad.get('es_piel') is False:
        return jsonify({
            'rechazado': True,
            'motivo': 'La imagen no parece una foto real de piel humana. '
                      'El modelo solo es valido sobre imagenes dermatoscopicas o de piel.',
            'calidad': calidad,
        })

    tensor = preprocesar(img_rgb)

    t0   = time.time()
    prob = inferir_onnx(tensor)
    ms   = round((time.time() - t0) * 1000, 1)

    clase  = 'Maligno' if prob >= umbral else 'Benigno'
    conf   = prob if clase == 'Maligno' else 1 - prob
    dudoso = abs(prob - umbral) <= MARGEN_DUDA

    return jsonify({
        'clase': clase,
        'confianza_pct': f'{conf*100:.1f}%',
        'latencia_ms': ms,
        'prob_raw': round(prob, 4),
        'modo': modo,
        'umbral_usado': umbral,
        'dudoso': dudoso,
        'riesgo': evaluar_riesgo(prob, umbral),
        'calidad': calidad,
    })


@bp.route('/predict', methods=['POST'])
def predict():
    return analizar()


@bp.route('/explicabilidad', methods=['POST'])
def explicabilidad():
    """Segundo paso, deliberadamente FUERA del Fast Path: agrupa los tres
    computos caros que /analizar ya no ejecuta —

      - Grad-CAM: requiere un GradientTape (backward pass completo).
      - ABCDE: segmentacion + contornos.
      - Visuales: segmentacion + render de matplotlib + k-means + Canny.

    El cliente llama esto DESPUES de mostrar el veredicto de /analizar, no en
    paralelo bloqueando la respuesta principal. Recibe la misma imagen porque
    el servidor no guarda estado entre peticiones (decodificar/preprocesar de
    nuevo es barato frente al costo real de estos tres computos)."""
    datos = request.get_json(silent=True)
    img_rgb, error = _decodificar_o_400(datos)
    if error:
        return error

    tensor = preprocesar(img_rgb)

    media_mc, disp_mc = incertidumbre_rapida(tensor, n=N_PASADAS_ABSTENCION)

    return jsonify({
        'abcde':    analisis_abcde(img_rgb),
        'gradcam':  generar_gradcam(tensor, img_rgb),
        'visuales': generar_visuales(img_rgb),
        'incertidumbre': {
            'prob_media_mc': round(media_mc, 4) if media_mc is not None else None,
            'dispersion': round(disp_mc, 4) if disp_mc is not None else None,
            'alta': disp_mc is not None and disp_mc > UMBRAL_ABSTENCION,
        },
    })
