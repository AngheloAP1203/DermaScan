"""Clasificación del nivel de riesgo a partir de la probabilidad del modelo."""
from .config import UMBRAL


def evaluar_riesgo(prob, umbral=UMBRAL):
    """Devuelve banda de riesgo, color y recomendación clínica según la probabilidad.

    Las bandas intermedias dependen del umbral de decisión activo, de modo que
    el modo screening (umbral más bajo) desplaza antes los casos a bandas altas.
    """
    if prob < min(0.30, umbral):
        return {'banda': 'Bajo', 'nivel': 1, 'color': '#06d6a0',
                'recomendacion': 'Probabilidad baja de malignidad. Mantenga vigilancia y autoexamen periodico.'}
    if prob < umbral:
        return {'banda': 'Moderado', 'nivel': 2, 'color': '#ffd166',
                'recomendacion': 'Riesgo moderado. Se sugiere control dermatologico de rutina.'}
    if prob < 0.85:
        return {'banda': 'Alto', 'nivel': 3, 'color': '#ff9a3c',
                'recomendacion': 'Riesgo alto. Consulte a un dermatologo a la brevedad para evaluacion.'}
    return {'banda': 'Muy alto', 'nivel': 4, 'color': '#ff4d6d',
            'recomendacion': 'Riesgo muy alto. Acuda a un especialista de forma prioritaria.'}
