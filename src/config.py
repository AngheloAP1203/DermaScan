"""Constantes y rutas de configuración del proyecto."""
import os

# Carpeta raíz del proyecto (un nivel arriba de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASES     = ['Benigno', 'Maligno']
IMG_SIZE   = 380
UMBRAL     = 0.62
MODEL_PATH = os.path.join(RAIZ, 'modelo_dermascan.keras')

# Modos de operación: el umbral de decisión cambia según la prioridad clínica.
# 'balanceado' usa el umbral óptimo hallado en validación (máxima exactitud).
# 'screening' baja el umbral para priorizar sensibilidad (menos falsos negativos).
MODOS = {
    'balanceado': 0.62,
    'screening':  0.45,
}

# Si |prob - umbral| <= MARGEN_DUDA el caso se marca como dudoso y se recomienda
# revisión prioritaria por un especialista.
MARGEN_DUDA = 0.10

# Metadatos del modelo (para la UI)
NOMBRE_MODELO = 'EfficientNetB4'
ACCURACY      = 0.8969
AUC_ROC       = 0.9479
