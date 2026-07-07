"""Constantes y rutas de configuración del proyecto."""
import os

# Carpeta raíz del proyecto (un nivel arriba de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASES     = ['Benigno', 'Maligno']
IMG_SIZE   = 380
UMBRAL     = 0.70
MODEL_PATH = os.path.join(RAIZ, 'modelo_dermascan.keras')

# Modos de operación: el umbral de decisión cambia según la prioridad clínica.
# 'balanceado' usa el umbral óptimo hallado en validación (máxima exactitud).
# 'screening' baja el umbral para priorizar sensibilidad (menos falsos negativos).
MODOS = {
    'balanceado': 0.70,
    'screening':  0.45,
}

# Si |prob - umbral| <= MARGEN_DUDA el caso se marca como dudoso y se recomienda
# revisión prioritaria por un especialista.
MARGEN_DUDA = 0.10

# Barrera de abstención por incertidumbre (Monte Carlo Dropout, solo la cabeza
# estocástica; el backbone y las capas BatchNormalization se mantienen en modo
# inferencia para no contaminar la dispersión con estadísticas de lote inestables
# -ver incertidumbre.incertidumbre_rapida). Si la desviación estándar de
# N_PASADAS_ABSTENCION pasadas supera este umbral, el sistema se abstiene de
# emitir Benigno/Maligno en vez de forzar un veredicto potencialmente engañoso.
#
# Calibrado empíricamente: con el backbone congelado y determinístico, la
# dispersión real observada en casos confiados es muy baja (<0.03), mientras que
# casos genuinamente ambiguos rondan 0.05+. MC-Dropout con un backbone congelado
# NO detecta de forma confiable imágenes fuera de dominio (p. ej. condiciones no
# pigmentadas): el modelo puede estar "confiadamente equivocado" en vez de dudoso
# ante ellas. Esta barrera captura casos de baja confianza cerca del umbral de
# decisión, no un detector general de fuera-de-distribución.
UMBRAL_ABSTENCION = 0.03
N_PASADAS_ABSTENCION = 6

# Metadatos del modelo (para la UI)
NOMBRE_MODELO = 'EfficientNetB4'
VERSION_MODELO = 'v3 (multi-dataset)'
DATASET       = 'HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)'
ACCURACY      = 0.9239
AUC_ROC       = 0.9706
