"""Constantes y rutas de configuración del proyecto."""
import os

# Carpeta raíz del proyecto (un nivel arriba de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASES     = ['Benigno', 'Maligno']
IMG_SIZE   = 380
UMBRAL     = 0.62
MODEL_PATH = os.path.join(RAIZ, 'modelo_dermascan.keras')

# Metadatos del modelo (para la UI)
NOMBRE_MODELO = 'EfficientNetB4'
ACCURACY      = 0.8969
AUC_ROC       = 0.9479
