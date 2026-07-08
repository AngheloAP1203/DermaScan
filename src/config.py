"""Configuración del proyecto.

Única fuente de verdad para el modelo servido y su umbral de decisión: se
resuelven en tiempo de arranque contra el Model Registry de MLflow (alias
"champion"), NUNCA se hardcodean aquí. Tanto `app.py` (Flask) como
`pipeline_batch.py` (Spark) importan estas constantes desde este módulo, así
que ambos quedan garantizados a usar exactamente el mismo modelo y el mismo
umbral — se elimina la deriva que existía entre config.py / pipeline_batch.py
/ registrar_experimento.py.

Variables de entorno requeridas:
    MLFLOW_TRACKING_URI   URI del servidor MLflow remoto (obligatoria, sin
                          default: si no está configurada, el arranque falla
                          de forma explícita en vez de caer a un mlruns/ local).
Variables de entorno opcionales:
    MLFLOW_MODEL_NAME     Nombre del modelo registrado (default: ver abajo).
    MLFLOW_MODEL_ALIAS    Alias a resolver (default: "champion").
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Carpeta raíz del proyecto (un nivel arriba de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASES   = ['Benigno', 'Maligno']
IMG_SIZE = 380

# ── Resolución contra el Model Registry ─────────────────────────────────────
try:
    MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
except KeyError as e:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI no está configurada. DermaScan no arranca sin un "
        "servidor MLflow remoto real (no se admite fallback a mlruns/ local en "
        "producción)."
    ) from e

MODEL_NAME  = os.environ.get('MLFLOW_MODEL_NAME', 'dermascan-clasificador-piel')
MODEL_ALIAS = os.environ.get('MLFLOW_MODEL_ALIAS', 'champion')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

try:
    _mv = _client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
except MlflowException as e:
    raise RuntimeError(
        f"No se pudo resolver el alias '{MODEL_ALIAS}' del modelo "
        f"'{MODEL_NAME}' en {MLFLOW_TRACKING_URI}. Verifica que exista una "
        f"version con ese alias asignado (ver registrar_experimento.py)."
    ) from e

MODEL_URI     = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
MODEL_VERSION = _mv.version

# Tags de la version del modelo: la única fuente de verdad para los umbrales
# de decisión. Si faltan, se falla de forma explícita en vez de asumir un
# valor por defecto silencioso — un umbral clínico mal calibrado que se cuela
# por un valor por defecto es peor que un arranque que revienta ruidosamente.
def _tag_float(nombre):
    try:
        return float(_mv.tags[nombre])
    except KeyError as e:
        raise RuntimeError(
            f"La version {_mv.version} de '{MODEL_NAME}' no tiene el tag "
            f"'{nombre}'. Debe fijarse al registrar el modelo "
            f"(ver registrar_experimento.py)."
        ) from e

UMBRAL = _tag_float('umbral_optimo')

# Modos de operación: el umbral de decisión cambia según la prioridad clínica.
# 'balanceado' = umbral óptimo hallado en validación (máxima exactitud, tag
# 'umbral_optimo'). 'screening' baja el umbral para priorizar sensibilidad
# (menos falsos negativos, tag 'umbral_screening'). Ambos vienen del registry,
# ligados a la versión concreta del modelo que está sirviendo.
MODOS = {
    'balanceado': UMBRAL,
    'screening':  _tag_float('umbral_screening'),
}

# Parámetros operativos de la barrera de abstención por incertidumbre (Monte
# Carlo Dropout). A diferencia de UMBRAL/MODOS, esto no es un umbral de
# decisión clínica del modelo sino un parámetro de ingeniería del mecanismo de
# abstención en sí (cuánta dispersión tolerar, cuántas pasadas correr) — se
# deja configurable por entorno para poder ajustarlo por despliegue sin
# reentrenar ni re-registrar el modelo.
#
# Ver incertidumbre.incertidumbre_rapida: el backbone y las capas
# BatchNormalization se mantienen en modo inferencia para no contaminar la
# dispersión con estadísticas de lote inestables. Calibrado empíricamente con
# el backbone congelado: dispersión en casos confiados <0.03, casos
# genuinamente ambiguos 0.05+.
MARGEN_DUDA           = float(os.environ.get('DERMASCAN_MARGEN_DUDA', '0.10'))
UMBRAL_ABSTENCION     = float(os.environ.get('DERMASCAN_UMBRAL_ABSTENCION', '0.03'))
N_PASADAS_ABSTENCION  = int(os.environ.get('DERMASCAN_N_PASADAS_ABSTENCION', '6'))

# Metadatos del modelo (para la UI) — leídos del run que produjo la versión
# "champion", no hardcodeados. Esto elimina la deriva que existía entre
# config.py, metricas.py y registrar_experimento.py (tres accuracies distintos
# para "el mismo modelo").
_run = _client.get_run(_mv.run_id)
NOMBRE_MODELO  = _run.data.params.get('arquitectura', 'desconocido')
VERSION_MODELO = f'v{_mv.version}'
DATASET        = _run.data.params.get('dataset', 'desconocido')
ACCURACY       = _run.data.metrics.get('accuracy')
AUC_ROC        = _run.data.metrics.get('auc_roc')
