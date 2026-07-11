"""Registra una version del modelo DermaScan en el Model Registry de MLflow.

NO escribe umbrales "porque si": cada umbral registrado tiene una fuente
explicita y trazable, de una de estas dos formas —

  (a) --predicciones <csv>   Se CALCULA desde predicciones reales (y_true,
                              y_score) de un set de validacion/prueba
                              (Youden's J para el umbral balanceado,
                              sensibilidad objetivo para screening).

  (b) --umbral-optimo / --umbral-screening + --fuente
                              Se usa un valor ya determinado por otro medio
                              (p. ej. el notebook de entrenamiento original),
                              pero WITH mandatory --fuente documentando de
                              donde sale — queda visible como tag en el
                              Registry, nunca como numero mudo en el codigo.

Un umbral sin (a) ni (b) simplemente no se puede registrar: el script se
niega, a proposito.

Uso (calculado):
    python registrar_experimento.py \
        --modelo modelo_dermascan.keras \
        --predicciones predicciones_val.csv \
        --dataset "HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)"

Uso (valor documentado, con fuente obligatoria):
    python registrar_experimento.py \
        --modelo modelo_dermascan.keras \
        --dataset "HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)" \
        --umbral-optimo 0.70 --umbral-screening 0.45 \
        --accuracy 0.9239 --auc-roc 0.9706 \
        --fuente "metricas.py / notebook de entrenamiento original (test set 1997 img).
                   No recalculado en este registro: archive/test (3k, Kaggle ISIC
                   benign/malignant) dio AUC 0.999 / umbral Youden 0.95, señal de
                   posible solapamiento con el corpus de entrenamiento original —
                   se descarto como fuente de calibracion." \
        --promover

Variables de entorno requeridas:
    MLFLOW_TRACKING_URI   URI del servidor MLflow remoto (obligatoria).

Al finalizar, la version registrada queda con el alias "champion" solo si se
pasa --promover explicitamente: promover a produccion es una decision, no un
efecto secundario automatico de correr el script.
"""
import argparse
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import keras
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score


def umbral_youden(y_true, y_score):
    """Umbral que maximiza el estadistico J de Youden (sensibilidad +
    especificidad - 1) sobre la curva ROC real. Este es el umbral
    'balanceado' — el punto de la ROC mas alejado de la diagonal de azar."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def umbral_para_sensibilidad(y_true, y_score, sensibilidad_objetivo=0.97):
    """Umbral mas alto que aun garantiza recall >= sensibilidad_objetivo
    sobre malignos. Usado para el modo 'screening': prioriza no dejar pasar
    malignos, a costa de mas falsos positivos."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    validos = np.where(tpr >= sensibilidad_objetivo)[0]
    if len(validos) == 0:
        raise ValueError(
            f"Ningun umbral de la curva ROC alcanza sensibilidad "
            f">= {sensibilidad_objetivo}. Revisa el modelo o baja el objetivo."
        )
    return float(thresholds[validos[-1]])


def _resolver_umbrales_y_metricas(args):
    """Devuelve (umbral_optimo, umbral_screening, metricas: dict, fuente: str,
    calculado: bool)."""
    if args.predicciones:
        df = pd.read_csv(args.predicciones)
        if not {'y_true', 'y_score'}.issubset(df.columns):
            raise ValueError("El CSV de predicciones debe tener columnas 'y_true' y 'y_score'.")

        y_true, y_score = df['y_true'].to_numpy(), df['y_score'].to_numpy()

        umbral_optimo    = umbral_youden(y_true, y_score)
        umbral_screening = umbral_para_sensibilidad(y_true, y_score, args.sensibilidad_screening)
        y_pred           = (y_score >= umbral_optimo).astype(int)

        metricas = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'auc_roc':  float(roc_auc_score(y_true, y_score)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        }
        fuente = f"calculado (Youden J) sobre {args.predicciones} ({len(df)} imagenes)"
        return umbral_optimo, umbral_screening, metricas, fuente, True

    if args.umbral_optimo is None or args.umbral_screening is None or not args.fuente:
        raise ValueError(
            "Sin --predicciones, hay que dar --umbral-optimo, --umbral-screening "
            "Y --fuente explicitamente. No hay un tercer camino con un default "
            "silencioso."
        )
    if args.accuracy is None or args.auc_roc is None:
        raise ValueError(
            "Con umbrales documentados (no calculados aqui) tambien hacen falta "
            "--accuracy y --auc-roc de la misma fuente, para que el Registry "
            "refleje metricas reales y no queden en None."
        )

    metricas = {'accuracy': args.accuracy, 'auc_roc': args.auc_roc}
    return args.umbral_optimo, args.umbral_screening, metricas, args.fuente, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modelo', required=True, help='Ruta al .keras entrenado')
    ap.add_argument('--predicciones', default=None,
                     help='CSV con columnas y_true,y_score del set de validacion/prueba')
    ap.add_argument('--umbral-optimo', type=float, default=None)
    ap.add_argument('--umbral-screening', type=float, default=None)
    ap.add_argument('--accuracy', type=float, default=None)
    ap.add_argument('--auc-roc', type=float, default=None)
    ap.add_argument('--fuente', default=None,
                     help='Obligatorio si no se usa --predicciones: de donde salen '
                          'los umbrales/metricas (p.ej. "notebook original, test set 1997 img")')
    ap.add_argument('--arquitectura', default='EfficientNetB4')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--modelo-registrado', default='dermascan-clasificador-piel')
    ap.add_argument('--experimento', default='DermaScan-EfficientNetB4')
    ap.add_argument('--sensibilidad-screening', type=float, default=0.97,
                     help='Recall minimo objetivo sobre Maligno para el umbral de screening (solo con --predicciones)')
    ap.add_argument('--promover', action='store_true',
                     help='Asigna el alias "champion" a la version registrada. '
                          'Sin este flag, el modelo queda registrado pero NO sirve produccion.')
    ap.add_argument('--skip-si-alias-existe', action='store_true',
                     help='No registra una nueva version si el alias "champion" ya existe. '
                          'Util para docker compose up repetidos.')
    args = ap.parse_args()

    tracking_uri = os.environ['MLFLOW_TRACKING_URI']  # sin default: falla explicito
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    if args.skip_si_alias_existe:
        try:
            mv_actual = client.get_model_version_by_alias(args.modelo_registrado, 'champion')
            print(
                f"[OK] Alias 'champion' ya existe para "
                f"{args.modelo_registrado} v{mv_actual.version}. No se registra otra version."
            )
            return
        except MlflowException:
            pass

    umbral_optimo, umbral_screening, metricas, fuente, calculado = _resolver_umbrales_y_metricas(args)

    print(f"[OK] Umbral optimo:     {umbral_optimo}")
    print(f"[OK] Umbral screening:  {umbral_screening}")
    print(f"[OK] Metricas:          {metricas}")
    print(f"[OK] Fuente:            {fuente}")
    print(f"[OK] Calculado en este registro: {calculado}")

    # ── Run de MLflow: modelo logueado con el flavor de Keras, no como artifact suelto ──
    mlflow.set_experiment(args.experimento)

    with mlflow.start_run(run_name=f"{args.arquitectura}-{args.dataset[:20]}") as run:
        mlflow.log_params({
            'arquitectura':        args.arquitectura,
            'dataset':             args.dataset,
            'img_size':            380,
            'umbral_fuente':       fuente,
            'umbral_fue_calculado': str(calculado),
        })
        mlflow.log_metrics({
            **metricas,
            'umbral_optimo':    umbral_optimo,
            'umbral_screening': umbral_screening,
        })

        modelo_keras = keras.models.load_model(args.modelo, compile=False)
        model_info = mlflow.keras.log_model(modelo_keras, artifact_path='model')

        run_id = run.info.run_id
        print(f"[OK] Run registrado con ID: {run_id}")

    # ── Model Registry ──
    # MLflow 3.x guarda el modelo como entidad "Logged Model" separada del
    # run (model_info.model_uri = "models:/m-<id>"), YA NO bajo
    # "runs:/<run_id>/model" (esa convencion de 2.x quedo obsoleta y produce
    # un source que no resuelve a ningun archivo real).
    model_uri = model_info.model_uri
    try:
        client.create_registered_model(args.modelo_registrado)
    except Exception:
        pass  # ya existe

    mv = client.create_model_version(name=args.modelo_registrado, source=model_uri, run_id=run_id)
    print(f"[OK] Version creada: v{mv.version}")

    # Umbrales como tags de la VERSION concreta: config.py los lee de aqui.
    # La fuente queda tambien como tag (no solo como param del run) para que
    # sea visible directamente en la pestaña de la version del modelo.
    client.set_model_version_tag(args.modelo_registrado, mv.version, 'umbral_optimo', str(umbral_optimo))
    client.set_model_version_tag(args.modelo_registrado, mv.version, 'umbral_screening', str(umbral_screening))
    client.set_model_version_tag(args.modelo_registrado, mv.version, 'umbral_fuente', fuente)
    print(f"[OK] Tags de umbral fijados en v{mv.version}")

    if args.promover:
        client.set_registered_model_alias(args.modelo_registrado, 'champion', mv.version)
        print(f"[OK] Alias 'champion' -> v{mv.version}. Este modelo pasara a servir en el proximo arranque.")
    else:
        print(f"[INFO] v{mv.version} registrada SIN promover. Corre con --promover para asignarle 'champion'.")


if __name__ == '__main__':
    main()
