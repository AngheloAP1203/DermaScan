"""Métricas reales del modelo, evaluadas sobre el conjunto de prueba del
notebook de entrenamiento (1503 imágenes). Datos fijos para mostrar en la UI
en la interfaz web."""

METRICAS_MODELO = {
    'modelo': 'EfficientNetB4',
    'resolucion': '380x380',
    'transfer_learning': 'ImageNet',
    'dataset': 'HAM10000 (10015 imagenes, 7470 pacientes)',
    'umbral': 0.62,
    'latencia_prod_ms': 21,
    'latencia_tta_ms': 199,
    'mlflow_run': '29b592ad41314811972776395b58c68e',

    # Métricas globales (test set, 1503 imágenes)
    'globales': {
        'accuracy':        0.8969,
        'auc_roc':         0.9479,
        'f1_macro':        0.8423,
        'precision_macro': 0.8299,
        'recall_macro':    0.8571,
    },

    # Métricas por clase
    'por_clase': [
        {'clase': 'Benigno', 'precision': 0.95, 'recall': 0.92, 'f1': 0.94, 'support': 1210},
        {'clase': 'Maligno', 'precision': 0.71, 'recall': 0.79, 'f1': 0.75, 'support': 293},
    ],

    # Matriz de confusión [[VN, FP], [FN, VP]] con Maligno como positivo
    'matriz_confusion': {
        'benignas_ok': 1116, 'falsos_positivos': 94,    # 1116+94 = 1210
        'malignas_ok': 232,  'falsos_negativos': 61,     # 232+61  = 293
    },

    # Comparativa contra el baseline clásico
    'baseline': {
        'nombre': 'HOG + SVM',
        'accuracy': 0.7944,
        'f1_macro': 0.6069,
        'precision_macro': 0.6460,
        'recall_macro': 0.5943,
    },
}
