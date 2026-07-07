"""Métricas reales del modelo v3, evaluadas sobre el conjunto de prueba del
notebook de entrenamiento (1997 imágenes, datasets HAM10000 + ISIC fusionados).
Datos fijos para mostrar en la UI en la interfaz web."""

METRICAS_MODELO = {
    'modelo': 'EfficientNetB4',
    'resolucion': '380x380',
    'transfer_learning': 'ImageNet',
    'dataset': 'HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)',
    'umbral': 0.70,
    'latencia_prod_ms': 16,
    'latencia_tta_ms': 186,
    'mlflow_run': 'bbbdad49a1354e70b0e63ffe66e0c106',

    # Métricas globales (test set, 1997 imágenes)
    'globales': {
        'accuracy':        0.9239,
        'auc_roc':         0.9706,
        'f1_macro':        0.9009,
        'precision_macro': 0.9009,
        'recall_macro':    0.9009,
    },

    # Métricas por clase
    'por_clase': [
        {'clase': 'Benigno', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 1479},
        {'clase': 'Maligno', 'precision': 0.85, 'recall': 0.85, 'f1': 0.85, 'support': 518},
    ],

    # Matriz de confusión [[VN, FP], [FN, VP]] con Maligno como positivo
    'matriz_confusion': {
        'benignas_ok': 1403, 'falsos_positivos': 76,    # 1403+76 = 1479
        'malignas_ok': 442,  'falsos_negativos': 76,     # 442+76  = 518
    },

    # Comparativa contra el baseline clásico
    'baseline': {
        'nombre': 'HOG + SVM',
        'accuracy': 0.7426,
        'f1_macro': 0.6387,
        'precision_macro': 0.6543,
        'recall_macro': 0.6306,
    },
}
