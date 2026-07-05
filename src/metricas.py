"""Métricas reales del modelo v3, evaluadas sobre el conjunto de prueba del
notebook de entrenamiento (1997 imágenes, datasets HAM10000 + ISIC fusionados).
Datos fijos para mostrar en la UI en la interfaz web."""

METRICAS_MODELO = {
    'modelo': 'EfficientNetB4',
    'resolucion': '380x380',
    'transfer_learning': 'ImageNet',
    'dataset': 'HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)',
    'umbral': 0.62,
    'latencia_prod_ms': 20,
    'latencia_tta_ms': 243,
    'mlflow_run': '4b35210effc343c3beec411b7dd5251a',

    # Métricas globales (test set, 1997 imágenes)
    'globales': {
        'accuracy':        0.9234,
        'auc_roc':         0.9747,
        'f1_macro':        0.9019,
        'precision_macro': 0.8957,
        'recall_macro':    0.9088,
    },

    # Métricas por clase
    'por_clase': [
        {'clase': 'Benigno', 'precision': 0.96, 'recall': 0.94, 'f1': 0.95, 'support': 1479},
        {'clase': 'Maligno', 'precision': 0.83, 'recall': 0.88, 'f1': 0.86, 'support': 518},
    ],

    # Matriz de confusión [[VN, FP], [FN, VP]] con Maligno como positivo
    'matriz_confusion': {
        'benignas_ok': 1389, 'falsos_positivos': 90,    # 1389+90 = 1479
        'malignas_ok': 455,  'falsos_negativos': 63,     # 455+63  = 518
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
