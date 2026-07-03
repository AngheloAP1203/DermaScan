"""Métricas reales del modelo v2, evaluadas sobre el conjunto de prueba del
notebook de entrenamiento (1997 imágenes, datasets HAM10000 + ISIC fusionados).
Datos fijos para mostrar en la UI en la interfaz web."""

METRICAS_MODELO = {
    'modelo': 'EfficientNetB4',
    'resolucion': '380x380',
    'transfer_learning': 'ImageNet',
    'dataset': 'HAM10000 + ISIC (13309 imagenes fusionadas y deduplicadas)',
    'umbral': 0.56,
    'latencia_prod_ms': 21,
    'latencia_tta_ms': 199,
    'mlflow_run': 'b87c28a8b36c462191648db33d90efba',

    # Métricas globales (test set, 1997 imágenes)
    'globales': {
        'accuracy':        0.9214,
        'auc_roc':         0.9743,
        'f1_macro':        0.8988,
        'precision_macro': 0.8948,
        'recall_macro':    0.9030,
    },

    # Métricas por clase
    'por_clase': [
        {'clase': 'Benigno', 'precision': 0.95, 'recall': 0.94, 'f1': 0.95, 'support': 1479},
        {'clase': 'Maligno', 'precision': 0.84, 'recall': 0.86, 'f1': 0.85, 'support': 518},
    ],

    # Matriz de confusión [[VN, FP], [FN, VP]] con Maligno como positivo
    'matriz_confusion': {
        'benignas_ok': 1392, 'falsos_positivos': 87,    # 1392+87 = 1479
        'malignas_ok': 448,  'falsos_negativos': 70,     # 448+70  = 518
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
