---
title: DermaScan
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DermaScan — Detección Temprana de Lesiones de Piel

Herramienta académica de análisis de lesiones cutáneas basada en EfficientNetB4
(92.39% accuracy, AUC 0.9706, entrenado sobre 13,309 imágenes HAM10000 + ISIC
fusionadas y deduplicadas). Incluye múltiples técnicas de explicabilidad
(Grad-CAM, Saliency, Oclusión, Integrated Gradients), análisis morfológico
ABCDE, estimación de incertidumbre (MC Dropout, TTA) y chequeo de
calidad/validez de la imagen subida.

Ofrece dos modos de operación seleccionables: **Balanceado** (umbral 0.70,
máxima exactitud) y **Screening** (umbral 0.45, prioriza sensibilidad para
detectar más lesiones malignas). En producción estos umbrales se leen desde
MLflow Model Registry mediante los tags de la versión `champion`; en desarrollo
local se usan los mismos valores por defecto desde `src/config.py`. Los casos
con probabilidad cercana al umbral se marcan como dudosos y se recomienda
revisión prioritaria por un especialista.

Proyecto UPAO 2026 — Percepción Computacional.

**Aviso:** herramienta académica, no sustituye el diagnóstico de un dermatólogo
certificado.

Ver [README.txt](README.txt) para documentación técnica detallada.
