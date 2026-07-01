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
(89.69% accuracy, entrenado sobre HAM10000). Incluye múltiples técnicas de
explicabilidad (Grad-CAM, Saliency, Oclusión, Integrated Gradients), análisis
morfológico ABCDE, estimación de incertidumbre (MC Dropout, TTA) y chequeo de
calidad/validez de la imagen subida.

Proyecto UPAO 2026 — Percepción Computacional.

**Aviso:** herramienta académica, no sustituye el diagnóstico de un dermatólogo
certificado.

Ver [README.txt](README.txt) para documentación técnica detallada.
