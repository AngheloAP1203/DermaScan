"""Curva ROC del modelo. Como el notebook no exportó los arreglos FPR/TPR, se
estima una curva binormal calibrada al AUC real (0.9747) y se superpone el
PUNTO DE OPERACIÓN REAL medido en el test set (umbral 0.62). Se etiqueta
claramente como curva estimada para ser honestos."""
import base64
import io
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .metricas import METRICAS_MODELO

_CACHE = {}


def curva_roc_b64():
    if 'img' in _CACHE:
        return _CACHE['img']

    g   = METRICAS_MODELO['globales']
    mc  = METRICAS_MODELO['matriz_confusion']
    auc = g['auc_roc']

    # Punto de operación real (Maligno = positivo)
    tpr_real = mc['malignas_ok'] / (mc['malignas_ok'] + mc['falsos_negativos'])      # recall
    fpr_real = mc['falsos_positivos'] / (mc['falsos_positivos'] + mc['benignas_ok'])

    # Curva binormal de varianza igual: AUC = Phi(a/sqrt(2)) -> a = sqrt(2)*Phi^-1(AUC)
    N = NormalDist()
    a = (2 ** 0.5) * N.inv_cdf(auc)
    fpr = [i / 200 for i in range(1, 200)]
    tpr = [N.cdf(a + N.inv_cdf(f)) for f in fpr]

    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=100)
    fig.patch.set_facecolor('#12122a')
    ax.set_facecolor('#12122a')
    ax.plot([0, 1], [0, 1], '--', color='#555577', linewidth=1, label='Azar (AUC 0.5)')
    ax.plot(fpr, tpr, color='#6c63ff', linewidth=2.2, label=f'EfficientNetB4 (AUC {auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#6c63ff')
    ax.scatter([fpr_real], [tpr_real], color='#ff4d6d', s=70, zorder=5,
               label=f'Umbral 0.62 (TPR {tpr_real:.2f}, FPR {fpr_real:.2f})')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Tasa de falsos positivos (FPR)', color='#c4b5fd', fontsize=9)
    ax.set_ylabel('Tasa de verdaderos positivos (TPR)', color='#c4b5fd', fontsize=9)
    ax.set_title('Curva ROC estimada · punto de operación real', color='#e8e8ff', fontsize=10)
    ax.tick_params(colors='#8888bb', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#333366')
    ax.grid(alpha=0.12)
    ax.legend(facecolor='#1a1a3e', edgecolor='#333366', labelcolor='#e8e8ff', fontsize=7.5, loc='lower right')
    fig.tight_layout()

    bio = io.BytesIO()
    fig.savefig(bio, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    bio.seek(0)
    _CACHE['img'] = base64.b64encode(bio.read()).decode('utf-8')
    return _CACHE['img']
