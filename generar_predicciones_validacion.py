"""Genera el CSV (y_true, y_score) que consume registrar_experimento.py, a
partir de un split de validacion/prueba real con etiquetas por carpeta
(benigno/ vs maligno/). NO decide ningun umbral — solo corre el modelo local
sobre imagenes reales y guarda las probabilidades crudas.

Uso:
    python generar_predicciones_validacion.py \
        --modelo modelo_dermascan.keras \
        --benigno "<ruta>/archive/test/benign" \
        --maligno "<ruta>/archive/test/malignant" \
        --salida predicciones_val.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import keras

IMG_SIZE = 380  # tamano de entrada del modelo (propiedad de la arquitectura, no un umbral)
EXTENSIONES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _listar_imagenes(directorio):
    rutas = []
    for nombre in sorted(os.listdir(directorio)):
        if os.path.splitext(nombre)[1].lower() in EXTENSIONES:
            rutas.append(os.path.join(directorio, nombre))
    return rutas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modelo', required=True)
    ap.add_argument('--benigno', required=True, help='Carpeta con imagenes etiquetadas benignas (y_true=0)')
    ap.add_argument('--maligno', required=True, help='Carpeta con imagenes etiquetadas malignas (y_true=1)')
    ap.add_argument('--salida', default='predicciones_val.csv')
    args = ap.parse_args()

    print(f"Cargando modelo local: {args.modelo}")
    modelo = keras.models.load_model(args.modelo, compile=False)

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
    def inferir(x):
        return modelo(x, training=False)

    filas = []
    for etiqueta, carpeta in [(0, args.benigno), (1, args.maligno)]:
        rutas = _listar_imagenes(carpeta)
        print(f"{'Benigno' if etiqueta == 0 else 'Maligno'}: {len(rutas)} imagenes en {carpeta}")
        for i, ruta in enumerate(rutas):
            try:
                datos = np.fromfile(ruta, dtype=np.uint8)
                img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError('no se pudo decodificar')
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')
                tensor = tf.expand_dims(tf.convert_to_tensor(resized), 0)
                prob = float(inferir(tensor)[0][0])
                filas.append({'imagen': os.path.basename(ruta), 'y_true': etiqueta, 'y_score': prob})
            except Exception as e:
                print(f"  [WARN] {ruta}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  ...{i + 1}/{len(rutas)}")

    df = pd.DataFrame(filas)
    df.to_csv(args.salida, index=False)
    print(f"\n[OK] {len(df)} predicciones guardadas en {args.salida}")
    print(f"     Benignas: {(df.y_true == 0).sum()}  Malignas: {(df.y_true == 1).sum()}")


if __name__ == '__main__':
    main()
