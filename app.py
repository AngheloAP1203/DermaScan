"""Punto de entrada de DermaScan.

Crea la aplicación Flask y registra las rutas. Toda la lógica de negocio
(modelo, Grad-CAM, ABCDE, riesgo, visualizaciones) vive en el paquete `src/`.
"""
from flask import Flask
from src.rutas import bp


def crear_app():
    app = Flask(__name__, template_folder='templates')
    # 15 MB: cubre de sobra una imagen redimensionada en base64 (el cliente ya
    # la reescala a <=1024px antes de enviarla). Sin este limite, un payload
    # enorme se procesaba igual y presionaba memoria antes de fallar.
    app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
    app.register_blueprint(bp)
    return app


app = crear_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
