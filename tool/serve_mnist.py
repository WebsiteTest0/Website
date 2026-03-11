import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parents[1] / 'mnist_model.keras'
HOST = '127.0.0.1'
PORT = 8000


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Modell nicht gefunden: {MODEL_PATH}. Fuehre zuerst python tool/train_mnist.py aus.'
        )
    return tf.keras.models.load_model(MODEL_PATH)


MODEL = load_model()


class MnistHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {'ok': True})

    def do_GET(self) -> None:
        if self.path == '/health':
            self._send_json(200, {'status': 'ok'})
            return
        self._send_json(404, {'error': 'Not found'})

    def do_POST(self) -> None:
        if self.path != '/predict':
            self._send_json(404, {'error': 'Not found'})
            return

        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length)

        try:
            data = json.loads(raw_body.decode('utf-8'))
            pixels = data.get('pixels')
            if not isinstance(pixels, list) or len(pixels) != 28 * 28:
                raise ValueError('pixels muss eine Liste mit 784 Werten sein.')

            array = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1)
            prediction = MODEL.predict(array, verbose=0)[0]
            probabilities = [float(value) for value in prediction.tolist()]
            best_index = int(np.argmax(prediction))

            self._send_json(
                200,
                {
                    'prediction': best_index,
                    'probabilities': probabilities,
                },
            )
        except Exception as exc:
            self._send_json(400, {'error': str(exc)})


if __name__ == '__main__':
    server = ThreadingHTTPServer((HOST, PORT), MnistHandler)
    print(f'MNIST-Server laeuft auf http://{HOST}:{PORT}')
    server.serve_forever()
