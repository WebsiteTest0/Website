import json
from pathlib import Path

import numpy as np
import tensorflow as tf

KERAS_MODEL_PATH = Path(__file__).resolve().parents[1] / 'mnist_model.keras'
EXPORT_PATH = Path(__file__).resolve().parents[1] / 'assets' / 'models' / 'mnist_dense.json'


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.RandomTranslation(
                height_factor=0.12,
                width_factor=0.12,
                fill_mode='constant',
                fill_value=0.0,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.18, 0.12),
                width_factor=(-0.18, 0.12),
                fill_mode='constant',
                fill_value=0.0,
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def export_for_flutter(model: tf.keras.Model) -> None:
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dense1, dense2, dense3 = dense_layers
    kernel1, bias1 = dense1.get_weights()
    kernel2, bias2 = dense2.get_weights()
    kernel3, bias3 = dense3.get_weights()

    payload = {
        'inputSize': 784,
        'hidden1Size': 256,
        'hidden2Size': 128,
        'outputSize': 10,
        'dense1Kernel': kernel1.astype(np.float32).ravel().tolist(),
        'dense1Bias': bias1.astype(np.float32).tolist(),
        'dense2Kernel': kernel2.astype(np.float32).ravel().tolist(),
        'dense2Bias': bias2.astype(np.float32).tolist(),
        'dense3Kernel': kernel3.astype(np.float32).ravel().tolist(),
        'dense3Bias': bias3.astype(np.float32).tolist(),
    }

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(payload), encoding='utf-8')
    print(f'Browser-Modell exportiert nach: {EXPORT_PATH}')


def train_or_load_model() -> tf.keras.Model:
    retrain = True
    model = None

    if KERAS_MODEL_PATH.exists():
        try:
            loaded = tf.keras.models.load_model(KERAS_MODEL_PATH)
            dense_layers = [layer for layer in loaded.layers if isinstance(layer, tf.keras.layers.Dense)]
            if len(dense_layers) == 3:
                print(f'Lade vorhandenes Modell: {KERAS_MODEL_PATH}')
                model = loaded
                retrain = False
            else:
                print('Vorhandenes Modell nutzt eine aeltere Architektur und wird neu trainiert.')
        except Exception as exc:
            print(f'Vorhandenes Modell konnte nicht geladen werden: {exc}')

    if retrain:
        print('Trainiere robusteres MNIST-Modell fuer Flutter Web und GitHub Pages...')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train.astype('float32') / 255.0, axis=-1)
        x_test = np.expand_dims(x_test.astype('float32') / 255.0, axis=-1)

        model = build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=2,
                restore_best_weights=True,
            )
        ]
        model.fit(
            x_train,
            y_train,
            validation_split=0.1,
            epochs=12,
            batch_size=128,
            verbose=1,
            callbacks=callbacks,
        )
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f'Test Accuracy: {test_accuracy:.4f}')
        model.save(KERAS_MODEL_PATH)
        print(f'Modell gespeichert unter: {KERAS_MODEL_PATH}')

    export_for_flutter(model)
    return model


if __name__ == '__main__':
    train_or_load_model()
