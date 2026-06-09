import json 
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.model import ARCHITECTURES

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dataset(name: str):
    
    if name == 'cifar10':
        (x_tr, y_tr), (x_test, y_test) = keras.datasets.cifar10.load_data()
        input_shape = (32, 32, 3)
    elif name == "mnist":
        (x_tr, y_tr), (x_test, y_test) = keras.datasets.mnist.load_data()
        # MNIST - чёрно-белый, добавляем размерность канала: (28,28) to (28,28,1)
        x_tr = x_tr[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        input_shape = (28, 28, 1)
    else:
        raise ValueError(f"Неизвестный датасет: {name}")
    
    # Нормализуем пиксели
    x_tr = x_tr.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encoding
    y_tr = keras.utils.to_categorical(y_tr, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_tr, y_tr), (x_test, y_test), input_shape

def train(dataset: str, arch: str, epochs: int = 10, batch_size: int = 64):

    (x_tr, y_tr), (x_te, y_te), input_shape = load_dataset(dataset)
    build_fn = ARCHITECTURES[arch]
    model = build_fn(input_shape, num_classes=10)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",  # стандарт для мультиклассовой классификации
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # 10% train → val, сеть их не учит
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)

    result = {
        "dataset":    dataset,
        "arch":       arch,
        "epochs":     epochs,
        "batch_size": batch_size,
        "train_acc":  [round(v, 4) for v in history.history["accuracy"]],
        "val_acc":    [round(v, 4) for v in history.history["val_accuracy"]],
        "train_loss": [round(v, 4) for v in history.history["loss"]],
        "val_loss":   [round(v, 4) for v in history.history["val_loss"]],
        "test_acc":   round(float(test_acc), 4),
        "test_loss":  round(float(test_loss), 4),
        "params":     model.count_params(),
    }

    out_path = os.path.join(RESULTS_DIR, f"{dataset}_{arch}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
