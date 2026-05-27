"""
Task 3 - Handwritten Character Recognition

Identify handwritten characters or digits using a CNN model.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required for this task. Install it with: pip install tensorflow"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handwritten character recognition.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "emnist"],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Limit training samples (0 means no limit).",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit test samples (0 means no limit).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full classification report (can be large).",
    )
    return parser.parse_args()


def limit_array(arr: np.ndarray, limit: int) -> np.ndarray:
    if limit and limit > 0:
        return arr[:limit]
    return arr


def load_mnist(
    train_limit: int, test_limit: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = limit_array(x_train, train_limit)
    y_train = limit_array(y_train, train_limit)
    x_test = limit_array(x_test, test_limit)
    y_test = limit_array(y_test, test_limit)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
    if x_test.ndim == 3:
        x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test


def load_emnist(
    train_limit: int, test_limit: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "tensorflow-datasets is required for EMNIST. Install it with: "
            "pip install tensorflow-datasets"
        ) from exc

    ds_train, ds_test = tfds.load(
        "emnist/byclass",
        split=["train", "test"],
        as_supervised=True,
    )

    def to_numpy(ds, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for idx, (image, label) in enumerate(tfds.as_numpy(ds)):
            if limit and idx >= limit:
                break
            images.append(image)
            labels.append(int(label))
        return np.stack(images), np.array(labels)

    x_train, y_train = to_numpy(ds_train, train_limit)
    x_test, y_test = to_numpy(ds_test, test_limit)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    args = parse_args()

    if args.dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist(
            args.train_limit, args.test_limit
        )
        num_classes = 10
    else:
        x_train, y_train, x_test, y_test = load_emnist(
            args.train_limit, args.test_limit
        )
        num_classes = int(np.max(y_train)) + 1

    model = build_model(x_train.shape[1:], num_classes)
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    if args.report:
        print(classification_report(y_test, y_pred, digits=4))

    return 0


if __name__ == "__main__":
    sys.exit(main())
