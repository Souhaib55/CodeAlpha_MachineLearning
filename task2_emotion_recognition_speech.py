"""
Task 2 - Emotion Recognition From Speech

Recognize human emotions from speech audio using MFCC features and a CNN model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "librosa is required for audio feature extraction. Install it with: "
        "pip install librosa soundfile"
    ) from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required for the CNN model. Install it with: pip install tensorflow"
    ) from exc


RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMODB_EMOTIONS = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech emotion recognition with MFCC + CNN."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ravdess",
        choices=["ravdess", "tess", "emodb", "custom"],
    )
    parser.add_argument("--cache", type=str, default="data/processed/task2_mfcc.npz")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of audio files (0 means no limit).",
    )
    return parser.parse_args()


def list_audio_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.rglob("*.wav"))


def label_from_ravdess(path: Path) -> Optional[str]:
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return RAVDESS_EMOTIONS.get(emotion_code)


def label_from_tess(path: Path) -> Optional[str]:
    name = path.stem.lower()
    parts = name.split("_")
    if "ps" in parts or "pleasant" in name or "surprise" in name:
        return "surprised"
    for label in ["angry", "disgust", "fear", "happy", "neutral", "sad"]:
        if label in name:
            return label
    return None


def label_from_emodb(path: Path) -> Optional[str]:
    # EMO-DB filename format: [speaker2][text2][emotion1][version1].wav
    # e.g. '03a01Wa' -> speaker='03', text='a01', emotion='W', version='a'
    # The emotion letter is at index 5 (0-based) in a 7-char stem.
    stem = path.stem
    if len(stem) < 6:
        return None
    emotion_code = stem[5].upper()
    return EMODB_EMOTIONS.get(emotion_code)


def label_from_custom(path: Path) -> Optional[str]:
    if path.parent.name:
        return path.parent.name.lower()
    return None


def build_feature(
    path: Path,
    sample_rate: int,
    n_mfcc: int,
    max_len: int,
) -> np.ndarray:
    signal, sr = librosa.load(path, sr=sample_rate)
    
    # Trim silence from beginning/end (common in EMO-DB/RAVDESS)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    # Normalize EACH coefficient (row) individually BEFORE padding so padded zeros remain exactly zero
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mean) / (std + 1e-8)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.astype(np.float32)


def load_dataset(
    data_dir: Path,
    dataset: str,
    sample_rate: int,
    n_mfcc: int,
    max_len: int,
    max_files: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    label_fn = {
        "ravdess": label_from_ravdess,
        "tess": label_from_tess,
        "emodb": label_from_emodb,
        "custom": label_from_custom,
    }[dataset]

    files = list_audio_files(data_dir)
    if max_files > 0:
        files = files[:max_files]

    features: List[np.ndarray] = []
    labels: List[str] = []

    for path in files:
        label = label_fn(path)
        if label is None:
            continue
        try:
            features.append(build_feature(path, sample_rate, n_mfcc, max_len))
            labels.append(label)
        except Exception:
            continue

    if not features:
        raise ValueError("No audio features extracted. Check dataset path and format.")

    X = np.stack(features, axis=0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return X, y, encoder.classes_.tolist()


def load_or_build_features(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["label_names"].tolist()

    X, y, label_names = load_dataset(
        Path(args.data_dir),
        args.dataset,
        args.sample_rate,
        args.n_mfcc,
        args.max_len,
        args.max_files,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, label_names=np.array(label_names))
    return X, y, label_names


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    args = parse_args()

    X, y, label_names = load_or_build_features(args)
    X = X[..., np.newaxis]

    # Use stratify only when every class has at least 2 samples.
    from collections import Counter
    from sklearn.utils.class_weight import compute_class_weight

    label_counts = Counter(y.tolist())
    print("\nClass distribution:")
    for idx, name in enumerate(label_names):
        print(f"  {name:12s}: {label_counts[idx]} samples")

    use_stratify = all(c >= 2 for c in label_counts.values())
    
    # Split to full train and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    # Create stratified validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full if use_stratify else None,
    )

    # Compute class weights to handle imbalanced emotion classes
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    print("\nClass weights:", {label_names[k]: round(v, 3) for k, v in class_weight.items()})

    model = build_model(X_train.shape[1:], len(label_names))
    callbacks = [
        # Increased patience so the model gets enough time to learn rare emotions
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)

    report = classification_report(
        y_test, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    print("\nClassification report:\n", report)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Training history (last epoch):")
    print(json.dumps({k: float(v[-1]) for k, v in history.history.items()}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
