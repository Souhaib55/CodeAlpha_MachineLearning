"""
Task 1 - Credit Scoring Model

Predict an individual's creditworthiness using past financial data.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
    "german.data"
)

GERMAN_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

GERMAN_NUMERIC_COLS = {
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "people_liable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit scoring model with classic classification algorithms."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file with a binary target column.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["logreg", "tree", "rf", "all"],
        help="Which model to train.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--download-cache",
        type=str,
        default="data/credit",
        help="Cache folder for downloaded datasets.",
    )
    return parser.parse_args()


def download_german_credit(cache_dir: Path) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "german.data"
    if data_path.exists():
        return data_path

    try:
        print(f"Downloading German Credit dataset to {data_path}...")
        urllib.request.urlretrieve(GERMAN_CREDIT_URL, data_path)
        return data_path
    except Exception as exc:  # pragma: no cover - network may be unavailable
        print(f"Download failed: {exc}")
        return None


def load_custom_csv(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must include at least one feature column and one target.")

    if "target" in df.columns:
        target = df.pop("target")
    elif "credit_risk" in df.columns:
        target = df.pop("credit_risk")
    else:
        target = df.iloc[:, -1]
        df = df.iloc[:, :-1]

    mask = target.notna()
    df = df.loc[mask].reset_index(drop=True)
    target = target.loc[mask].reset_index(drop=True)

    y = normalize_target(target)
    return df, y


def load_german_credit(cache_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    data_path = download_german_credit(cache_dir)
    if data_path is None:
        raise FileNotFoundError(
            "German Credit dataset could not be downloaded. Provide --data instead."
        )

    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    target = df.pop("credit_risk")
    y = (target == 2).astype(int).to_numpy()
    return df, y


def load_credit_data(data_path: Optional[str], cache_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    if data_path:
        return load_custom_csv(Path(data_path))

    try:
        return load_german_credit(cache_dir)
    except Exception:
        print("Falling back to a synthetic dataset.")
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=2000,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            weights=[0.7, 0.3],
            class_sep=1.0,
            random_state=42,
        )
        columns = [
            "income",
            "debts",
            "payment_history",
            "credit_utilization",
            "age",
            "employment_length",
            "savings",
            "open_accounts",
        ]
        return pd.DataFrame(X, columns=columns), y


def normalize_target(series: pd.Series) -> np.ndarray:
    s = pd.Series(series).copy()
    unique = sorted(s.unique().tolist())

    if len(unique) != 2:
        raise ValueError("Target column must be binary.")

    if all(isinstance(val, (int, np.integer, float, np.floating)) for val in unique):
        if set(unique) <= {0, 1}:
            return s.astype(int).to_numpy()
        if set(unique) <= {1, 2}:
            return (s == 2).astype(int).to_numpy()

    s_lower = s.astype(str).str.strip().str.lower()
    positive = {
        "bad",
        "default",
        "yes",
        "1",
        "true",
        "high_risk",
        "risk",
    }
    return s_lower.isin(positive).astype(int).to_numpy()


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        if col in GERMAN_NUMERIC_COLS:
            numeric_cols.append(col)
            continue

        if df[col].dtype.kind in "if":
            numeric_cols.append(col)
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() > 0.9:
            df[col] = coerced
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
) -> Dict[str, float]:
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = None
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except ValueError:
            roc_auc = None

    print("\n" + "=" * 72)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if roc_auc is not None:
        metrics["roc_auc"] = float(roc_auc)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.download_cache)

    try:
        X, y = load_credit_data(args.data, cache_dir)
    except Exception as exc:
        print(f"Failed to load data: {exc}")
        return 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "tree": DecisionTreeClassifier(random_state=args.random_state),
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=args.random_state
        ),
    }

    selected = models if args.model == "all" else {args.model: models[args.model]}
    for name, model in selected.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor)

    return 0


if __name__ == "__main__":
    sys.exit(main())
