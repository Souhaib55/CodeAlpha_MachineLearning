"""
Task 4 - Disease Prediction From Medical Data

Predict the possibility of diseases based on patient data using classifiers.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.svm import SVC

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disease prediction on medical data.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional CSV file with target column.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name (required when using --data unless last column).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["logreg", "rf", "svm", "xgboost", "all"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_default_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def load_custom_dataset(path: Path, target_col: Optional[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must include at least one feature column and one target.")

    if target_col and target_col in df.columns:
        target = df.pop(target_col)
    else:
        target = df.iloc[:, -1]
        df = df.iloc[:, :-1]

    mask = target.notna()
    df = df.loc[mask].reset_index(drop=True)
    target = target.loc[mask].reset_index(drop=True)

    y = normalize_target(target)
    return df, y


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
    positive = {"yes", "1", "true", "positive", "disease", "bad"}
    return s_lower.isin(positive).astype(int).to_numpy()


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
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

    try:
        if args.data:
            X, y = load_custom_dataset(Path(args.data), args.target)
        else:
            X, y = load_default_dataset()
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
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=args.random_state
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=args.random_state),
    }

    if xgb is not None:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=args.random_state,
        )

    selected = models if args.model == "all" else {args.model: models[args.model]}
    for name, model in selected.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor)

    return 0


if __name__ == "__main__":
    sys.exit(main())
