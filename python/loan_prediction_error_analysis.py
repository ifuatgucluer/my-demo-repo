#!/usr/bin/env python3
"""Evaluate probability calibration, thresholds, and out-of-fold errors.

This is an independent analysis utility. It trains a preprocessing-safe
Logistic Regression pipeline, creates out-of-fold probabilities, scans
classification thresholds, and writes both a metrics report and an error CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", transformer),
        ("model", LogisticRegression(max_iter=2500, C=0.5, random_state=42)),
    ])


def threshold_table(y: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.25, 0.751, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y, predictions),
            "f1": f1_score(y, predictions, zero_division=0),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
        })
    return pd.DataFrame(rows).sort_values(["accuracy", "f1"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--report", default="reports/error_analysis_metrics.csv")
    parser.add_argument("--errors", default="reports/out_of_fold_errors.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    if "Loan_Status" not in train.columns:
        raise ValueError("The training file must contain Loan_Status.")

    y = train["Loan_Status"].map({"Y": 1, "N": 0})
    if y.isna().any():
        raise ValueError("Loan_Status must contain only Y and N values.")
    y = y.astype(int)
    id_column = "Loan_ID" if "Loan_ID" in train.columns else None
    feature_columns = [column for column in train.columns if column not in {id_column, "Loan_Status"}]
    X = train[feature_columns]

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_pipeline(X)
    probabilities = cross_val_predict(model, X, y, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
    threshold_metrics = threshold_table(y, probabilities)
    threshold_metrics.to_csv(args.report, index=False)

    best_threshold = float(threshold_metrics.iloc[0]["threshold"])
    predictions = (probabilities >= best_threshold).astype(int)
    errors = train[[column for column in [id_column] if column]].copy() if id_column else pd.DataFrame(index=train.index)
    errors["actual_status"] = np.where(y == 1, "Y", "N")
    errors["predicted_status"] = np.where(predictions == 1, "Y", "N")
    errors["predicted_probability"] = probabilities.round(6)
    errors["confidence"] = np.where(probabilities >= 0.5, probabilities, 1 - probabilities).round(6)
    errors["is_error"] = errors["actual_status"] != errors["predicted_status"]
    errors.sort_values(["is_error", "confidence"], ascending=[False, True]).to_csv(args.errors, index=False)

    print(f"OOF ROC-AUC: {roc_auc_score(y, probabilities):.4f}")
    print(f"Best threshold by accuracy: {best_threshold:.2f}")
    print(f"Best accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"OOF errors: {int(errors['is_error'].sum())}")
    print(f"Saved metrics: {Path(args.report).resolve()}")
    print(f"Saved errors: {Path(args.errors).resolve()}")


if __name__ == "__main__":
    main()
