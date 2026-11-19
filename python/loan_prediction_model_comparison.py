#!/usr/bin/env python3
"""Loan Prediction model comparison utility.

This independent example compares three preprocessing/model pipelines using
stratified cross-validation and writes a compact validation report. It is
separate from the ensemble solution in loan_prediction_feature_engineered_solution.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    def pipeline(model):
        return Pipeline([("preprocess", make_preprocessor(X)), ("model", model)])

    return {
        "logistic_regression": pipeline(
            LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        ),
        "random_forest": pipeline(
            RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": pipeline(
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=1.5,
                random_state=42,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument(
        "--report",
        default="model_comparison_report.csv",
        help="Output path for validation metrics",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    if "Loan_Status" not in train.columns:
        raise ValueError("The training file must contain a Loan_Status column.")

    target = train["Loan_Status"].map({"Y": 1, "N": 0})
    if target.isna().any():
        raise ValueError("Loan_Status must contain only Y and N values.")

    feature_columns = [column for column in train.columns if column not in {"Loan_ID", "Loan_Status"}]
    X = train[feature_columns]
    y = target.astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    rows = []

    for name, model in build_models(X).items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "f1_mean": scores["test_f1"].mean(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )

    report = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    report.to_csv(args.report, index=False)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved report to: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
