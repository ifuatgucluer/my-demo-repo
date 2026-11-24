#!/usr/bin/env python3
"""Train, save, and serve a reusable Loan Prediction inference model.

The command supports two workflows:

1. ``train`` fits a preprocessing-safe Logistic Regression model and saves it
   with joblib.
2. ``predict`` loads the saved artifact and scores a JSON object or CSV file.

Example:
    python loan_prediction_inference.py train --train train.csv --model artifacts/loan_model.joblib
    python loan_prediction_inference.py predict --model artifacts/loan_model.joblib --input applicant.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X: pd.DataFrame) -> Pipeline:
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
        ("classifier", LogisticRegression(max_iter=2500, C=0.5, random_state=42)),
    ])


def train_model(train_path: Path, model_path: Path) -> None:
    data = pd.read_csv(train_path)
    if "Loan_Status" not in data.columns:
        raise ValueError("The training file must contain Loan_Status.")
    target = data["Loan_Status"].map({"Y": 1, "N": 0})
    if target.isna().any():
        raise ValueError("Loan_Status must contain only Y and N values.")
    ignored = {"Loan_Status", "Loan_ID"}
    features = [column for column in data.columns if column not in ignored]
    model = build_pipeline(data[features])
    model.fit(data[features], target.astype(int))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "classes": {0: "N", 1: "Y"}}, model_path)
    print(f"Saved model to: {model_path.resolve()}")
    print(f"Training rows: {len(data):,}; features: {len(features)}")


def load_records(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        return pd.DataFrame(payload)
    return pd.read_csv(input_path)


def predict(model_path: Path, input_path: Path, output_path: Path | None) -> None:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    records = load_records(input_path)
    missing = [column for column in features if column not in records.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    probabilities = model.predict_proba(records[features])[:, 1]
    result = records.copy()
    result["Loan_Status"] = ["Y" if value >= 0.5 else "N" for value in probabilities]
    result["approval_probability"] = probabilities.round(6)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path.resolve()}")
    else:
        print(result.to_json(orient="records", indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="fit and save a model")
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--model", default="artifacts/loan_model.joblib")

    predict_parser = subparsers.add_parser("predict", help="score JSON or CSV records")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output")

    args = parser.parse_args()
    if args.command == "train":
        train_model(Path(args.train), Path(args.model))
    else:
        predict(Path(args.model), Path(args.input), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
