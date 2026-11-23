#!/usr/bin/env python3
"""Prepare Loan Prediction CSV files for machine-learning training.

Imputation statistics and category vocabularies are learned from the training
file only. The same transformations are then applied to train and test data,
which avoids leaking test information into preprocessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_dependents(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"3+": 3}), errors="coerce")


def prepare(train_path: Path, test_path: Path, output_dir: Path) -> None:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = train["Loan_Status"].copy() if "Loan_Status" in train.columns else None
    id_column = "Loan_ID" if "Loan_ID" in train.columns else None
    feature_columns = [column for column in test.columns if column != id_column]
    train_features = train[feature_columns].copy()
    test_features = test[feature_columns].copy()

    if "Dependents" in train_features.columns:
        train_features["Dependents"] = clean_dependents(train_features["Dependents"])
        test_features["Dependents"] = clean_dependents(test_features["Dependents"])

    numeric_columns = train_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in train_features.columns if column not in numeric_columns]
    numeric_fill: dict[str, float] = {}
    categorical_fill: dict[str, str] = {}

    for column in numeric_columns:
        value = float(train_features[column].median())
        numeric_fill[column] = value
        train_features[column] = train_features[column].fillna(value)
        test_features[column] = test_features[column].fillna(value)

    for column in categorical_columns:
        mode = train_features[column].dropna().mode()
        fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
        categorical_fill[column] = fill_value
        train_features[column] = train_features[column].fillna(fill_value).astype(str)
        test_features[column] = test_features[column].fillna(fill_value).astype(str)

    # Learn category vocabularies from train only and apply identical one-hot columns.
    category_values: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = sorted(train_features[column].astype(str).unique().tolist())
        category_values[column] = values
        train_features[column] = pd.Categorical(train_features[column], categories=values)
        test_features[column] = pd.Categorical(test_features[column], categories=values)

    train_encoded = pd.get_dummies(train_features, columns=categorical_columns, dtype=int)
    test_encoded = pd.get_dummies(test_features, columns=categorical_columns, dtype=int)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    if id_column:
        train_encoded.insert(0, id_column, train[id_column].values)
        test_encoded.insert(0, id_column, test[id_column].values)
    if target is not None:
        train_encoded["Loan_Status"] = target.values

    train_output = output_dir / "train_processed.csv"
    test_output = output_dir / "test_processed.csv"
    summary_output = output_dir / "preprocessing_summary.json"
    train_encoded.to_csv(train_output, index=False)
    test_encoded.to_csv(test_output, index=False)

    summary = {
        "source_train": str(train_path),
        "source_test": str(test_path),
        "rows": {"train": len(train_encoded), "test": len(test_encoded)},
        "feature_count_after_encoding": len(train_encoded.columns) - int(id_column is not None) - int(target is not None),
        "numeric_fill_values": numeric_fill,
        "categorical_fill_values": categorical_fill,
        "category_values_learned_from_train": category_values,
        "outputs": {"train": str(train_output), "test": str(test_output)},
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {train_output}")
    print(f"Saved: {test_output}")
    print(f"Saved: {summary_output}")
    print(f"Encoded feature count: {summary['feature_count_after_encoding']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory")
    args = parser.parse_args()
    prepare(Path(args.train), Path(args.test), Path(args.output_dir))


if __name__ == "__main__":
    main()
