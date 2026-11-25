#!/usr/bin/env python3
"""Detect train/test distribution drift for Loan Prediction data.

The script calculates Population Stability Index (PSI) for numeric columns and
absolute proportion differences for categorical columns. It produces a CSV
report so data-quality issues can be reviewed before model training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty or expected.nunique() < 2:
        return 0.0
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    # Include test values outside the training range in the first/last bin.
    # Without open-ended edges, np.histogram can produce zero total counts for
    # a completely shifted test distribution and the PSI becomes NaN.
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)
    expected_share = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
    actual_share = np.clip(actual_counts / actual_counts.sum(), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def numeric_report(train: pd.DataFrame, test: pd.DataFrame, column: str) -> dict[str, object]:
    value = psi(train[column], test[column])
    return {
        "column": column,
        "type": "numeric",
        "metric": "PSI",
        "score": round(value, 6),
        "severity": "high" if value >= 0.25 else "medium" if value >= 0.10 else "low",
    }


def categorical_report(train: pd.DataFrame, test: pd.DataFrame, column: str) -> list[dict[str, object]]:
    train_share = train[column].fillna("<MISSING>").astype(str).value_counts(normalize=True)
    test_share = test[column].fillna("<MISSING>").astype(str).value_counts(normalize=True)
    categories = sorted(set(train_share.index) | set(test_share.index))
    rows = []
    for category in categories:
        difference = abs(float(test_share.get(category, 0.0)) - float(train_share.get(category, 0.0)))
        rows.append({
            "column": column,
            "type": "categorical",
            "metric": "absolute_share_difference",
            "category": category,
            "score": round(difference, 6),
            "severity": "high" if difference >= 0.20 else "medium" if difference >= 0.10 else "low",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--output", default="reports/data_drift_report.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    common = [column for column in train.columns if column in test.columns and column != "Loan_ID"]
    rows: list[dict[str, object]] = []
    for column in common:
        if pd.api.types.is_numeric_dtype(train[column]):
            rows.append(numeric_report(train, test, column))
        else:
            rows.extend(categorical_report(train, test, column))

    report = pd.DataFrame(rows).sort_values("score", ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f"Compared {len(common)} shared columns.")
    print(f"Saved drift report to: {output.resolve()}")
    if not report.empty:
        print(report.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
