#!/usr/bin/env python3
"""Evaluate classification predictions against labeled ground truth.

The truth and prediction files may be CSVs or JSON/JSONL files. By default,
the script looks for ``Loan_Status`` and optionally uses a probability column
to calculate ROC-AUC and threshold metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload if isinstance(payload, list) else [payload])
    return pd.read_csv(path)


def normalize_labels(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.upper().map({"Y": 1, "N": 0, "1": 1, "0": 0, "YES": 1, "NO": 0})
    if mapped.isna().any():
        unknown = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported labels found: {unknown}")
    return mapped.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Labeled CSV or JSON file")
    parser.add_argument("--predictions", required=True, help="Predictions CSV or JSON file")
    parser.add_argument("--truth-column", default="Loan_Status")
    parser.add_argument("--prediction-column", default="Loan_Status")
    parser.add_argument("--probability-column", default="approval_probability")
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()

    truth = read_table(Path(args.truth))
    predictions = read_table(Path(args.predictions))
    if args.truth_column not in truth.columns:
        raise ValueError(f"Truth column not found: {args.truth_column}")
    if args.prediction_column not in predictions.columns:
        raise ValueError(f"Prediction column not found: {args.prediction_column}")
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction files must contain the same number of rows.")

    y_true = normalize_labels(truth[args.truth_column])
    y_pred = normalize_labels(predictions[args.prediction_column])
    result: dict[str, object] = {
        "rows_evaluated": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["N", "Y"], output_dict=True, zero_division=0),
    }
    if args.probability_column in predictions.columns:
        probability = pd.to_numeric(predictions[args.probability_column], errors="coerce")
        if probability.notna().all() and y_true.nunique() == 2:
            result["roc_auc"] = roc_auc_score(y_true, probability)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"Saved evaluation to: {output.resolve()}")


if __name__ == "__main__":
    main()
