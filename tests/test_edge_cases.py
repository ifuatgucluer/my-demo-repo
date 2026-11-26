from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluation
import loan_prediction_data_audit as audit
import loan_prediction_drift_check as drift
import loan_prediction_error_analysis as error_analysis
import loan_prediction_inference as inference
import loan_prediction_preprocess as preprocess


def test_evaluation_rejects_mismatched_row_counts(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame({"Loan_Status": ["Y", "N"]}).to_csv(truth_path, index=False)
    pd.DataFrame({"Loan_Status": ["Y"]}).to_csv(prediction_path, index=False)
    with pytest.raises(ValueError, match="same number of rows"):
        old_argv = __import__("sys").argv
        __import__("sys").argv = ["evaluation.py", "--truth", str(truth_path), "--predictions", str(prediction_path)]
        try:
            evaluation.main()
        finally:
            __import__("sys").argv = old_argv


def test_evaluation_skips_roc_auc_for_single_class(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"Loan_Status": ["Y", "Y"]}).to_csv(truth, index=False)
    pd.DataFrame({"Loan_Status": ["Y", "Y"], "approval_probability": [0.8, 0.9]}).to_csv(predictions, index=False)
    output = tmp_path / "result.json"
    old_argv = __import__("sys").argv
    __import__("sys").argv = ["evaluation.py", "--truth", str(truth), "--predictions", str(predictions), "--output", str(output)]
    try:
        evaluation.main()
    finally:
        __import__("sys").argv = old_argv
    assert "roc_auc" not in json.loads(output.read_text())


def test_audit_handles_frame_without_target_or_numeric_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({"A": ["x", "y", None], "B": ["x", "x", "y"]})
    report = tmp_path / "audit.md"
    audit.build_report(frame, report, tmp_path / "charts")
    text = report.read_text()
    assert "Dataset overview" in text
    assert "_No rows._" in text
    assert "No categorical columns" not in text
    assert not (tmp_path / "charts" / "target_distribution.png").exists()


def test_drift_psi_handles_missing_and_constant_values() -> None:
    expected = pd.Series([1, None, 1, 1])
    actual = pd.Series([1, None, 1, 1])
    assert drift.psi(expected, actual) == 0.0
    assert drift.psi(pd.Series([None, None]), pd.Series([1, 2])) == 0.0


def test_preprocess_handles_unseen_test_category(tmp_path: Path) -> None:
    train = pd.DataFrame({"Loan_ID": ["A", "B"], "Category": ["known", "known"], "Income": [1.0, None], "Loan_Status": ["Y", "N"]})
    test = pd.DataFrame({"Loan_ID": ["C"], "Category": ["new"], "Income": [3.0]})
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    output = tmp_path / "processed"
    preprocess.prepare(train_path, test_path, output)
    train_out = pd.read_csv(output / "train_processed.csv")
    test_out = pd.read_csv(output / "test_processed.csv")
    assert list(test_out.columns) == [column for column in train_out.columns if column != "Loan_Status"]
    assert test_out.filter(regex="Category_").sum().sum() == 0


def test_inference_train_and_predict_round_trip(tmp_path: Path, loan_train: pd.DataFrame) -> None:
    train_path = tmp_path / "train.csv"
    loan_train.to_csv(train_path, index=False)
    model_path = tmp_path / "model.joblib"
    inference.train_model(train_path, model_path)
    input_path = tmp_path / "applicant.json"
    row = loan_train.drop(columns=["Loan_ID", "Loan_Status"]).iloc[0].to_dict()
    input_path.write_text(json.dumps(row, default=str))
    output_path = tmp_path / "predictions.csv"
    inference.predict(model_path, input_path, output_path)
    result = pd.read_csv(output_path)
    assert result.loc[0, "Loan_Status"] in {"Y", "N"}
    assert 0.0 <= result.loc[0, "approval_probability"] <= 1.0


def test_error_analysis_threshold_table_covers_requested_grid() -> None:
    table = error_analysis.threshold_table(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert table["threshold"].min() == 0.25
    assert table["threshold"].max() <= 0.75
    assert {"accuracy", "f1", "precision", "recall"}.issubset(table.columns)


def test_preprocess_summary_is_valid_json(tmp_path: Path, csv_pair: tuple[Path, Path]) -> None:
    train_path, test_path = csv_pair
    output = tmp_path / "processed"
    preprocess.prepare(train_path, test_path, output)
    summary = json.loads((output / "preprocessing_summary.json").read_text())
    assert summary["outputs"]["train"].endswith("train_processed.csv")
    assert summary["numeric_fill_values"]
    assert "category_values_learned_from_train" in summary


def test_drift_report_contains_numeric_and_categorical_rows(tmp_path: Path, csv_pair: tuple[Path, Path]) -> None:
    train_path, test_path = csv_pair
    output = tmp_path / "drift.csv"
    old_argv = __import__("sys").argv
    __import__("sys").argv = ["drift.py", "--train", str(train_path), "--test", str(test_path), "--output", str(output)]
    try:
        drift.main()
    finally:
        __import__("sys").argv = old_argv
    report = pd.read_csv(output)
    assert {"numeric", "categorical"}.issubset(set(report["type"]))
    assert report["score"].notna().all()


def test_feature_engineering_replaces_infinite_values() -> None:
    module = pytest.importorskip("loan_prediction_feature_engineered_solution")
    frame = pd.DataFrame({
        "ApplicantIncome": [0], "CoapplicantIncome": [0], "LoanAmount": [0], "Loan_Amount_Term": [0],
        "Dependents": ["0"], "Credit_History": [1.0], "Education": ["Graduate"],
        "Property_Area": ["Urban"], "Self_Employed": ["No"], "Married": ["No"],
    })
    result = module.feature_engineer(frame)
    assert not np.isinf(result.select_dtypes(include="number")).any().any()
