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
import loan_prediction_model_comparison as comparison
import loan_prediction_preprocess as preprocess


def test_evaluation_normalizes_supported_labels() -> None:
    result = evaluation.normalize_labels(pd.Series(["Y", "n", "Yes", "0", "1"]))
    assert result.tolist() == [1, 0, 1, 0, 1]


def test_evaluation_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unsupported labels"):
        evaluation.normalize_labels(pd.Series(["Y", "MAYBE"]))


def test_evaluation_reads_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    json_path = tmp_path / "rows.json"
    frame = pd.DataFrame({"Loan_Status": ["Y", "N"]})
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps([{"Loan_Status": "Y"}, {"Loan_Status": "N"}]))
    assert evaluation.read_table(csv_path).equals(frame)
    assert evaluation.read_table(json_path).equals(frame)


def test_evaluation_main_writes_metrics(tmp_path: Path) -> None:
    truth = pd.DataFrame({"Loan_ID": ["A", "B", "C", "D"], "Loan_Status": ["Y", "N", "Y", "N"]})
    predictions = pd.DataFrame({"Loan_ID": ["A", "B", "C", "D"], "Loan_Status": ["Y", "N", "N", "N"], "approval_probability": [0.9, 0.1, 0.4, 0.2]})
    truth_path, predictions_path = tmp_path / "truth.csv", tmp_path / "predictions.csv"
    output = tmp_path / "reports" / "evaluation.json"
    truth.to_csv(truth_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    import sys
    old = sys.argv
    sys.argv = ["evaluation.py", "--truth", str(truth_path), "--predictions", str(predictions_path), "--output", str(output)]
    try:
        evaluation.main()
    finally:
        sys.argv = old
    result = json.loads(output.read_text())
    assert result["rows_evaluated"] == 4
    assert result["accuracy"] == 0.75
    assert "roc_auc" in result


def test_drift_psi_detects_change() -> None:
    expected = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    actual = pd.Series([90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
    assert drift.psi(expected, actual) > 0.25
    assert drift.psi(pd.Series([1, 1]), pd.Series([1, 1])) == 0.0


def test_drift_categorical_report_contains_all_categories() -> None:
    train = pd.DataFrame({"Area": ["Urban", "Urban", "Rural"]})
    test = pd.DataFrame({"Area": ["Rural", "Semiurban"]})
    rows = drift.categorical_report(train, test, "Area")
    categories = {row["category"] for row in rows}
    assert categories == {"Urban", "Rural", "Semiurban"}
    assert any(row["severity"] in {"medium", "high"} for row in rows)


def test_audit_creates_report_and_charts(tmp_path: Path, loan_train: pd.DataFrame) -> None:
    report = tmp_path / "audit.md"
    charts = tmp_path / "charts"
    audit.build_report(loan_train, report, charts)
    assert report.exists()
    text = report.read_text()
    assert "Missing values and column profile" in text
    assert "Generated visualizations" in text
    assert (charts / "missing_values.png").exists()
    assert (charts / "target_distribution.png").exists()
    assert (charts / "numeric_distributions.png").exists()
    assert (charts / "correlation_heatmap.png").exists()


def test_preprocess_is_leakage_safe_and_aligns_columns(tmp_path: Path, csv_pair: tuple[Path, Path]) -> None:
    train_path, test_path = csv_pair
    output_dir = tmp_path / "processed"
    preprocess.prepare(train_path, test_path, output_dir)
    train_out = pd.read_csv(output_dir / "train_processed.csv")
    test_out = pd.read_csv(output_dir / "test_processed.csv")
    summary = json.loads((output_dir / "preprocessing_summary.json").read_text())
    assert list(train_out.columns) == list(test_out.columns) + ["Loan_Status"]
    assert not train_out.drop(columns=["Loan_ID", "Loan_Status"]).isna().any().any()
    assert not test_out.drop(columns=["Loan_ID"]).isna().any().any()
    assert summary["rows"]["train"] == 8


def test_error_analysis_threshold_table_prefers_good_threshold() -> None:
    y = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    table = error_analysis.threshold_table(y, probabilities)
    assert table.iloc[0]["accuracy"] == 1.0
    assert 0.2 <= table.iloc[0]["threshold"] <= 0.8


def test_feature_engineering_adds_domain_columns() -> None:
    module = pytest.importorskip("loan_prediction_feature_engineered_solution")
    source = pd.DataFrame({
        "Dependents": ["3+", "0"],
        "ApplicantIncome": [5000, 3000],
        "CoapplicantIncome": [1000, 0],
        "LoanAmount": [120.0, 80.0],
        "Loan_Amount_Term": [360.0, 360.0],
        "Credit_History": [1.0, 0.0],
        "Education": ["Graduate", "Not Graduate"],
        "Property_Area": ["Urban", "Rural"],
        "Self_Employed": ["No", "Yes"],
        "Married": ["Yes", "No"],
    })
    result = module.feature_engineer(source)
    assert result["Dependents"].tolist() == [3.0, 0.0]
    for column in ["TotalIncome", "LoanToIncome", "PaymentToIncome", "LogTotalIncome", "Credit_Education"]:
        assert column in result.columns


def test_model_comparison_builds_three_pipelines(loan_train: pd.DataFrame) -> None:
    X = loan_train.drop(columns=["Loan_ID", "Loan_Status"])
    models = comparison.build_models(X)
    assert set(models) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    for model in models.values():
        model.fit(X, pd.Series([1, 0, 0, 1, 1, 0, 1, 1]))
        assert len(model.predict(X)) == len(X)


def test_inference_loads_json_records(tmp_path: Path) -> None:
    path = tmp_path / "applicant.json"
    path.write_text(json.dumps({"ApplicantIncome": 5000, "Education": "Graduate"}))
    records = inference.load_records(path)
    assert records.shape == (1, 2)
    assert records.iloc[0]["Education"] == "Graduate"


def test_inference_pipeline_can_fit_small_data(loan_train: pd.DataFrame) -> None:
    features = [column for column in loan_train.columns if column not in {"Loan_ID", "Loan_Status"}]
    model = inference.build_pipeline(loan_train[features])
    X = loan_train[features]
    y = loan_train["Loan_Status"].map({"Y": 1, "N": 0})
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
