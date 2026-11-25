# Python API and Usage Documentation

This document describes every Python command-line utility in `my-demo-repo`. Unless noted otherwise, commands are run from the repository root with the virtual environment activated.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm tabulate joblib
```

The official competition dataset should be downloaded separately. The repository does not include competition data.

## Common data contract

Training data normally contains `Loan_Status` with `Y`/`N` labels. `Loan_ID` is an identifier and should not be used as a model feature. Typical predictors include `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, and `Property_Area`.

---

## `loan_prediction_data_audit.py`

### Purpose

Profiles a training CSV and generates a Markdown report plus PNG visualizations. The script uses Matplotlib's non-interactive `Agg` backend, so it works on servers without a display.

### Command

```bash
python python/loan_prediction_data_audit.py \
  --train data/train.csv \
  --output reports/data_audit.md \
  --chart-dir reports/charts
```

### Arguments

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--train` | yes | — | Training CSV path. |
| `--output` | no | `loan_prediction_data_audit.md` | Markdown report path. |
| `--chart-dir` | no | `audit_charts` | Directory for generated PNG files. |

### Outputs

`missing_values.png`, `target_distribution.png`, `numeric_distributions.png`, and `correlation_heatmap.png` are created when their required columns/data are available. The report includes row/column counts, duplicate counts, target balance, missingness, numeric summaries, and categorical frequencies.

---

## `loan_prediction_preprocess.py`

### Purpose

Creates model-ready train and test matrices while preventing preprocessing leakage. Medians, modes, and category vocabularies are learned from training data only.

### Command

```bash
python python/loan_prediction_preprocess.py \
  --train data/train.csv \
  --test data/test.csv \
  --output-dir processed_data
```

### Outputs

- `train_processed.csv`: one-hot encoded training features plus `Loan_ID` and `Loan_Status`.
- `test_processed.csv`: matching one-hot encoded test features plus `Loan_ID`.
- `preprocessing_summary.json`: learned fill values, category vocabularies, row counts, and feature count.

The `Dependents` value `3+` is converted to numeric `3`. Unknown test categories are handled by reindexing the test matrix to the training feature columns.

---

## `loan_prediction_model_comparison.py`

### Purpose

Compares Logistic Regression, Random Forest, and HistGradientBoosting using five-fold stratified cross-validation.

### Command

```bash
python python/loan_prediction_model_comparison.py \
  --train data/train.csv \
  --report reports/model_comparison.csv
```

### Output schema

The report contains `model`, `accuracy_mean`, `accuracy_std`, `f1_mean`, and `roc_auc_mean`. Models use an imputation, scaling, and one-hot encoding pipeline fitted inside each cross-validation fold.

---

## `loan_prediction_feature_engineered_solution.py`

### Purpose

Generates a competition submission using feature engineering and a weighted XGBoost, LightGBM, and Logistic Regression ensemble.

### Command

```bash
python python/loan_prediction_feature_engineered_solution.py \
  --train data/train.csv \
  --test data/test.csv \
  --output submission_feature_engineered.csv
```

### Feature groups

The script creates total income, income ratios, loan-to-income burden, estimated payment burden, dependent-adjusted loan amount, log transformations, and categorical interactions involving credit history, education, property area, employment, marital status, and dependents. It selects a probability threshold from seven-fold out-of-fold predictions.

### Output schema

The submission contains exactly `Loan_ID` and `Loan_Status` with `Y`/`N` predictions.

---

## `loan_prediction_error_analysis.py`

### Purpose

Measures out-of-fold classification quality across a threshold grid and exports validation errors.

### Command

```bash
python python/loan_prediction_error_analysis.py \
  --train data/train.csv \
  --report reports/error_analysis_metrics.csv \
  --errors reports/out_of_fold_errors.csv
```

### Outputs

The metrics CSV contains `threshold`, `accuracy`, `f1`, `precision`, and `recall`. The errors CSV includes the identifier when available, actual/predicted status, predicted probability, confidence, and `is_error`.

---

## `loan_prediction_inference.py`

### Purpose

Trains a reusable Logistic Regression pipeline and scores new applicant records from CSV or JSON. The fitted preprocessing and classifier are stored together in a Joblib artifact.

### Train a model

```bash
python python/loan_prediction_inference.py train \
  --train data/train.csv \
  --model artifacts/loan_model.joblib
```

### Score records

```bash
python python/loan_prediction_inference.py predict \
  --model artifacts/loan_model.joblib \
  --input examples/applicant.json \
  --output reports/predictions.csv
```

If `--output` is omitted, predictions are printed as JSON. The output contains `Loan_Status` and `approval_probability`.

---

## `loan_prediction_drift_check.py`

### Purpose

Checks whether train and test distributions differ before modeling. Numeric columns use PSI based on train quantile bins. Categorical columns use absolute category-share differences.

### Command

```bash
python python/loan_prediction_drift_check.py \
  --train examples/drift_sample/train.csv \
  --test examples/drift_sample/test.csv \
  --output reports/data_drift_report.csv
```

### Severity interpretation

- Numeric PSI `< 0.10`: low drift; `0.10–0.25`: medium; `>= 0.25`: high.
- Categorical share difference `< 0.10`: low; `0.10–0.20`: medium; `>= 0.20`: high.

The thresholds are screening heuristics, not proof of data quality. Investigate high scores with domain context.

---

## `evaluation.py`

### Purpose

Evaluates predictions against labeled ground truth. It supports CSV and JSON files, normalizes `Y/N`, `Yes/No`, and `1/0` labels, and calculates classification metrics.

### Command

```bash
python python/evaluation.py \
  --truth validation_truth.csv \
  --predictions validation_predictions.csv \
  --truth-column Loan_Status \
  --prediction-column Loan_Status \
  --probability-column approval_probability \
  --output reports/evaluation.json
```

### Output metrics

The JSON report contains row count, accuracy, precision, recall, F1, confusion matrix, a per-class classification report, and ROC-AUC when a valid probability column and both classes are present.

---

## Reproducibility notes

Use the same source data, Python package versions, and command arguments when comparing results. Competition leaderboard scores may differ from local metrics because the evaluation labels are hidden. These scripts are educational examples and should not be used as the sole basis for lending decisions.
