# Loan Prediction ML Toolkit

A reproducible machine-learning toolkit for the Analytics Vidhya Loan Prediction dataset. The repository contains independent utilities for data auditing, visualization, leakage-safe preprocessing, model comparison, and ensemble-based prediction.

> **Dataset note:** The competition CSV files are intentionally not committed to this repository. Download them from the authorized competition source and pass their local paths to the scripts.

## Highlights

- Leakage-safe preprocessing: imputation statistics and category vocabularies are learned from `train.csv` only.
- Automated data-quality report with missing-value, duplicate, target-balance, and category analysis.
- Matplotlib charts generated in a headless environment.
- Reproducible model comparison with stratified cross-validation.
- Feature-engineered XGBoost + LightGBM + Logistic Regression ensemble.
- Clear command-line interfaces and machine-readable outputs.

## Repository structure

```text
.
├── python/
│   ├── loan_prediction_data_audit.py
│   ├── loan_prediction_preprocess.py
│   ├── loan_prediction_model_comparison.py
│   └── loan_prediction_feature_engineered_solution.py
├── README.md
├── pom.xml
└── deneme.iml
```

## Installation

Python 3.10+ is recommended:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm tabulate
```

## 1. Generate a data audit and visualizations

```bash
python python/loan_prediction_data_audit.py \
  --train path/to/train.csv \
  --output reports/loan_prediction_data_audit.md \
  --chart-dir reports/charts
```

The command creates:

- a Markdown data-quality report,
- `missing_values.png`,
- `target_distribution.png`,
- `numeric_distributions.png`, and
- `correlation_heatmap.png` when the relevant columns are available.

## 2. Preprocess train and test data

```bash
python python/loan_prediction_preprocess.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --output-dir processed_data
```

Outputs:

- `processed_data/train_processed.csv`
- `processed_data/test_processed.csv`
- `processed_data/preprocessing_summary.json`

Numeric missing values are filled with training-set medians. Categorical missing values are filled with training-set modes, and categorical variables are one-hot encoded using categories learned from the training data. `Loan_ID` is preserved for submissions and `Loan_Status` is preserved only in the processed training file.

## 3. Compare baseline models

```bash
python python/loan_prediction_model_comparison.py \
  --train path/to/train.csv \
  --report reports/model_comparison_report.csv
```

The utility evaluates Logistic Regression, Random Forest, and HistGradientBoosting using five-fold stratified cross-validation and reports accuracy, F1, ROC-AUC, and variability.

## 4. Train the feature-engineered ensemble

```bash
python python/loan_prediction_feature_engineered_solution.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --output submission_feature_engineered.csv
```

The ensemble adds income, repayment-burden, log-transformed, and categorical-interaction features. It combines XGBoost, LightGBM, and Logistic Regression probabilities and selects a classification threshold from out-of-fold predictions.

## Reproducibility and responsible use

All examples use fixed random seeds where supported. Cross-validation is used for model comparison, but the final competition score can differ from local validation because the evaluation labels are hidden. This project is for educational and research use; predictions should not be used as the sole basis for real lending or credit decisions. Review fairness, data quality, and applicable regulations before any real-world deployment.

## License

Unless a separate license is added, the code is provided for educational use in this repository. Review the competition's dataset and usage terms before redistributing data or submitting results.
