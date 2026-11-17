# my-demo-repo

## Feature-Engineered Loan Prediction

This repository includes an original machine-learning solution for the Analytics Vidhya Loan Prediction dataset.

The solution combines XGBoost, LightGBM, and Logistic Regression with:

- domain-based income and repayment features,
- categorical interaction features,
- missing-value imputation,
- one-hot encoding,
- seven-fold stratified cross-validation,
- out-of-fold threshold optimization, and
- weighted probability ensembling.

### Run

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
```

Run the solution with the competition CSV files:

```bash
python python/loan_prediction_feature_engineered_solution.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --output submission_feature_engineered.csv
```

The generated submission contains the required `Loan_ID` and `Loan_Status` columns.
