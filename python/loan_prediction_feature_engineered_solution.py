#!/usr/bin/env python3
"""Feature-engineered Loan Prediction solution.

Adds domain features, outlier-resistant transformations, XGBoost/LightGBM
blending, OOF threshold tuning and a submission CSV.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Normalise the special string category used by the source data.
    if "Dependents" in out.columns:
        out["Dependents"] = out["Dependents"].replace({"3+": 3}).astype(float)

    eps = 1e-6
    applicant = out.get("ApplicantIncome", 0).fillna(0)
    coapplicant = out.get("CoapplicantIncome", 0).fillna(0)
    loan = out.get("LoanAmount", pd.Series(0, index=out.index)).fillna(out.get("LoanAmount", 0).median())
    term = out.get("Loan_Amount_Term", pd.Series(360, index=out.index)).fillna(360)

    out["TotalIncome"] = applicant + coapplicant
    out["IncomeDifference"] = applicant - coapplicant
    out["IncomeRatio"] = applicant / (coapplicant + 1.0)
    out["CoapplicantShare"] = coapplicant / (out["TotalIncome"] + 1.0)
    out["LoanToIncome"] = loan / (out["TotalIncome"] + 1.0)
    out["MonthlyIncome"] = out["TotalIncome"] / 12.0
    out["EstimatedMonthlyPayment"] = loan * 1000.0 / (term + eps)
    out["PaymentToIncome"] = out["EstimatedMonthlyPayment"] / (out["MonthlyIncome"] + 1.0)
    out["LoanAmountPerDependent"] = loan / (out.get("Dependents", pd.Series(0, index=out.index)).fillna(0) + 1.0)

    # Log transforms reduce the influence of extreme income and loan amounts.
    out["LogApplicantIncome"] = np.log1p(applicant.clip(lower=0))
    out["LogCoapplicantIncome"] = np.log1p(coapplicant.clip(lower=0))
    out["LogTotalIncome"] = np.log1p(out["TotalIncome"].clip(lower=0))
    out["LogLoanAmount"] = np.log1p(loan.clip(lower=0))

    # Useful categorical interactions.
    if "Credit_History" in out.columns:
        credit = out["Credit_History"].fillna(-1).astype(str)
        out["Credit_Education"] = credit + "_" + out["Education"].fillna("Unknown").astype(str)
        out["Credit_Property"] = credit + "_" + out["Property_Area"].fillna("Unknown").astype(str)
        out["Credit_SelfEmployed"] = credit + "_" + out["Self_Employed"].fillna("Unknown").astype(str)
    if "Married" in out.columns and "Dependents" in out.columns:
        out["Married_Dependents"] = out["Married"].fillna("Unknown").astype(str) + "_" + out["Dependents"].fillna(-1).astype(str)

    return out.replace([np.inf, -np.inf], np.nan)


def prep(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def best_threshold(y: pd.Series, p: np.ndarray) -> tuple[float, float]:
    grid = np.arange(0.25, 0.751, 0.0025)
    scores = np.array([accuracy_score(y, (p >= t).astype(int)) for t in grid])
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_u6lujuX_CVtuZ9i.csv")
    ap.add_argument("--test", default="test_Y3wMUE5_7gLdaTN.csv")
    ap.add_argument("--output", default="submission_feature_engineered.csv")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    y = train["Loan_Status"].map({"Y": 1, "N": 0}).astype(int)
    test_ids = test["Loan_ID"]
    raw_features = [c for c in test.columns if c != "Loan_ID"]
    X = feature_engineer(train[raw_features])
    X_test = feature_engineer(test[raw_features])

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=123)
    xgb = Pipeline([
        ("prep", prep(X)),
        ("model", XGBClassifier(n_estimators=650, learning_rate=0.02, max_depth=3, min_child_weight=3, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=3.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=123, n_jobs=-1)),
    ])
    lgbm = Pipeline([
        ("prep", prep(X)),
        ("model", LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=12, max_depth=4, min_child_samples=25, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, reg_alpha=0.05, reg_lambda=3.0, verbosity=-1, random_state=123, n_jobs=-1)),
    ])
    logistic = Pipeline([
        ("prep", prep(X)),
        ("model", LogisticRegression(max_iter=2500, C=0.35, class_weight=None, random_state=123)),
    ])

    xoof = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    loof = cross_val_predict(lgbm, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    goof = cross_val_predict(logistic, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    oof = 0.45 * xoof + 0.40 * loof + 0.15 * goof
    threshold, score = best_threshold(y, oof)

    xgb.fit(X, y)
    lgbm.fit(X, y)
    logistic.fit(X, y)
    p = 0.45 * xgb.predict_proba(X_test)[:, 1] + 0.40 * lgbm.predict_proba(X_test)[:, 1] + 0.15 * logistic.predict_proba(X_test)[:, 1]
    pred = np.where(p >= threshold, "Y", "N")
    submission = pd.DataFrame({"Loan_ID": test_ids, "Loan_Status": pred})
    submission.to_csv(args.output, index=False)
    print(f"Feature count: {X.shape[1]}")
    print(f"OOF accuracy: {score:.4f}; threshold: {threshold:.4f}")
    print(f"Wrote {len(submission)} predictions to {args.output}")
    print(submission["Loan_Status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
