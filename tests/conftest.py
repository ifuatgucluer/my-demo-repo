from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.fixture
def loan_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Loan_ID": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
            "Gender": ["Male", "Female", "Male", "Male", None, "Female", "Male", "Female"],
            "Married": ["Yes", "No", "Yes", "Yes", "No", None, "Yes", "No"],
            "Dependents": ["0", "1", "2", "3+", "0", "1", None, "0"],
            "Education": ["Graduate", "Graduate", "Not Graduate", "Graduate", "Graduate", "Graduate", "Not Graduate", "Graduate"],
            "Self_Employed": ["No", "No", "Yes", "No", None, "No", "Yes", "No"],
            "ApplicantIncome": [5000, 3200, 2800, 6200, 4100, 2500, 7100, 3600],
            "CoapplicantIncome": [1500, 0, 1200, 1800, 900, 0, 2200, 0],
            "LoanAmount": [120.0, 80.0, 95.0, 150.0, None, 70.0, 180.0, 90.0],
            "Loan_Amount_Term": [360.0, 360.0, 360.0, 360.0, 360.0, None, 360.0, 360.0],
            "Credit_History": [1.0, 1.0, 0.0, 1.0, 1.0, None, 1.0, 1.0],
            "Property_Area": ["Urban", "Rural", "Semiurban", "Urban", "Semiurban", "Rural", "Urban", "Semiurban"],
            "Loan_Status": ["Y", "N", "N", "Y", "Y", "N", "Y", "Y"],
        }
    )


@pytest.fixture
def loan_test(loan_train: pd.DataFrame) -> pd.DataFrame:
    return loan_train.drop(columns=["Loan_Status"]).copy().assign(
        Loan_ID=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    )


@pytest.fixture
def csv_pair(tmp_path: Path, loan_train: pd.DataFrame, loan_test: pd.DataFrame) -> tuple[Path, Path]:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    loan_train.to_csv(train_path, index=False)
    loan_test.to_csv(test_path, index=False)
    return train_path, test_path
