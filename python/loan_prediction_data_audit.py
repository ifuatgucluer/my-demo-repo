#!/usr/bin/env python3
"""Create a reproducible data-quality and exploratory audit for Loan Prediction.

This script is intentionally separate from the model-training examples. It
profiles columns, missing values, duplicate IDs, class balance, numeric
summary statistics, and category frequencies, then writes a Markdown report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pct(value: float) -> str:
    return f"{value:.2f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


def build_report(train: pd.DataFrame, output_path: Path) -> None:
    id_column = "Loan_ID" if "Loan_ID" in train.columns else None
    target_column = "Loan_Status" if "Loan_Status" in train.columns else None

    missing = pd.DataFrame(
        {
            "column": train.columns,
            "dtype": [str(train[column].dtype) for column in train.columns],
            "missing_count": [int(train[column].isna().sum()) for column in train.columns],
            "missing_pct": [pct(train[column].isna().mean() * 100) for column in train.columns],
            "unique_count": [int(train[column].nunique(dropna=True)) for column in train.columns],
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])

    numeric = train.select_dtypes(include="number")
    numeric_summary = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    numeric_summary = numeric_summary.round(3)

    categorical_sections: list[str] = []
    categorical = train.select_dtypes(exclude="number")
    for column in categorical.columns:
        counts = train[column].fillna("<MISSING>").value_counts(dropna=False).head(15)
        table = pd.DataFrame({"value": counts.index.astype(str), "count": counts.values})
        categorical_sections.append(f"### `{column}`\n\n{markdown_table(table)}")

    lines = [
        "# Loan Prediction Data Audit",
        "",
        "## Dataset overview",
        "",
        f"- Rows: **{len(train):,}**",
        f"- Columns: **{len(train.columns)}**",
        f"- Duplicate rows: **{int(train.duplicated().sum()):,}**",
    ]
    if id_column:
        lines.append(f"- Duplicate `{id_column}` values: **{int(train[id_column].duplicated().sum()):,}**")
    if target_column:
        balance = train[target_column].fillna("<MISSING>").value_counts()
        lines.extend(
            [
                "",
                "## Target balance",
                "",
                markdown_table(
                    pd.DataFrame(
                        {
                            "class": balance.index.astype(str),
                            "count": balance.values,
                            "share": [pct(value / len(train) * 100) for value in balance.values],
                        }
                    )
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Missing values and column profile",
            "",
            markdown_table(missing),
            "",
            "## Numeric summary",
            "",
            markdown_table(numeric_summary),
            "",
            "## Categorical frequencies",
            "",
            "\n\n".join(categorical_sections) if categorical_sections else "_No categorical columns found._",
            "",
            "## Suggested checks",
            "",
            "- Review columns with missing values before training.",
            "- Check whether rare categories should be grouped into an `Other` class.",
            "- Keep the identifier column out of model features.",
            "- Compare target balance against the evaluation metric before selecting a decision threshold.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument(
        "--output",
        default="loan_prediction_data_audit.md",
        help="Path for the Markdown audit report",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    output_path = Path(args.output)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    train = pd.read_csv(train_path)
    build_report(train, output_path)
    print(f"Audited {len(train):,} rows and {len(train.columns)} columns.")
    print(f"Saved report to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
