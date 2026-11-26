#!/usr/bin/env python3
"""Create a reproducible data-quality audit and Matplotlib visualizations.

The script profiles missing values, duplicate IDs, class balance, numeric
summaries, and category frequencies. It also creates PNG charts in the output
directory without requiring a graphical desktop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pct(value: float) -> str:
    return f"{value:.2f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


def create_plots(train: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    missing = train.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing.sort_values().plot.barh(ax=ax, color="#2563eb")
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Number of missing values")
        ax.set_ylabel("Column")
        fig.tight_layout()
        path = output_dir / "missing_values.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)

    if "Loan_Status" in train.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        counts = train["Loan_Status"].fillna("<MISSING>").value_counts()
        counts.plot.bar(ax=ax, color=["#16a34a", "#dc2626", "#64748b"][: len(counts)])
        ax.set_title("Loan Status Distribution")
        ax.set_xlabel("Loan status")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        path = output_dir / "target_distribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)

    numeric = train.select_dtypes(include="number")
    if not numeric.empty:
        columns = list(numeric.columns[: min(6, len(numeric.columns))])
        fig, axes = plt.subplots(len(columns), 1, figsize=(9, max(3, 2.5 * len(columns))))
        if len(columns) == 1:
            axes = [axes]
        for ax, column in zip(axes, columns):
            numeric[column].dropna().plot.hist(ax=ax, bins=20, color="#7c3aed", alpha=0.82)
            ax.set_title(f"Distribution: {column}")
            ax.set_ylabel("Frequency")
        fig.tight_layout()
        path = output_dir / "numeric_distributions.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)

        if len(numeric.columns) >= 2:
            fig, ax = plt.subplots(figsize=(9, 7))
            correlation = numeric.corr(numeric_only=True)
            image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title("Numeric Feature Correlation")
            ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=70, ha="right")
            ax.set_yticks(range(len(correlation.columns)), correlation.columns)
            fig.colorbar(image, ax=ax, label="Correlation")
            fig.tight_layout()
            path = output_dir / "correlation_heatmap.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(path)

    return created


def build_report(train: pd.DataFrame, output_path: Path, chart_dir: Path) -> None:
    id_column = "Loan_ID" if "Loan_ID" in train.columns else None
    target_column = "Loan_Status" if "Loan_Status" in train.columns else None
    charts = create_plots(train, chart_dir)

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
    if numeric.empty:
        numeric_summary = pd.DataFrame(columns=["column"])
    else:
        numeric_summary = numeric.describe().T.reset_index().rename(columns={"index": "column"}).round(3)

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
        lines.extend(["", "## Target balance", "", markdown_table(pd.DataFrame({"class": balance.index.astype(str), "count": balance.values, "share": [pct(value / len(train) * 100) for value in balance.values]}))])

    chart_links = [f"- [{path.name}]({path.as_posix()})" for path in charts]
    lines.extend([
        "", "## Generated visualizations", "",
        "\n".join(chart_links) if chart_links else "_No charts were generated._",
        "", "## Missing values and column profile", "", markdown_table(missing),
        "", "## Numeric summary", "", markdown_table(numeric_summary),
        "", "## Categorical frequencies", "",
        "\n\n".join(categorical_sections) if categorical_sections else "_No categorical columns found._",
        "", "## Suggested checks", "",
        "- Review columns with missing values before training.",
        "- Check whether rare categories should be grouped into an `Other` class.",
        "- Keep the identifier column out of model features.",
        "- Compare target balance against the evaluation metric before selecting a decision threshold.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument("--output", default="loan_prediction_data_audit.md", help="Path for the Markdown report")
    parser.add_argument("--chart-dir", default="audit_charts", help="Directory for PNG visualizations")
    args = parser.parse_args()

    train_path = Path(args.train)
    output_path = Path(args.output)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    train = pd.read_csv(train_path)
    build_report(train, output_path, Path(args.chart_dir))
    print(f"Audited {len(train):,} rows and {len(train.columns)} columns.")
    print(f"Saved report to: {output_path.resolve()}")
    print(f"Saved charts to: {Path(args.chart_dir).resolve()}")


if __name__ == "__main__":
    main()
