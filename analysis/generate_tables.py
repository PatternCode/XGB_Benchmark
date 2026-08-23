
"""
Generate dataset-level Markdown performance tables for the XGB Benchmark.

The script:
1. Reads the combined benchmark results.
2. Keeps the manuscript downstream models:
   LR, XGBoost, DT3, DT4, DT5, DT6.
3. Summarizes Gain, Weight, Cover, and SHAP across CV folds.
4. Treats Random correctly:
   - first averages the 20 random repetitions within each fold/condition;
   - then averages those fold-level random means across folds.
5. Adds the all-features reference using the same downstream models.
6. Writes one Markdown file per requested metric, containing one table
   for each dataset.

Default repository paths:
  input:
    results/summaries/combined_results_9_datasets.csv
  output:
    results/summaries/dataset_performance_<metric>.md

Example:
  python analysis/generate_tables.py

  python analysis/generate_tables.py \
      --metrics f1_macro accuracy balanced_accuracy

  python analysis/generate_tables.py \
      --show-std
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/summaries/combined_results_9_datasets.csv")
DEFAULT_OUTPUT_DIR = Path("results/summaries")

RANKING_METHODS = ["gain", "weight", "cover", "shap", "random"]
RANKING_LABELS = {
    "gain": "Gain",
    "weight": "Weight",
    "cover": "Cover",
    "shap": "SHAP",
    "random": "Random",
    "all_features": "All features",
}

SUPPORTED_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1_macro": "Macro-F1",
    "f1_weighted": "Weighted F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

DATASET_LABELS = {
    "adult_income": "Adult Income",
    "bank_marketing": "Bank Marketing",
    "breast_cancer_wisconsin": "Breast Cancer Wisconsin",
    "cic_ids2017": "CIC-IDS2017",
    "covertype": "Covertype",
    "credit_card_fraud": "Credit Card Fraud",
    "dry_bean": "Dry Bean",
    "steel_plates_faults": "Steel Plates Faults",
    "unsw_nb15": "UNSW-NB15",
}

# Manuscript-facing models requested for the dataset-level tables.
MODEL_ORDER = ["LR", "XGBoost", "DT3", "DT4", "DT5", "DT6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown dataset-level benchmark tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Combined results CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1_macro"],
        choices=SUPPORTED_METRICS,
        help="Metric(s) to export. Default: f1_macro",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places shown in tables. Default: 3",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help=(
            "Show mean ± sample SD across the five fold-level values. "
            "For Random, SD is computed across fold-level means after "
            "averaging the 20 repetitions within each fold."
        ),
    )
    return parser.parse_args()


def model_label(row: pd.Series) -> str | None:
    """Map raw model configuration to manuscript-facing model label."""
    model = row["model"]

    if model == "logistic_regression":
        return "LR"

    if model == "xgboost":
        return "XGBoost"

    if model == "decision_tree":
        depth = row["max_depth"]
        if pd.notna(depth) and int(depth) in {3, 4, 5, 6}:
            return f"DT{int(depth)}"

    # Excludes DT1, DT2, and unrestricted DT from these manuscript tables.
    return None


def validate_input(df: pd.DataFrame, metrics: Iterable[str]) -> None:
    required = {
        "dataset",
        "outer_fold",
        "selection_method",
        "n_selected_features",
        "random_repetition",
        "model",
        "max_depth",
    } | set(metrics)

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    unknown_methods = set(df["selection_method"].dropna().unique()) - {
        "gain", "weight", "cover", "shap", "random", "all_features"
    }
    if unknown_methods:
        raise ValueError(
            "Unexpected selection method(s): "
            + ", ".join(sorted(map(str, unknown_methods)))
        )


def validate_random_repetitions(df: pd.DataFrame) -> None:
    """
    Confirm that every included Random fold/subset/model condition has
    20 independent repetitions.
    """
    random_df = df[df["selection_method"] == "random"].copy()
    if random_df.empty:
        raise ValueError("No random-baseline rows were found.")

    counts = (
        random_df.groupby(
            ["dataset", "outer_fold", "n_selected_features", "model_label"],
            dropna=False,
        )["random_repetition"]
        .nunique()
    )

    bad = counts[counts != 20]
    if not bad.empty:
        sample = bad.head(10)
        raise ValueError(
            "Expected exactly 20 random repetitions for every included "
            "fold/subset/model condition. Example mismatches:\n"
            f"{sample}"
        )


def fold_level_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Return one value per dataset/fold/method/k/model.

    Non-random ranking methods:
      the stored result is already one observation per fold/condition.

    Random:
      average the 20 repetitions *within the fold* first.
    """
    keys = [
        "dataset",
        "outer_fold",
        "selection_method",
        "n_selected_features",
        "model_label",
    ]

    ranked = df[df["selection_method"].isin(["gain", "weight", "cover", "shap"])].copy()

    ranked_fold = (
        ranked.groupby(keys, as_index=False, dropna=False)[metric]
        .mean()
    )

    random_df = df[df["selection_method"] == "random"].copy()
    random_fold = (
        random_df.groupby(keys, as_index=False, dropna=False)[metric]
        .mean()
    )

    return pd.concat([ranked_fold, random_fold], ignore_index=True)


def dataset_level_summary(
    fold_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Aggregate fold-level values into dataset-level mean and SD."""
    keys = [
        "dataset",
        "selection_method",
        "n_selected_features",
        "model_label",
    ]

    return (
        fold_df.groupby(keys, as_index=False, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def all_features_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Summarize the all-features reference across folds.

    There is no subset size for this condition, so it is summarized
    only by dataset and downstream model.
    """
    all_df = df[df["selection_method"] == "all_features"].copy()

    return (
        all_df.groupby(
            ["dataset", "model_label"],
            as_index=False,
            dropna=False,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def format_value(mean: float, std: float, decimals: int, show_std: bool) -> str:
    if pd.isna(mean):
        return "—"

    if show_std and pd.notna(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    return f"{mean:.{decimals}f}"


def markdown_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def make_dataset_table(
    dataset: str,
    selected_summary: pd.DataFrame,
    all_summary: pd.DataFrame,
    decimals: int,
    show_std: bool,
) -> str:
    ds_selected = selected_summary[selected_summary["dataset"] == dataset].copy()
    ds_all = all_summary[all_summary["dataset"] == dataset].copy()

    if ds_selected.empty:
        return ""

    ks = sorted(
        int(k)
        for k in ds_selected["n_selected_features"].dropna().unique()
    )

    # Infer full feature count from all-features rows in the original summary
    # is not possible here because all_summary intentionally does not retain k.
    # It is added by the caller through the table header text when available.

    headers = ["Ranking", "Model"] + [f"k={k}" for k in ks] + ["All"]
    lines = [
        markdown_row(headers),
        markdown_row(["---", "---"] + ["---:"] * (len(ks) + 1)),
    ]

    for method in RANKING_METHODS:
        method_df = ds_selected[ds_selected["selection_method"] == method]

        for model_index, model in enumerate(MODEL_ORDER):
            row = [
                RANKING_LABELS[method] if model_index == 0 else "",
                model,
            ]

            for k in ks:
                cell = method_df[
                    (method_df["model_label"] == model)
                    & (method_df["n_selected_features"] == k)
                ]
                if cell.empty:
                    row.append("—")
                else:
                    r = cell.iloc[0]
                    row.append(
                        format_value(
                            r["mean"], r["std"], decimals, show_std
                        )
                    )

            # All-features values are shown only in the All-features block
            # to avoid repeating the same reference under every ranking.
            row.append("—")
            lines.append(markdown_row(row))

    # Add one compact all-features block at the bottom of the same table.
    for model_index, model in enumerate(MODEL_ORDER):
        cell = ds_all[ds_all["model_label"] == model]

        row = [
            "All features" if model_index == 0 else "",
            model,
        ] + ["—"] * len(ks)

        if cell.empty:
            row.append("—")
        else:
            r = cell.iloc[0]
            row.append(
                format_value(r["mean"], r["std"], decimals, show_std)
            )

        lines.append(markdown_row(row))

    return "\n".join(lines)


def full_feature_counts(df: pd.DataFrame) -> dict[str, int]:
    """Extract full feature count from all-features rows."""
    counts = (
        df[df["selection_method"] == "all_features"]
        .groupby("dataset")["n_selected_features"]
        .unique()
    )

    result: dict[str, int] = {}
    for dataset, vals in counts.items():
        vals = [int(v) for v in vals if pd.notna(v)]
        if len(set(vals)) != 1:
            raise ValueError(
                f"Expected one all-feature count for {dataset}; found {vals}"
            )
        result[dataset] = vals[0]

    return result


def generate_metric_markdown(
    df: pd.DataFrame,
    metric: str,
    decimals: int,
    show_std: bool,
) -> str:
    fold_df = fold_level_summary(df, metric)
    selected_summary = dataset_level_summary(fold_df, metric)
    all_summary = all_features_summary(df, metric)
    feature_counts = full_feature_counts(df)

    metric_label = METRIC_LABELS[metric]

    lines = [
        f"# Dataset-level benchmark tables — {metric_label}",
        "",
        (
            f"Values report mean **{metric_label}** across the five "
            "stratified cross-validation folds."
        ),
        (
            "For the **Random** baseline, the 20 random repetitions are first "
            "averaged within each fold and matched experimental condition; "
            "the resulting fold-level means are then averaged across folds."
        ),
        (
            "**LR** = logistic regression; **DT3–DT6** = decision trees with "
            "maximum depths 3–6. DT1, DT2, and the unrestricted decision tree "
            "are intentionally omitted from these manuscript-facing tables."
        ),
    ]

    if show_std:
        lines.append(
            "Cells are reported as **mean ± sample standard deviation across folds**."
        )
    else:
        lines.append(
            "Cells report fold means only; variability remains available from "
            "the fold-level benchmark results."
        )

    lines.extend(
        [
            "",
            (
                "Subset columns use the **actual number of selected features "
                "($k$)** rather than requested percentages."
            ),
            "",
        ]
    )

    dataset_order = [
        d for d in DATASET_LABELS
        if d in set(df["dataset"].unique())
    ]
    extra = [
        d for d in df["dataset"].unique()
        if d not in DATASET_LABELS
    ]
    dataset_order.extend(sorted(extra))

    for dataset in dataset_order:
        title = DATASET_LABELS.get(dataset, dataset)
        full_k = feature_counts.get(dataset)

        lines.extend(
            [
                f"## {title}",
                "",
                (
                    f"All-features reference: **k={full_k}**."
                    if full_k is not None
                    else "All-features reference."
                ),
                "",
                make_dataset_table(
                    dataset,
                    selected_summary,
                    all_summary,
                    decimals,
                    show_std,
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    validate_input(df, args.metrics)

    df = df.copy()
    df["model_label"] = df.apply(model_label, axis=1)
    df = df[df["model_label"].notna()].copy()

    validate_random_repetitions(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        markdown = generate_metric_markdown(
            df=df,
            metric=metric,
            decimals=args.decimals,
            show_std=args.show_std,
        )

        output_path = (
            args.output_dir / f"dataset_performance_{metric}.md"
        )
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()