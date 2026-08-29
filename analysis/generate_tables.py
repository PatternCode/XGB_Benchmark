#!/usr/bin/env python3
"""
Generate manuscript-facing Markdown tables for the XGB Benchmark.

Outputs
-------
1. dataset_performance_<metric>.md
   Dataset-level tables for Gain, Weight, Cover, SHAP, Random, and the
   all-features reference. Random repetitions are averaged within each fold
   before fold-level means and sample standard deviations are calculated.

2. cross_dataset_shap_comparison_<metric>.md
   Cross-dataset comparison of Gain, Weight, and Cover against SHAP using
   matched fold-wise differences.

   A matched fold-level comparison uses exactly the same:
       dataset × outer fold × selected-feature count × downstream model.

   For native method m:
       delta = score(m) - score(SHAP)

   For each dataset × downstream model × selected-feature count, the five
   fold-wise deltas are averaged before assigning:
       Win  : mean delta > +equivalence_margin
       Tie  : -equivalence_margin <= mean delta <= +equivalence_margin
       Loss : mean delta < -equivalence_margin

   Win/tie/loss rates are calculated within each dataset first and then
   averaged across datasets so that every dataset receives equal weight.
   Per-dataset summaries report mean paired differences, while a detailed\n   condition-level section reports mean ± sample SD across the five folds.\n
3. compact_decision_tree_summary_<metric>.md
   Descriptive compact decision-tree examples. Candidate trees use DT3-DT6
   with Gain/Weight/Cover/SHAP subsets. A candidate is eligible when:
       - its mean score is no more than equivalence_margin below the
         same-depth all-features tree; and
       - its mean score is at least as high as the matched random baseline.

   Among eligible candidates, preference is given to:
       smaller k -> lower maximum depth -> fewer mean nodes ->
       higher predictive score -> method name.

   These are descriptive examples identified from the same cross-validation
   results, not independently validated optimal models.

Random baseline treatment
-------------------------
The 20 random repetitions are always averaged within the same
dataset/fold/k/model condition before any cross-fold summary is calculated.

Default manuscript usage
------------------------
python analysis/generate_tables.py \
    --metrics f1_macro \
    --primary-metric f1_macro \
    --show-std \
    --equivalence-margin 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT = Path("results/summaries/combined_results_9_datasets.csv")
DEFAULT_OUTPUT_DIR = Path("results/summaries")

RANKING_METHODS = ["gain", "weight", "cover", "shap"]
NATIVE_METHODS = ["gain", "weight", "cover"]
DISPLAY_METHODS = ["gain", "weight", "cover", "shap", "random"]

RANKING_LABELS = {
    "gain": "Gain",
    "weight": "Weight",
    "cover": "Cover",
    "shap": "SHAP",
    "random": "Random",
    "all_features": "All",
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

MODEL_ORDER = ["LR", "XGBoost", "DT3", "DT4", "DT5", "DT6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-facing XGB Benchmark tables."
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
        help="Dataset-level table metrics. Default: f1_macro",
    )
    parser.add_argument(
        "--primary-metric",
        default="f1_macro",
        choices=SUPPORTED_METRICS,
        help="Metric for SHAP comparison and compact-tree summary.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places shown in output tables. Default: 3",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Show mean ± sample SD across folds in dataset-level tables.",
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.01,
        help=(
            "Absolute metric margin used for practical equivalence. "
            "Default: 0.01"
        ),
    )
    args = parser.parse_args()

    if args.decimals < 0:
        parser.error("--decimals must be non-negative.")

    if args.equivalence_margin < 0:
        parser.error("--equivalence-margin must be non-negative.")

    return args


def model_label(row: pd.Series) -> str | None:
    if row["model"] == "logistic_regression":
        return "LR"

    if row["model"] == "xgboost":
        return "XGBoost"

    if row["model"] == "decision_tree":
        depth = row["max_depth"]
        if pd.notna(depth) and int(depth) in {3, 4, 5, 6}:
            return f"DT{int(depth)}"

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
        "actual_tree_depth",
        "n_tree_nodes",
        "n_tree_leaves",
        "n_tree_features_used",
    } | set(metrics)

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}"
        )


def validate_random_repetitions(df: pd.DataFrame) -> None:
    random_df = df[df["selection_method"] == "random"].copy()
    if random_df.empty:
        raise ValueError("No random-baseline rows were found.")

    counts = (
        random_df.groupby(
            [
                "dataset",
                "outer_fold",
                "n_selected_features",
                "model_label",
            ],
            dropna=False,
        )["random_repetition"]
        .nunique()
    )

    bad = counts[counts != 20]
    if not bad.empty:
        raise ValueError(
            "Expected exactly 20 random repetitions for every included "
            "dataset/fold/k/model condition. Example mismatches:\n"
            f"{bad.head(10)}"
        )


def validate_five_folds(df: pd.DataFrame) -> None:
    counts = df.groupby("dataset")["outer_fold"].nunique()
    bad = counts[counts != 5]

    if not bad.empty:
        raise ValueError(
            "Expected exactly five outer folds for every dataset. "
            f"Found:\n{bad}"
        )


def dataset_order(df: pd.DataFrame) -> list[str]:
    present = set(df["dataset"].unique())
    ordered = [dataset for dataset in DATASET_LABELS if dataset in present]
    ordered.extend(sorted(present - set(DATASET_LABELS)))
    return ordered


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def full_feature_counts(df: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    all_df = df[df["selection_method"] == "all_features"]

    for dataset, group in all_df.groupby("dataset"):
        values = sorted(
            {
                int(value)
                for value in group["n_selected_features"].dropna().unique()
            }
        )

        if len(values) != 1:
            raise ValueError(
                f"Expected one all-feature count for {dataset}; "
                f"found {values}"
            )

        result[dataset] = values[0]

    return result


def format_value(
    mean: float,
    std: float | None,
    decimals: int,
    show_std: bool,
) -> str:
    if pd.isna(mean):
        return "—"

    if show_std and std is not None and pd.notna(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    return f"{mean:.{decimals}f}"


def format_signed(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.{decimals}f}"


# ---------------------------------------------------------------------------
# Dataset-level tables
# ---------------------------------------------------------------------------

def fold_level_selected_and_random(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    keys = [
        "dataset",
        "outer_fold",
        "selection_method",
        "n_selected_features",
        "model_label",
    ]

    ranked = df[df["selection_method"].isin(RANKING_METHODS)].copy()
    ranked_fold = (
        ranked.groupby(keys, as_index=False, dropna=False)[metric]
        .mean()
    )

    # Average the 20 random repetitions inside each fold/condition first.
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
    keys = [
        "dataset",
        "selection_method",
        "n_selected_features",
        "model_label",
    ]

    return (
        fold_df.groupby(keys, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def all_features_summary(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    all_df = df[df["selection_method"] == "all_features"].copy()

    return (
        all_df.groupby(
            ["dataset", "model_label"],
            dropna=False,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def dataset_markdown(
    df: pd.DataFrame,
    metric: str,
    decimals: int,
    show_std: bool,
) -> str:
    fold_df = fold_level_selected_and_random(df, metric)
    summary = dataset_level_summary(fold_df, metric)
    all_summary = all_features_summary(df, metric)
    full_counts = full_feature_counts(df)
    label = METRIC_LABELS[metric]

    lines = [
        f"# Dataset-level benchmark tables — {label}",
        "",
        f"Values report mean **{label}** across five stratified CV folds.",
        (
            "For **Random**, the 20 repetitions are first averaged within "
            "each fold/k/model condition."
        ),
        (
            "**LR** = logistic regression; **DT3–DT6** = decision trees "
            "with maximum depths 3–6."
        ),
        (
            "Subset size is shown using the actual number of selected "
            "features **k**."
        ),
    ]

    if show_std:
        lines.append(
            "Cells are reported as **mean ± sample SD across the five folds**."
        )

    lines.append("")

    for dataset in dataset_order(df):
        dataset_summary = summary[
            summary["dataset"] == dataset
        ].copy()

        dataset_all = all_summary[
            all_summary["dataset"] == dataset
        ].copy()

        feature_counts = sorted(
            int(value)
            for value in dataset_summary[
                "n_selected_features"
            ].dropna().unique()
        )

        lines += [
            f"## {DATASET_LABELS.get(dataset, dataset)}",
            "",
            f"All-features reference: **k={full_counts[dataset]}**.",
            "",
            md_row(["Ranking", "k"] + MODEL_ORDER),
            md_row(
                ["---", "---:"] + ["---:"] * len(MODEL_ORDER)
            ),
        ]

        for method in DISPLAY_METHODS:
            method_df = dataset_summary[
                dataset_summary["selection_method"] == method
            ]

            for row_index, k in enumerate(feature_counts):
                row = [
                    RANKING_LABELS[method] if row_index == 0 else "",
                    str(k),
                ]

                for model in MODEL_ORDER:
                    cell = method_df[
                        (method_df["n_selected_features"] == k)
                        & (method_df["model_label"] == model)
                    ]

                    if cell.empty:
                        row.append("—")
                        continue

                    result = cell.iloc[0]
                    row.append(
                        format_value(
                            result["mean"],
                            result["std"],
                            decimals,
                            show_std,
                        )
                    )

                lines.append(md_row(row))

        all_row = ["All features", str(full_counts[dataset])]

        for model in MODEL_ORDER:
            cell = dataset_all[
                dataset_all["model_label"] == model
            ]

            if cell.empty:
                all_row.append("—")
                continue

            result = cell.iloc[0]
            all_row.append(
                format_value(
                    result["mean"],
                    result["std"],
                    decimals,
                    show_std,
                )
            )

        lines += [md_row(all_row), ""]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Paired native-method comparisons against SHAP
# ---------------------------------------------------------------------------

def paired_fold_differences(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Return fold-wise native-minus-SHAP differences.

    Pairing keys:
        dataset × outer_fold × k × downstream model.
    """
    ranked_fold = fold_level_selected_and_random(df, metric)
    ranked_fold = ranked_fold[
        ranked_fold["selection_method"].isin(RANKING_METHODS)
    ].copy()

    condition_keys = [
        "dataset",
        "outer_fold",
        "n_selected_features",
        "model_label",
    ]

    pivot = ranked_fold.pivot(
        index=condition_keys,
        columns="selection_method",
        values=metric,
    )

    missing_columns = [
        method
        for method in RANKING_METHODS
        if method not in pivot.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing ranking methods from matched comparison table: "
            f"{missing_columns}"
        )

    incomplete = pivot[RANKING_METHODS].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            "Some matched fold-level conditions do not contain all four "
            "ranking methods. Example:\n"
            f"{pivot.loc[incomplete, RANKING_METHODS].head(10)}"
        )

    base = pivot.reset_index()
    rows: list[pd.DataFrame] = []

    for method in NATIVE_METHODS:
        method_df = base[condition_keys].copy()
        method_df["selection_method"] = method
        method_df["difference"] = (
            base[method] - base["shap"]
        )
        rows.append(method_df)

    return pd.concat(rows, ignore_index=True)


def paired_condition_statistics(
    paired_df: pd.DataFrame,
    equivalence_margin: float,
) -> pd.DataFrame:
    """
    Summarize the five paired fold differences for each dataset/model/k.

    Win/tie/loss classification is made only after averaging the five
    fold-level paired differences.
    """
    keys = [
        "dataset",
        "model_label",
        "n_selected_features",
        "selection_method",
    ]

    stats = (
        paired_df.groupby(keys)["difference"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_difference",
                "std": "sd_difference",
                "count": "fold_count",
            }
        )
    )

    bad = stats[stats["fold_count"] != 5]
    if not bad.empty:
        raise ValueError(
            "Expected exactly five paired fold differences for every "
            "dataset/model/k/method condition. Example:\n"
            f"{bad.head(10)}"
        )

    def classify(delta: float) -> str:
        if delta > equivalence_margin:
            return "win"
        if delta < -equivalence_margin:
            return "loss"
        return "tie"

    stats["outcome"] = stats["mean_difference"].map(classify)

    return stats


def paired_dataset_model_statistics(
    paired_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize fold-wise paired differences within each dataset/model.

    Every k has the same five folds, so averaging the fold-wise differences
    gives equal weight to each evaluated k within that dataset/model.
    """
    return (
        paired_df.groupby(
            ["dataset", "model_label", "selection_method"]
        )["difference"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_difference",
                "std": "sd_difference",
                "count": "paired_count",
            }
        )
    )


def win_tie_loss_statistics(
    condition_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate win/tie/loss rates with equal dataset weighting.

    Returns
    -------
    per_dataset_model
        One row per dataset × model × native method.
    by_model
        Dataset-equal-weight rates for each model × native method.
    """
    working = condition_stats.copy()

    working["win"] = (working["outcome"] == "win").astype(float)
    working["tie"] = (working["outcome"] == "tie").astype(float)
    working["loss"] = (working["outcome"] == "loss").astype(float)

    per_dataset_model = (
        working.groupby(
            ["dataset", "model_label", "selection_method"],
            as_index=False,
        )
        .agg(
            win_rate=("win", "mean"),
            tie_rate=("tie", "mean"),
            loss_rate=("loss", "mean"),
            n_k=("outcome", "size"),
        )
    )

    # Every dataset contributes equally to a model-specific summary,
    # regardless of how many unique k values that dataset contains.
    by_model = (
        per_dataset_model.groupby(
            ["model_label", "selection_method"],
            as_index=False,
        )
        .agg(
            win_rate=("win_rate", "mean"),
            tie_rate=("tie_rate", "mean"),
            loss_rate=("loss_rate", "mean"),
            datasets=("dataset", "nunique"),
        )
    )

    return per_dataset_model, by_model

def equal_weight_paired_differences(
    paired_dataset_model: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate dataset-equal-weight paired differences by downstream model.

    Dataset-level mean paired differences are averaged so that every
    dataset contributes equally within each model × native-method summary.
    """
    return (
        paired_dataset_model.groupby(
            ["model_label", "selection_method"],
            as_index=False,
        )
        .agg(
            mean_difference=("mean_difference", "mean"),
            dataset_sd=("mean_difference", "std"),
            datasets=("dataset", "nunique"),
        )
    )

def cross_dataset_shap_comparison_markdown(
    df: pd.DataFrame,
    metric: str,
    decimals: int,
    equivalence_margin: float,
) -> str:
    paired = paired_fold_differences(df, metric)
    condition_stats = paired_condition_statistics(
        paired,
        equivalence_margin,
    )
    paired_dataset_model = paired_dataset_model_statistics(paired)

    _, wtl_by_model = win_tie_loss_statistics(
        condition_stats
    )
    delta_by_model = equal_weight_paired_differences(
        paired_dataset_model
    )

    label = METRIC_LABELS[metric]

    by_model = wtl_by_model.merge(
        delta_by_model[
            [
                "model_label",
                "selection_method",
                "mean_difference",
                "dataset_sd",
            ]
        ],
        on=["model_label", "selection_method"],
        how="left",
        validate="one_to_one",
    )

    lines = [
        f"# Cross-dataset native-method comparison against SHAP — {label}",
        "",
        (
            "All paired differences are defined as **native method − SHAP** "
            "for exactly matched dataset/fold/k/downstream-model conditions."
        ),
        (
            "For each dataset/model/k condition, the five fold-wise paired "
            "differences are averaged before assigning a practical outcome."
        ),
        (
            f"With equivalence margin **δ={equivalence_margin:.3f}**, a native "
            "method is a **Win** when mean Δ > +δ, a **Tie** when "
            "−δ ≤ mean Δ ≤ +δ, and a **Loss** when mean Δ < −δ."
        ),
        (
            "Win/tie/loss rates are calculated within each dataset first and "
            "then averaged across datasets, preventing datasets with more "
            "distinct k values from receiving greater weight."
        ),
        (
            "The reported mean Δ values are descriptive paired differences, "
            "not statistical significance tests."
        ),
        "",
        "## Summary by downstream model",
        "",
        md_row(
            [
                "Model",
                "Method",
                f"Mean Δ vs SHAP ({label})",
                "Wins (%)",
                "Ties (%)",
                "Losses (%)",
            ]
        ),
        md_row(["---", "---", "---:", "---:", "---:", "---:"]),
    ]

    for model in MODEL_ORDER:
        model_df = by_model[by_model["model_label"] == model]

        for method in NATIVE_METHODS:
            row = model_df[
                model_df["selection_method"] == method
            ].iloc[0]

            lines.append(
                md_row(
                    [
                        model,
                        RANKING_LABELS[method],
                        format_signed(
                            row["mean_difference"],
                            decimals,
                        ),
                        f"{100 * row['win_rate']:.1f}",
                        f"{100 * row['tie_rate']:.1f}",
                        f"{100 * row['loss_rate']:.1f}",
                    ]
                )
            )

    lines += [
        "",
        "## Per-dataset mean paired differences",
        "",
        (
            "Values below report the mean native-minus-SHAP difference "
            "across all matched fold and subset-size comparisons for the "
            "indicated dataset and model. Positive values favour the native "
            "method."
        ),
        "",
    ]

    for model in MODEL_ORDER:
        model_df = paired_dataset_model[
            paired_dataset_model["model_label"] == model
        ]

        pivot = model_df.pivot(
            index="dataset",
            columns="selection_method",
            values="mean_difference",
        )

        lines += [
            f"### {model}",
            "",
            md_row(
                ["Dataset"]
                + [
                    f"{RANKING_LABELS[method]} − SHAP"
                    for method in NATIVE_METHODS
                ]
            ),
            md_row(
                ["---"] + ["---:"] * len(NATIVE_METHODS)
            ),
        ]

        for dataset in dataset_order(df):
            row = [DATASET_LABELS.get(dataset, dataset)]

            for method in NATIVE_METHODS:
                if (
                    dataset not in pivot.index
                    or method not in pivot.columns
                    or pd.isna(pivot.loc[dataset, method])
                ):
                    row.append("—")
                else:
                    row.append(
                        format_signed(
                            pivot.loc[dataset, method],
                            decimals,
                        )
                    )

            lines.append(md_row(row))

        lines.append("")

    lines += [
        "## Detailed condition-level paired differences",
        "",
        (
            "For reproducibility and fold-wise variability reporting, each "
            "entry below corresponds to one fixed dataset/model/k condition. "
            "Values are the mean ± sample SD of the five matched fold-wise "
            "differences, where Δ = native method − SHAP."
        ),
        "",
    ]

    for model in MODEL_ORDER:
        model_stats = condition_stats[
            condition_stats["model_label"] == model
        ].copy()

        if model_stats.empty:
            continue

        lines += [
            f"### {model}",
            "",
            md_row(
                [
                    "Dataset",
                    "k",
                    "Method",
                    "Mean Δ",
                    "Fold SD",
                    "Outcome",
                ]
            ),
            md_row(
                ["---", "---:", "---", "---:", "---:", "---"]
            ),
        ]

        for dataset in dataset_order(df):
            dataset_stats = model_stats[
                model_stats["dataset"] == dataset
            ].copy()

            if dataset_stats.empty:
                continue

            dataset_stats = dataset_stats.sort_values(
                ["n_selected_features", "selection_method"]
            )

            for _, row in dataset_stats.iterrows():
                lines.append(
                    md_row(
                        [
                            DATASET_LABELS.get(dataset, dataset),
                            str(int(row["n_selected_features"])),
                            RANKING_LABELS[row["selection_method"]],
                            format_signed(
                                row["mean_difference"],
                                decimals,
                            ),
                            (
                                f"{row['sd_difference']:.{decimals}f}"
                                if pd.notna(row["sd_difference"])
                                else "—"
                            ),
                            str(row["outcome"]).capitalize(),
                        ]
                    )
                )

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Compact decision-tree summary
# ---------------------------------------------------------------------------

def compact_tree_statistics(
    df: pd.DataFrame,
    metric: str,
    equivalence_margin: float,
) -> pd.DataFrame:
    """
    Identify one descriptive compact DT example per dataset.

    Eligibility:
        selected_mean >= same-depth all-feature mean - equivalence_margin

        AND

        selected_mean >= matched random-subset mean

    Selection among eligible candidates:
        1. smaller k
        2. smaller configured maximum depth
        3. fewer mean tree nodes
        4. higher predictive metric
        5. method name

    No fallback is used if a dataset has no qualifying reduced-feature
    configuration; that absence is reported explicitly.
    """
    dt = df[
        (df["model"] == "decision_tree")
        & (df["max_depth"].isin([3, 4, 5, 6]))
    ].copy()

    candidates = dt[
        dt["selection_method"].isin(RANKING_METHODS)
    ].copy()

    candidate_stats = (
        candidates.groupby(
            [
                "dataset",
                "selection_method",
                "n_selected_features",
                "max_depth",
            ]
        )
        .agg(
            selected_mean=(metric, "mean"),
            selected_sd=(metric, "std"),
            selected_n=(metric, "count"),
            selected_actual_depth=("actual_tree_depth", "mean"),
            selected_nodes=("n_tree_nodes", "mean"),
            selected_leaves=("n_tree_leaves", "mean"),
            selected_features_used=("n_tree_features_used", "mean"),
        )
        .reset_index()
    )

    bad_candidates = candidate_stats[
        candidate_stats["selected_n"] != 5
    ]
    if not bad_candidates.empty:
        raise ValueError(
            "Expected five folds for every ranked DT candidate. Example:\n"
            f"{bad_candidates.head(10)}"
        )

    # Random repetitions -> fold means -> dataset-level mean/SD.
    random_fold = (
        dt[dt["selection_method"] == "random"]
        .groupby(
            [
                "dataset",
                "outer_fold",
                "n_selected_features",
                "max_depth",
            ],
            as_index=False,
        )[metric]
        .mean()
    )

    random_stats = (
        random_fold.groupby(
            ["dataset", "n_selected_features", "max_depth"]
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "random_mean",
                "std": "random_sd",
                "count": "random_n",
            }
        )
    )

    bad_random = random_stats[random_stats["random_n"] != 5]
    if not bad_random.empty:
        raise ValueError(
            "Expected five fold-level random means for every DT condition. "
            f"Example:\n{bad_random.head(10)}"
        )

    all_stats = (
        dt[dt["selection_method"] == "all_features"]
        .groupby(["dataset", "max_depth"])
        .agg(
            all_mean=(metric, "mean"),
            all_sd=(metric, "std"),
            all_n=(metric, "count"),
            all_actual_depth=("actual_tree_depth", "mean"),
            all_nodes=("n_tree_nodes", "mean"),
            all_leaves=("n_tree_leaves", "mean"),
            all_features_used=("n_tree_features_used", "mean"),
            full_feature_count=("n_selected_features", "first"),
        )
        .reset_index()
    )

    bad_all = all_stats[all_stats["all_n"] != 5]
    if not bad_all.empty:
        raise ValueError(
            "Expected five folds for every all-features DT reference. "
            f"Example:\n{bad_all.head(10)}"
        )

    merged = (
        candidate_stats
        .merge(
            random_stats,
            on=[
                "dataset",
                "n_selected_features",
                "max_depth",
            ],
            how="left",
            validate="many_to_one",
        )
        .merge(
            all_stats,
            on=["dataset", "max_depth"],
            how="left",
            validate="many_to_one",
        )
    )

    if merged[
        [
            "random_mean",
            "all_mean",
            "full_feature_count",
        ]
    ].isna().any().any():
        raise ValueError(
            "Could not match every DT candidate to random and "
            "same-depth all-features references."
        )

    merged["eligible"] = (
        (
            merged["selected_mean"]
            >= merged["all_mean"] - equivalence_margin
        )
        & (
            merged["selected_mean"]
            >= merged["random_mean"]
        )
    )

    full_counts = full_feature_counts(df)
    rows: list[dict[str, object]] = []

    for dataset in dataset_order(df):
        dataset_candidates = merged[
            merged["dataset"] == dataset
        ].copy()

        eligible = dataset_candidates[
            dataset_candidates["eligible"]
        ].copy()

        if eligible.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "selection_method": None,
                    "k": None,
                    "d": full_counts[dataset],
                    "max_depth": None,
                    "actual_depth": None,
                    "features_used": None,
                    "nodes": None,
                    "leaves": None,
                    "selected_mean": None,
                    "selected_sd": None,
                    "random_mean": None,
                    "random_sd": None,
                    "all_mean": None,
                    "all_sd": None,
                    "status": "no_qualifying_configuration",
                }
            )
            continue

        chosen = (
            eligible.sort_values(
                [
                    "n_selected_features",
                    "max_depth",
                    "selected_nodes",
                    "selected_mean",
                    "selection_method",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        rows.append(
            {
                "dataset": dataset,
                "selection_method": chosen["selection_method"],
                "k": int(chosen["n_selected_features"]),
                "d": int(chosen["full_feature_count"]),
                "max_depth": int(chosen["max_depth"]),
                "actual_depth": chosen["selected_actual_depth"],
                "features_used": chosen["selected_features_used"],
                "nodes": chosen["selected_nodes"],
                "leaves": chosen["selected_leaves"],
                "selected_mean": chosen["selected_mean"],
                "selected_sd": chosen["selected_sd"],
                "random_mean": chosen["random_mean"],
                "random_sd": chosen["random_sd"],
                "all_mean": chosen["all_mean"],
                "all_sd": chosen["all_sd"],
                "status": "qualifying_configuration",
            }
        )

    return pd.DataFrame(rows)


def compact_tree_markdown(
    df: pd.DataFrame,
    metric: str,
    decimals: int,
    equivalence_margin: float,
) -> str:
    summary = compact_tree_statistics(
        df,
        metric,
        equivalence_margin,
    )
    label = METRIC_LABELS[metric]

    lines = [
        f"# Illustrative compact decision-tree configurations — {label}",
        "",
        (
            "Candidate trees use DT3–DT6 with subsets ranked by Gain, Weight, "
            "Cover, or SHAP."
        ),
        (
            f"A candidate qualifies when its mean {label} is no more than "
            f"{equivalence_margin:.3f} below the same-depth all-features tree "
            "and is at least as high as the matched random-feature baseline."
        ),
        (
            "Among qualifying candidates, the descriptive rule prioritizes "
            "fewer selected features, lower maximum depth, fewer mean nodes, "
            "higher predictive performance, and finally method name."
        ),
        (
            "These configurations are descriptive examples identified from "
            "the same cross-validation results; they are not independently "
            "validated optimal models."
        ),
        (
            "Predictive values are reported as mean ± sample SD across the "
            "five folds. Random SD is calculated after averaging the 20 "
            "repetitions within each fold."
        ),
        "",
        md_row(
            [
                "Dataset",
                "Ranking",
                "k/d",
                "DT max",
                "Actual depth",
                "Features used",
                "Nodes",
                label,
                f"Random {label}",
                f"All-feature {label}",
            ]
        ),
        md_row(
            [
                "---",
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        ),
    ]

    no_candidate: list[str] = []

    for _, result in summary.iterrows():
        dataset = result["dataset"]
        dataset_label = DATASET_LABELS.get(dataset, dataset)

        if result["status"] != "qualifying_configuration":
            no_candidate.append(dataset_label)
            lines.append(
                md_row(
                    [
                        dataset_label,
                        "—",
                        f"—/{int(result['d'])}",
                        "—",
                        "—",
                        "—",
                        "—",
                        "—",
                        "—",
                        "—",
                    ]
                )
            )
            continue

        lines.append(
            md_row(
                [
                    dataset_label,
                    RANKING_LABELS[result["selection_method"]],
                    f"{int(result['k'])}/{int(result['d'])}",
                    str(int(result["max_depth"])),
                    f"{result['actual_depth']:.1f}",
                    f"{result['features_used']:.1f}",
                    f"{result['nodes']:.1f}",
                    format_value(
                        result["selected_mean"],
                        result["selected_sd"],
                        decimals,
                        True,
                    ),
                    format_value(
                        result["random_mean"],
                        result["random_sd"],
                        decimals,
                        True,
                    ),
                    format_value(
                        result["all_mean"],
                        result["all_sd"],
                        decimals,
                        True,
                    ),
                ]
            )
        )

    if no_candidate:
        lines += [
            "",
            (
                "**No qualifying reduced-feature configuration under the "
                f"δ={equivalence_margin:.3f} criterion:** "
                + ", ".join(no_candidate)
                + "."
            ),
        ]

    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    metrics_needed = set(args.metrics) | {args.primary_metric}

    df = pd.read_csv(args.input)
    validate_input(df, metrics_needed)

    df = df.copy()
    df["model_label"] = df.apply(model_label, axis=1)
    df = df[df["model_label"].notna()].copy()

    validate_five_folds(df)
    validate_random_repetitions(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        path = args.output_dir / f"dataset_performance_{metric}.md"
        path.write_text(
            dataset_markdown(
                df,
                metric,
                args.decimals,
                args.show_std,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {path}")

    metric = args.primary_metric

    shap_path = (
        args.output_dir
        / f"cross_dataset_shap_comparison_{metric}.md"
    )
    shap_path.write_text(
        cross_dataset_shap_comparison_markdown(
            df,
            metric,
            args.decimals,
            args.equivalence_margin,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {shap_path}")

    tree_path = (
        args.output_dir
        / f"compact_decision_tree_summary_{metric}.md"
    )
    tree_path.write_text(
        compact_tree_markdown(
            df,
            metric,
            args.decimals,
            args.equivalence_margin,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {tree_path}")


if __name__ == "__main__":
    main()