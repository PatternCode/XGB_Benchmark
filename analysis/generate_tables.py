#!/usr/bin/env python3
"""
Generate manuscript-facing Markdown tables for the XGB Benchmark.

Outputs
-------
1. dataset_performance_<metric>.md
   One compact table per dataset. Rows are ranking method / selected-feature
   count combinations and columns are downstream models.

2. cross_dataset_best_subset_<metric>.md
   Two-panel cross-dataset table for Logistic Regression and XGBoost.
   For Gain, Weight, Cover, SHAP, and Random, each cell reports:
       mean metric (k=<selected features>)
   using a compact one-standard-error rule:
       - find the subset size with the highest mean score;
       - compute the standard error of that best subset across CV folds;
       - among subset sizes whose mean score is at least
         best_mean - best_SE, choose the smallest k.
   The All column reports the mean all-feature result as:
       mean metric (k=<all features>)

   Random is handled correctly:
       - average the 20 random repetitions within each fold/k/model first;
       - then summarize those five fold-level means across folds.

3. compact_decision_tree_summary_<metric>.md
   One compact competitive decision-tree configuration per dataset.
   Candidate trees use DT3-DT6 with Gain/Weight/Cover/SHAP subsets.
   A candidate is eligible when:
       - its mean score is within one standard error of the all-feature tree
         at the same maximum depth; and
       - it is not worse than the matched averaged random baseline.
   Among eligible candidates, select:
       smallest k -> lower max depth -> fewer mean nodes -> higher score.

The one-standard-error rules are descriptive model-selection criteria, not
statistical significance tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/summaries/combined_results_9_datasets.csv")
DEFAULT_OUTPUT_DIR = Path("results/summaries")

RANKING_METHODS = ["gain", "weight", "cover", "shap"]
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
CROSS_DATASET_MODELS = ["LR", "XGBoost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset-level and cross-dataset benchmark tables."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1_macro"],
        choices=SUPPORTED_METRICS,
    )
    parser.add_argument(
        "--primary-metric",
        default="f1_macro",
        choices=SUPPORTED_METRICS,
    )
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--show-std", action="store_true")
    return parser.parse_args()


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
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def validate_random_repetitions(df: pd.DataFrame) -> None:
    random_df = df[df["selection_method"] == "random"].copy()
    counts = (
        random_df.groupby(
            ["dataset", "outer_fold", "n_selected_features", "model_label"],
            dropna=False,
        )["random_repetition"]
        .nunique()
    )
    bad = counts[counts != 20]
    if not bad.empty:
        raise ValueError(
            "Expected exactly 20 random repetitions for every included "
            f"condition. Example mismatches:\n{bad.head(10)}"
        )


def dataset_order(df: pd.DataFrame) -> list[str]:
    present = set(df["dataset"].unique())
    ordered = [d for d in DATASET_LABELS if d in present]
    ordered.extend(sorted(present - set(DATASET_LABELS)))
    return ordered


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def full_feature_counts(df: pd.DataFrame) -> dict[str, int]:
    result = {}
    all_df = df[df["selection_method"] == "all_features"]
    for dataset, group in all_df.groupby("dataset"):
        vals = sorted({int(v) for v in group["n_selected_features"].dropna().unique()})
        if len(vals) != 1:
            raise ValueError(f"Expected one all-feature count for {dataset}; found {vals}")
        result[dataset] = vals[0]
    return result


def fold_level_selected_and_random(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = [
        "dataset",
        "outer_fold",
        "selection_method",
        "n_selected_features",
        "model_label",
    ]

    ranked = df[df["selection_method"].isin(RANKING_METHODS)].copy()
    ranked_fold = ranked.groupby(keys, as_index=False, dropna=False)[metric].mean()

    random_df = df[df["selection_method"] == "random"].copy()
    random_fold = random_df.groupby(keys, as_index=False, dropna=False)[metric].mean()

    return pd.concat([ranked_fold, random_fold], ignore_index=True)


def dataset_level_summary(fold_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["dataset", "selection_method", "n_selected_features", "model_label"]
    return (
        fold_df.groupby(keys, as_index=False, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def all_features_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    all_df = df[df["selection_method"] == "all_features"].copy()
    return (
        all_df.groupby(["dataset", "model_label"], as_index=False, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def format_value(mean, std, decimals, show_std):
    if pd.isna(mean):
        return "—"
    if show_std and pd.notna(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def dataset_markdown(df, metric, decimals, show_std):
    fold_df = fold_level_selected_and_random(df, metric)
    summary = dataset_level_summary(fold_df, metric)
    all_summary = all_features_summary(df, metric)
    full_counts = full_feature_counts(df)
    label = METRIC_LABELS[metric]

    lines = [
        f"# Dataset-level benchmark tables — {label}",
        "",
        f"Values report mean **{label}** across five stratified CV folds.",
        "For **Random**, the 20 repetitions are first averaged within each fold/k/model condition.",
        "**LR** = logistic regression; **DT3–DT6** = decision trees with maximum depths 3–6.",
        "Subset size is shown using the actual number of selected features **k**.",
        "",
    ]

    for dataset in dataset_order(df):
        ds = summary[summary["dataset"] == dataset].copy()
        ds_all = all_summary[all_summary["dataset"] == dataset].copy()
        ks = sorted(int(k) for k in ds["n_selected_features"].unique())

        lines += [
            f"## {DATASET_LABELS.get(dataset, dataset)}",
            "",
            f"All-features reference: **k={full_counts[dataset]}**.",
            "",
            md_row(["Ranking", "k"] + MODEL_ORDER),
            md_row(["---", "---:"] + ["---:"] * len(MODEL_ORDER)),
        ]

        for method in DISPLAY_METHODS:
            method_df = ds[ds["selection_method"] == method]
            for row_i, k in enumerate(ks):
                row = [RANKING_LABELS[method] if row_i == 0 else "", str(k)]
                for model in MODEL_ORDER:
                    cell = method_df[
                        (method_df["n_selected_features"] == k)
                        & (method_df["model_label"] == model)
                    ]
                    if cell.empty:
                        row.append("—")
                    else:
                        r = cell.iloc[0]
                        row.append(format_value(r["mean"], r["std"], decimals, show_std))
                lines.append(md_row(row))

        all_row = ["All features", str(full_counts[dataset])]
        for model in MODEL_ORDER:
            cell = ds_all[ds_all["model_label"] == model]
            if cell.empty:
                all_row.append("—")
            else:
                r = cell.iloc[0]
                all_row.append(format_value(r["mean"], r["std"], decimals, show_std))
        lines += [md_row(all_row), ""]

    return "\n".join(lines).rstrip() + "\n"


def choose_compact_best_subset(subset_stats: pd.DataFrame) -> pd.Series:
    best_idx = subset_stats["mean"].idxmax()
    best_row = subset_stats.loc[best_idx]

    best_se = (
        best_row["std"] / np.sqrt(best_row["count"])
        if pd.notna(best_row["std"]) and best_row["count"] > 0
        else 0.0
    )
    threshold = best_row["mean"] - best_se
    eligible = subset_stats[subset_stats["mean"] >= threshold].copy()

    chosen = eligible.sort_values(
        ["n_selected_features", "mean"],
        ascending=[True, False],
    ).iloc[0].copy()

    chosen["best_mean"] = best_row["mean"]
    chosen["best_k"] = best_row["n_selected_features"]
    chosen["best_se"] = best_se
    return chosen


def cross_dataset_best_subset_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    fold_df = fold_level_selected_and_random(df, metric)
    fold_df = fold_df[fold_df["model_label"].isin(CROSS_DATASET_MODELS)].copy()

    subset_stats = (
        fold_df.groupby(
            ["dataset", "selection_method", "model_label", "n_selected_features"],
            as_index=False,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    rows = []
    for dataset in dataset_order(df):
        for model in CROSS_DATASET_MODELS:
            for method in DISPLAY_METHODS:
                group = subset_stats[
                    (subset_stats["dataset"] == dataset)
                    & (subset_stats["model_label"] == model)
                    & (subset_stats["selection_method"] == method)
                ].copy()
                if group.empty:
                    continue

                chosen = choose_compact_best_subset(group)
                rows.append(
                    {
                        "dataset": dataset,
                        "model_label": model,
                        "selection_method": method,
                        "selected_mean": chosen["mean"],
                        "selected_k": int(chosen["n_selected_features"]),
                    }
                )

    full_counts = full_feature_counts(df)
    all_df = (
        df[
            (df["selection_method"] == "all_features")
            & (df["model_label"].isin(CROSS_DATASET_MODELS))
        ]
        .groupby(["dataset", "model_label"], as_index=False)[metric]
        .mean()
    )

    for _, r in all_df.iterrows():
        rows.append(
            {
                "dataset": r["dataset"],
                "model_label": r["model_label"],
                "selection_method": "all_features",
                "selected_mean": r[metric],
                "selected_k": full_counts[r["dataset"]],
            }
        )

    return pd.DataFrame(rows)


def cross_dataset_best_subset_markdown(df, metric, decimals):
    stats = cross_dataset_best_subset_stats(df, metric)
    label = METRIC_LABELS[metric]

    lines = [
        f"# Cross-dataset compact-subset comparison — {label}",
        "",
        f"Each cell reports **mean {label} (k)**.",
        (
            "For Gain, Weight, Cover, SHAP, and Random, k is selected using a descriptive "
            "one-standard-error rule: choose the smallest subset whose mean performance "
            "lies within one standard error of that method's best observed subset "
            "performance for the same dataset and downstream model."
        ),
        (
            "For Random, the 20 repetitions are averaged within each fold/k/model "
            "condition before the fold-level values are summarized."
        ),
        (
            "The **All** column reports the mean performance using the complete feature set."
        ),
        "",
    ]

    columns = RANKING_METHODS + ["random", "all_features"]

    for model in CROSS_DATASET_MODELS:
        lines += [
            f"## {model}",
            "",
            md_row(["Dataset"] + [RANKING_LABELS[c] for c in columns]),
            md_row(["---"] + ["---:"] * len(columns)),
        ]

        model_df = stats[stats["model_label"] == model]
        for dataset in dataset_order(df):
            row = [DATASET_LABELS.get(dataset, dataset)]
            for method in columns:
                cell = model_df[
                    (model_df["dataset"] == dataset)
                    & (model_df["selection_method"] == method)
                ]
                if cell.empty:
                    row.append("—")
                else:
                    r = cell.iloc[0]
                    row.append(f"{r['selected_mean']:.{decimals}f} (k={int(r['selected_k'])})")
            lines.append(md_row(row))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def compact_tree_statistics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    dt = df[
        (df["model"] == "decision_tree")
        & (df["max_depth"].isin([3, 4, 5, 6]))
    ].copy()

    candidates = dt[dt["selection_method"].isin(RANKING_METHODS)]

    candidate_stats = (
        candidates.groupby(
            ["dataset", "selection_method", "n_selected_features", "max_depth"],
            as_index=False,
        )
        .agg(
            selected_mean=(metric, "mean"),
            selected_actual_depth=("actual_tree_depth", "mean"),
            selected_nodes=("n_tree_nodes", "mean"),
            selected_leaves=("n_tree_leaves", "mean"),
            selected_features_used=("n_tree_features_used", "mean"),
        )
    )

    random_fold = (
        dt[dt["selection_method"] == "random"]
        .groupby(
            ["dataset", "outer_fold", "n_selected_features", "max_depth"],
            as_index=False,
        )[metric]
        .mean()
    )

    random_stats = (
        random_fold.groupby(
            ["dataset", "n_selected_features", "max_depth"],
            as_index=False,
        )[metric]
        .mean()
        .rename(columns={metric: "random_mean"})
    )

    all_stats = (
        dt[dt["selection_method"] == "all_features"]
        .groupby(["dataset", "max_depth"], as_index=False)
        .agg(
            all_mean=(metric, "mean"),
            all_sd=(metric, "std"),
            all_n=(metric, "count"),
            all_features_used=("n_tree_features_used", "mean"),
            all_nodes=("n_tree_nodes", "mean"),
            full_feature_count=("n_selected_features", "first"),
        )
    )
    all_stats["all_se"] = all_stats["all_sd"] / np.sqrt(all_stats["all_n"])

    merged = (
        candidate_stats
        .merge(
            random_stats,
            on=["dataset", "n_selected_features", "max_depth"],
            how="left",
        )
        .merge(all_stats, on=["dataset", "max_depth"], how="left")
    )

    merged["eligible"] = (
        (merged["selected_mean"] >= merged["all_mean"] - merged["all_se"])
        & (merged["selected_mean"] >= merged["random_mean"])
    )

    rows = []
    for dataset in dataset_order(df):
        ds = merged[merged["dataset"] == dataset].copy()
        eligible = ds[ds["eligible"]].copy()

        if not eligible.empty:
            chosen = eligible.sort_values(
                ["n_selected_features", "max_depth", "selected_nodes", "selected_mean"],
                ascending=[True, True, True, False],
            ).iloc[0]
        else:
            chosen = ds.sort_values(
                ["selected_mean", "n_selected_features", "max_depth", "selected_nodes"],
                ascending=[False, True, True, True],
            ).iloc[0]

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
                "selected_metric": chosen["selected_mean"],
                "random_metric": chosen["random_mean"],
                "all_metric": chosen["all_mean"],
            }
        )

    return pd.DataFrame(rows)


def compact_tree_markdown(df, metric, decimals):
    summary = compact_tree_statistics(df, metric)
    label = METRIC_LABELS[metric]

    lines = [
        f"# Compact decision-tree summary — {label}",
        "",
        (
            "Each row identifies one compact competitive decision-tree configuration. "
            "Candidates use DT3–DT6 with Gain, Weight, Cover, or SHAP subsets."
        ),
        "",
        md_row(
            [
                "Dataset", "Ranking", "k/d", "DT max", "Actual depth",
                "Features used", "Nodes", label, f"Random {label}", f"All-feature {label}"
            ]
        ),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]

    for _, r in summary.iterrows():
        lines.append(
            md_row(
                [
                    DATASET_LABELS.get(r["dataset"], r["dataset"]),
                    RANKING_LABELS[r["selection_method"]],
                    f"{int(r['k'])}/{int(r['d'])}",
                    str(int(r["max_depth"])),
                    f"{r['actual_depth']:.1f}",
                    f"{r['features_used']:.1f}",
                    f"{r['nodes']:.1f}",
                    f"{r['selected_metric']:.{decimals}f}",
                    f"{r['random_metric']:.{decimals}f}",
                    f"{r['all_metric']:.{decimals}f}",
                ]
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    metrics_needed = set(args.metrics) | {args.primary_metric}

    df = pd.read_csv(args.input)
    validate_input(df, metrics_needed)

    df = df.copy()
    df["model_label"] = df.apply(model_label, axis=1)
    df = df[df["model_label"].notna()].copy()

    validate_random_repetitions(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        path = args.output_dir / f"dataset_performance_{metric}.md"
        path.write_text(
            dataset_markdown(df, metric, args.decimals, args.show_std),
            encoding="utf-8",
        )
        print(f"Wrote {path}")

    metric = args.primary_metric

    cross_path = args.output_dir / f"cross_dataset_best_subset_{metric}.md"
    cross_path.write_text(
        cross_dataset_best_subset_markdown(df, metric, args.decimals),
        encoding="utf-8",
    )
    print(f"Wrote {cross_path}")

    tree_path = args.output_dir / f"compact_decision_tree_summary_{metric}.md"
    tree_path.write_text(
        compact_tree_markdown(df, metric, args.decimals),
        encoding="utf-8",
    )
    print(f"Wrote {tree_path}")


if __name__ == "__main__":
    main()