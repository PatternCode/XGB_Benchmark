"""
Inspect standardized raw benchmark datasets.

This script reads each dataset listed in src/data/registry.py and produces
a diagnostic report about the raw data quality and structure.

It does NOT modify the datasets.

Expected raw data structure:
    data/raw/<dataset_name>/data.csv
    data/raw/<dataset_name>/metadata.json

Generated reports:
    data/reports/inspection_report.json
    data/reports/inspection_summary.csv

Usage:
    python src/data/inspect_raw.py
    python src/data/inspect_raw.py --max-rows 50000
    python src/data/inspect_raw.py --full
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from registry import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_JSON = REPORT_DIR / "inspection_report.json"
REPORT_CSV = REPORT_DIR / "inspection_summary.csv"

MISSING_TOKENS = {"?", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "", " "}


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of the raw data file."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_feature_names_hash(columns: list[str]) -> str:
    """Compute SHA-256 hash of feature names to detect schema changes."""
    text = ",".join(columns)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset_preview(
    path: Path,
    dataset_key: str,
    max_rows: int | None,
) -> pd.DataFrame:
    """Load either the full dataset or the first max_rows rows."""
    if dataset_key == "higgs":
        columns = ["class"] + [f"feature_{i}" for i in range(1, 29)]
        return pd.read_csv(path, header=None, names=columns, nrows=max_rows)

    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def detect_missing_tokens(df: pd.DataFrame) -> dict:
    """Detect string values that may represent missing data."""
    found = {}

    for col in df.select_dtypes(include="object").columns:
        values = set(df[col].dropna().astype(str).str.strip().unique())
        tokens = sorted(values.intersection(MISSING_TOKENS))

        if tokens:
            found[col] = tokens

    return found


def count_infinite_values(df: pd.DataFrame) -> int:
    """Count positive and negative infinity values in numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    return int(np.isinf(numeric_df).sum().sum())


def find_zero_variance_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Find feature columns that contain only one unique value."""
    feature_df = df.drop(columns=[target], errors="ignore")

    return [
        col
        for col in feature_df.columns
        if feature_df[col].nunique(dropna=False) <= 1
    ]


def find_duplicate_columns(df: pd.DataFrame, target: str) -> list[dict]:
    """Find groups of feature columns with exactly identical values."""
    feature_df = df.drop(columns=[target], errors="ignore")

    duplicate_groups = []
    used = set()
    columns = list(feature_df.columns)

    for i, col_a in enumerate(columns):
        if col_a in used:
            continue

        group = [col_a]

        for col_b in columns[i + 1:]:
            if col_b in used:
                continue

            if feature_df[col_a].equals(feature_df[col_b]):
                group.append(col_b)
                used.add(col_b)

        if len(group) > 1:
            duplicate_groups.append(
                {
                    "keep": group[0],
                    "duplicates": group[1:],
                }
            )

    return duplicate_groups


def find_binary_indicator_columns(X: pd.DataFrame) -> list[str]:
    """Find numeric feature columns containing only binary values 0 and 1."""
    binary_cols = []

    for col in X.select_dtypes(include="number").columns:
        values = set(X[col].dropna().unique())

        if values and values.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)

    return binary_cols


def compute_numeric_summary(
    X: pd.DataFrame,
    binary_indicator_columns: list[str],
) -> dict:
    """Compute basic summary statistics for continuous numeric features only."""
    numeric_df = X.select_dtypes(include="number").drop(
        columns=binary_indicator_columns,
        errors="ignore",
    )

    summary = {}

    for col in numeric_df.columns:
        series = numeric_df[col]

        summary[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }

    return summary


def inspect_dataset(dataset_key: str, config: dict, max_rows: int | None) -> dict:
    """Inspect one raw dataset and return a detailed report dictionary."""
    data_path = RAW_DIR / dataset_key / "data.csv"
    target = config["target"]

    report = {
        "dataset_key": dataset_key,
        "dataset_name": config["name"],
        "domain": config["domain"],
        "task": config["task"],
        "size": config["size"],
        "raw_file_exists": data_path.exists(),
        "target_column": target,
        "inspection_mode": "full" if max_rows is None else "quick",
        "max_rows": max_rows,
        "status": "ok",
        "notes": [],
    }

    if not data_path.exists():
        report["status"] = "missing_raw_file"
        report["notes"].append("Expected data.csv was not found.")
        return report

    try:
        df = load_dataset_preview(data_path, dataset_key, max_rows)
    except Exception as exc:
        report["status"] = "read_error"
        report["notes"].append(str(exc))
        return report

    report["sha256"] = compute_sha256(data_path)
    report["num_rows_inspected"] = int(df.shape[0])
    report["num_columns"] = int(df.shape[1])
    report["column_names"] = list(df.columns)
    report["target_exists"] = target in df.columns
    report["target_dtype"] = str(df[target].dtype) if target in df.columns else None

    if target not in df.columns:
        report["status"] = "target_missing"
        report["notes"].append(f"Target column '{target}' was not found.")
        return report

    X = df.drop(columns=[target])
    y = df[target]

    numeric_columns = list(X.select_dtypes(include="number").columns)
    categorical_columns = list(X.select_dtypes(exclude="number").columns)
    binary_indicator_columns = find_binary_indicator_columns(X)
    class_counts = y.value_counts(dropna=False).to_dict()

    report["feature_names_hash"] = compute_feature_names_hash(list(X.columns))
    report["num_features"] = int(X.shape[1])
    report["numeric_features"] = len(numeric_columns)
    report["categorical_features"] = len(categorical_columns)
    report["binary_indicator_features"] = len(binary_indicator_columns)
    report["continuous_numeric_features"] = (
        report["numeric_features"] - report["binary_indicator_features"]
    )

    report["numeric_columns"] = numeric_columns
    report["categorical_columns"] = categorical_columns
    report["binary_indicator_columns"] = binary_indicator_columns

    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    report["missing_values_total"] = int(df.isna().sum().sum())
    report["potential_missing_tokens"] = detect_missing_tokens(df)
    report["infinite_values_total"] = count_infinite_values(df)
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["zero_variance_columns"] = find_zero_variance_columns(df, target)
    report["duplicate_columns"] = find_duplicate_columns(df, target)
    report["memory_usage_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 3)

    report["class_labels"] = [str(label) for label in class_counts.keys()]
    report["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}

    report["categorical_cardinality"] = {
        col: int(X[col].nunique(dropna=False)) for col in categorical_columns
    }

    report["numeric_summary"] = compute_numeric_summary(
        X,
        binary_indicator_columns,
    )

    if len(class_counts) > 1:
        report["imbalance_ratio"] = round(
            max(class_counts.values()) / min(class_counts.values()), 3
        )
    else:
        report["imbalance_ratio"] = None
        report["notes"].append("Only one class found in inspected rows.")

    if report["missing_values_total"] > 0:
        report["notes"].append("Missing values detected.")

    if report["infinite_values_total"] > 0:
        report["notes"].append("Infinite values detected.")

    if report["zero_variance_columns"]:
        report["notes"].append("Zero-variance feature columns detected.")

    if report["duplicate_columns"]:
        report["notes"].append("Duplicate feature columns detected.")

    if report["potential_missing_tokens"]:
        report["notes"].append("Potential string-based missing tokens detected.")

    return report


def flatten_for_csv(report: dict) -> dict:
    """Flatten the detailed report into a compact row for the CSV summary."""
    return {
        "dataset_key": report.get("dataset_key"),
        "dataset_name": report.get("dataset_name"),
        "domain": report.get("domain"),
        "task": report.get("task"),
        "size": report.get("size"),
        "status": report.get("status"),
        "inspection_mode": report.get("inspection_mode"),
        "target_dtype": report.get("target_dtype"),
        "num_rows_inspected": report.get("num_rows_inspected"),
        "num_columns": report.get("num_columns"),
        "num_features": report.get("num_features"),
        "numeric_features": report.get("numeric_features"),
        "continuous_numeric_features": report.get("continuous_numeric_features"),
        "binary_indicator_features": report.get("binary_indicator_features"),
        "categorical_features": report.get("categorical_features"),
        "missing_values_total": report.get("missing_values_total"),
        "infinite_values_total": report.get("infinite_values_total"),
        "duplicate_rows": report.get("duplicate_rows"),
        "zero_variance_columns_count": len(report.get("zero_variance_columns", [])),
        "duplicate_column_groups_count": len(report.get("duplicate_columns", [])),
        "num_classes": len(report.get("class_labels", [])),
        "imbalance_ratio": report.get("imbalance_ratio"),
        "memory_usage_mb": report.get("memory_usage_mb"),
        "sha256": report.get("sha256"),
        "feature_names_hash": report.get("feature_names_hash"),
        "notes": "; ".join(report.get("notes", [])),
    }


def save_reports(reports: list[dict]) -> pd.DataFrame:
    """Save detailed JSON and compact CSV inspection reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)

    summary_df = pd.DataFrame([flatten_for_csv(report) for report in reports])
    summary_df.to_csv(REPORT_CSV, index=False)

    return summary_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect raw benchmark datasets.")

    parser.add_argument(
        "--full",
        action="store_true",
        help="Inspect the full dataset instead of a preview.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=10_000,
        help="Maximum number of rows to inspect in quick mode.",
    )

    return parser.parse_args()


def main() -> None:
    """Run raw dataset inspection for all registered datasets."""
    args = parse_args()
    max_rows = None if args.full else args.max_rows

    reports = []

    for dataset_key, config in DATASETS.items():
        print(f"Inspecting {dataset_key}...")
        reports.append(inspect_dataset(dataset_key, config, max_rows))

    summary_df = save_reports(reports)

    print(f"\nSaved detailed report to: {REPORT_JSON}")
    print(f"Saved summary report to: {REPORT_CSV}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()