from pathlib import Path
import argparse
import json

import pandas as pd

from registry import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_JSON = REPORT_DIR / "inspection_report.json"
REPORT_CSV = REPORT_DIR / "inspection_summary.csv"

MISSING_TOKENS = {
    "?",
    "NA",
    "N/A",
    "na",
    "n/a",
    "null",
    "NULL",
    "None",
    "none",
    "",
    " ",
}


def load_dataset_preview(path: Path, max_rows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def detect_missing_tokens(df: pd.DataFrame) -> dict:
    found = {}

    for col in df.select_dtypes(include="object").columns:
        values = set(df[col].dropna().astype(str).str.strip().unique())
        tokens = sorted(values.intersection(MISSING_TOKENS))

        if tokens:
            found[col] = tokens

    return found


def find_zero_variance_columns(df: pd.DataFrame, target: str) -> list[str]:
    feature_df = df.drop(columns=[target], errors="ignore")

    return [
        col
        for col in feature_df.columns
        if feature_df[col].nunique(dropna=False) <= 1
    ]


def find_duplicate_columns(df: pd.DataFrame, target: str) -> list[dict]:
    feature_df = df.drop(columns=[target], errors="ignore")

    duplicate_groups = []
    used = set()
    columns = list(feature_df.columns)

    for i, col_a in enumerate(columns):
        if col_a in used:
            continue

        group = [col_a]

        for col_b in columns[i + 1 :]:
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


def inspect_dataset(dataset_key: str, config: dict, max_rows: int | None) -> dict:
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
        df = load_dataset_preview(data_path, max_rows)
    except Exception as exc:
        report["status"] = "read_error"
        report["notes"].append(str(exc))
        return report

    report["num_rows_inspected"] = int(df.shape[0])
    report["num_columns"] = int(df.shape[1])
    report["column_names"] = list(df.columns)
    report["target_exists"] = target in df.columns

    if target not in df.columns:
        report["status"] = "target_missing"
        report["notes"].append(f"Target column '{target}' was not found.")
        return report

    X = df.drop(columns=[target])
    y = df[target]

    numeric_columns = list(X.select_dtypes(include="number").columns)
    categorical_columns = list(X.select_dtypes(exclude="number").columns)

    class_counts = y.value_counts(dropna=False).to_dict()

    report["num_features"] = int(X.shape[1])
    report["numeric_features"] = len(numeric_columns)
    report["categorical_features"] = len(categorical_columns)
    report["numeric_columns"] = numeric_columns
    report["categorical_columns"] = categorical_columns
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    report["missing_values_total"] = int(df.isna().sum().sum())
    report["potential_missing_tokens"] = detect_missing_tokens(df)
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["zero_variance_columns"] = find_zero_variance_columns(df, target)
    report["duplicate_columns"] = find_duplicate_columns(df, target)
    report["memory_usage_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 3)
    report["class_labels"] = [str(label) for label in class_counts.keys()]
    report["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}
    report["categorical_cardinality"] = {
        col: int(X[col].nunique(dropna=False)) for col in categorical_columns
    }

    if len(class_counts) > 1:
        report["imbalance_ratio"] = round(
            max(class_counts.values()) / min(class_counts.values()), 3
        )
    else:
        report["imbalance_ratio"] = None
        report["notes"].append("Only one class found in inspected rows.")

    if report["zero_variance_columns"]:
        report["notes"].append("Zero-variance feature columns detected.")

    if report["duplicate_columns"]:
        report["notes"].append("Duplicate feature columns detected.")

    if report["potential_missing_tokens"]:
        report["notes"].append("Potential string-based missing tokens detected.")

    return report


def flatten_for_csv(report: dict) -> dict:
    return {
        "dataset_key": report.get("dataset_key"),
        "dataset_name": report.get("dataset_name"),
        "domain": report.get("domain"),
        "task": report.get("task"),
        "size": report.get("size"),
        "status": report.get("status"),
        "inspection_mode": report.get("inspection_mode"),
        "num_rows_inspected": report.get("num_rows_inspected"),
        "num_columns": report.get("num_columns"),
        "num_features": report.get("num_features"),
        "numeric_features": report.get("numeric_features"),
        "categorical_features": report.get("categorical_features"),
        "missing_values_total": report.get("missing_values_total"),
        "duplicate_rows": report.get("duplicate_rows"),
        "zero_variance_columns_count": len(report.get("zero_variance_columns", [])),
        "duplicate_column_groups_count": len(report.get("duplicate_columns", [])),
        "num_classes": len(report.get("class_labels", [])),
        "imbalance_ratio": report.get("imbalance_ratio"),
        "memory_usage_mb": report.get("memory_usage_mb"),
        "notes": "; ".join(report.get("notes", [])),
    }


def save_reports(reports: list[dict]) -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)

    summary_df = pd.DataFrame([flatten_for_csv(report) for report in reports])
    summary_df.to_csv(REPORT_CSV, index=False)

    return summary_df


def parse_args() -> argparse.Namespace:
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