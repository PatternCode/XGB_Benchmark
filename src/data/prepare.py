"""
Prepare canonical benchmark datasets.

This script converts every raw dataset registered in ``src/data/registry.py``
into a deterministic canonical representation suitable for downstream
benchmark experiments.

Expected raw structure:
    data/raw/<dataset_key>/data.csv
    data/raw/<dataset_key>/metadata.json

Generated processed structure:
    data/processed/<dataset_key>/data.csv
    data/processed/<dataset_key>/metadata.json

Examples:
    python src/data/prepare.py
    python src/data/prepare.py --dataset adult_income
    python src/data/prepare.py --overwrite
    python src/data/prepare.py --dataset adult_income --overwrite

Canonical preparation performs only deterministic, representation-level
transformations:

- standardize column names
- standardize missing-value representations
- replace positive and negative infinity with NaN
- remove rows with missing target values
- remove zero-variance features
- retain one representative from each group of exactly duplicate features
- encode target labels as consecutive integers
- preserve categorical predictors without encoding
- save detailed preparation metadata

It intentionally does not perform imputation, scaling, normalization,
resampling, train/test splitting, or supervised feature selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from registry import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PREPARATION_VERSION = "1.0"

MISSING_TOKENS = {
    "",
    "?",
    "na",
    "n/a",
    "null",
    "none",
}


CIC_IDS2017_LABEL_FIXES = {
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
}


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values into JSON-serializable Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_raw_dataset(path: Path, dataset_key: str) -> pd.DataFrame:
    """Load a complete raw dataset.

    HIGGS is handled separately because its raw CSV file has no header row.
    """
    if dataset_key == "higgs":
        columns = ["class"] + [f"feature_{i}" for i in range(1, 29)]
        return pd.read_csv(
            path,
            header=None,
            names=columns,
            low_memory=False,
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def standardize_column_name(name: Any) -> str:
    """Convert one column name to a readable snake_case identifier."""
    text = str(name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_").lower()

    return text or "unnamed_column"


def make_unique(names: list[str]) -> list[str]:
    """Make standardized column names unique while preserving their order."""
    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for name in names:
        count = counts.get(name, 0)
        unique_name = name if count == 0 else f"{name}_{count}"
        counts[name] = count + 1

        while unique_name in unique_names:
            count = counts[name]
            unique_name = f"{name}_{count}"
            counts[name] = count + 1

        unique_names.append(unique_name)

    return unique_names


def standardize_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Standardize all column names and return an auditable name mapping."""
    original_names = [str(column) for column in df.columns]
    standardized_names = make_unique(
        [standardize_column_name(column) for column in original_names]
    )

    mapping = [
        {
            "original": original,
            "standardized": standardized,
        }
        for original, standardized in zip(
            original_names,
            standardized_names,
            strict=True,
        )
    ]

    result = df.copy()
    result.columns = standardized_names
    return result, mapping


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip string values and convert known textual missing tokens to NaN.

    Values such as ``unknown`` are deliberately retained because they may
    represent meaningful categories in published datasets.
    """
    result = df.copy()

    for column in result.select_dtypes(include=["object", "string"]).columns:
        series = result[column]

        def clean_value(value: Any) -> Any:
            if pd.isna(value):
                return np.nan

            if not isinstance(value, str):
                return value

            stripped = value.strip()
            if stripped.lower() in MISSING_TOKENS:
                return np.nan

            return stripped

        result[column] = series.map(clean_value)

    return result


def normalize_target_labels(
    df: pd.DataFrame,
    target: str,
    dataset_key: str,
) -> pd.DataFrame:
    """Apply narrowly scoped, auditable repairs to known target labels."""
    result = df.copy()

    if dataset_key == "cic_ids2017":
        result[target] = result[target].replace(CIC_IDS2017_LABEL_FIXES)

        corrupted_mask = result[target].astype("string").str.contains(
            "\ufffd",
            regex=False,
            na=False,
        )

        if corrupted_mask.any():
            corrupted_labels = sorted(
                result.loc[corrupted_mask, target]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "CIC-IDS2017 still contains target labels with the Unicode "
                f"replacement character U+FFFD: {corrupted_labels}"
            )

    return result


def replace_infinite_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace positive and negative infinity with NaN."""
    result = df.copy()
    numeric_columns = result.select_dtypes(include="number").columns

    if len(numeric_columns) == 0:
        return result, 0

    infinite_count = int(
        np.isinf(result[numeric_columns].to_numpy(dtype=float, copy=False))
        .sum()
    )

    result[numeric_columns] = result[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result, infinite_count


def remove_missing_target_rows(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, int]:
    """Remove observations whose target value is missing."""
    missing_mask = df[target].isna()
    removed_count = int(missing_mask.sum())

    return df.loc[~missing_mask].copy(), removed_count


def find_zero_variance_features(X: pd.DataFrame) -> list[str]:
    """Return features containing at most one distinct value."""
    return [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]


def series_fingerprint(series: pd.Series) -> str:
    """Build a stable content fingerprint for duplicate-column screening."""
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(str(series.dtype).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def find_duplicate_features(X: pd.DataFrame) -> list[dict[str, Any]]:
    """Find groups of exactly identical feature columns.

    Hashing narrows candidate groups before exact equality checks, which avoids
    repeatedly comparing every pair of full columns in large datasets.
    """
    fingerprint_groups: dict[str, list[str]] = {}

    for column in X.columns:
        fingerprint = series_fingerprint(X[column])
        fingerprint_groups.setdefault(fingerprint, []).append(column)

    duplicate_groups: list[dict[str, Any]] = []

    for candidates in fingerprint_groups.values():
        if len(candidates) < 2:
            continue

        representatives: list[str] = []
        grouped_duplicates: dict[str, list[str]] = {}

        for column in candidates:
            matched_representative = None

            for representative in representatives:
                if X[representative].equals(X[column]):
                    matched_representative = representative
                    break

            if matched_representative is None:
                representatives.append(column)
            else:
                grouped_duplicates.setdefault(
                    matched_representative,
                    [],
                ).append(column)

        for retained, duplicates in grouped_duplicates.items():
            duplicate_groups.append(
                {
                    "retained": retained,
                    "removed": duplicates,
                }
            )

    return duplicate_groups


def remove_redundant_features(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    """Remove zero-variance and exactly duplicate feature columns."""
    zero_variance_features = find_zero_variance_features(X)
    reduced = X.drop(columns=zero_variance_features)

    duplicate_groups = find_duplicate_features(reduced)
    duplicate_features = [
        duplicate
        for group in duplicate_groups
        for duplicate in group["removed"]
    ]

    reduced = reduced.drop(columns=duplicate_features)

    return reduced, zero_variance_features, duplicate_groups


def sorted_labels(values: pd.Series) -> list[Any]:
    """Return target labels in a deterministic order."""
    labels = [to_python_scalar(value) for value in values.dropna().unique()]

    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=lambda value: (type(value).__name__, str(value)))


def is_consecutive_zero_based(labels: list[Any]) -> bool:
    """Check whether labels are integers 0, 1, ..., n-1."""
    if not labels:
        return False

    if any(
        isinstance(label, bool)
        or not isinstance(label, (int, np.integer))
        for label in labels
    ):
        return False

    integer_labels = sorted(int(label) for label in labels)
    return integer_labels == list(range(len(integer_labels)))


def normalize_explicit_mapping(mapping: dict[Any, Any]) -> dict[Any, int]:
    """Validate and normalize an explicit target mapping."""
    normalized = {
        to_python_scalar(label): int(encoded)
        for label, encoded in mapping.items()
    }

    encoded_values = sorted(normalized.values())
    expected_values = list(range(len(encoded_values)))

    if encoded_values != expected_values:
        raise ValueError(
            "Explicit target_mapping values must be consecutive integers "
            "starting at 0."
        )

    return normalized


def encode_target(
    y: pd.Series,
    explicit_mapping: dict[Any, Any] | None = None,
) -> tuple[pd.Series, dict[Any, int]]:
    """Encode target labels deterministically as consecutive integers."""
    labels = sorted_labels(y)

    if explicit_mapping is not None:
        mapping = normalize_explicit_mapping(explicit_mapping)
        missing_labels = [
            label
            for label in labels
            if label not in mapping
        ]

        if missing_labels:
            raise ValueError(
                "Explicit target_mapping does not contain all observed labels: "
                f"{missing_labels}"
            )
    elif is_consecutive_zero_based(labels):
        mapping = {label: int(label) for label in labels}
    else:
        mapping = {
            label: encoded
            for encoded, label in enumerate(labels)
        }

    encoded = y.map(mapping)

    if encoded.isna().any():
        unmapped = sorted_labels(y.loc[encoded.isna()])
        raise ValueError(f"Unmapped target labels detected: {unmapped}")

    return encoded.astype("int64"), mapping


def find_binary_indicator_features(X: pd.DataFrame) -> list[str]:
    """Return numeric or Boolean features whose observed values are only 0/1."""
    binary_features: list[str] = []

    for column in X.columns:
        series = X[column]

        if not (is_numeric_dtype(series) or is_bool_dtype(series)):
            continue

        values = set(
            to_python_scalar(value)
            for value in series.dropna().unique()
        )

        if values and values.issubset({0, 1, False, True}):
            binary_features.append(column)

    return binary_features


def identify_feature_types(X: pd.DataFrame) -> dict[str, list[str]]:
    """Identify numeric, categorical, and binary-indicator features."""
    numeric_features = [
        column
        for column in X.columns
        if is_numeric_dtype(X[column]) and not is_bool_dtype(X[column])
    ]
    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]
    binary_indicator_features = find_binary_indicator_features(X)

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_indicator_features": binary_indicator_features,
    }


def json_target_mapping(mapping: dict[Any, int]) -> list[dict[str, Any]]:
    """Represent target mappings without losing original label types."""
    return [
        {
            "original_label": to_python_scalar(label),
            "encoded_label": int(encoded),
        }
        for label, encoded in sorted(
            mapping.items(),
            key=lambda item: item[1],
        )
    ]


def build_processed_metadata(
    *,
    dataset_key: str,
    config: dict[str, Any],
    raw_path: Path,
    processed_path: Path,
    source_sha256: str,
    processed_sha256: str,
    original_target: str,
    standardized_original_target: str,
    target_mapping: dict[Any, int],
    column_name_mapping: list[dict[str, str]],
    raw_rows: int,
    processed_rows: int,
    raw_feature_count: int,
    X: pd.DataFrame,
    removed_zero_variance: list[str],
    removed_duplicate_groups: list[dict[str, Any]],
    rows_removed_missing_target: int,
    infinite_values_replaced: int,
) -> dict[str, Any]:
    """Create fully traceable metadata for one processed dataset."""
    feature_types = identify_feature_types(X)

    return {
        "preparation_version": PREPARATION_VERSION,
        "dataset_key": dataset_key,
        "dataset_name": config.get("name"),
        "domain": config.get("domain"),
        "task": config.get("task"),
        "size": config.get("size"),
        "source_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "source_sha256": source_sha256,
        "processed_file": str(processed_path.relative_to(PROJECT_ROOT)),
        "processed_sha256": processed_sha256,
        "original_target_column": original_target,
        "standardized_original_target_column": standardized_original_target,
        "target_column": "target",
        "target_mapping": json_target_mapping(target_mapping),
        "column_name_mapping": column_name_mapping,
        "num_rows_raw": raw_rows,
        "num_rows_processed": processed_rows,
        "num_features_raw": raw_feature_count,
        "num_features_processed": int(X.shape[1]),
        "feature_names": list(X.columns),
        **feature_types,
        "removed_zero_variance_features": removed_zero_variance,
        "removed_duplicate_features": removed_duplicate_groups,
        "rows_removed_missing_target": rows_removed_missing_target,
        "infinite_values_replaced": infinite_values_replaced,
        "missing_predictor_values_total": int(X.isna().sum().sum()),
    }


def output_exists(output_dir: Path) -> bool:
    """Return True when either canonical output file already exists."""
    return (
        (output_dir / "data.csv").exists()
        or (output_dir / "metadata.json").exists()
    )


def prepare_dataset(
    dataset_key: str,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Prepare one registered dataset and return a compact status record."""
    raw_path = RAW_DIR / dataset_key / "data.csv"
    output_dir = PROCESSED_DIR / dataset_key
    processed_path = output_dir / "data.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    if output_exists(output_dir) and not overwrite:
        return {
            "dataset_key": dataset_key,
            "status": "skipped",
            "message": "Processed files already exist. Use --overwrite.",
        }

    original_target = str(config["target"])
    source_sha256 = compute_sha256(raw_path)

    df = load_raw_dataset(raw_path, dataset_key)
    raw_rows = int(df.shape[0])
    raw_feature_count = int(df.shape[1] - 1)

    df, column_name_mapping = standardize_column_names(df)

    standardized_target = standardize_column_name(original_target)

    if standardized_target not in df.columns:
        raise KeyError(
            f"Target column '{original_target}' was not found in the raw data. "
            f"Expected standardized name: '{standardized_target}'."
        )

    df = standardize_missing_values(df)
    df = normalize_target_labels(df, standardized_target, dataset_key)
    df, infinite_values_replaced = replace_infinite_values(df)
    df, rows_removed_missing_target = remove_missing_target_rows(
        df,
        standardized_target,
    )

    X = df.drop(columns=[standardized_target])
    y = df[standardized_target]

    X, removed_zero_variance, removed_duplicate_groups = (
        remove_redundant_features(X)
    )

    y_encoded, target_mapping = encode_target(
        y,
        explicit_mapping=config.get("target_mapping"),
    )

    processed_df = X.copy()
    processed_df["target"] = y_encoded.to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    processed_sha256 = compute_sha256(processed_path)

    metadata = build_processed_metadata(
        dataset_key=dataset_key,
        config=config,
        raw_path=raw_path,
        processed_path=processed_path,
        source_sha256=source_sha256,
        processed_sha256=processed_sha256,
        original_target=original_target,
        standardized_original_target=standardized_target,
        target_mapping=target_mapping,
        column_name_mapping=column_name_mapping,
        raw_rows=raw_rows,
        processed_rows=int(processed_df.shape[0]),
        raw_feature_count=raw_feature_count,
        X=X,
        removed_zero_variance=removed_zero_variance,
        removed_duplicate_groups=removed_duplicate_groups,
        rows_removed_missing_target=rows_removed_missing_target,
        infinite_values_replaced=infinite_values_replaced,
    )

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return {
        "dataset_key": dataset_key,
        "status": "prepared",
        "message": (
            f"{processed_df.shape[0]} rows, "
            f"{X.shape[1]} features"
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare canonical benchmark datasets."
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        help="Prepare only one registered dataset.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed files.",
    )

    return parser.parse_args()


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a compact preparation summary."""
    print("\nPreparation summary")
    print("-" * 80)

    width = max(len(result["dataset_key"]) for result in results)

    for result in results:
        dataset_key = result["dataset_key"].ljust(width)
        status = result["status"].upper().ljust(8)
        print(f"{dataset_key}  {status}  {result['message']}")


def main() -> None:
    """Prepare selected or all registered datasets."""
    args = parse_args()

    selected_keys = (
        [args.dataset]
        if args.dataset
        else list(DATASETS.keys())
    )

    results: list[dict[str, Any]] = []
    failed = False

    for dataset_key in selected_keys:
        print(f"Preparing {dataset_key}...")

        try:
            result = prepare_dataset(
                dataset_key,
                DATASETS[dataset_key],
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failed = True
            result = {
                "dataset_key": dataset_key,
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

        results.append(result)

    print_summary(results)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()