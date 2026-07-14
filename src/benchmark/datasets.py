"""Load and validate processed datasets for XGB Benchmark experiments."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetError(Exception):
    """Raised when a processed dataset is missing or invalid."""


@dataclass(frozen=True)
class Dataset:
    """Processed tabular dataset used by the benchmark.

    Attributes
    ----------
    name
        Dataset identifier used in the repository.
    X
        Feature matrix with feature names preserved as columns.
    y
        Target labels.
    metadata
        Metadata loaded from the processed dataset directory.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    @property
    def n_features(self) -> int:
        """Return the number of input features."""
        return self.X.shape[1]

    @property
    def feature_names(self) -> list[str]:
        """Return feature names in their original column order."""
        return self.X.columns.tolist()

    @property
    def target_name(self) -> str:
        """Return the target column name."""
        return str(self.y.name)

    @property
    def n_classes(self) -> int:
        """Return the number of distinct target classes."""
        return int(self.y.nunique())


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata from a JSON file."""
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise DatasetError(
            f"Invalid JSON in metadata file '{metadata_path}': {error}"
        ) from error
    except OSError as error:
        raise DatasetError(
            f"Could not read metadata file '{metadata_path}': {error}"
        ) from error

    if not isinstance(metadata, dict):
        raise DatasetError(
            f"Metadata must contain a JSON object: {metadata_path}"
        )

    return metadata


def _get_target_name(metadata: dict[str, Any]) -> str:
    """Return the target column name recorded in metadata."""
    target_name = metadata.get("target_column", "target")

    if not isinstance(target_name, str) or not target_name:
        raise DatasetError(
            "metadata.target_column must be a non-empty string."
        )

    return target_name


def load_dataset(
    dataset_name: str,
    data_root: str | Path = "data/processed",
) -> Dataset:
    """Load and validate a processed tabular dataset.

    This function loads the complete processed dataset. It deliberately does
    not split, scale, impute, select features, or create model-specific data
    structures. Those operations must be performed later using training data
    only.

    Parameters
    ----------
    dataset_name
        Name of the processed dataset directory.
    data_root
        Root directory containing processed datasets.

    Returns
    -------
    Dataset
        Loaded dataset with its feature matrix, target, and metadata.

    Raises
    ------
    DatasetError
        If required files are missing or the dataset is inconsistent.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise DatasetError("dataset_name must be a non-empty string.")

    dataset_directory = Path(data_root) / dataset_name
    data_path = dataset_directory / "data.csv"
    metadata_path = dataset_directory / "metadata.json"

    if not dataset_directory.is_dir():
        raise DatasetError(
            f"Processed dataset directory does not exist: "
            f"{dataset_directory}"
        )

    if not data_path.is_file():
        raise DatasetError(
            f"Processed data file does not exist: {data_path}"
        )

    if not metadata_path.is_file():
        raise DatasetError(
            f"Processed metadata file does not exist: {metadata_path}"
        )

    metadata = _load_metadata(metadata_path)

    try:
        data = pd.read_csv(data_path)
    except (OSError, pd.errors.ParserError) as error:
        raise DatasetError(
            f"Could not load processed data file '{data_path}': {error}"
        ) from error

    if data.empty:
        raise DatasetError(f"Processed dataset is empty: {data_path}")

    if not data.columns.is_unique:
        duplicated_columns = data.columns[data.columns.duplicated()].tolist()
        raise DatasetError(
            f"Processed dataset contains duplicate columns: "
            f"{duplicated_columns}"
        )

    target_name = _get_target_name(metadata)

    if target_name not in data.columns:
        raise DatasetError(
            f"Target column '{target_name}' was not found in {data_path}."
        )

    X = data.drop(columns=target_name)
    y = data[target_name].copy()

    if X.shape[1] == 0:
        raise DatasetError(
            f"Dataset '{dataset_name}' contains no feature columns."
        )

    if y.isna().any():
        raise DatasetError(
            f"Target column '{target_name}' contains missing values."
        )

    return Dataset(
        name=dataset_name,
        X=X,
        y=y,
        metadata=metadata,
    )