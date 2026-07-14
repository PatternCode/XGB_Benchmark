"""Tests for processed dataset loading and validation."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from benchmark.datasets import DatasetError, load_dataset


def write_processed_dataset(
    root: Path,
    dataset_name: str = "sample_dataset",
    data: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a temporary processed dataset for testing."""
    dataset_directory = root / dataset_name
    dataset_directory.mkdir(parents=True)

    if data is None:
        data = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

    if metadata is None:
        metadata = {
            "target_column": "target",
        }

    data.to_csv(dataset_directory / "data.csv", index=False)

    with (dataset_directory / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file)

    return dataset_directory


def test_load_dataset_returns_expected_dataset(tmp_path: Path) -> None:
    """Load a valid processed dataset."""
    write_processed_dataset(tmp_path)

    dataset = load_dataset(
        "sample_dataset",
        data_root=tmp_path,
    )

    assert dataset.name == "sample_dataset"
    assert dataset.n_samples == 3
    assert dataset.n_features == 2
    assert dataset.n_classes == 2
    assert dataset.target_name == "target"
    assert dataset.feature_names == ["feature_a", "feature_b"]
    assert dataset.X.shape == (3, 2)
    assert dataset.y.tolist() == [0, 1, 0]


def test_load_dataset_uses_default_target_name(tmp_path: Path) -> None:
    """Use 'target' when metadata does not specify a target column."""
    write_processed_dataset(
        tmp_path,
        metadata={},
    )

    dataset = load_dataset(
        "sample_dataset",
        data_root=tmp_path,
    )

    assert dataset.target_name == "target"


def test_load_dataset_rejects_empty_dataset_name(tmp_path: Path) -> None:
    """Reject an empty dataset name."""
    with pytest.raises(
        DatasetError,
        match="dataset_name must be a non-empty string",
    ):
        load_dataset("", data_root=tmp_path)


def test_load_dataset_rejects_missing_directory(tmp_path: Path) -> None:
    """Reject a dataset directory that does not exist."""
    with pytest.raises(
        DatasetError,
        match="Processed dataset directory does not exist",
    ):
        load_dataset("missing_dataset", data_root=tmp_path)


def test_load_dataset_rejects_missing_data_file(tmp_path: Path) -> None:
    """Reject a processed dataset without data.csv."""
    dataset_directory = tmp_path / "sample_dataset"
    dataset_directory.mkdir()

    with (dataset_directory / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({"target_column": "target"}, file)

    with pytest.raises(
        DatasetError,
        match="Processed data file does not exist",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_missing_metadata_file(
    tmp_path: Path,
) -> None:
    """Reject a processed dataset without metadata.json."""
    dataset_directory = tmp_path / "sample_dataset"
    dataset_directory.mkdir()

    pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    ).to_csv(
        dataset_directory / "data.csv",
        index=False,
    )

    with pytest.raises(
        DatasetError,
        match="Processed metadata file does not exist",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_invalid_metadata_json(
    tmp_path: Path,
) -> None:
    """Reject invalid metadata JSON."""
    dataset_directory = write_processed_dataset(tmp_path)

    (dataset_directory / "metadata.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetError,
        match="Invalid JSON in metadata file",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_non_object_metadata(
    tmp_path: Path,
) -> None:
    """Reject metadata whose top-level value is not an object."""
    dataset_directory = write_processed_dataset(tmp_path)

    with (dataset_directory / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(["target"], file)

    with pytest.raises(
        DatasetError,
        match="Metadata must contain a JSON object",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_missing_target_column(
    tmp_path: Path,
) -> None:
    """Reject a dataset whose target column is absent."""
    write_processed_dataset(
        tmp_path,
        metadata={"target_column": "label"},
    )

    with pytest.raises(
        DatasetError,
        match="Target column 'label' was not found",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_empty_data_file(tmp_path: Path) -> None:
    """Reject an empty processed dataset."""
    dataset_directory = tmp_path / "sample_dataset"
    dataset_directory.mkdir()

    (dataset_directory / "data.csv").write_text(
        "feature,target\n",
        encoding="utf-8",
    )

    with (dataset_directory / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({"target_column": "target"}, file)

    with pytest.raises(
        DatasetError,
        match="Processed dataset is empty",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_no_feature_columns(
    tmp_path: Path,
) -> None:
    """Reject a dataset containing only the target column."""
    write_processed_dataset(
        tmp_path,
        data=pd.DataFrame(
            {
                "target": [0, 1, 0],
            }
        ),
    )

    with pytest.raises(
        DatasetError,
        match="contains no feature columns",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)


def test_load_dataset_rejects_missing_target_values(
    tmp_path: Path,
) -> None:
    """Reject missing values in the target column."""
    write_processed_dataset(
        tmp_path,
        data=pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0],
                "target": [0, None, 1],
            }
        ),
    )

    with pytest.raises(
        DatasetError,
        match="Target column 'target' contains missing values",
    ):
        load_dataset("sample_dataset", data_root=tmp_path)
