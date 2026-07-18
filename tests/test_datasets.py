"""Tests for processed dataset loading and validation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmark.datasets import DatasetError, load_dataset


def _write_dataset(
    data_root: Path,
    dataset_name: str = "example_dataset",
    data: pd.DataFrame | None = None,
    metadata: dict | None = None,
) -> Path:
    """Create a temporary processed dataset for testing."""
    dataset_directory = data_root / dataset_name
    dataset_directory.mkdir(parents=True)

    if data is None:
        data = pd.DataFrame(
            {
                "numeric_feature": [1.0, 2.0, 3.0, 4.0],
                "categorical_feature": [0, 1, 0, 1],
                "target": [0, 1, 0, 1],
            }
        )

    if metadata is None:
        metadata = {
            "target_column": "target",
            "numeric_features": ["numeric_feature"],
            "categorical_features": ["categorical_feature"],
        }

    data.to_csv(
        dataset_directory / "data.csv",
        index=False,
    )

    with (
        dataset_directory / "metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file)

    return dataset_directory


def test_load_dataset_returns_expected_dataset(
    tmp_path: Path,
) -> None:
    """A valid processed dataset should load successfully."""
    _write_dataset(tmp_path)

    dataset = load_dataset(
        dataset_name="example_dataset",
        data_root=tmp_path,
    )

    assert dataset.name == "example_dataset"
    assert dataset.n_samples == 4
    assert dataset.n_features == 2
    assert dataset.n_classes == 2
    assert dataset.target_name == "target"

    assert dataset.feature_names == [
        "numeric_feature",
        "categorical_feature",
    ]

    assert dataset.numeric_features == [
        "numeric_feature",
    ]

    assert dataset.categorical_features == [
        "categorical_feature",
    ]

    pd.testing.assert_frame_equal(
        dataset.X,
        pd.DataFrame(
            {
                "numeric_feature": [1.0, 2.0, 3.0, 4.0],
                "categorical_feature": [0, 1, 0, 1],
            }
        ),
    )

    pd.testing.assert_series_equal(
        dataset.y,
        pd.Series(
            [0, 1, 0, 1],
            name="target",
        ),
    )


def test_load_dataset_preserves_missing_predictor_values(
    tmp_path: Path,
) -> None:
    """Loading must not impute or remove missing predictors."""
    data = pd.DataFrame(
        {
            "numeric_feature": [1.0, None, 3.0, 4.0],
            "categorical_feature": [0, 1, None, 1],
            "target": [0, 1, 0, 1],
        }
    )

    _write_dataset(
        data_root=tmp_path,
        data=data,
    )

    dataset = load_dataset(
        dataset_name="example_dataset",
        data_root=tmp_path,
    )

    assert dataset.n_samples == 4
    assert dataset.X.isna().sum().sum() == 2
    assert pd.isna(
        dataset.X.loc[1, "numeric_feature"]
    )
    assert pd.isna(
        dataset.X.loc[2, "categorical_feature"]
    )


def test_load_dataset_uses_metadata_target_column(
    tmp_path: Path,
) -> None:
    """The target name should be read from metadata."""
    data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "label": [0, 1, 0, 1],
        }
    )

    metadata = {
        "target_column": "label",
        "numeric_features": ["feature"],
        "categorical_features": [],
    }

    _write_dataset(
        data_root=tmp_path,
        data=data,
        metadata=metadata,
    )

    dataset = load_dataset(
        dataset_name="example_dataset",
        data_root=tmp_path,
    )

    assert dataset.target_name == "label"
    assert dataset.feature_names == ["feature"]
    assert dataset.numeric_features == ["feature"]
    assert dataset.categorical_features == []


def test_load_dataset_defaults_target_column_to_target(
    tmp_path: Path,
) -> None:
    """The conventional target name should remain supported."""
    metadata = {
        "numeric_features": ["numeric_feature"],
        "categorical_features": ["categorical_feature"],
    }

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    dataset = load_dataset(
        dataset_name="example_dataset",
        data_root=tmp_path,
    )

    assert dataset.target_name == "target"


def test_load_dataset_rejects_empty_dataset_name(
    tmp_path: Path,
) -> None:
    """Dataset names must be non-empty strings."""
    with pytest.raises(
        DatasetError,
        match="dataset_name must be a non-empty string",
    ):
        load_dataset(
            dataset_name="",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """The processed dataset directory must exist."""
    with pytest.raises(
        DatasetError,
        match="Processed dataset directory does not exist",
    ):
        load_dataset(
            dataset_name="missing_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_missing_data_file(
    tmp_path: Path,
) -> None:
    """The processed CSV file must exist."""
    dataset_directory = tmp_path / "example_dataset"
    dataset_directory.mkdir()

    with (
        dataset_directory / "metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "target_column": "target",
                "numeric_features": [],
                "categorical_features": [],
            },
            file,
        )

    with pytest.raises(
        DatasetError,
        match="Processed data file does not exist",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_missing_metadata_file(
    tmp_path: Path,
) -> None:
    """The metadata JSON file must exist."""
    dataset_directory = tmp_path / "example_dataset"
    dataset_directory.mkdir()

    pd.DataFrame(
        {
            "feature": [1.0, 2.0],
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
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_invalid_metadata_json(
    tmp_path: Path,
) -> None:
    """Metadata must contain valid JSON."""
    dataset_directory = tmp_path / "example_dataset"
    dataset_directory.mkdir()

    pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "target": [0, 1],
        }
    ).to_csv(
        dataset_directory / "data.csv",
        index=False,
    )

    (
        dataset_directory / "metadata.json"
    ).write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetError,
        match="Invalid JSON in metadata file",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_non_object_metadata(
    tmp_path: Path,
) -> None:
    """The top-level metadata value must be a JSON object."""
    dataset_directory = tmp_path / "example_dataset"
    dataset_directory.mkdir()

    pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "target": [0, 1],
        }
    ).to_csv(
        dataset_directory / "data.csv",
        index=False,
    )

    with (
        dataset_directory / "metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(["not", "an", "object"], file)

    with pytest.raises(
        DatasetError,
        match="Metadata must contain a JSON object",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_empty_data(
    tmp_path: Path,
) -> None:
    """A processed dataset must contain at least one row."""
    data = pd.DataFrame(
        columns=[
            "numeric_feature",
            "categorical_feature",
            "target",
        ]
    )

    _write_dataset(
        data_root=tmp_path,
        data=data,
    )

    with pytest.raises(
        DatasetError,
        match="Processed dataset is empty",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_missing_target_column(
    tmp_path: Path,
) -> None:
    """The target column recorded in metadata must exist."""
    data = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0],
            "categorical_feature": [0, 1],
        }
    )

    _write_dataset(
        data_root=tmp_path,
        data=data,
    )

    with pytest.raises(
        DatasetError,
        match="Target column 'target' was not found",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_missing_target_values(
    tmp_path: Path,
) -> None:
    """Missing target labels must not be accepted."""
    data = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, 3.0],
            "categorical_feature": [0, 1, 0],
            "target": [0, None, 1],
        }
    )

    _write_dataset(
        data_root=tmp_path,
        data=data,
    )

    with pytest.raises(
        DatasetError,
        match="Target column 'target' contains missing values",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_dataset_without_features(
    tmp_path: Path,
) -> None:
    """At least one predictor column is required."""
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
        }
    )

    metadata = {
        "target_column": "target",
        "numeric_features": [],
        "categorical_features": [],
    }

    _write_dataset(
        data_root=tmp_path,
        data=data,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match="contains no feature columns",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "numeric_features",
        "categorical_features",
    ],
)
def test_load_dataset_requires_feature_type_lists(
    tmp_path: Path,
    field_name: str,
) -> None:
    """Both metadata feature-type fields must be lists."""
    metadata = {
        "target_column": "target",
        "numeric_features": ["numeric_feature"],
        "categorical_features": ["categorical_feature"],
    }

    metadata.pop(field_name)

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=rf"metadata\.{field_name} must be a list",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("numeric_features", "numeric_feature"),
        ("categorical_features", "categorical_feature"),
        ("numeric_features", None),
        ("categorical_features", None),
    ],
)
def test_load_dataset_rejects_non_list_feature_metadata(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    """Feature-type metadata fields must be JSON arrays."""
    metadata = {
        "target_column": "target",
        "numeric_features": ["numeric_feature"],
        "categorical_features": ["categorical_feature"],
    }

    metadata[field_name] = invalid_value

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=rf"metadata\.{field_name} must be a list",
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_non_string_feature_names(
    tmp_path: Path,
) -> None:
    """Feature lists must contain only valid column names."""
    metadata = {
        "target_column": "target",
        "numeric_features": ["numeric_feature", 12],
        "categorical_features": ["categorical_feature"],
    }

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=(
            r"metadata\.numeric_features must contain only "
            "non-empty strings"
        ),
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_duplicate_metadata_features(
    tmp_path: Path,
) -> None:
    """A feature must not appear twice in one metadata list."""
    metadata = {
        "target_column": "target",
        "numeric_features": [
            "numeric_feature",
            "numeric_feature",
        ],
        "categorical_features": ["categorical_feature"],
    }

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=(
            r"metadata\.numeric_features contains duplicate "
            "feature names"
        ),
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_overlapping_feature_types(
    tmp_path: Path,
) -> None:
    """A feature cannot be both numeric and categorical."""
    metadata = {
        "target_column": "target",
        "numeric_features": [
            "numeric_feature",
            "categorical_feature",
        ],
        "categorical_features": [
            "categorical_feature",
        ],
    }

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=(
            "numeric_features and "
            "metadata.categorical_features overlap"
        ),
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_unknown_metadata_features(
    tmp_path: Path,
) -> None:
    """Metadata must not reference columns absent from the CSV."""
    metadata = {
        "target_column": "target",
        "numeric_features": [
            "numeric_feature",
            "unknown_feature",
        ],
        "categorical_features": [
            "categorical_feature",
        ],
    }

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=(
            "Feature metadata contains columns that are not present"
        ),
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )


def test_load_dataset_rejects_unclassified_features(
    tmp_path: Path,
) -> None:
    """Every predictor must have one metadata feature type."""
    metadata = {
        "target_column": "target",
        "numeric_features": ["numeric_feature"],
        "categorical_features": [],
    }

    _write_dataset(
        data_root=tmp_path,
        metadata=metadata,
    )

    with pytest.raises(
        DatasetError,
        match=(
            "Feature metadata does not classify all processed "
            "features"
        ),
    ):
        load_dataset(
            dataset_name="example_dataset",
            data_root=tmp_path,
        )