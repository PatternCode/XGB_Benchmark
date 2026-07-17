"""Tests for benchmark result conversion and saving."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from benchmark.results import (
    ResultError,
    load_progress,
    results_to_frame,
    save_experiment_output,
    save_results,
    write_progress,
)


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """Return a small collection of flat result records."""
    return [
        {
            "experiment": "smoke_test",
            "dataset": "sample_dataset",
            "outer_fold": 1,
            "selection_method": "gain",
            "requested_percentage": 10.0,
            "actual_percentage": 10.0,
            "n_selected_features": 3,
            "random_repetition": None,
            "model": "decision_tree",
            "max_depth": 2,
            "accuracy": 0.90,
            "f1_macro": 0.88,
        },
        {
            "experiment": "smoke_test",
            "dataset": "sample_dataset",
            "outer_fold": 2,
            "selection_method": "all_features",
            "requested_percentage": None,
            "actual_percentage": 100.0,
            "n_selected_features": 30,
            "random_repetition": None,
            "model": "decision_tree",
            "max_depth": 2,
            "accuracy": 0.92,
            "f1_macro": 0.90,
        },
    ]


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Return a minimal benchmark configuration."""
    return {
        "experiment": {
            "name": "smoke_test",
            "random_seed": 42,
        },
        "datasets": {
            "sample_dataset": {
                "enabled": True,
            },
        },
        "models": {
            "decision_tree": {
                "enabled": True,
            },
        },
    }


@pytest.fixture
def sample_experiment_output(
    sample_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return sample result, ranking, and selected-feature records."""
    rankings = [
        {
            "experiment": "smoke_test",
            "dataset": "sample_dataset",
            "outer_fold": 1,
            "ranking_method": "gain",
            "feature": "feature_a",
            "rank": 1,
            "importance_score": 4.5,
        },
        {
            "experiment": "smoke_test",
            "dataset": "sample_dataset",
            "outer_fold": 1,
            "ranking_method": "gain",
            "feature": "feature_b",
            "rank": 2,
            "importance_score": 1.2,
        },
    ]

    selected_features = [
        {
            "experiment": "smoke_test",
            "dataset": "sample_dataset",
            "outer_fold": 1,
            "selection_method": "gain",
            "requested_percentage": 10.0,
            "actual_percentage": 10.0,
            "random_repetition": None,
            "feature": "feature_a",
            "selection_rank": 1,
            "importance_score": 4.5,
        },
        {
            "experiment": "smoke_test",
            "dataset": "sample_dataset",
            "outer_fold": 1,
            "selection_method": "gain",
            "requested_percentage": 10.0,
            "actual_percentage": 10.0,
            "random_repetition": None,
            "feature": "feature_b",
            "selection_rank": 2,
            "importance_score": 1.2,
        },
    ]

    return {
        "results": sample_results,
        "rankings": rankings,
        "selected_features": selected_features,
    }


def test_results_to_frame_returns_dataframe(
    sample_results: list[dict[str, Any]],
) -> None:
    """Convert flat result records into a DataFrame."""
    results_frame = results_to_frame(sample_results)

    assert isinstance(results_frame, pd.DataFrame)
    assert len(results_frame) == 2
    assert results_frame["accuracy"].tolist() == [0.90, 0.92]


def test_results_to_frame_preserves_columns(
    sample_results: list[dict[str, Any]],
) -> None:
    """Preserve the result fields as DataFrame columns."""
    results_frame = results_to_frame(sample_results)

    assert set(results_frame.columns) == set(
        sample_results[0]
    )


@pytest.mark.parametrize(
    "results",
    [
        [],
        [1, 2],
        ["invalid"],
    ],
)
def test_results_to_frame_rejects_invalid_records(
    results: list[object],
) -> None:
    """Reject empty or non-dictionary result collections."""
    with pytest.raises(ResultError):
        results_to_frame(
            results,  # type: ignore[arg-type]
        )


def test_results_to_frame_rejects_non_list() -> None:
    """Require results to be supplied as a list."""
    with pytest.raises(
        ResultError,
        match="results must be provided as a list",
    ):
        results_to_frame(  # type: ignore[arg-type]
            {"accuracy": 0.9}
        )


def test_save_results_creates_csv_file(
    tmp_path: Path,
    sample_results: list[dict[str, Any]],
) -> None:
    """Save experiment results to a timestamped CSV file."""
    results_path = save_results(
        results=sample_results,
        output_directory=tmp_path,
        experiment_name="smoke_test",
    )

    assert results_path.is_file()
    assert results_path.name == "results.csv"
    assert results_path.parent.parent == tmp_path
    assert results_path.parent.name.startswith(
        "smoke_test_"
    )


def test_save_results_csv_contains_expected_data(
    tmp_path: Path,
    sample_results: list[dict[str, Any]],
) -> None:
    """Write all result rows and fields to the CSV file."""
    results_path = save_results(
        results=sample_results,
        output_directory=tmp_path,
        experiment_name="smoke_test",
    )

    saved_frame = pd.read_csv(results_path)

    assert len(saved_frame) == 2
    assert saved_frame["outer_fold"].tolist() == [1, 2]
    assert saved_frame["accuracy"].tolist() == pytest.approx(
        [0.90, 0.92]
    )


def test_save_results_creates_separate_run_directories(
    tmp_path: Path,
    sample_results: list[dict[str, Any]],
) -> None:
    """Avoid overwriting results from previous runs."""
    first_path = save_results(
        results=sample_results,
        output_directory=tmp_path,
        experiment_name="smoke_test",
    )

    second_path = save_results(
        results=sample_results,
        output_directory=tmp_path,
        experiment_name="smoke_test",
    )

    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()


@pytest.mark.parametrize(
    "experiment_name",
    [
        "",
        "   ",
        42,
        None,
    ],
)
def test_save_results_rejects_invalid_experiment_name(
    tmp_path: Path,
    sample_results: list[dict[str, Any]],
    experiment_name: object,
) -> None:
    """Require a non-empty experiment name."""
    with pytest.raises(
        ResultError,
        match="experiment_name must be a non-empty string",
    ):
        save_results(
            results=sample_results,
            output_directory=tmp_path,
            experiment_name=experiment_name,  # type: ignore[arg-type]
        )


def test_save_experiment_output_creates_all_output_files(
    tmp_path: Path,
    sample_experiment_output: dict[
        str,
        list[dict[str, Any]],
    ],
    sample_config: dict[str, Any],
) -> None:
    """Save all experiment output files."""
    saved_paths = save_experiment_output(
        experiment_output=sample_experiment_output,
        output_directory=tmp_path,
        experiment_name="smoke_test",
        config=sample_config,
    )

    assert saved_paths["run_directory"].is_dir()
    assert saved_paths["results"].is_file()
    assert saved_paths["rankings"].is_file()
    assert saved_paths["selected_features"].is_file()
    assert saved_paths["metadata"].is_file()

    assert saved_paths["results"].name == "results.csv"
    assert saved_paths["rankings"].name == "rankings.csv"
    assert (
        saved_paths["selected_features"].name
        == "selected_features.csv"
    )
    assert saved_paths["metadata"].name == "run_metadata.json"


def test_save_experiment_output_writes_expected_rankings(
    tmp_path: Path,
    sample_experiment_output: dict[
        str,
        list[dict[str, Any]],
    ],
    sample_config: dict[str, Any],
) -> None:
    """Write ranking fields and rows to rankings.csv."""
    saved_paths = save_experiment_output(
        experiment_output=sample_experiment_output,
        output_directory=tmp_path,
        experiment_name="smoke_test",
        config=sample_config,
    )

    rankings_frame = pd.read_csv(
        saved_paths["rankings"]
    )

    assert len(rankings_frame) == 2
    assert rankings_frame["feature"].tolist() == [
        "feature_a",
        "feature_b",
    ]
    assert rankings_frame["rank"].tolist() == [1, 2]
    assert rankings_frame[
        "importance_score"
    ].tolist() == pytest.approx([4.5, 1.2])


def test_save_experiment_output_writes_selected_features(
    tmp_path: Path,
    sample_experiment_output: dict[
        str,
        list[dict[str, Any]],
    ],
    sample_config: dict[str, Any],
) -> None:
    """Write selected-feature records to selected_features.csv."""
    saved_paths = save_experiment_output(
        experiment_output=sample_experiment_output,
        output_directory=tmp_path,
        experiment_name="smoke_test",
        config=sample_config,
    )

    selected_frame = pd.read_csv(
        saved_paths["selected_features"]
    )

    assert len(selected_frame) == 2
    assert selected_frame["feature"].tolist() == [
        "feature_a",
        "feature_b",
    ]
    assert selected_frame["selection_rank"].tolist() == [1, 2]
    assert selected_frame[
        "importance_score"
    ].tolist() == pytest.approx([4.5, 1.2])


def test_save_experiment_output_writes_metadata(
    tmp_path: Path,
    sample_experiment_output: dict[
        str,
        list[dict[str, Any]],
    ],
    sample_config: dict[str, Any],
) -> None:
    """Write reproducibility information to run_metadata.json."""
    saved_paths = save_experiment_output(
        experiment_output=sample_experiment_output,
        output_directory=tmp_path,
        experiment_name="smoke_test",
        config=sample_config,
    )

    with saved_paths["metadata"].open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata["experiment"] == "smoke_test"
    assert metadata["random_seed"] == 42
    assert metadata["enabled_datasets"] == [
        "sample_dataset"
    ]
    assert metadata["enabled_models"] == [
        "decision_tree"
    ]
    assert metadata["n_result_rows"] == 2
    assert metadata["n_ranking_rows"] == 2
    assert metadata["n_selected_feature_rows"] == 2
    assert metadata["configuration"] == sample_config


def test_save_experiment_output_uses_same_run_directory(
    tmp_path: Path,
    sample_experiment_output: dict[
        str,
        list[dict[str, Any]],
    ],
    sample_config: dict[str, Any],
) -> None:
    """Store all output files in one timestamped directory."""
    saved_paths = save_experiment_output(
        experiment_output=sample_experiment_output,
        output_directory=tmp_path,
        experiment_name="smoke_test",
        config=sample_config,
    )

    run_directory = saved_paths["run_directory"]

    assert saved_paths["results"].parent == run_directory
    assert saved_paths["rankings"].parent == run_directory
    assert (
        saved_paths["selected_features"].parent
        == run_directory
    )
    assert saved_paths["metadata"].parent == run_directory


@pytest.mark.parametrize(
    "missing_key",
    [
        "results",
        "rankings",
        "selected_features",
    ],
)
def test_save_experiment_output_rejects_missing_collection(
    tmp_path: Path,
    sample_experiment_output: dict[
        str,
        list[dict[str, Any]],
    ],
    sample_config: dict[str, Any],
    missing_key: str,
) -> None:
    """Require every experiment-output collection."""
    invalid_output = dict(sample_experiment_output)
    invalid_output.pop(missing_key)

    with pytest.raises(
        ResultError,
        match=f"missing '{missing_key}'",
    ):
        save_experiment_output(
            experiment_output=invalid_output,
            output_directory=tmp_path,
            experiment_name="smoke_test",
            config=sample_config,
        )


def test_save_experiment_output_rejects_non_dictionary(
    tmp_path: Path,
    sample_config: dict[str, Any],
) -> None:
    """Require the complete output to be a dictionary."""
    with pytest.raises(
        ResultError,
        match="experiment_output must be a dictionary",
    ):
        save_experiment_output(
            experiment_output=[],  # type: ignore[arg-type]
            output_directory=tmp_path,
            experiment_name="smoke_test",
            config=sample_config,
        )

def test_write_progress_creates_progress_file(
    tmp_path: Path,
) -> None:
    """Write dataset completion progress as JSON."""
    progress_path = write_progress(
        run_directory=tmp_path,
        experiment_name="smoke_test",
        completed_datasets=[
            "breast_cancer_wisconsin",
            "steel_plates_faults",
        ],
    )

    assert progress_path == tmp_path / "progress.json"
    assert progress_path.is_file()


def test_write_progress_writes_expected_content(
    tmp_path: Path,
) -> None:
    """Write experiment and completed dataset information."""
    progress_path = write_progress(
        run_directory=tmp_path,
        experiment_name="smoke_test",
        completed_datasets=[
            "breast_cancer_wisconsin",
        ],
    )

    with progress_path.open(
        "r",
        encoding="utf-8",
    ) as progress_file:
        progress = json.load(progress_file)

    assert progress["experiment"] == "smoke_test"
    assert progress["completed_datasets"] == [
        "breast_cancer_wisconsin",
    ]
    assert isinstance(
        progress["last_updated_utc"],
        str,
    )
    assert progress["last_updated_utc"]


def test_write_progress_overwrites_existing_progress(
    tmp_path: Path,
) -> None:
    """Replace progress with the latest completed datasets."""
    write_progress(
        run_directory=tmp_path,
        experiment_name="smoke_test",
        completed_datasets=[
            "breast_cancer_wisconsin",
        ],
    )

    progress_path = write_progress(
        run_directory=tmp_path,
        experiment_name="smoke_test",
        completed_datasets=[
            "breast_cancer_wisconsin",
            "steel_plates_faults",
        ],
    )

    with progress_path.open(
        "r",
        encoding="utf-8",
    ) as progress_file:
        progress = json.load(progress_file)

    assert progress["completed_datasets"] == [
        "breast_cancer_wisconsin",
        "steel_plates_faults",
    ]


def test_write_progress_allows_no_completed_datasets(
    tmp_path: Path,
) -> None:
    """Allow progress to represent a newly created run."""
    progress_path = write_progress(
        run_directory=tmp_path,
        experiment_name="smoke_test",
        completed_datasets=[],
    )

    progress = load_progress(tmp_path)

    assert progress_path.is_file()
    assert progress["completed_datasets"] == []


@pytest.mark.parametrize(
    "experiment_name",
    [
        "",
        "   ",
        42,
        None,
    ],
)
def test_write_progress_rejects_invalid_experiment_name(
    tmp_path: Path,
    experiment_name: object,
) -> None:
    """Require a non-empty experiment name."""
    with pytest.raises(
        ResultError,
        match="experiment_name must be a non-empty string",
    ):
        write_progress(
            run_directory=tmp_path,
            experiment_name=experiment_name,  # type: ignore[arg-type]
            completed_datasets=[],
        )


@pytest.mark.parametrize(
    "completed_datasets",
    [
        "sample_dataset",
        [""],
        ["   "],
        [42],
        ["sample_dataset", "sample_dataset"],
    ],
)
def test_write_progress_rejects_invalid_completed_datasets(
    tmp_path: Path,
    completed_datasets: object,
) -> None:
    """Require unique, non-empty dataset names."""
    with pytest.raises(ResultError):
        write_progress(
            run_directory=tmp_path,
            experiment_name="smoke_test",
            completed_datasets=completed_datasets,  # type: ignore[arg-type]
        )


def test_write_progress_rejects_missing_run_directory(
    tmp_path: Path,
) -> None:
    """Require an existing run directory."""
    missing_directory = tmp_path / "missing"

    with pytest.raises(
        ResultError,
        match="Run directory does not exist",
    ):
        write_progress(
            run_directory=missing_directory,
            experiment_name="smoke_test",
            completed_datasets=[],
        )


def test_load_progress_returns_saved_progress(
    tmp_path: Path,
) -> None:
    """Load valid progress from a run directory."""
    write_progress(
        run_directory=tmp_path,
        experiment_name="smoke_test",
        completed_datasets=[
            "breast_cancer_wisconsin",
        ],
    )

    progress = load_progress(tmp_path)

    assert progress["experiment"] == "smoke_test"
    assert progress["completed_datasets"] == [
        "breast_cancer_wisconsin",
    ]
    assert progress["last_updated_utc"]


def test_load_progress_rejects_missing_progress_file(
    tmp_path: Path,
) -> None:
    """Require progress.json to exist."""
    with pytest.raises(
        ResultError,
        match="Progress file does not exist",
    ):
        load_progress(tmp_path)


def test_load_progress_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Reject malformed progress JSON."""
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ResultError,
        match="invalid JSON",
    ):
        load_progress(tmp_path)


def test_load_progress_rejects_non_object_json(
    tmp_path: Path,
) -> None:
    """Require progress JSON to contain an object."""
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        '["sample_dataset"]',
        encoding="utf-8",
    )

    with pytest.raises(
        ResultError,
        match="must contain a JSON object",
    ):
        load_progress(tmp_path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "experiment",
        "completed_datasets",
        "last_updated_utc",
    ],
)
def test_load_progress_rejects_missing_required_fields(
    tmp_path: Path,
    missing_field: str,
) -> None:
    """Require all progress fields."""
    progress = {
        "experiment": "smoke_test",
        "completed_datasets": [],
        "last_updated_utc": (
            "2026-07-17T18:30:15+00:00"
        ),
    }

    del progress[missing_field]

    progress_path = tmp_path / "progress.json"

    with progress_path.open(
        "w",
        encoding="utf-8",
    ) as progress_file:
        json.dump(progress, progress_file)

    with pytest.raises(
        ResultError,
        match="missing required fields",
    ):
        load_progress(tmp_path)