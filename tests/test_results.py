"""Tests for benchmark result conversion and saving."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from benchmark.results import (
    ResultError,
    results_to_frame,
    save_results,
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