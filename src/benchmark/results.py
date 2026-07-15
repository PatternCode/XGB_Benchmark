"""Save benchmark experiment results."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class ResultError(Exception):
    """Raised when benchmark results cannot be saved."""


def results_to_frame(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert experiment result records into a pandas DataFrame.

    Parameters
    ----------
    results
        Flat result dictionaries returned by ``run_experiment()``.

    Returns
    -------
    pandas.DataFrame
        Tabular experiment results.

    Raises
    ------
    ResultError
        If the result collection is empty or invalid.
    """
    if not isinstance(results, list):
        raise ResultError(
            "results must be provided as a list."
        )

    if not results:
        raise ResultError(
            "results must not be empty."
        )

    if not all(isinstance(result, dict) for result in results):
        raise ResultError(
            "Every result record must be a dictionary."
        )

    results_frame = pd.DataFrame(results)

    if results_frame.empty:
        raise ResultError(
            "Could not create a non-empty results table."
        )

    return results_frame


def save_results(
    results: list[dict[str, Any]],
    output_directory: str | Path,
    experiment_name: str,
) -> Path:
    """Save experiment results as a CSV file.

    A separate timestamped directory is created for every run so that
    previous results are not overwritten.

    Parameters
    ----------
    results
        Result records returned by ``run_experiment()``.
    output_directory
        Parent directory for benchmark runs.
    experiment_name
        Name used to identify the experiment run.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.

    Raises
    ------
    ResultError
        If the inputs are invalid or the file cannot be written.
    """
    if (
        not isinstance(experiment_name, str)
        or not experiment_name.strip()
    ):
        raise ResultError(
            "experiment_name must be a non-empty string."
        )

    results_frame = results_to_frame(results)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    run_directory = (
        Path(output_directory)
        / f"{experiment_name}_{timestamp}"
    )

    results_path = run_directory / "results.csv"

    try:
        run_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        results_frame.to_csv(
            results_path,
            index=False,
        )
    except OSError as error:
        raise ResultError(
            f"Could not save experiment results to "
            f"'{results_path}': {error}"
        ) from error

    return results_path