"""Save benchmark experiment results."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class ResultError(Exception):
    """Raised when benchmark results cannot be saved."""


def results_to_frame(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert experiment result records into a pandas DataFrame."""
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
    """Save experiment results as a CSV file."""
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


def _get_git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        completed_process = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    commit = completed_process.stdout.strip()

    return commit or None


def _build_run_metadata(
    *,
    experiment_name: str,
    experiment_output: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Create reproducibility metadata for one experiment run."""
    enabled_datasets = [
        dataset_name
        for dataset_name, settings in config["datasets"].items()
        if settings.get("enabled", False)
    ]

    enabled_models = [
        model_name
        for model_name, settings in config["models"].items()
        if settings.get("enabled", False)
    ]

    return {
        "experiment": experiment_name,
        "timestamp_utc": timestamp,
        "random_seed": config["experiment"]["random_seed"],
        "enabled_datasets": enabled_datasets,
        "enabled_models": enabled_models,
        "n_result_rows": len(experiment_output["results"]),
        "n_ranking_rows": len(experiment_output["rankings"]),
        "n_selected_feature_rows": len(
            experiment_output["selected_features"]
        ),
        "python_version": platform.python_version(),
        "git_commit": _get_git_commit(),
        "configuration": config,
    }


def save_experiment_output(
    experiment_output: dict[str, list[dict[str, Any]]],
    output_directory: str | Path,
    experiment_name: str,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Save benchmark results and reproducibility records."""
    if (
        not isinstance(experiment_name, str)
        or not experiment_name.strip()
    ):
        raise ResultError(
            "experiment_name must be a non-empty string."
        )

    if not isinstance(experiment_output, dict):
        raise ResultError(
            "experiment_output must be a dictionary."
        )

    if "results" not in experiment_output:
        raise ResultError(
            "experiment_output is missing 'results'."
        )

    if "rankings" not in experiment_output:
        raise ResultError(
            "experiment_output is missing 'rankings'."
        )

    if "selected_features" not in experiment_output:
        raise ResultError(
            "experiment_output is missing 'selected_features'."
        )

    results_frame = results_to_frame(
        experiment_output["results"]
    )

    rankings_frame = results_to_frame(
        experiment_output["rankings"]
    )

    selected_features_frame = results_to_frame(
        experiment_output["selected_features"]
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    metadata = _build_run_metadata(
        experiment_name=experiment_name,
        experiment_output=experiment_output,
        config=config,
        timestamp=timestamp,
    )

    run_directory = (
        Path(output_directory)
        / f"{experiment_name}_{timestamp}"
    )

    results_path = run_directory / "results.csv"
    rankings_path = run_directory / "rankings.csv"
    selected_features_path = (
        run_directory / "selected_features.csv"
    )
    metadata_path = run_directory / "run_metadata.json"

    try:
        run_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        results_frame.to_csv(
            results_path,
            index=False,
        )

        rankings_frame.to_csv(
            rankings_path,
            index=False,
        )

        selected_features_frame.to_csv(
            selected_features_path,
            index=False,
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as error:
        raise ResultError(
            f"Could not save experiment output in "
            f"'{run_directory}': {error}"
        ) from error

    return {
        "run_directory": run_directory,
        "results": results_path,
        "rankings": rankings_path,
        "selected_features": selected_features_path,
        "metadata": metadata_path,
    }