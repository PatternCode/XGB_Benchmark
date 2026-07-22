"""Save and load benchmark experiment results."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ExperimentOutput = dict[str, list[dict[str, Any]]]


class ResultError(Exception):
    """Raised when benchmark results cannot be saved or loaded."""


def results_to_frame(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert experiment result records into a pandas DataFrame."""
    if not isinstance(results, list):
        raise ResultError("results must be provided as a list.")
    if not results:
        raise ResultError("results must not be empty.")
    if not all(isinstance(result, dict) for result in results):
        raise ResultError("Every result record must be a dictionary.")

    results_frame = pd.DataFrame(results)
    if results_frame.empty:
        raise ResultError("Could not create a non-empty results table.")

    return results_frame


def save_results(
    results: list[dict[str, Any]],
    output_directory: str | Path,
    experiment_name: str,
) -> Path:
    """Save experiment results as a CSV file."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ResultError("experiment_name must be a non-empty string.")

    results_frame = results_to_frame(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = Path(output_directory) / f"{experiment_name}_{timestamp}"
    results_path = run_directory / "results.csv"

    try:
        run_directory.mkdir(parents=True, exist_ok=False)
        results_frame.to_csv(results_path, index=False)
    except OSError as error:
        raise ResultError(
            f"Could not save experiment results to '{results_path}': {error}"
        ) from error

    return results_path


def _get_git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        completed_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = completed_process.stdout.strip()
    return commit or None


def _build_run_metadata(
    *,
    experiment_name: str,
    experiment_output: ExperimentOutput,
    config: dict[str, Any],
    timestamp: str,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    runtime_seconds: float | None = None,
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
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime_seconds": runtime_seconds,
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


def create_run_directory(
    output_directory: str | Path,
    experiment_name: str,
) -> Path:
    """Create and return a timestamped experiment run directory."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ResultError("experiment_name must be a non-empty string.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = Path(output_directory) / f"{experiment_name}_{timestamp}"

    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ResultError(
            f"Could not create experiment run directory "
            f"'{run_directory}': {error}"
        ) from error

    return run_directory


def write_experiment_output(
    *,
    experiment_output: ExperimentOutput,
    run_directory: str | Path,
    experiment_name: str,
    config: dict[str, Any],
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Path]:
    """Write experiment output files into an existing run directory."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ResultError("experiment_name must be a non-empty string.")
    if not isinstance(experiment_output, dict):
        raise ResultError("experiment_output must be a dictionary.")

    for required_key in ("results", "rankings", "selected_features"):
        if required_key not in experiment_output:
            raise ResultError(
                f"experiment_output is missing '{required_key}'."
            )

    results_frame = results_to_frame(experiment_output["results"])
    rankings_frame = results_to_frame(experiment_output["rankings"])
    selected_features_frame = results_to_frame(
        experiment_output["selected_features"]
    )

    run_directory = Path(run_directory)
    if not run_directory.is_dir():
        raise ResultError(
            f"Run directory does not exist: '{run_directory}'."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    metadata = _build_run_metadata(
        experiment_name=experiment_name,
        experiment_output=experiment_output,
        config=config,
        timestamp=timestamp,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        runtime_seconds=runtime_seconds,
    )

    results_path = run_directory / "results.csv"
    rankings_path = run_directory / "rankings.csv"
    selected_features_path = run_directory / "selected_features.csv"
    metadata_path = run_directory / "run_metadata.json"

    try:
        results_frame.to_csv(results_path, index=False)
        rankings_frame.to_csv(rankings_path, index=False)
        selected_features_frame.to_csv(
            selected_features_path,
            index=False,
        )
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as error:
        raise ResultError(
            f"Could not write experiment output in "
            f"'{run_directory}': {error}"
        ) from error

    return {
        "run_directory": run_directory,
        "results": results_path,
        "rankings": rankings_path,
        "selected_features": selected_features_path,
        "metadata": metadata_path,
    }


def save_experiment_output(
    experiment_output: ExperimentOutput,
    output_directory: str | Path,
    experiment_name: str,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Create a run directory and save experiment output files."""
    run_directory = create_run_directory(
        output_directory=output_directory,
        experiment_name=experiment_name,
    )

    return write_experiment_output(
        experiment_output=experiment_output,
        run_directory=run_directory,
        experiment_name=experiment_name,
        config=config,
    )


def _validate_completed_datasets(
    completed_datasets: object,
) -> list[str]:
    """Validate and return completed dataset names."""
    if not isinstance(completed_datasets, list):
        raise ResultError("completed_datasets must be provided as a list.")

    if not all(
        isinstance(dataset_name, str) and dataset_name.strip()
        for dataset_name in completed_datasets
    ):
        raise ResultError(
            "Every completed dataset name must be a non-empty string."
        )

    if len(completed_datasets) != len(set(completed_datasets)):
        raise ResultError(
            "completed_datasets must not contain duplicates."
        )

    return completed_datasets


def write_progress(
    *,
    run_directory: str | Path,
    experiment_name: str,
    completed_datasets: list[str],
) -> Path:
    """Write dataset completion progress for an experiment run."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ResultError("experiment_name must be a non-empty string.")

    validated_datasets = _validate_completed_datasets(completed_datasets)
    run_directory = Path(run_directory)

    if not run_directory.is_dir():
        raise ResultError(
            f"Run directory does not exist: '{run_directory}'."
        )

    progress_path = run_directory / "progress.json"
    progress = {
        "experiment": experiment_name,
        "completed_datasets": validated_datasets,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with progress_path.open("w", encoding="utf-8") as progress_file:
            json.dump(
                progress,
                progress_file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as error:
        raise ResultError(
            f"Could not write experiment progress to "
            f"'{progress_path}': {error}"
        ) from error

    return progress_path


def load_progress(
    run_directory: str | Path,
) -> dict[str, Any]:
    """Load and validate dataset completion progress."""
    run_directory = Path(run_directory)

    if not run_directory.is_dir():
        raise ResultError(
            f"Run directory does not exist: '{run_directory}'."
        )

    progress_path = run_directory / "progress.json"
    if not progress_path.is_file():
        raise ResultError(
            f"Progress file does not exist: '{progress_path}'."
        )

    try:
        with progress_path.open("r", encoding="utf-8") as progress_file:
            progress = json.load(progress_file)
    except OSError as error:
        raise ResultError(
            f"Could not read experiment progress from "
            f"'{progress_path}': {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ResultError(
            f"Progress file contains invalid JSON: '{progress_path}'."
        ) from error

    if not isinstance(progress, dict):
        raise ResultError("Progress file must contain a JSON object.")

    required_fields = {
        "experiment",
        "completed_datasets",
        "last_updated_utc",
    }
    missing_fields = required_fields - set(progress)

    if missing_fields:
        missing_fields_text = ", ".join(sorted(missing_fields))
        raise ResultError(
            "Progress file is missing required fields: "
            f"{missing_fields_text}."
        )

    experiment_name = progress["experiment"]
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ResultError("Progress experiment must be a non-empty string.")

    progress["completed_datasets"] = _validate_completed_datasets(
        progress["completed_datasets"]
    )

    last_updated_utc = progress["last_updated_utc"]
    if not isinstance(last_updated_utc, str) or not last_updated_utc.strip():
        raise ResultError(
            "Progress last_updated_utc must be a non-empty string."
        )

    return progress


def load_run_metadata(
    run_directory: str | Path,
) -> dict[str, Any]:
    """Load and validate metadata from an existing experiment run."""
    run_directory = Path(run_directory)

    if not run_directory.is_dir():
        raise ResultError(
            f"Run directory does not exist: '{run_directory}'."
        )

    metadata_path = run_directory / "run_metadata.json"
    if not metadata_path.is_file():
        raise ResultError(
            f"Run metadata file does not exist: '{metadata_path}'."
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except OSError as error:
        raise ResultError(
            f"Could not read run metadata from "
            f"'{metadata_path}': {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ResultError(
            f"Run metadata contains invalid JSON: '{metadata_path}'."
        ) from error

    if not isinstance(metadata, dict):
        raise ResultError("Run metadata must contain a JSON object.")

    required_fields = {
        "experiment",
        "started_at_utc",
        "completed_at_utc",
        "runtime_seconds",
        "configuration",
    }
    missing_fields = required_fields - set(metadata)

    if missing_fields:
        missing_fields_text = ", ".join(sorted(missing_fields))
        raise ResultError(
            "Run metadata is missing required fields: "
            f"{missing_fields_text}."
        )

    experiment_name = metadata["experiment"]
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ResultError(
            "Run metadata experiment must be a non-empty string."
        )

    configuration = metadata["configuration"]
    if not isinstance(configuration, dict):
        raise ResultError(
            "Run metadata configuration must be a dictionary."
        )

    started_at_utc = metadata["started_at_utc"]
    if not isinstance(started_at_utc, str) or not started_at_utc.strip():
        raise ResultError(
            "Run metadata started_at_utc must be a non-empty string."
        )

    completed_at_utc = metadata["completed_at_utc"]
    if completed_at_utc is not None and (
        not isinstance(completed_at_utc, str)
        or not completed_at_utc.strip()
    ):
        raise ResultError(
            "Run metadata completed_at_utc must be null or "
            "a non-empty string."
        )

    runtime_seconds = metadata["runtime_seconds"]
    if (
        not isinstance(runtime_seconds, (int, float))
        or isinstance(runtime_seconds, bool)
        or runtime_seconds < 0
    ):
        raise ResultError(
            "Run metadata runtime_seconds must be a non-negative number."
        )

    return metadata


def validate_resume_configuration(
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    progress: dict[str, Any],
) -> None:
    """Ensure an existing run is compatible with the current config."""
    experiment_name = config["experiment"]["name"]

    if metadata["experiment"] != experiment_name:
        raise ResultError(
            "The experiment name in run_metadata.json does not match "
            "the supplied configuration."
        )

    if progress["experiment"] != experiment_name:
        raise ResultError(
            "The experiment name in progress.json does not match "
            "the supplied configuration."
        )

    if metadata["configuration"] != config:
        raise ResultError(
            "The supplied configuration does not match the configuration "
            "stored in the resumed run."
        )

    enabled_datasets = {
        dataset_name
        for dataset_name, settings in config["datasets"].items()
        if settings.get("enabled", False)
    }

    unknown_completed_datasets = (
        set(progress["completed_datasets"]) - enabled_datasets
    )

    if unknown_completed_datasets:
        unknown_text = ", ".join(
            sorted(unknown_completed_datasets)
        )
        raise ResultError(
            "progress.json contains completed datasets that are not "
            f"enabled in the supplied configuration: {unknown_text}."
        )

    if metadata["completed_at_utc"] is not None:
        raise ResultError(
            "The selected run is already marked as completed and "
            "cannot be resumed."
        )


def _load_csv_records(
    path: Path,
    description: str,
) -> list[dict[str, Any]]:
    """Load CSV rows as dictionaries while restoring missing values."""
    if not path.is_file():
        raise ResultError(
            f"{description} file does not exist: '{path}'."
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ResultError(
            f"Could not read {description} from '{path}': {error}"
        ) from error

    if frame.empty:
        raise ResultError(
            f"{description} file must contain at least one row: '{path}'."
        )

    object_frame = frame.astype(object).where(
        pd.notna(frame),
        None,
    )
    return object_frame.to_dict(orient="records")


def load_experiment_output(
    run_directory: str | Path,
    completed_datasets: list[str] | None = None,
) -> ExperimentOutput:
    """Load saved output from an existing experiment run.

    When completed_datasets is provided, rows belonging to datasets not
    recorded as complete are removed. This prevents partially checkpointed
    or inconsistent rows from being duplicated during resume.
    """
    run_directory = Path(run_directory)

    if not run_directory.is_dir():
        raise ResultError(
            f"Run directory does not exist: '{run_directory}'."
        )

    experiment_output: ExperimentOutput = {
        "results": _load_csv_records(
            run_directory / "results.csv",
            "results",
        ),
        "rankings": _load_csv_records(
            run_directory / "rankings.csv",
            "rankings",
        ),
        "selected_features": _load_csv_records(
            run_directory / "selected_features.csv",
            "selected features",
        ),
    }

    if completed_datasets is None:
        return experiment_output

    validated_datasets = _validate_completed_datasets(
        completed_datasets
    )
    completed_dataset_set = set(validated_datasets)

    for collection_name, records in experiment_output.items():
        for record in records:
            if "dataset" not in record:
                raise ResultError(
                    f"Saved {collection_name} records must include "
                    "a 'dataset' field."
                )

        experiment_output[collection_name] = [
            record
            for record in records
            if record["dataset"] in completed_dataset_set
        ]

    if validated_datasets:
        for collection_name, records in experiment_output.items():
            represented_datasets = {
                str(record["dataset"])
                for record in records
            }
            missing_datasets = (
                completed_dataset_set - represented_datasets
            )

            if missing_datasets:
                missing_text = ", ".join(
                    sorted(missing_datasets)
                )
                raise ResultError(
                    f"Saved {collection_name} output is missing rows "
                    f"for completed datasets: {missing_text}."
                )

    return experiment_output