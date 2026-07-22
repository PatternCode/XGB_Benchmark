"""Command-line entry point for XGB Benchmark experiments."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from benchmark.config import load_config
from benchmark.experiment import run_experiment
from benchmark.results import (
    create_run_directory,
    load_experiment_output,
    load_progress,
    load_run_metadata,
    validate_resume_configuration,
    write_experiment_output,
    write_progress,
)


ExperimentOutput = dict[str, list[dict[str, Any]]]


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {remaining_seconds} s"
    if minutes:
        return f"{minutes} min {remaining_seconds} s"
    return f"{remaining_seconds} s"


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume leakage-free XGB Benchmark experiments."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a benchmark YAML configuration file.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        help=(
            "Existing incomplete run directory to resume. "
            "Completed datasets are skipped."
        ),
    )
    return parser.parse_args(arguments)


def _prepare_new_run(
    *,
    config: dict[str, Any],
    experiment_name: str,
) -> tuple[Path, list[str], ExperimentOutput, str, float]:
    """Create state for a new experiment run."""
    run_directory = create_run_directory(
        output_directory=config["output"]["directory"],
        experiment_name=experiment_name,
    )

    started_at_utc = datetime.now(timezone.utc).isoformat()

    return (
        run_directory,
        [],
        {
            "results": [],
            "rankings": [],
            "selected_features": [],
        },
        started_at_utc,
        0.0,
    )


def _prepare_resumed_run(
    *,
    run_directory: Path,
    config: dict[str, Any],
) -> tuple[Path, list[str], ExperimentOutput, str, float]:
    """Load and validate state for an incomplete experiment run."""
    progress = load_progress(run_directory)
    metadata = load_run_metadata(run_directory)

    validate_resume_configuration(
        config=config,
        metadata=metadata,
        progress=progress,
    )

    completed_datasets = list(progress["completed_datasets"])
    existing_output = load_experiment_output(
        run_directory,
        completed_datasets=completed_datasets,
    )

    return (
        run_directory,
        completed_datasets,
        existing_output,
        metadata["started_at_utc"],
        float(metadata["runtime_seconds"]),
    )


def main(
    arguments: Sequence[str] | None = None,
) -> Path:
    """Run or resume an experiment with dataset-level checkpoints."""
    args = parse_arguments(arguments)

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    if args.resume_run is None:
        (
            run_directory,
            completed_datasets,
            initial_output,
            started_at_utc,
            previous_runtime_seconds,
        ) = _prepare_new_run(
            config=config,
            experiment_name=experiment_name,
        )
        is_resumed_run = False
    else:
        (
            run_directory,
            completed_datasets,
            initial_output,
            started_at_utc,
            previous_runtime_seconds,
        ) = _prepare_resumed_run(
            run_directory=args.resume_run,
            config=config,
        )
        is_resumed_run = True

        completed_text = (
            ", ".join(completed_datasets)
            if completed_datasets
            else "none"
        )
        print(
            f"Resuming run: {run_directory}\n"
            f"Completed datasets: {completed_text}",
            flush=True,
        )

    session_start = perf_counter()

    def current_total_runtime() -> float:
        """Return runtime accumulated across all run sessions."""
        return (
            previous_runtime_seconds
            + perf_counter()
            - session_start
        )

    def write_dataset_checkpoint(
        dataset_name: str,
        experiment_output: ExperimentOutput,
    ) -> None:
        """Write cumulative output after a dataset completes."""
        checkpoint_runtime = current_total_runtime()

        write_experiment_output(
            experiment_output=experiment_output,
            run_directory=run_directory,
            experiment_name=experiment_name,
            config=config,
            started_at_utc=started_at_utc,
            completed_at_utc=None,
            runtime_seconds=checkpoint_runtime,
        )

        if dataset_name not in completed_datasets:
            completed_datasets.append(dataset_name)

        write_progress(
            run_directory=run_directory,
            experiment_name=experiment_name,
            completed_datasets=completed_datasets,
        )

        print(
            f"Dataset checkpoint saved: {dataset_name}",
            flush=True,
        )

    if is_resumed_run:
        experiment_output = run_experiment(
            config,
            on_dataset_complete=write_dataset_checkpoint,
            completed_datasets=completed_datasets,
            initial_output=initial_output,
        )
    else:
        experiment_output = run_experiment(
            config,
            on_dataset_complete=write_dataset_checkpoint,
        )

    completed_at_utc = datetime.now(timezone.utc).isoformat()
    total_runtime = current_total_runtime()

    saved_paths = write_experiment_output(
        experiment_output=experiment_output,
        run_directory=run_directory,
        experiment_name=experiment_name,
        config=config,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        runtime_seconds=total_runtime,
    )

    print(
        f"Experiment completed successfully: {experiment_name}",
        flush=True,
    )
    print(
        f"Total runtime: {_format_duration(total_runtime)}",
        flush=True,
    )
    print(
        f"Result rows: {len(experiment_output['results'])}",
        flush=True,
    )
    print(
        f"Ranking rows: {len(experiment_output['rankings'])}",
        flush=True,
    )
    print(
        f"Run directory: {saved_paths['run_directory']}",
        flush=True,
    )
    print(
        f"Results saved to: {saved_paths['results']}",
        flush=True,
    )
    print(
        f"Rankings saved to: {saved_paths['rankings']}",
        flush=True,
    )

    return saved_paths["run_directory"]


if __name__ == "__main__":
    main()