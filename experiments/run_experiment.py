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
    write_experiment_output,
    write_progress,
)


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
        description="Run leakage-free XGB Benchmark experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a benchmark YAML configuration file.",
    )
    return parser.parse_args(arguments)


def main(
    arguments: Sequence[str] | None = None,
) -> Path:
    """Run an experiment with dataset-level checkpointing."""
    args = parse_arguments(arguments)

    started_at = datetime.now(timezone.utc)
    run_start = perf_counter()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    run_directory = create_run_directory(
        output_directory=config["output"]["directory"],
        experiment_name=experiment_name,
    )

    completed_datasets: list[str] = []

    def write_dataset_checkpoint(
        dataset_name: str,
        experiment_output: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Write cumulative output after a dataset completes."""
        checkpoint_runtime = perf_counter() - run_start

        write_experiment_output(
            experiment_output=experiment_output,
            run_directory=run_directory,
            experiment_name=experiment_name,
            config=config,
            started_at_utc=started_at.isoformat(),
            completed_at_utc=None,
            runtime_seconds=checkpoint_runtime,
        )

        completed_datasets.append(dataset_name)

        write_progress(
            run_directory=run_directory,
            experiment_name=experiment_name,
            completed_datasets=completed_datasets,
        )

        print(f"Dataset checkpoint saved: {dataset_name}")

    experiment_output = run_experiment(
        config,
        on_dataset_complete=write_dataset_checkpoint,
    )

    completed_at = datetime.now(timezone.utc)
    total_runtime = perf_counter() - run_start

    saved_paths = write_experiment_output(
        experiment_output=experiment_output,
        run_directory=run_directory,
        experiment_name=experiment_name,
        config=config,
        started_at_utc=started_at.isoformat(),
        completed_at_utc=completed_at.isoformat(),
        runtime_seconds=total_runtime,
    )

    print(f"Experiment completed successfully: {experiment_name}")
    print(f"Total runtime: {_format_duration(total_runtime)}")
    print(f"Result rows: {len(experiment_output['results'])}")
    print(f"Ranking rows: {len(experiment_output['rankings'])}")
    print(f"Run directory: {saved_paths['run_directory']}")
    print(f"Results saved to: {saved_paths['results']}")
    print(f"Rankings saved to: {saved_paths['rankings']}")

    return saved_paths["run_directory"]


if __name__ == "__main__":
    main()