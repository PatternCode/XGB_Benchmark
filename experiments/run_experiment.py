"""Command-line entry point for XGB Benchmark experiments."""

import argparse
from pathlib import Path
from typing import Any, Sequence

from benchmark.config import load_config
from benchmark.experiment import run_experiment
from benchmark.results import (
    create_run_directory,
    write_experiment_output,
    write_progress,
)


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

    config = load_config(args.config)

    experiment_name = config["experiment"]["name"]

    run_directory = create_run_directory(
        output_directory=config["output"]["directory"],
        experiment_name=experiment_name,
    )

    completed_datasets: list[str] = []

    def write_dataset_checkpoint(
        dataset_name: str,
        experiment_output: dict[
            str,
            list[dict[str, Any]],
        ],
    ) -> None:
        """Write cumulative output after a dataset completes."""
        write_experiment_output(
            experiment_output=experiment_output,
            run_directory=run_directory,
            experiment_name=experiment_name,
            config=config,
        )

        completed_datasets.append(dataset_name)

        write_progress(
            run_directory=run_directory,
            experiment_name=experiment_name,
            completed_datasets=completed_datasets,
        )

        print(
            "Dataset checkpoint saved: "
            f"{dataset_name}"
        )

    experiment_output = run_experiment(
        config,
        on_dataset_complete=write_dataset_checkpoint,
    )

    saved_paths = write_experiment_output(
        experiment_output=experiment_output,
        run_directory=run_directory,
        experiment_name=experiment_name,
        config=config,
    )

    print(
        "Experiment completed successfully: "
        f"{experiment_name}"
    )
    print(
        "Result rows: "
        f"{len(experiment_output['results'])}"
    )
    print(
        "Ranking rows: "
        f"{len(experiment_output['rankings'])}"
    )
    print(f"Run directory: {saved_paths['run_directory']}")
    print(f"Results saved to: {saved_paths['results']}")
    print(f"Rankings saved to: {saved_paths['rankings']}")

    return saved_paths["run_directory"]


if __name__ == "__main__":
    main()