"""Command-line entry point for XGB Benchmark experiments."""

import argparse
from pathlib import Path
from typing import Sequence

from benchmark.config import load_config
from benchmark.experiment import run_experiment
from benchmark.results import save_experiment_output


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
    """Run an experiment and save its output files."""
    args = parse_arguments(arguments)

    config = load_config(args.config)

    experiment_output = run_experiment(config)

    saved_paths = save_experiment_output(
        experiment_output=experiment_output,
        output_directory=config["output"]["directory"],
        experiment_name=config["experiment"]["name"],
        config=config,
)

    print(
        "Experiment completed successfully: "
        f"{config['experiment']['name']}"
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