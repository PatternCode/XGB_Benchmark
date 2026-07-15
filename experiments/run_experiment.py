"""Command-line entry point for XGB Benchmark experiments."""

import argparse
from pathlib import Path
from typing import Sequence

from benchmark.config import load_config
from benchmark.experiment import run_experiment
from benchmark.results import save_results


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
    """Run an experiment and save its results."""
    args = parse_arguments(arguments)

    config = load_config(args.config)

    results = run_experiment(config)

    results_path = save_results(
        results=results,
        output_directory=config["output"]["directory"],
        experiment_name=config["experiment"]["name"],
    )

    print(
        f"Experiment completed successfully: "
        f"{config['experiment']['name']}"
    )
    print(f"Result rows: {len(results)}")
    print(f"Results saved to: {results_path}")

    return results_path


if __name__ == "__main__":
    main()