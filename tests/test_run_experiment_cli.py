"""Tests for the experiment command-line entry point."""

from pathlib import Path
from typing import Any

import pytest

from experiments import run_experiment as run_experiment_cli


@pytest.fixture
def experiment_output() -> dict[str, list[dict[str, Any]]]:
    """Return a minimal valid experiment output."""
    return {
        "results": [
            {
                "dataset": "breast_cancer_wisconsin",
                "accuracy": 0.95,
            }
        ],
        "rankings": [
            {
                "dataset": "breast_cancer_wisconsin",
                "feature": "mean_radius",
                "rank": 1,
            }
        ],
        "selected_features": [
            {
                "dataset": "breast_cancer_wisconsin",
                "feature": "mean_radius",
            }
        ],
    }


def test_main_writes_dataset_checkpoint_and_final_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_output: dict[str, list[dict[str, Any]]],
) -> None:
    """Write cumulative output during and after execution."""
    config = {
        "experiment": {
            "name": "smoke_test",
        },
        "output": {
            "directory": str(tmp_path),
        },
    }

    run_directory = tmp_path / "smoke_test_run"

    write_calls: list[
        dict[str, list[dict[str, Any]]]
    ] = []

    def fake_load_config(
        config_path: Path,
    ) -> dict[str, Any]:
        assert config_path == Path("configs/smoke_test.yaml")
        return config

    def fake_create_run_directory(
        *,
        output_directory: str | Path,
        experiment_name: str,
    ) -> Path:
        assert output_directory == str(tmp_path)
        assert experiment_name == "smoke_test"
        return run_directory

    def fake_write_experiment_output(
        *,
        experiment_output: dict[
            str,
            list[dict[str, Any]],
        ],
        run_directory: str | Path,
        experiment_name: str,
        config: dict[str, Any],
    ) -> dict[str, Path]:
        write_calls.append(experiment_output)

        return {
            "run_directory": Path(run_directory),
            "results": Path(run_directory) / "results.csv",
            "rankings": Path(run_directory) / "rankings.csv",
            "selected_features": (
                Path(run_directory)
                / "selected_features.csv"
            ),
            "metadata": (
                Path(run_directory)
                / "run_metadata.json"
            ),
        }

    def fake_run_experiment(
        received_config: dict[str, Any],
        on_dataset_complete: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert received_config is config
        assert on_dataset_complete is not None

        on_dataset_complete(
            "breast_cancer_wisconsin",
            experiment_output,
        )

        return experiment_output

    monkeypatch.setattr(
        run_experiment_cli,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "create_run_directory",
        fake_create_run_directory,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_experiment_output",
        fake_write_experiment_output,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "run_experiment",
        fake_run_experiment,
    )

    returned_directory = run_experiment_cli.main(
        [
            "--config",
            "configs/smoke_test.yaml",
        ]
    )

    assert returned_directory == run_directory
    assert len(write_calls) == 2
    assert write_calls[0] is experiment_output
    assert write_calls[1] is experiment_output


def test_main_creates_run_directory_before_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_output: dict[str, list[dict[str, Any]]],
) -> None:
    """Create the run directory before experiment execution."""
    config = {
        "experiment": {
            "name": "smoke_test",
        },
        "output": {
            "directory": str(tmp_path),
        },
    }

    call_order: list[str] = []
    run_directory = tmp_path / "smoke_test_run"

    def fake_load_config(
        config_path: Path,
    ) -> dict[str, Any]:
        return config

    def fake_create_run_directory(
        *,
        output_directory: str | Path,
        experiment_name: str,
    ) -> Path:
        call_order.append("create_run_directory")
        return run_directory

    def fake_run_experiment(
        received_config: dict[str, Any],
        on_dataset_complete: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        call_order.append("run_experiment")
        return experiment_output

    def fake_write_experiment_output(
        *,
        experiment_output: dict[
            str,
            list[dict[str, Any]],
        ],
        run_directory: str | Path,
        experiment_name: str,
        config: dict[str, Any],
    ) -> dict[str, Path]:
        return {
            "run_directory": Path(run_directory),
            "results": Path(run_directory) / "results.csv",
            "rankings": Path(run_directory) / "rankings.csv",
            "selected_features": (
                Path(run_directory)
                / "selected_features.csv"
            ),
            "metadata": (
                Path(run_directory)
                / "run_metadata.json"
            ),
        }

    monkeypatch.setattr(
        run_experiment_cli,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "create_run_directory",
        fake_create_run_directory,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "run_experiment",
        fake_run_experiment,
    )
    monkeypatch.setattr(
        run_experiment_cli,
        "write_experiment_output",
        fake_write_experiment_output,
    )

    run_experiment_cli.main(
        [
            "--config",
            "configs/smoke_test.yaml",
        ]
    )

    assert call_order == [
        "create_run_directory",
        "run_experiment",
    ]